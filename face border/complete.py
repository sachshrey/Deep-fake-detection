import cv2
import numpy as np
import mediapipe as mp
import os
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def extract_face(frame):
    results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1 = max(int(bbox.xmin * w), 0)
        y1 = max(int(bbox.ymin * h), 0)
        x2 = min(int((bbox.xmin + bbox.width) * w), w)
        y2 = min(int((bbox.ymin + bbox.height) * h), h)
        face = frame[y1:y2, x1:x2]
        if face.size > 0:
            return cv2.resize(face, (128, 128))
    return None

def process_video(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_count-1, max_frames, dtype=int)
    faces = []
    for idx in tqdm(indices, desc=f"Extracting frames from {os.path.basename(video_path)}", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        face = extract_face(frame)
        if face is not None:
            faces.append(face)
    cap.release()
    return np.array(faces)

def prepare_dataset(base_path, max_frames=100):
    video_paths, labels = [], []
    for label, folder in enumerate(['real', 'fake']):
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue
        for video_file in glob(os.path.join(folder_path, '*.mp4')):
            video_paths.append(video_file)
            labels.append(label)
    X, y = [], []
    for path, label in tqdm(list(zip(video_paths, labels)), desc="Processing videos"):
        faces = process_video(path, max_frames)
        if faces.shape[0] == 0:
            print(f"Skipped {path} (no valid faces)")
            continue
        for face in faces:
            X.append(face)
            y.append(label)
    return np.array(X), np.array(y)

def build_transfer_model(input_shape=(128,128,3)):
    base = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    out = Dense(2, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    base_path = r"C:\Users\Pavan\Downloads\FF++"
    X, y = prepare_dataset(base_path, max_frames=100)
    if len(X) == 0:
        print("No data extracted. Check your dataset path and video files.")
    else:
        X = X.astype('float32') / 255.0
        y_cat = to_categorical(y, num_classes=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)
        model = build_transfer_model(input_shape=(128,128,3))
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint("best_efficientnet_face.h5", save_best_only=True)
        ]
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=callbacks)
        loss, acc = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {acc:.4f}")