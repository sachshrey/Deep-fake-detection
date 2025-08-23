import cv2
import numpy as np
import mediapipe as mp
import pywt
import os
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- 1. Landmark Detection and Mask Creation ---
mp_face_mesh = mp.solutions.face_mesh

LANDMARK_GROUPS = {
    "left_eye": list(range(33, 133)),
    "right_eye": list(range(263, 362)),
    "lips": list(range(61, 81)) + list(range(291, 311)),
    "jaw": list(range(0, 17)),
    "face_oval": list(range(0, 17)) + list(range(17, 27)),
}

def get_attention_mask(frame, landmarks, dilate_px=7):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in LANDMARK_GROUPS.values():
        points = np.array([(int(landmarks[i][0]*w), int(landmarks[i][1]*h)) for i in region if i < len(landmarks)])
        if len(points) > 0:
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8))
    return mask

def extract_landmarks(frame, face_mesh):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        return [(lm.x, lm.y) for lm in face_landmarks.landmark]
    return None

# --- 2. Feature Extraction: Only One Wavelet Transform ---
def extract_wavelet_masked(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    # Resize for consistency
    masked = cv2.resize(masked, (128, 128))
    # Wavelet transform (level 1, Haar)
    cA, (cH, cV, cD) = pywt.dwt2(masked, 'haar')
    # Normalize cA to 0-255 and resize to 128x128
    cA_norm = cv2.normalize(cA, None, 0, 255, cv2.NORM_MINMAX)
    cA_norm = cA_norm.astype(np.uint8)
    cA_resized = cv2.resize(cA_norm, (128, 128))
    # Stack as 2-channel image: [masked, cA_resized]
    stacked = np.stack([masked, cA_resized], axis=-1)
    return stacked

# --- 3. Frame and Video Processing ---
def process_video(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_count-1, max_frames, dtype=int)
    frames = []
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            landmarks = extract_landmarks(frame, face_mesh)
            if landmarks is None:
                continue
            mask = get_attention_mask(frame, landmarks)
            if np.sum(mask) == 0:
                continue
            stacked = extract_wavelet_masked(frame, mask)
            frames.append(stacked)
    cap.release()
    return np.array(frames)  # shape: (frames, 128, 128, 2)

# --- 4. Dataset Preparation ---
def prepare_dataset(base_path, max_frames=10):
    video_paths, labels = [], []
    for label, folder in enumerate(['real', 'fake']):
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue
        for video_file in glob(os.path.join(folder_path, '*.mp4')):
            video_paths.append(video_file)
            labels.append(label)
    X, y = [], []
    for path, label in zip(video_paths, labels):
        frames = process_video(path, max_frames)
        if frames.shape[0] == 0:
            print(f"Skipped {path} (no valid frames with landmarks)")
            continue
        for frame in frames:
            X.append(frame)
            y.append(label)
    return np.array(X), np.array(y)

# --- 5. CNN Model ---
def build_cnn_model(input_shape=(128, 128, 2)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 6. Main ---
if __name__ == "__main__":
    base_path = r"C:\Users\Pavan\Downloads\FF++"  # Change to your dataset path
    X, y = prepare_dataset(base_path, max_frames=10)
    if len(X) == 0:
        print("No data extracted. Check your dataset path and video files.")
    else:
        # Normalize and split
        X = X.astype('float32') / 255.0
        y_cat = to_categorical(y, num_classes=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)
        # Build and train CNN
        model = build_cnn_model(input_shape=(128, 128, 2))
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint("best_landmark_wavelet_cnn.h5", save_best_only=True)
        ]
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=callbacks)
        loss, acc = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {acc:.4f}")