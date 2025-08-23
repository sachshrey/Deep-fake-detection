import cv2
import numpy as np
import mediapipe as mp
import os
import random
from matplotlib import pyplot as plt

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# --- Helper Functions ---
def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        return [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
    return None

def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    for x, y in landmarks:
        cx, cy = int(x * w), int(y * h)
        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
    return frame

def crop_face(frame, landmarks):
    h, w = frame.shape[:2]
    xs = [int(x * w) for x, y in landmarks]
    ys = [int(y * h) for x, y in landmarks]
    x_min, x_max = max(0, min(xs)), min(w, max(xs))
    y_min, y_max = max(0, min(ys)), min(h, max(ys))
    if (x_max - x_min) > 10 and (y_max - y_min) > 10:
        return frame[y_min:y_max, x_min:x_max]
    return None

# --- Extract and Save Frames ---
def sample_frames_with_landmarks(video_path, label, out_dir='samples', max_frames=10):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"⚠ Video not loaded properly: {video_path}")
        return
    selected = sorted(random.sample(range(total_frames), min(max_frames, total_frames)))
    
    saved = 0
    for idx in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        landmarks = extract_landmarks(frame)
        if landmarks is None:
            continue
        
        # Save landmarks on original frame
        frame_landmarked = draw_landmarks(frame.copy(), landmarks)
        cv2.imwrite(os.path.join(out_dir, f"{label}landmark{saved}.jpg"), frame_landmarked)

        # Save cropped face
        cropped = crop_face(frame, landmarks)
        if cropped is not None:
            cv2.imwrite(os.path.join(out_dir, f"{label}cropped{saved}.jpg"), cropped)
            saved += 1

        if saved >= max_frames:
            break
    cap.release()

# --- Visualization ---
def plot_comparison(out_dir='samples', max_pairs=8):
    real_landmarks = sorted([f for f in os.listdir(out_dir) if f.startswith("real_landmark")])[:max_pairs]
    fake_landmarks = sorted([f for f in os.listdir(out_dir) if f.startswith("fake_landmark")])[:max_pairs]
    
    real_crops = sorted([f for f in os.listdir(out_dir) if f.startswith("real_cropped")])[:max_pairs]
    fake_crops = sorted([f for f in os.listdir(out_dir) if f.startswith("fake_cropped")])[:max_pairs]

    min_pairs = min(len(real_landmarks), len(fake_landmarks), len(real_crops), len(fake_crops), max_pairs)
    if min_pairs == 0:
        print("⚠ Not enough images to plot.")
        return

    plt.figure(figsize=(12, 4 * min_pairs))

    for i in range(min_pairs):
        # Landmark Comparisons
        real_img = cv2.imread(os.path.join(out_dir, real_landmarks[i]))
        fake_img = cv2.imread(os.path.join(out_dir, fake_landmarks[i]))
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)

        plt.subplot(min_pairs, 4, 4*i + 1)
        plt.imshow(real_img)
        plt.title(f"REAL - Landmarks {i}")
        plt.axis('off')

        plt.subplot(min_pairs, 4, 4*i + 2)
        plt.imshow(fake_img)
        plt.title(f"FAKE - Landmarks {i}")
        plt.axis('off')

        # Cropped Face Comparisons
        real_crop = cv2.imread(os.path.join(out_dir, real_crops[i]))
        fake_crop = cv2.imread(os.path.join(out_dir, fake_crops[i]))
        real_crop = cv2.cvtColor(real_crop, cv2.COLOR_BGR2RGB)
        fake_crop = cv2.cvtColor(fake_crop, cv2.COLOR_BGR2RGB)

        plt.subplot(min_pairs, 4, 4*i + 3)
        plt.imshow(real_crop)
        plt.title(f"REAL - Cropped Face {i}")
        plt.axis('off')

        plt.subplot(min_pairs, 4, 4*i + 4)
        plt.imshow(fake_crop)
        plt.title(f"FAKE - Cropped Face {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
def main():
    real_video = r"C:\Users\Pavan\Downloads\FF++\real\01__meeting_serious.mp4"
    fake_video = r"C:\Users\Pavan\Downloads\FF++\fake\01_11__meeting_serious__9OM3VE0Y.mp4"

    print("⏳ Processing real video...")
    sample_frames_with_landmarks(real_video, label="real", out_dir="samples", max_frames=8)

    print("⏳ Processing fake video...")
    sample_frames_with_landmarks(fake_video, label="fake", out_dir="samples", max_frames=8)

    print("📊 Plotting comparisons...")
    plot_comparison(out_dir="samples", max_pairs=8)

if __name__ == "__main__":
    main()