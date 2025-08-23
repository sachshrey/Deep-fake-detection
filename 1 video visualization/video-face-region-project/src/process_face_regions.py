import cv2
import os
import numpy as np
import mediapipe as mp

def process_face_region(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    mask = np.zeros(image.shape, dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Example: Draw circles at landmark points for eyes, nose, mouth, jawline
            important_indices = [
                # Left eye
                33, 133, 159, 145, 153, 154, 155, 246,
                # Right eye
                362, 263, 386, 374, 380, 381, 382, 466,
                # Nose
                1, 2, 98, 327, 168, 197, 195, 5, 4, 51, 280, 309, 291,
                # Mouth (outer and inner)
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 13, 312, 311, 310, 415,
                # Jawline/face border
                234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 447, 366, 10
            ]
            h, w, _ = image.shape
            for idx in important_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(mask, (x, y), 10, (255, 255, 255), -1)

    face_mesh.close()
    # Keep only the masked regions
    processed_face = cv2.bitwise_and(image, mask)
    return processed_face

def process_frames(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust based on extracted frame formats
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            processed_image = process_face_region(image)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_frames_folder = os.path.join(base_dir, "data", "frames", "real")
    fake_frames_folder = os.path.join(base_dir, "data", "frames", "fake")
    processed_real_folder = os.path.join(base_dir, "data", "processed_frames", "real")
    processed_fake_folder = os.path.join(base_dir, "data", "processed_frames", "fake")

    process_frames(real_frames_folder, processed_real_folder)
    process_frames(fake_frames_folder, processed_fake_folder)