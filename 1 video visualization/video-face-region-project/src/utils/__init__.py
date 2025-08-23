def extract_frames(video_path, output_folder):
    import cv2
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

def process_face_regions(image_path, output_folder):
    import cv2
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]
        output_filename = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_filename, face_region)

def main():
    # This function can be used to call extract_frames and process_face_regions
    pass

__all__ = ['extract_frames', 'process_face_regions', 'main']