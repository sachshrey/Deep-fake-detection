import cv2
import os

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Opening video: {video_path}")
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            print(f"Finished reading video: {video_path}")
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    video_capture.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_frames_dir = os.path.join(base_dir, "data", "frames", "real")
    fake_frames_dir = os.path.join(base_dir, "data", "frames", "fake")

    real_video_path = r"C:\Users\Pavan\Downloads\FF++\real\15__kitchen_still.mp4"
    fake_video_path = r"C:\Users\Pavan\Downloads\FF++\fake\10_19__kitchen_still__IDX76N5R.mp4"
    
    extract_frames(real_video_path, real_frames_dir)
    extract_frames(fake_video_path, fake_frames_dir)