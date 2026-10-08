import cv2
import os

video_path = "input_video.mp4"
output_folder = "footwork_frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_number = 0
saved_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % 3 == 0:
        filename = os.path.join(
            output_folder,
            f"frame_{saved_number:04d}.jpg"
        )
        cv2.imwrite(filename, frame)
        saved_number += 1

    frame_number += 1

cap.release()

print("Total video frames:", frame_number)
print("Extracted frames:", saved_number)
print("Saved in:", output_folder)