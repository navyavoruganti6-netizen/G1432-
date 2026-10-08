import cv2
import os

input_folder = "position_frames"
output_video = "final_footwork_video.mp4"

files = sorted(
    [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]
)

if not files:
    print("No frames found!")
    exit()

first_frame = cv2.imread(os.path.join(input_folder, files[0]))

height, width = first_frame.shape[:2]

fps = 10

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

for filename in files:
    frame = cv2.imread(os.path.join(input_folder, filename))

    if frame is not None:
        writer.write(frame)
        print("Added:", filename)

writer.release()

print("VIDEO CREATED SUCCESSFULLY!")
print("Output:", output_video)
