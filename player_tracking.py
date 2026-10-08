from ultralytics import YOLO
import cv2
import os

model = YOLO("yolo11n.pt")

input_video = "final_footwork_video.mp4"
output_video = "tracked_footwork_video.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

frame_count = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.50,
        verbose=False
    )

    annotated_frame = results[0].plot()

    writer.write(annotated_frame)

    frame_count += 1

    print("Tracked frame:", frame_count)

cap.release()
writer.release()

print("PLAYER TRACKING COMPLETED!")
print("Frames tracked:", frame_count)
print("Output:", output_video)
