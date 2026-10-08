from ultralytics import YOLO
import cv2
import csv

model = YOLO("yolo11n.pt")

input_video = "final_footwork_video.mp4"
output_csv = "position_history.csv"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

frame_number = 0
rows = []

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

    if results[0].boxes is not None and len(results[0].boxes) > 0:

        box = results[0].boxes[0]
        coordinates = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = coordinates

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        rows.append([
            frame_number,
            center_x,
            center_y
        ])

    frame_number += 1

cap.release()

with open(output_csv, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Frame",
        "X_Position",
        "Y_Position"
    ])

    writer.writerows(rows)

print("POSITION HISTORY COMPLETED!")
print("Frames analyzed:", frame_number)
print("Position records:", len(rows))
print("Output:", output_csv)
