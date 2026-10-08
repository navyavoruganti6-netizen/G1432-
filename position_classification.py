import cv2
import os
import numpy as np

input_folder = "processed_frames"
output_folder = "position_frames"

os.makedirs(output_folder, exist_ok=True)

# Court area
court_polygon = np.array([
    (100, 100),
    (900, 100),
    (1000, 700),
    (50, 700)
], dtype=np.int32)

for filename in sorted(os.listdir(input_folder)):

    if not filename.lower().endswith(".jpg"):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    image = cv2.imread(input_path)

    if image is None:
        continue

    h, w = image.shape[:2]

    # Scale court polygon according to image size
    polygon = np.array([
        (int(0.10 * w), int(0.15 * h)),
        (int(0.90 * w), int(0.15 * h)),
        (int(0.95 * w), int(0.95 * h)),
        (int(0.05 * w), int(0.95 * h))
    ], dtype=np.int32)

    # Draw court boundary
    cv2.polylines(image, [polygon], True, (255, 255, 0), 3)

    # Find green/red foot markers
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(
        hsv,
        np.array([35, 80, 50]),
        np.array([85, 255, 255])
    )

    red_mask1 = cv2.inRange(
        hsv,
        np.array([0, 80, 50]),
        np.array([10, 255, 255])
    )

    red_mask2 = cv2.inRange(
        hsv,
        np.array([170, 80, 50]),
        np.array([180, 255, 255])
    )

    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_points = cv2.findNonZero(green_mask)
    red_points = cv2.findNonZero(red_mask)

    points = []

    if green_points is not None:
        points.extend(green_points.reshape(-1, 2).tolist())

    if red_points is not None:
        points.extend(red_points.reshape(-1, 2).tolist())

    if points:

        foot_x = int(np.mean([p[0] for p in points]))
        foot_y = int(np.mean([p[1] for p in points]))

        # Divide court into 3 position zones
        if foot_y < 0.40 * h:
            position = "BACK COURT"
        elif foot_y < 0.70 * h:
            position = "MID COURT"
        else:
            position = "FRONT COURT"

        cv2.circle(image, (foot_x, foot_y), 10, (255, 0, 255), -1)

        cv2.putText(
            image,
            "Position: " + position,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3
        )

    else:
        cv2.putText(
            image,
            "Position: Not Detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3
        )

    cv2.imwrite(output_path, image)

    print("Processed:", filename)

print("COURT POSITION CLASSIFICATION COMPLETED!")
print("Output folder:", output_folder)
