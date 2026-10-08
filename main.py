import cv2
import mediapipe as mp
import os

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_full.task"
    ),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = PoseLandmarker.create_from_options(options)

input_folder = "frames"
output_folder = "processed_frames"

os.makedirs(output_folder, exist_ok=True)

for filename in sorted(os.listdir(input_folder)):

    if not filename.endswith(".jpg"):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    image = cv2.imread(input_path)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    result = landmarker.detect(mp_image)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        left_foot = landmarks[31]
        right_foot = landmarks[32]

        h, w, _ = image.shape

        left_x = int(left_foot.x * w)
        left_y = int(left_foot.y * h)

        right_x = int(right_foot.x * w)
        right_y = int(right_foot.y * h)

        cv2.circle(image, (left_x, left_y), 8, (0, 255, 0), -1)
        cv2.circle(image, (right_x, right_y), 8, (0, 0, 255), -1)

        cv2.putText(
            image,
            "Left Foot",
            (left_x + 10, left_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            "Right Foot",
            (right_x + 10, right_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    cv2.imwrite(output_path, image)

    print("Processed:", filename)

landmarker.close()

print("All 200 frames processed successfully!")