import csv
import json
from pathlib import Path

import cv2
import numpy as np
from inference import get_model
import supervision as sv


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "venu-sudha/badminton-singles-players-5-yolo26n-t1"

INPUT_VIDEO = Path("video/badminton_test_10sec.mp4")

RESULTS_DIR = Path("results")
CSV_FILE = RESULTS_DIR / "badminton_coordinates_v6.csv"
JSON_FILE = RESULTS_DIR / "badminton_coordinates_v6.json"
OUTPUT_VIDEO = RESULTS_DIR / "badminton_tracking_v6.mp4"

# Court dimensions
COURT_WIDTH_M = 5.18
COURT_LENGTH_M = 13.40

# Minimum confidence for player detections
CONFIDENCE_THRESHOLD = 0.30

# ------------------------------------------------------------
# EXACT ROBoflow CALIBRATION
#
# Image/pixel points:
#   near-left
#   near-right
#   far-right
#   far-left
#
# World/court points:
#   (0, 0)
#   (5.18, 0)
#   (5.18, 13.40)
#   (0, 13.40)
# ------------------------------------------------------------

PIXEL_POINTS = np.array(
    [
        [447, 1007],
        [1482, 1007],
        [1277, 455],
        [639, 455],
    ],
    dtype=np.float32,
)

WORLD_POINTS = np.array(
    [
        [0.0, 0.0],
        [5.18, 0.0],
        [5.18, 13.40],
        [0.0, 13.40],
    ],
    dtype=np.float32,
)


# ============================================================
# SETUP
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("BADMINTON AI - LOCAL V6 PIPELINE")
print("=" * 70)

print()
print("Model:")
print(MODEL_ID)

print()
print("Input:")
print(INPUT_VIDEO)

print()
print("Calibration:")
for pixel, world in zip(PIXEL_POINTS, WORLD_POINTS):
    print(
        f"  Pixel ({pixel[0]:.0f}, {pixel[1]:.0f}) "
        f"-> Court ({world[0]:.2f}, {world[1]:.2f}) m"
    )

print()


# ============================================================
# CREATE PERSPECTIVE TRANSFORMATION
# ============================================================

# Maps image pixels -> court coordinates in metres.
TRANSFORM_MATRIX = cv2.getPerspectiveTransform(
    PIXEL_POINTS,
    WORLD_POINTS,
)

print("Perspective transformation created successfully.")
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Roboflow model...")

model = get_model(MODEL_ID)

print("MODEL LOADED SUCCESSFULLY")
print()


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(INPUT_VIDEO))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {INPUT_VIDEO}"
    )

FPS = cap.get(cv2.CAP_PROP_FPS)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video information:")
print(f"  Resolution : {FRAME_WIDTH} x {FRAME_HEIGHT}")
print(f"  FPS        : {FPS}")
print(f"  Frames     : {TOTAL_FRAMES}")
print()


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    FPS,
    (FRAME_WIDTH, FRAME_HEIGHT),
)

if not video_writer.isOpened():
    cap.release()
    raise RuntimeError(
        f"Could not create output video: {OUTPUT_VIDEO}"
    )


# ============================================================
# BYTE TRACK
# ============================================================

tracker = sv.ByteTrack()


# ============================================================
# CSV / JSON STORAGE
# ============================================================

csv_rows = []
json_rows = []

csv_file = open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8",
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow(
    [
        "frame",
        "time_seconds",
        "tracker_id",
        "pixel_x",
        "pixel_y",
        "court_x_m",
        "court_y_m",
        "confidence",
        "in_court",
    ]
)


# ============================================================
# HELPER: PIXEL -> COURT
# ============================================================

def pixel_to_court(pixel_x, pixel_y):
    """
    Convert one image pixel position into court coordinates
    using the exact Roboflow calibration.
    """

    point = np.array(
        [[[float(pixel_x), float(pixel_y)]]],
        dtype=np.float32,
    )

    transformed = cv2.perspectiveTransform(
        point,
        TRANSFORM_MATRIX,
    )

    court_x = float(transformed[0, 0, 0])
    court_y = float(transformed[0, 0, 1])

    return court_x, court_y


# ============================================================
# HELPER: CHECK COURT BOUNDS
# ============================================================

def is_in_court(court_x, court_y):
    """
    Allow a tiny numerical tolerance around the court boundary.
    """

    tolerance = 0.05

    return (
        -tolerance <= court_x <= COURT_WIDTH_M + tolerance
        and
        -tolerance <= court_y <= COURT_LENGTH_M + tolerance
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0
frames_with_players = 0
frames_with_two_players = 0
total_coordinate_rows = 0

print("Starting video processing...")
print()

while True:

    ok, frame = cap.read()

    if not ok:
        break

    frame_number += 1

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    try:
        results = model.infer(frame)

    except Exception as e:
        print(
            f"WARNING: inference failed on frame "
            f"{frame_number}: {e}"
        )

        video_writer.write(frame)
        continue

    # --------------------------------------------------------
    # EXTRACT MODEL PREDICTIONS
    # --------------------------------------------------------

    if not results:
        video_writer.write(frame)
        continue

    result = results[0]

    predictions = getattr(
        result,
        "predictions",
        None,
    )

    if predictions is None:
        video_writer.write(frame)
        continue

    xyxy_list = []
    confidence_list = []
    class_id_list = []

    for prediction in predictions:

        confidence = float(
            getattr(
                prediction,
                "confidence",
                0.0,
            )
        )

        class_name = str(
            getattr(
                prediction,
                "class_name",
                "",
            )
        ).lower()

        # Only badminton player detections.
        if class_name != "player":
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x = float(
            getattr(
                prediction,
                "x",
                0.0,
            )
        )

        y = float(
            getattr(
                prediction,
                "y",
                0.0,
            )
        )

        width = float(
            getattr(
                prediction,
                "width",
                0.0,
            )
        )

        height = float(
            getattr(
                prediction,
                "height",
                0.0,
            )
        )

        x1 = x - width / 2.0
        y1 = y - height / 2.0
        x2 = x + width / 2.0
        y2 = y + height / 2.0

        xyxy_list.append(
            [x1, y1, x2, y2]
        )

        confidence_list.append(
            confidence
        )

        class_id_list.append(0)

    # --------------------------------------------------------
    # CREATE SUPERVISION DETECTIONS
    # --------------------------------------------------------

    if xyxy_list:

        detections = sv.Detections(
            xyxy=np.array(
                xyxy_list,
                dtype=np.float32,
            ),
            confidence=np.array(
                confidence_list,
                dtype=np.float32,
            ),
            class_id=np.array(
                class_id_list,
                dtype=np.int32,
            ),
        )

    else:

        detections = sv.Detections.empty()

    # --------------------------------------------------------
    # BYTE TRACK
    # --------------------------------------------------------

    tracked_detections = tracker.update_with_detections(
        detections
    )

    # --------------------------------------------------------
    # TRACKED PLAYER COUNT
    # --------------------------------------------------------

    player_count = len(
        tracked_detections.xyxy
    )

    if player_count > 0:
        frames_with_players += 1

    if player_count >= 2:
        frames_with_two_players += 1

    # --------------------------------------------------------
    # DRAW TRACKING + COURT COORDINATES
    # --------------------------------------------------------

    output_frame = frame.copy()

    for i, box in enumerate(
        tracked_detections.xyxy
    ):

        x1, y1, x2, y2 = box

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        # Bottom-center point = player's ground/foot position.
        pixel_x = (x1 + x2) / 2.0
        pixel_y = float(y2)

        court_x, court_y = pixel_to_court(
            pixel_x,
            pixel_y,
        )

        in_court = is_in_court(
            court_x,
            court_y,
        )

        confidence = float(
            tracked_detections.confidence[i]
        )

        tracker_id = int(
            tracked_detections.tracker_id[i]
        )

        # ----------------------------------------------------
        # SAVE CSV ROW
        # ----------------------------------------------------

        csv_writer.writerow(
            [
                frame_number,
                round(
                    frame_number / FPS,
                    4,
                ),
                tracker_id,
                round(pixel_x, 2),
                round(pixel_y, 2),
                round(court_x, 4),
                round(court_y, 4),
                round(confidence, 4),
                in_court,
            ]
        )

        total_coordinate_rows += 1

        # ----------------------------------------------------
        # SAVE JSON ROW
        # ----------------------------------------------------

        json_rows.append(
            {
                "frame": frame_number,
                "time_seconds": round(
                    frame_number / FPS,
                    4,
                ),
                "tracker_id": tracker_id,
                "pixel_x": round(
                    pixel_x,
                    2,
                ),
                "pixel_y": round(
                    pixel_y,
                    2,
                ),
                "court_x_m": round(
                    court_x,
                    4,
                ),
                "court_y_m": round(
                    court_y,
                    4,
                ),
                "confidence": round(
                    confidence,
                    4,
                ),
                "in_court": in_court,
            }
        )

        # ----------------------------------------------------
        # DRAW PLAYER BOX
        # ----------------------------------------------------

        cv2.rectangle(
            output_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # Bottom-center point.
        cv2.circle(
            output_frame,
            (
                int(round(pixel_x)),
                int(round(pixel_y)),
            ),
            6,
            (0, 0, 255),
            -1,
        )

        # ----------------------------------------------------
        # DRAW TRACKER + COURT POSITION
        # ----------------------------------------------------

        label = (
            f"ID {tracker_id} "
            f"({court_x:.2f}, {court_y:.2f})m"
        )

        cv2.putText(
            output_frame,
            label,
            (
                x1,
                max(25, y1 - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    status = (
        f"Frame {frame_number}/{TOTAL_FRAMES} | "
        f"Players: {player_count}"
    )

    cv2.putText(
        output_frame,
        status,
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    video_writer.write(output_frame)

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_number % 30 == 0:

        print(
            f"Processed "
            f"{frame_number}/{TOTAL_FRAMES} frames | "
            f"tracked players: {player_count} | "
            f"coordinate rows: {total_coordinate_rows}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
video_writer.release()
csv_file.close()


# ============================================================
# SAVE JSON
# ============================================================

output_json = {
    "video": str(INPUT_VIDEO),
    "model": MODEL_ID,
    "fps": FPS,
    "total_frames": TOTAL_FRAMES,
    "resolution": [
        FRAME_WIDTH,
        FRAME_HEIGHT,
    ],
    "court": {
        "width_m": COURT_WIDTH_M,
        "length_m": COURT_LENGTH_M,
    },
    "calibration": {
        "pixel_points": PIXEL_POINTS.tolist(),
        "world_points": WORLD_POINTS.tolist(),
    },
    "frames_with_players": frames_with_players,
    "frames_with_two_players": frames_with_two_players,
    "total_coordinate_rows": total_coordinate_rows,
    "coordinates": json_rows,
}


with open(
    JSON_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output_json,
        f,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("V6 PROCESSING COMPLETE")
print("=" * 70)

print()
print(f"Frames processed          : {frame_number}")
print(f"Frames with players       : {frames_with_players}")
print(f"Frames with 2+ players    : {frames_with_two_players}")
print(f"Coordinate rows           : {total_coordinate_rows}")

print()
print("OUTPUT FILES:")

print(f"CSV   : {CSV_FILE}")
print(f"JSON  : {JSON_FILE}")
print(f"VIDEO : {OUTPUT_VIDEO}")

print()
print("=" * 70)
