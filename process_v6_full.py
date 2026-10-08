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

INPUT_VIDEO = Path("video/badminton_test.mp4")

RESULTS_DIR = Path("results")
CSV_FILE = RESULTS_DIR / "badminton_coordinates_v6_full.csv"
JSON_FILE = RESULTS_DIR / "badminton_coordinates_v6_full.json"
OUTPUT_VIDEO = RESULTS_DIR / "badminton_tracking_v6_full.mp4"

# Court dimensions
COURT_WIDTH_M = 5.18
COURT_LENGTH_M = 13.40

# Minimum confidence for player detections
CONFIDENCE_THRESHOLD = 0.30

# ------------------------------------------------------------
# SPIKE FILTER
#
# We DO NOT remove every high-speed movement.
#
# A point is considered suspicious only when:
#
#   1. Current speed is very high
#   2. Previous speed is much lower
#   3. Next speed is also much lower
#
# This catches:
#
#     normal -> huge spike -> normal
#
# while preserving:
#
#     fast -> fast -> fast -> fast
#
# ------------------------------------------------------------

SPIKE_SPEED_THRESHOLD = 15.0
SPIKE_RATIO = 2.0

# Maximum allowed interpolation gap for tracker points
MAX_INTERPOLATION_GAP = 1


# ------------------------------------------------------------
# EXACT ROBOFLOW CALIBRATION
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
print("BADMINTON AI - V6 TRACKER-STABILIZED PIPELINE")
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
# HELPERS
# ============================================================

def pixel_to_court(pixel_x, pixel_y):
    """
    Convert one image pixel position into court coordinates.
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


def is_in_court(court_x, court_y):
    """
    Check whether a court coordinate lies inside the court.
    """

    tolerance = 0.05

    return (
        -tolerance <= court_x <= COURT_WIDTH_M + tolerance
        and
        -tolerance <= court_y <= COURT_LENGTH_M + tolerance
    )


def distance_between(p1, p2):
    """
    Euclidean distance between two court-coordinate points.
    """

    return float(
        np.sqrt(
            (p1[0] - p2[0]) ** 2
            +
            (p1[1] - p2[1]) ** 2
        )
    )


# ============================================================
# FIRST PASS
#
# Run YOLO + ByteTrack and store detections.
# We do NOT calculate final movement yet.
# ============================================================

raw_rows = []

frame_number = 0
frames_with_players = 0
frames_with_two_players = 0

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
    # NO RESULTS
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

    # --------------------------------------------------------
    # EXTRACT PLAYER PREDICTIONS
    # --------------------------------------------------------

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

        # Only badminton players
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

    player_count = len(
        tracked_detections.xyxy
    )

    if player_count > 0:
        frames_with_players += 1

    if player_count >= 2:
        frames_with_two_players += 1

    # --------------------------------------------------------
    # STORE TRACKED DETECTIONS
    # --------------------------------------------------------

    for i, box in enumerate(
        tracked_detections.xyxy
    ):

        x1, y1, x2, y2 = box

        # Keep floating-point values.
        # Do NOT round before coordinate calculation.
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        # ----------------------------------------------------
        # PLAYER GROUND POINT
        # ----------------------------------------------------

        pixel_x = (
            x1 + x2
        ) / 2.0

        pixel_y = y2

        # ----------------------------------------------------
        # COURT COORDINATES
        # ----------------------------------------------------

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

        raw_rows.append(
            {
                "frame": frame_number,
                "time_seconds": round(
                    frame_number / FPS,
                    4,
                ),

                "tracker_id": tracker_id,

                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,

                "box_width": x2 - x1,
                "box_height": y2 - y1,

                "pixel_x": pixel_x,
                "pixel_y": pixel_y,

                "court_x_m": court_x,
                "court_y_m": court_y,

                "confidence": confidence,

                "in_court": in_court,

                "spike_corrected": False,
            }
        )

        # ----------------------------------------------------
        # DRAW ORIGINAL DETECTION
        # ----------------------------------------------------

        draw_x1 = int(round(x1))
        draw_y1 = int(round(y1))
        draw_x2 = int(round(x2))
        draw_y2 = int(round(y2))

        cv2.rectangle(
            frame,
            (draw_x1, draw_y1),
            (draw_x2, draw_y2),
            (0, 255, 0),
            2,
        )

        # Ground point
        cv2.circle(
            frame,
            (
                int(round(pixel_x)),
                int(round(pixel_y)),
            ),
            6,
            (0, 0, 255),
            -1,
        )

        label = (
            f"ID {tracker_id} "
            f"({court_x:.2f}, {court_y:.2f})m"
        )

        cv2.putText(
            frame,
            label,
            (
                draw_x1,
                max(
                    25,
                    draw_y1 - 10,
                ),
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
        frame,
        status,
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    video_writer.write(frame)

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_number % 30 == 0:

        print(
            f"Processed "
            f"{frame_number}/{TOTAL_FRAMES} frames | "
            f"tracked players: {player_count} | "
            f"raw coordinate rows: {len(raw_rows)}"
        )


# ============================================================
# CLOSE VIDEO
# ============================================================

cap.release()
video_writer.release()


print()
print("Raw detection pass complete.")
print(f"Raw coordinate rows: {len(raw_rows)}")
print()


# ============================================================
# CONVERT TO NUMPY / SORT
# ============================================================

raw_rows.sort(
    key=lambda r: (
        r["tracker_id"],
        r["frame"],
    )
)


# ============================================================
# SPIKE CORRECTION
#
# IMPORTANT:
#
# A spike belongs to the CURRENT/MIDDLE POINT when the movement
# INTO that point is abnormally large and the movement immediately
# AFTER that point drops back down.
#
# Example:
#
#     frame 301       frame 302       frame 303
#       normal       ABNORMAL         normal
#          \            |              /
#           \____ large movement ____/
#
# The correction is therefore written to curr_index (frame 302
# in the example), never to prev_index (frame 301).
#
# We deliberately use the incoming segment:
#
#     previous -> current
#
# as the spike speed.  The old logic used current -> next as the
# primary "current speed", which can shift the correction one frame.
# ============================================================

corrected_count = 0

# Group row indices by tracker ID
tracker_groups = {}

for index, row in enumerate(raw_rows):

    tracker_id = row["tracker_id"]

    if tracker_id not in tracker_groups:
        tracker_groups[tracker_id] = []

    tracker_groups[tracker_id].append(index)


for tracker_id, indices in tracker_groups.items():

    if len(indices) < 3:
        continue

    for position in range(1, len(indices) - 1):

        prev_index = indices[position - 1]
        curr_index = indices[position]
        next_index = indices[position + 1]

        prev_row = raw_rows[prev_index]
        curr_row = raw_rows[curr_index]
        next_row = raw_rows[next_index]

        # ----------------------------------------------------
        # REQUIRE CONSECUTIVE FRAMES
        #
        # This prevents interpolation across tracker gaps.
        # ----------------------------------------------------

        if curr_row["frame"] - prev_row["frame"] != 1:
            continue

        if next_row["frame"] - curr_row["frame"] != 1:
            continue

        # ----------------------------------------------------
        # CALCULATE MOVEMENT AROUND THE CURRENT POINT
        #
        # incoming = previous -> current
        # outgoing = current -> next
        #
        # The incoming movement is what identifies the current
        # point as the suspicious/middle point.
        # ----------------------------------------------------

        prev_point = (
            prev_row["court_x_m"],
            prev_row["court_y_m"],
        )

        curr_point = (
            curr_row["court_x_m"],
            curr_row["court_y_m"],
        )

        next_point = (
            next_row["court_x_m"],
            next_row["court_y_m"],
        )

        incoming_distance = distance_between(
            prev_point,
            curr_point,
        )

        outgoing_distance = distance_between(
            curr_point,
            next_point,
        )

        incoming_speed = incoming_distance * FPS
        outgoing_speed = outgoing_distance * FPS

        # ----------------------------------------------------
        # IS THE CURRENT/MIDDLE POINT THE SPIKE?
        #
        # We want:
        #
        #   previous -> current = unusually fast
        #   current  -> next    = noticeably slower
        #
        # This identifies the point AFTER the abnormal jump,
        # i.e. the actual middle sample.
        #
        # The 35% drop prevents normal continuous movement from
        # being treated as an isolated spike.
        # ----------------------------------------------------

        speed_drop = (
            outgoing_speed < incoming_speed * 0.65
        )

        isolated_spike = (
            incoming_speed > SPIKE_SPEED_THRESHOLD
            and
            outgoing_speed < SPIKE_SPEED_THRESHOLD
            and
            speed_drop
        )

        should_correct = isolated_spike

        # ----------------------------------------------------
        # SECONDARY GEOMETRIC CHECK
        #
        # If the incoming movement is a strong jump but the
        # outgoing movement is still moderately high, check
        # whether the CURRENT point is far from the linear
        # midpoint between its neighbors.
        #
        # This is intentionally symmetric around curr_row.
        # ----------------------------------------------------

        if not should_correct:

            expected_x = (
                prev_row["court_x_m"]
                +
                next_row["court_x_m"]
            ) / 2.0

            expected_y = (
                prev_row["court_y_m"]
                +
                next_row["court_y_m"]
            ) / 2.0

            midpoint_error = np.sqrt(
                (
                    curr_row["court_x_m"]
                    -
                    expected_x
                ) ** 2
                +
                (
                    curr_row["court_y_m"]
                    -
                    expected_y
                ) ** 2
            )

            should_correct = (
                incoming_speed > SPIKE_SPEED_THRESHOLD
                and
                incoming_speed > outgoing_speed * SPIKE_RATIO
                and
                midpoint_error > 0.30
            )

        # ----------------------------------------------------
        # CORRECT ONLY THE CURRENT / MIDDLE POINT
        # ----------------------------------------------------

        if should_correct:

            corrected_x = (
                prev_row["court_x_m"]
                +
                next_row["court_x_m"]
            ) / 2.0

            corrected_y = (
                prev_row["court_y_m"]
                +
                next_row["court_y_m"]
            ) / 2.0

            raw_rows[curr_index]["court_x_m"] = corrected_x
            raw_rows[curr_index]["court_y_m"] = corrected_y
            raw_rows[curr_index]["spike_corrected"] = True

            corrected_count += 1

            print(
                f"Spike corrected: tracker {tracker_id}, "
                f"frame {curr_row['frame']} "
                f"(incoming {incoming_speed:.2f} m/s, "
                f"outgoing {outgoing_speed:.2f} m/s)"
            )


# RECALCULATE MOVEMENT
#
# Movement is calculated PER TRACKER ID.
#
# This prevents:
#
# tracker 5 -> tracker 11
#
# from being treated as one continuous trajectory.
# ============================================================

for tracker_id, indices in tracker_groups.items():

    previous_index = None

    for index in indices:

        row = raw_rows[index]

        # First point of this tracker
        if previous_index is None:

            row["dx_m"] = 0.0
            row["dy_m"] = 0.0
            row["distance_m"] = 0.0
            row["speed_mps"] = 0.0

            previous_index = index

            continue

        previous_row = raw_rows[
            previous_index
        ]

        # ----------------------------------------------------
        # If there is a frame gap, don't create artificial
        # speed across the gap.
        # ----------------------------------------------------

        frame_gap = (
            row["frame"]
            -
            previous_row["frame"]
        )

        if frame_gap != 1:

            row["dx_m"] = 0.0
            row["dy_m"] = 0.0
            row["distance_m"] = 0.0
            row["speed_mps"] = 0.0

            previous_index = index

            continue

        dx = (
            row["court_x_m"]
            -
            previous_row["court_x_m"]
        )

        dy = (
            row["court_y_m"]
            -
            previous_row["court_y_m"]
        )

        distance = np.sqrt(
            dx ** 2
            +
            dy ** 2
        )

        dt = (
            row["time_seconds"]
            -
            previous_row["time_seconds"]
        )

        if dt > 0:

            speed = distance / dt

        else:

            speed = 0.0

        row["dx_m"] = dx
        row["dy_m"] = dy
        row["distance_m"] = distance
        row["speed_mps"] = speed

        previous_index = index


# ============================================================
# FINAL CSV
# ============================================================

csv_columns = [
    "frame",
    "time_seconds",
    "tracker_id",

    "x1",
    "y1",
    "x2",
    "y2",

    "box_width",
    "box_height",

    "pixel_x",
    "pixel_y",

    "court_x_m",
    "court_y_m",

    "confidence",
    "in_court",

    "dx_m",
    "dy_m",
    "distance_m",
    "speed_mps",

    "spike_corrected",
]


with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=csv_columns,
    )

    csv_writer.writeheader()

    for row in raw_rows:

        output_row = {}

        for column in csv_columns:

            value = row.get(
                column,
                "",
            )

            if isinstance(value, float):

                if column in [
                    "court_x_m",
                    "court_y_m",
                    "dx_m",
                    "dy_m",
                    "distance_m",
                    "speed_mps",
                ]:
                    value = round(
                        value,
                        4,
                    )

                else:
                    value = round(
                        value,
                        2,
                    )

            output_row[column] = value

        csv_writer.writerow(
            output_row
        )


# ============================================================
# FINAL JSON
# ============================================================

json_coordinates = []

for row in raw_rows:

    json_coordinates.append(
        {
            "frame": row["frame"],
            "time_seconds": row[
                "time_seconds"
            ],

            "tracker_id": row[
                "tracker_id"
            ],

            "x1": round(
                row["x1"],
                2,
            ),

            "y1": round(
                row["y1"],
                2,
            ),

            "x2": round(
                row["x2"],
                2,
            ),

            "y2": round(
                row["y2"],
                2,
            ),

            "box_width": round(
                row["box_width"],
                2,
            ),

            "box_height": round(
                row["box_height"],
                2,
            ),

            "pixel_x": round(
                row["pixel_x"],
                2,
            ),

            "pixel_y": round(
                row["pixel_y"],
                2,
            ),

            "court_x_m": round(
                row["court_x_m"],
                4,
            ),

            "court_y_m": round(
                row["court_y_m"],
                4,
            ),

            "confidence": round(
                row["confidence"],
                4,
            ),

            "in_court": row[
                "in_court"
            ],

            "dx_m": round(
                row["dx_m"],
                4,
            ),

            "dy_m": round(
                row["dy_m"],
                4,
            ),

            "distance_m": round(
                row["distance_m"],
                4,
            ),

            "speed_mps": round(
                row["speed_mps"],
                4,
            ),

            "spike_corrected": row[
                "spike_corrected"
            ],
        }
    )


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

    "frames_with_two_players": (
        frames_with_two_players
    ),

    "total_coordinate_rows": len(
        raw_rows
    ),

    "spike_corrections": corrected_count,

    "coordinates": json_coordinates,
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
print("V6 TRACKER-STABILIZED PROCESSING COMPLETE")
print("=" * 70)

print()

print(
    f"Frames processed          : "
    f"{frame_number}"
)

print(
    f"Frames with players       : "
    f"{frames_with_players}"
)

print(
    f"Frames with 2+ players    : "
    f"{frames_with_two_players}"
)

print(
    f"Coordinate rows           : "
    f"{len(raw_rows)}"
)

print(
    f"Spike corrections         : "
    f"{corrected_count}"
)

print()

print("OUTPUT FILES:")

print(
    f"CSV   : {CSV_FILE}"
)

print(
    f"JSON  : {JSON_FILE}"
)

print(
    f"VIDEO : {OUTPUT_VIDEO}"
)

print()
print("=" * 70)