
import csv
import json
from pathlib import Path

from inference import InferencePipeline

WORKSPACE = "venu-sudha"

# CURRENT V5 TRACKING WORKFLOW
WORKFLOW_ID = "badminton-singles-player-tracking-1789540968597"

INPUT_VIDEO = Path("video/badminton_test_10sec.mp4")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS_DIR / "badminton_coordinates_v5_test.csv"
JSON_FILE = RESULTS_DIR / "badminton_coordinates_v5_test.json"

frame_counter = 0
frames_with_coordinates = 0
all_frames = []
csv_rows = []


def on_prediction(prediction, video_frame):
    global frame_counter
    global frames_with_coordinates

    frame_counter += 1

    if frame_counter % 30 == 0:
        print(f"Processed frame: {frame_counter}")

    # Current workflow structure:
    # court_coordinates
    #   -> players
    #       -> tracker_id
    #       -> x_m
    #       -> y_m
    #       -> confidence
    #       -> in_court

    court_coordinates = prediction.get("court_coordinates", {})

    if not isinstance(court_coordinates, dict):
        court_coordinates = {}

    players = court_coordinates.get("players", [])

    if not isinstance(players, list):
        players = []

    if len(players) > 0:
        frames_with_coordinates += 1

    frame_record = {
        "frame": frame_counter,
        "court_width_m": court_coordinates.get("court_width_m"),
        "court_length_m": court_coordinates.get("court_length_m"),
        "players": []
    }

    for player in players:

        player_record = {
            "tracker_id": player.get("tracker_id"),
            "court_x_m": player.get("x_m"),
            "court_y_m": player.get("y_m"),
            "foot_x_px": player.get("foot_x_px"),
            "foot_y_px": player.get("foot_y_px"),
            "confidence": player.get("confidence"),
            "in_court": player.get("in_court"),
        }

        frame_record["players"].append(player_record)

        csv_rows.append({
            "frame": frame_counter,
            "tracker_id": player.get("tracker_id"),
            "court_x_m": player.get("x_m"),
            "court_y_m": player.get("y_m"),
            "foot_x_px": player.get("foot_x_px"),
            "foot_y_px": player.get("foot_y_px"),
            "confidence": player.get("confidence"),
            "in_court": player.get("in_court"),
        })

    all_frames.append(frame_record)


if not INPUT_VIDEO.exists():
    raise FileNotFoundError(
        f"Input video not found:\n{INPUT_VIDEO.resolve()}"
    )


print("=" * 75)
print("BADMINTON AI - V5 PLAYER TRACKING + COURT COORDINATES")
print("=" * 75)

print(f"Workspace : {WORKSPACE}")
print(f"Workflow  : {WORKFLOW_ID}")
print(f"Video     : {INPUT_VIDEO.resolve()}")
print(f"CSV       : {CSV_FILE.resolve()}")
print(f"JSON      : {JSON_FILE.resolve()}")
print()

print("Initializing Roboflow workflow...")

pipeline = InferencePipeline.init_with_workflow(
    video_reference=str(INPUT_VIDEO),
    workspace_name=WORKSPACE,
    workflow_id=WORKFLOW_ID,
    on_prediction=on_prediction,
)

print("Workflow initialized.")
print()
print("Starting video processing...")
print()

pipeline.start()
pipeline.join()

print()
print("=" * 75)
print("PROCESSING COMPLETE")
print("=" * 75)

with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(all_frames, f, indent=2)

with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "frame",
            "tracker_id",
            "court_x_m",
            "court_y_m",
            "foot_x_px",
            "foot_y_px",
            "confidence",
            "in_court",
        ],
    )

    writer.writeheader()
    writer.writerows(csv_rows)


print(f"Frames processed        : {frame_counter}")
print(f"Frames with players     : {frames_with_coordinates}")
print(f"Coordinate rows         : {len(csv_rows)}")
print()
print(f"JSON saved              : {JSON_FILE.resolve()}")
print(f"CSV saved               : {CSV_FILE.resolve()}")
print("=" * 75)
| Set-Content -Encoding UTF8 process_v5.py