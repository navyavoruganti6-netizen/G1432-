from pathlib import Path
import pandas as pd

INPUT = Path("results/badminton_coordinates_v6_full.csv")
OUTPUT = Path("results/player_tracks_v1.csv")

df = pd.read_csv(INPUT)

# Keep only valid court detections
if "in_court" in df.columns:
    df = df[df["in_court"] == True].copy()

df = df.dropna(
    subset=[
        "frame",
        "time_seconds",
        "tracker_id",
        "court_x_m",
        "court_y_m",
    ]
).copy()

# Assign court side
MID_Y = 6.7

df["player"] = df["court_y_m"].apply(
    lambda y: "Player_A" if y >= MID_Y else "Player_B"
)

# One detection per frame/player.
# Highest confidence is retained.
df = (
    df.sort_values(
        ["frame", "player", "confidence"],
        ascending=[True, True, False],
    )
    .drop_duplicates(
        subset=["frame", "player"],
        keep="first",
    )
)

df = df.sort_values(
    ["frame", "player"]
).reset_index(drop=True)

# Movement columns
df["dx_m"] = 0.0
df["dy_m"] = 0.0
df["distance_m"] = 0.0
df["speed_mps"] = 0.0

# IMPORTANT:
# Calculate movement separately for each TRACKER ID.
# Do not connect tracker ID 5 directly to tracker ID 11.
for tracker_id, group in df.groupby("tracker_id"):

    indices = group.sort_values("frame").index

    previous_index = None

    for index in indices:

        if previous_index is None:
            previous_index = index
            continue

        current = df.loc[index]
        previous = df.loc[previous_index]

        # Never calculate movement across a frame gap
        if current["frame"] - previous["frame"] != 1:
            previous_index = index
            continue

        dx = (
            current["court_x_m"]
            - previous["court_x_m"]
        )

        dy = (
            current["court_y_m"]
            - previous["court_y_m"]
        )

        distance = (
            dx ** 2 + dy ** 2
        ) ** 0.5

        dt = (
            current["time_seconds"]
            - previous["time_seconds"]
        )

        speed = (
            distance / dt
            if dt > 0
            else 0.0
        )

        df.loc[index, "dx_m"] = dx
        df.loc[index, "dy_m"] = dy
        df.loc[index, "distance_m"] = distance
        df.loc[index, "speed_mps"] = speed

        previous_index = index

# Save
df.to_csv(
    OUTPUT,
    index=False,
)

print()
print("PLAYER TRACK DATA CREATED")
print("-------------------------")
print(f"Input:  {INPUT}")
print(f"Output: {OUTPUT}")
print(f"Rows: {len(df)}")

print()
print("Rows by player:")
print(df["player"].value_counts())

print()
print("Unique frames by player:")
print(
    df.groupby("player")["frame"]
    .nunique()
)

print()
print("Total unique frames:")
print(df["frame"].nunique())

print()
print("Tracker IDs used by each player:")
print(
    df.groupby("player")["tracker_id"]
    .unique()
)

print()
print("Missing player frames:")

for player in ["Player_A", "Player_B"]:

    frames = set(
        df[df["player"] == player]["frame"]
    )

    missing = TOTAL_FRAMES = 1802 - len(frames)

    print(
        f"{player} : {missing} frames missing"
    )

print()
print("DONE")