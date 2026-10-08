from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("results/player_tracks_v1.csv")
OUTPUT = Path("results/player_tracks_v3.csv")

MAX_GAP = 3

# --------------------------------------------------
# OUTLIER SETTINGS
# --------------------------------------------------

# Maximum physically plausible instantaneous speed.
# This is used only as a first-pass filter.
MAX_SPEED = 12.0

# A jump is considered suspicious when the movement
# is much larger than the local movement around it.
LOCAL_MULTIPLIER = 2.5

# Small smoothing window.
SMOOTH_WINDOW = 3


df = pd.read_csv(INPUT)

all_players = []


# ==================================================
# PROCESS EACH PLAYER
# ==================================================

for player in ["Player_A", "Player_B"]:

    p = df[df["player"] == player].copy()
    p = p.sort_values("frame").reset_index(drop=True)

    # Original observations
    p["interpolated"] = False
    p["outlier_removed"] = False

    # --------------------------------------------------
    # STEP 1: DETECT SUSPICIOUS POSITION JUMPS
    # --------------------------------------------------

    x = p["court_x_m"].to_numpy(dtype=float)
    y = p["court_y_m"].to_numpy(dtype=float)
    t = p["time_seconds"].to_numpy(dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    dt[dt <= 0] = np.nan

    distance = np.sqrt(dx**2 + dy**2)
    speed = distance / dt

    suspicious = np.zeros(len(p), dtype=bool)

    for i in range(1, len(p) - 1):

        current_speed = speed[i - 1]

        if not np.isfinite(current_speed):
            continue

        # --------------------------------------------------
        # LOCAL MOTION BEFORE AND AFTER
        # --------------------------------------------------

        before_speed = speed[i - 2] if i >= 2 else np.nan
        after_speed = speed[i] if i < len(speed) else np.nan

        local_values = [
            v for v in [before_speed, after_speed]
            if np.isfinite(v)
        ]

        if not local_values:
            continue

        local_speed = np.median(local_values)

        # --------------------------------------------------
        # RULE 1: ABSOLUTE SPEED LIMIT
        # --------------------------------------------------

        too_fast = current_speed > MAX_SPEED

        # --------------------------------------------------
        # RULE 2: LARGE JUMP RELATIVE TO LOCAL MOTION
        # --------------------------------------------------

        much_larger = (
            local_speed > 0
            and current_speed > local_speed * LOCAL_MULTIPLIER
        )

        # --------------------------------------------------
        # RULE 3: CHECK WHETHER POSITION JUMPS
        # AND THEN RETURNS TOWARD PREVIOUS TRAJECTORY
        # --------------------------------------------------

        return_jump = False

        if i >= 1 and i < len(p) - 1:

            prev_x = x[i - 1]
            prev_y = y[i - 1]

            curr_x = x[i]
            curr_y = y[i]

            next_x = x[i + 1]
            next_y = y[i + 1]

            jump_before = np.sqrt(
                (curr_x - prev_x) ** 2
                + (curr_y - prev_y) ** 2
            )

            jump_after = np.sqrt(
                (next_x - curr_x) ** 2
                + (next_y - curr_y) ** 2
            )

            normal_step = np.sqrt(
                (next_x - prev_x) ** 2
                + (next_y - prev_y) ** 2
            )

            # If the current point creates two large jumps
            # while the surrounding trajectory is relatively small.
            if (
                jump_before > 0.35
                and jump_after > 0.35
                and normal_step < jump_before * 0.75
            ):
                return_jump = True

        if too_fast and (much_larger or return_jump):
            suspicious[i] = True

    # --------------------------------------------------
    # SPECIAL CHECK FOR EXTREME SPEEDS
    # --------------------------------------------------

    extreme = speed > 20

    for i in range(1, len(p) - 1):

        if i - 1 < len(extreme) and extreme[i - 1]:
            suspicious[i] = True

    # --------------------------------------------------
    # APPLY OUTLIER MASK
    # --------------------------------------------------

    p.loc[suspicious, "outlier_removed"] = True

    print()
    print(player)
    print("-" * len(player))
    print("Original rows:", len(p))
    print("Outliers removed:", int(suspicious.sum()))

    if suspicious.sum() > 0:
        print()
        print("Removed frames:")

        print(
            p.loc[
                suspicious,
                [
                    "frame",
                    "court_x_m",
                    "court_y_m"
                ]
            ].to_string(index=False)
        )

    # Replace suspicious coordinates with NaN
    p.loc[suspicious, "court_x_m"] = np.nan
    p.loc[suspicious, "court_y_m"] = np.nan

    # --------------------------------------------------
    # STEP 2: INTERPOLATE REMOVED POSITIONS
    # --------------------------------------------------

    p["court_x_m"] = (
        p["court_x_m"]
        .interpolate(method="linear", limit=MAX_GAP)
    )

    p["court_y_m"] = (
        p["court_y_m"]
        .interpolate(method="linear", limit=MAX_GAP)
    )

    # Mark newly reconstructed rows
    p.loc[suspicious, "interpolated"] = True

    # --------------------------------------------------
    # STEP 3: SMALL TRAJECTORY SMOOTHING
    # --------------------------------------------------

    p["court_x_m"] = (
        p["court_x_m"]
        .rolling(
            window=SMOOTH_WINDOW,
            center=True,
            min_periods=1
        )
        .median()
    )

    p["court_y_m"] = (
        p["court_y_m"]
        .rolling(
            window=SMOOTH_WINDOW,
            center=True,
            min_periods=1
        )
        .median()
    )

    # --------------------------------------------------
    # RESET TRACKER INFO FOR RECONSTRUCTED ROWS
    # --------------------------------------------------

    reconstructed = p["interpolated"]

    p.loc[reconstructed, "tracker_id"] = -1
    p.loc[reconstructed, "confidence"] = np.nan
    p.loc[reconstructed, "pixel_x"] = np.nan
    p.loc[reconstructed, "pixel_y"] = np.nan

    all_players.append(p)


# ==================================================
# COMBINE PLAYERS
# ==================================================

out = pd.concat(all_players, ignore_index=True)

out = out.sort_values(
    ["frame", "player"]
).reset_index(drop=True)


# ==================================================
# RECALCULATE MOVEMENT FEATURES
# ==================================================

out["dx_m"] = 0.0
out["dy_m"] = 0.0
out["distance_m"] = 0.0
out["speed_mps"] = 0.0


for player in ["Player_A", "Player_B"]:

    mask = out["player"] == player

    p = (
        out.loc[mask]
        .sort_values("frame")
        .copy()
    )

    dx = p["court_x_m"].diff()
    dy = p["court_y_m"].diff()
    dt = p["time_seconds"].diff()

    distance = np.sqrt(
        dx**2 + dy**2
    )

    speed = (
        distance /
        dt.replace(0, np.nan)
    )

    out.loc[p.index, "dx_m"] = dx.fillna(0)
    out.loc[p.index, "dy_m"] = dy.fillna(0)
    out.loc[p.index, "distance_m"] = distance.fillna(0)
    out.loc[p.index, "speed_mps"] = speed.fillna(0)


# ==================================================
# REPORT
# ==================================================

print()
print("=" * 45)
print("PLAYER TRACKS V3")
print("=" * 45)

print()
print("Output:", OUTPUT)
print("Rows:", len(out))

print()
print("Rows by player:")
print(out["player"].value_counts())

print()
print("Interpolated / reconstructed rows:")
print(
    out.groupby(
        ["player", "interpolated"]
    ).size()
)

print()
print(
    "Total interpolated/reconstructed:",
    int(out["interpolated"].sum())
)

print()
print("Outliers removed:")
print(
    out.groupby("player")["outlier_removed"]
       .sum()
)

print()
print("Maximum speed:")
print(
    out.groupby("player")["speed_mps"]
       .max()
)

print()
print("95th percentile speed:")
print(
    out.groupby("player")["speed_mps"]
       .quantile(0.95)
)

print()
print("99th percentile speed:")
print(
    out.groupby("player")["speed_mps"]
       .quantile(0.99)
)

print()
print("Mean speed:")
print(
    out.groupby("player")["speed_mps"]
       .mean()
)


# ==================================================
# SAVE
# ==================================================

out.to_csv(OUTPUT, index=False)

print()
print("Saved successfully.")
print("DONE")