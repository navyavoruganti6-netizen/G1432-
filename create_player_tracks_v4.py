from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("results/player_tracks_v3.csv")
OUTPUT = Path("results/player_tracks_v4.csv")

MAX_SPEED = 12.0
MAX_GAP = 3

df = pd.read_csv(INPUT)

print()
print("CREATING PLAYER TRACKS V4")
print("=========================")

all_players = []

for player in ["Player_A", "Player_B"]:

    p = df[df["player"] == player].sort_values("frame").copy()
    p = p.reset_index(drop=True)

    print()
    print(player)
    print("-------------------------")

    # ---------------------------------------------------------
    # Calculate raw frame-to-frame movement
    # ---------------------------------------------------------

    p["raw_dx"] = p["court_x_m"].diff()
    p["raw_dy"] = p["court_y_m"].diff()

    p["raw_distance"] = np.sqrt(
        p["raw_dx"] ** 2 +
        p["raw_dy"] ** 2
    )

    p["raw_speed"] = p["raw_distance"] / p["time_seconds"].diff()

    # ---------------------------------------------------------
    # Detect isolated jumps
    #
    # A point is suspicious when:
    # 1. Its speed is high
    # 2. Movement before/after the point is much smaller
    # ---------------------------------------------------------

    bad_frames = []

    for i in range(1, len(p) - 1):

        current_speed = p.loc[i, "raw_speed"]

        if pd.isna(current_speed):
            continue

        prev_distance = p.loc[i, "raw_distance"]

        next_dx = p.loc[i + 1, "court_x_m"] - p.loc[i, "court_x_m"]
        next_dy = p.loc[i + 1, "court_y_m"] - p.loc[i, "court_y_m"]

        dt_next = (
            p.loc[i + 1, "time_seconds"] -
            p.loc[i, "time_seconds"]
        )

        if dt_next <= 0:
            continue

        next_speed = (
            np.sqrt(next_dx ** 2 + next_dy ** 2) /
            dt_next
        )

        # A large jump followed by a large reversal
        reversal_distance = np.sqrt(
            (
                p.loc[i + 1, "court_x_m"] -
                p.loc[i - 1, "court_x_m"]
            ) ** 2 +
            (
                p.loc[i + 1, "court_y_m"] -
                p.loc[i - 1, "court_y_m"]
            ) ** 2
        )

        dt_total = (
            p.loc[i + 1, "time_seconds"] -
            p.loc[i - 1, "time_seconds"]
        )

        surrounding_speed = (
            reversal_distance / dt_total
            if dt_total > 0 else 0
        )

        # Strong isolated jump
        isolated_jump = (
            current_speed > MAX_SPEED and
            surrounding_speed < current_speed * 0.55
        )

        # Previous + next movement disagree strongly
        direction_change = (
            current_speed > MAX_SPEED and
            surrounding_speed < 6.0
        )

        if isolated_jump or direction_change:
            bad_frames.append(i)

    print("Suspicious points detected:", len(bad_frames))

    # ---------------------------------------------------------
    # Remove suspicious coordinates
    # ---------------------------------------------------------

    p["v4_removed"] = False

    for i in bad_frames:
        p.loc[i, "court_x_m"] = np.nan
        p.loc[i, "court_y_m"] = np.nan
        p.loc[i, "v4_removed"] = True

    # ---------------------------------------------------------
    # Interpolate only short gaps
    # ---------------------------------------------------------

    p["court_x_m"] = (
        p["court_x_m"]
        .interpolate(limit=MAX_GAP)
        .bfill()
        .ffill()
    )

    p["court_y_m"] = (
        p["court_y_m"]
        .interpolate(limit=MAX_GAP)
        .bfill()
        .ffill()
    )

    # ---------------------------------------------------------
    # Mark reconstructed rows
    # ---------------------------------------------------------

    p["interpolated"] = (
        p["interpolated"].fillna(False) |
        p["v4_removed"]
    )

    for i in bad_frames:
        p.loc[i, "tracker_id"] = -1
        p.loc[i, "confidence"] = np.nan
        p.loc[i, "pixel_x"] = np.nan
        p.loc[i, "pixel_y"] = np.nan

    # ---------------------------------------------------------
    # Recalculate movement
    # ---------------------------------------------------------

    p["dx_m"] = p["court_x_m"].diff()
    p["dy_m"] = p["court_y_m"].diff()

    p["distance_m"] = np.sqrt(
        p["dx_m"] ** 2 +
        p["dy_m"] ** 2
    )

    dt = p["time_seconds"].diff()

    p["speed_mps"] = (
        p["distance_m"] / dt
    )

    p["speed_mps"] = (
        p["speed_mps"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    all_players.append(p)

    print("Rows:", len(p))
    print("Removed/reconstructed:", len(bad_frames))
    print("Maximum speed:",
          round(p["speed_mps"].max(), 3), "m/s")
    print("95th percentile:",
          round(p["speed_mps"].quantile(0.95), 3), "m/s")
    print("99th percentile:",
          round(p["speed_mps"].quantile(0.99), 3), "m/s")


# -------------------------------------------------------------
# Combine players
# -------------------------------------------------------------

result = pd.concat(all_players, ignore_index=True)

# Remove temporary columns
for col in [
    "raw_dx",
    "raw_dy",
    "raw_distance",
    "raw_speed",
    "v4_removed"
]:
    if col in result.columns:
        result.drop(columns=col, inplace=True)

result = result.sort_values(
    ["player", "frame"]
).reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print()
print("=========================")
print("V4 COMPLETE")
print("=========================")
print("Total rows:", len(result))
print("Saved to:", OUTPUT)