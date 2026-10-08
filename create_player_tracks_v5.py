from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("results/player_tracks_v3.csv")
OUTPUT = Path("results/player_tracks_v5.csv")

MAX_GAP = 3
MAX_INTERPOLATED_SPEED = 10.0

df = pd.read_csv(INPUT)

print()
print("CREATING PLAYER TRACKS V5")
print("=========================")

all_players = []

for player in ["Player_A", "Player_B"]:

    p = df[df["player"] == player].sort_values("frame").copy()
    p = p.reset_index(drop=True)

    print()
    print(player)
    print("-------------------------")

    # Keep original coordinates
    original_x = p["court_x_m"].copy()
    original_y = p["court_y_m"].copy()

    # ---------------------------------------------------------
    # Identify rows already reconstructed by V3
    # ---------------------------------------------------------

    reconstructed = p["interpolated"].fillna(False).astype(bool)

    # ---------------------------------------------------------
    # Rebuild suspicious V3 interpolated points
    #
    # If an interpolated point creates a very large movement,
    # replace it using the midpoint between surrounding points.
    # ---------------------------------------------------------

    corrected = 0

    for i in range(1, len(p) - 1):

        if not reconstructed.iloc[i]:
            continue

        prev_x = p.loc[i - 1, "court_x_m"]
        prev_y = p.loc[i - 1, "court_y_m"]

        next_x = p.loc[i + 1, "court_x_m"]
        next_y = p.loc[i + 1, "court_y_m"]

        if any(pd.isna(v) for v in [
            prev_x, prev_y, next_x, next_y
        ]):
            continue

        # Expected midpoint
        midpoint_x = (prev_x + next_x) / 2
        midpoint_y = (prev_y + next_y) / 2

        # Distance of current point from midpoint
        error = np.sqrt(
            (p.loc[i, "court_x_m"] - midpoint_x) ** 2 +
            (p.loc[i, "court_y_m"] - midpoint_y) ** 2
        )

        # Movement from previous to current
        incoming = np.sqrt(
            (p.loc[i, "court_x_m"] - prev_x) ** 2 +
            (p.loc[i, "court_y_m"] - prev_y) ** 2
        )

        # Movement from current to next
        outgoing = np.sqrt(
            (next_x - p.loc[i, "court_x_m"]) ** 2 +
            (next_y - p.loc[i, "court_y_m"]) ** 2
        )

        # If interpolation creates an unusually large jump,
        # use the midpoint instead.
        if (
            incoming > 0.30 or
            outgoing > 0.30 or
            error > 0.20
        ):
            p.loc[i, "court_x_m"] = midpoint_x
            p.loc[i, "court_y_m"] = midpoint_y

            corrected += 1

    print("Interpolation points corrected:", corrected)

    # ---------------------------------------------------------
    # Recalculate movement
    # ---------------------------------------------------------

    p["dx_m"] = p["court_x_m"].diff()
    p["dy_m"] = p["court_y_m"].diff()

    dt = p["time_seconds"].diff()

    p["distance_m"] = np.sqrt(
        p["dx_m"] ** 2 +
        p["dy_m"] ** 2
    )

    p["speed_mps"] = (
        p["distance_m"] / dt
    )

    p["speed_mps"] = (
        p["speed_mps"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ---------------------------------------------------------
    # Reconstructed rows remain marked
    # ---------------------------------------------------------

    p["interpolated"] = reconstructed

    all_players.append(p)

    print("Rows:", len(p))
    print(
        "Maximum speed:",
        round(p["speed_mps"].max(), 3),
        "m/s"
    )
    print(
        "95th percentile:",
        round(p["speed_mps"].quantile(0.95), 3),
        "m/s"
    )
    print(
        "99th percentile:",
        round(p["speed_mps"].quantile(0.99), 3),
        "m/s"
    )


# -------------------------------------------------------------
# Combine
# -------------------------------------------------------------

result = pd.concat(
    all_players,
    ignore_index=True
)

# Remove temporary columns if present
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

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT,
    index=False
)

print()
print("=========================")
print("V5 COMPLETE")
print("=========================")
print("Total rows:", len(result))
print("Saved to:", OUTPUT)