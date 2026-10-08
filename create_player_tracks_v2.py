from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("results/player_tracks_v1.csv")
OUTPUT = Path("results/player_tracks_v2.csv")

MAX_GAP = 3

df = pd.read_csv(INPUT)

all_players = []

for player in ["Player_A", "Player_B"]:

    p = df[df["player"] == player].copy()
    p = p.sort_values("frame").reset_index(drop=True)

    # --------------------------------------------------
    # ORIGINAL OBSERVATIONS
    # --------------------------------------------------

    p["interpolated"] = False

    # --------------------------------------------------
    # BUILD OUTPUT ROWS
    # --------------------------------------------------

    rows = []

    for i in range(len(p) - 1):

        current = p.iloc[i]
        next_row = p.iloc[i + 1]

        rows.append(current.to_dict())

        f1 = int(current["frame"])
        f2 = int(next_row["frame"])

        gap = f2 - f1

        # Only interpolate short gaps
        if 1 < gap <= MAX_GAP:

            for frame in range(f1 + 1, f2):

                alpha = (frame - f1) / float(gap)

                x = (
                    current["court_x_m"]
                    + alpha *
                    (next_row["court_x_m"] - current["court_x_m"])
                )

                y = (
                    current["court_y_m"]
                    + alpha *
                    (next_row["court_y_m"] - current["court_y_m"])
                )

                # Interpolated timestamp
                t = (
                    current["time_seconds"]
                    + alpha *
                    (next_row["time_seconds"]
                     - current["time_seconds"])
                )

                row = current.to_dict()

                row["frame"] = frame
                row["time_seconds"] = t
                row["court_x_m"] = x
                row["court_y_m"] = y

                # These values were not actually observed
                row["tracker_id"] = -1
                row["confidence"] = np.nan
                row["pixel_x"] = np.nan
                row["pixel_y"] = np.nan

                row["interpolated"] = True

                rows.append(row)

    # Add final observed row
    rows.append(p.iloc[-1].to_dict())

    player_out = pd.DataFrame(rows)

    player_out = player_out.sort_values("frame").reset_index(drop=True)

    all_players.append(player_out)

# --------------------------------------------------
# COMBINE PLAYERS
# --------------------------------------------------

out = pd.concat(all_players, ignore_index=True)

out = out.sort_values(
    ["frame", "player"]
).reset_index(drop=True)

# --------------------------------------------------
# RECALCULATE MOVEMENT FEATURES
# --------------------------------------------------

out["dx_m"] = 0.0
out["dy_m"] = 0.0
out["distance_m"] = 0.0
out["speed_mps"] = 0.0

for player in ["Player_A", "Player_B"]:

    mask = out["player"] == player

    p = out.loc[mask].sort_values("frame").copy()

    dx = p["court_x_m"].diff()
    dy = p["court_y_m"].diff()
    dt = p["time_seconds"].diff()

    distance = np.sqrt(
        dx**2 + dy**2
    )

    speed = distance / dt.replace(0, np.nan)

    out.loc[p.index, "dx_m"] = dx.fillna(0)
    out.loc[p.index, "dy_m"] = dy.fillna(0)
    out.loc[p.index, "distance_m"] = distance.fillna(0)
    out.loc[p.index, "speed_mps"] = speed.fillna(0)

# --------------------------------------------------
# REPORT
# --------------------------------------------------

print()
print("PLAYER TRACKS V2")
print("================")

print("Output:", OUTPUT)
print("Rows:", len(out))

print()
print("Rows by player:")
print(out["player"].value_counts())

print()
print("Observed vs interpolated:")

print(
    out.groupby(
        ["player", "interpolated"]
    ).size()
)

print()
print("Interpolated rows:", int(out["interpolated"].sum()))

print()
print("Maximum speed:")
print(
    out.groupby("player")["speed_mps"]
       .max()
)

print()
print("Mean speed:")
print(
    out.groupby("player")["speed_mps"]
       .mean()
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

out.to_csv(OUTPUT, index=False)

print()
print("Saved successfully.")
print("DONE")
