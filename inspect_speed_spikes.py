from pathlib import Path
import pandas as pd

INPUT = Path("results/player_tracks_v2.csv")

df = pd.read_csv(INPUT)

print()
print("LOCAL INSPECTION OF SPEED SPIKES")
print("================================")

for player in ["Player_A", "Player_B"]:

    p = df[df["player"] == player].sort_values("frame").reset_index(drop=True)

    print()
    print(player)
    print("--------------------------------")

    # Get the 10 largest speeds
    spike_indices = (
        p["speed_mps"]
        .nlargest(10)
        .index
    )

    for idx in spike_indices:

        start = max(0, idx - 2)
        end = min(len(p), idx + 3)

        cols = [
            "frame",
            "time_seconds",
            "tracker_id",
            "court_x_m",
            "court_y_m",
            "speed_mps",
            "interpolated"
        ]

        print()
        print("SPIKE AT FRAME:", int(p.loc[idx, "frame"]))
        print(
            p.iloc[start:end][cols].to_string(index=False)
        )

print()
print("DONE")
