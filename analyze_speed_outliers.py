from pathlib import Path
import pandas as pd

INPUT = Path("results/player_tracks_v4.csv")

df = pd.read_csv(INPUT)

print()
print("V3 SPEED OUTLIER ANALYSIS")
print("=========================")

for player in ["Player_A", "Player_B"]:

    p = df[df["player"] == player].sort_values("frame").copy()

    print()
    print(player)
    print("----------------------")

    print("95th percentile speed:",
          round(p["speed_mps"].quantile(0.95), 3), "m/s")

    print("99th percentile speed:",
          round(p["speed_mps"].quantile(0.99), 3), "m/s")

    print("Maximum speed:",
          round(p["speed_mps"].max(), 3), "m/s")

    print()
    print("Top 15 speed observations:")

    cols = [
        "frame",
        "time_seconds",
        "tracker_id",
        "court_x_m",
        "court_y_m",
        "dx_m",
        "dy_m",
        "distance_m",
        "speed_mps",
        "interpolated",
        "outlier_removed"
    ]

    # Only include columns that actually exist
    cols = [c for c in cols if c in p.columns]

    top = (
        p.sort_values("speed_mps", ascending=False)
         .head(15)
    )

    print(top[cols].to_string(index=False))

print()
print("DONE")