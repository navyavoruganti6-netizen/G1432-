from pathlib import Path
import pandas as pd

INPUT = Path("results/player_tracks_v3.csv")

df = pd.read_csv(INPUT)

print()
print("V3 SUSPICIOUS SPEED INSPECTION")
print("==============================")

THRESHOLD = 12.0

for player in ["Player_A", "Player_B"]:

    p = df[df["player"] == player].sort_values("frame").copy()

    suspicious = p[p["speed_mps"] > THRESHOLD]

    print()
    print(player)
    print("------------------------------")
    print("Suspicious observations:", len(suspicious))

    for _, row in suspicious.iterrows():

        frame = int(row["frame"])

        nearby = p[
            (p["frame"] >= frame - 2) &
            (p["frame"] <= frame + 2)
        ]

        print()
        print(
            f"Frame {frame} | "
            f"Speed {row['speed_mps']:.2f} m/s | "
            f"Interpolated: {row['interpolated']}"
        )

        cols = [
            "frame",
            "court_x_m",
            "court_y_m",
            "speed_mps",
            "tracker_id",
            "interpolated",
            "outlier_removed"
        ]

        cols = [c for c in cols if c in nearby.columns]

        print(nearby[cols].to_string(index=False))

print()
print("DONE")