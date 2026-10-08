from pathlib import Path
import pandas as pd

INPUT = Path("results/player_tracks_v1.csv")

df = pd.read_csv(INPUT)

print()
print("TRACK GAP ANALYSIS")
print("==================")

for player in ["Player_A", "Player_B"]:

    p = df[df["player"] == player].sort_values("frame").copy()

    frames = p["frame"].astype(int)

    gaps = frames.diff()

    gap_info = pd.DataFrame({
        "frame": frames,
        "gap": gaps
    })

    print()
    print(player)
    print("------------------")

    print("Observed frames:", len(p))
    print("First frame:", frames.iloc[0])
    print("Last frame:", frames.iloc[-1])

    print("Number of gaps > 1 frame:", (gaps > 1).sum())
    print("Maximum frame gap:", int(gaps.max()))

    print()
    print("Gap distribution:")

    print(
        gaps.value_counts()
            .sort_index()
            .head(20)
    )

    print()
    print("Largest gaps:")

    largest = (
        gap_info[gap_info["gap"] > 1]
        .sort_values("gap", ascending=False)
        .head(15)
    )

    if len(largest):

        for _, row in largest.iterrows():

            frame_after = int(row["frame"])
            gap = int(row["gap"])
            frame_before = frame_after - gap

            print(
                "frames {} -> {} : gap {} frames".format(
                    frame_before,
                    frame_after,
                    gap
                )
            )

    else:
        print("No missing frames.")

print()
print("DONE")
