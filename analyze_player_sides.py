from pathlib import Path
import pandas as pd

CSV_FILE = Path("results/badminton_coordinates_v6_full.csv")

df = pd.read_csv(CSV_FILE)

print("\nTOTAL ROWS:", len(df))

# Court midpoint
MID_Y = 6.7

# Assign court side
df["court_side"] = df["court_y_m"].apply(
    lambda y: "UPPER" if y >= MID_Y else "LOWER"
)

print("\nCOURT SIDE COUNTS:")
print(df["court_side"].value_counts())

print("\nTRACKER IDS BY COURT SIDE:")
print(
    df.groupby(["court_side", "tracker_id"])
      .size()
      .sort_values(ascending=False)
)

# Frame information
if "frame" in df.columns:

    print("\nTOTAL UNIQUE FRAMES:", df["frame"].nunique())

    frame_side = (
        df.groupby(["frame", "court_side"])
          .size()
          .unstack(fill_value=0)
    )

    if "UPPER" not in frame_side.columns:
        frame_side["UPPER"] = 0

    if "LOWER" not in frame_side.columns:
        frame_side["LOWER"] = 0

    both_present = (
        (frame_side["UPPER"] > 0) &
        (frame_side["LOWER"] > 0)
    )

    upper_only = (
        (frame_side["UPPER"] > 0) &
        (frame_side["LOWER"] == 0)
    )

    lower_only = (
        (frame_side["LOWER"] > 0) &
        (frame_side["UPPER"] == 0)
    )

    print("\nFRAME ANALYSIS:")
    print("Both players detected:", both_present.sum())
    print("Upper only:", upper_only.sum())
    print("Lower only:", lower_only.sum())
    print("Total frames:", len(frame_side))

    print("\nPERCENTAGE:")
    print(
        "Both:",
        round(100 * both_present.sum() / len(frame_side), 2),
        "%"
    )

    print(
        "Upper only:",
        round(100 * upper_only.sum() / len(frame_side), 2),
        "%"
    )

    print(
        "Lower only:",
        round(100 * lower_only.sum() / len(frame_side), 2),
        "%"
    )

print("\nDONE")
