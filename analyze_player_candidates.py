from pathlib import Path
import pandas as pd

CSV_FILE = Path("results/badminton_coordinates_v6_full.csv")

df = pd.read_csv(CSV_FILE)

print("\nCOLUMNS:")
print(df.columns.tolist())

# Use the known clean court limits
df = df[
    df["court_x_m"].between(-0.05, 5.0) &
    df["court_y_m"].between(0.0, 12.85)
].copy()

print("\nCLEAN ROWS:", len(df))

MID_Y = 6.7

df["player_side"] = df["court_y_m"].apply(
    lambda y: "UPPER" if y >= MID_Y else "LOWER"
)

# --------------------------------------------------
# NUMBER OF DETECTIONS PER SIDE PER FRAME
# --------------------------------------------------

counts = (
    df.groupby(["frame", "player_side"])
      .size()
      .unstack(fill_value=0)
)

if "UPPER" not in counts.columns:
    counts["UPPER"] = 0

if "LOWER" not in counts.columns:
    counts["LOWER"] = 0

print("\nFRAME DETECTION COUNTS")
print("--------------------------------")

print(
    "Frames with exactly 1 upper:",
    (counts["UPPER"] == 1).sum()
)

print(
    "Frames with >1 upper:",
    (counts["UPPER"] > 1).sum()
)

print(
    "Frames with exactly 1 lower:",
    (counts["LOWER"] == 1).sum()
)

print(
    "Frames with >1 lower:",
    (counts["LOWER"] > 1).sum()
)

print(
    "Frames with no upper:",
    (counts["UPPER"] == 0).sum()
)

print(
    "Frames with no lower:",
    (counts["LOWER"] == 0).sum()
)

# --------------------------------------------------
# DUPLICATE DETECTION STATISTICS
# --------------------------------------------------

upper_multi = counts[counts["UPPER"] > 1]
lower_multi = counts[counts["LOWER"] > 1]

print("\nMULTIPLE DETECTIONS ON UPPER SIDE")
print("--------------------------------")

if len(upper_multi):
    print("Number of affected frames:", len(upper_multi))
    print("Maximum detections in one frame:", upper_multi["UPPER"].max())
else:
    print("None")

print("\nMULTIPLE DETECTIONS ON LOWER SIDE")
print("--------------------------------")

if len(lower_multi):
    print("Number of affected frames:", len(lower_multi))
    print("Maximum detections in one frame:", lower_multi["LOWER"].max())
else:
    print("None")

# --------------------------------------------------
# EXAMPLE PROBLEM FRAMES
# --------------------------------------------------

print("\nEXAMPLE FRAMES WITH MULTIPLE DETECTIONS")
print("--------------------------------")

problem_frames = sorted(
    set(upper_multi.index.tolist()[:10] +
        lower_multi.index.tolist()[:10])
)

for frame in problem_frames:

    temp = df[df["frame"] == frame]

    print("\nFRAME:", frame)

    print(
        temp[
            ["tracker_id", "court_x_m", "court_y_m"]
        ].to_string(index=False)
    )

print("\nDONE")
