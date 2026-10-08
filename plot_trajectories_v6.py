from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = Path("results/badminton_coordinates_v6_full.csv")
OUTPUT_FILE = Path("results/court_trajectories_v6.png")

COURT_WIDTH = 5.18
COURT_LENGTH = 13.40

df = pd.read_csv(CSV_FILE)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("Rows:", len(df))

required = ["tracker_id", "court_x_m", "court_y_m"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError("Missing columns: " + str(missing))

df = df.dropna(subset=required).copy()

print("Rows with coordinates:", len(df))

print()
print("TRACKER COUNTS:")
print(df["tracker_id"].value_counts().sort_index())

fig, ax = plt.subplots(figsize=(8, 12))

for tracker_id, group in df.groupby("tracker_id"):

    if "frame" in group.columns:
        group = group.sort_values("frame")

    ax.plot(
        group["court_x_m"],
        group["court_y_m"],
        linewidth=1.2,
        alpha=0.75,
        label="ID {} ({})".format(tracker_id, len(group))
    )

    ax.scatter(
        group["court_x_m"].iloc[0],
        group["court_y_m"].iloc[0],
        s=25
    )

ax.set_xlim(-0.3, COURT_WIDTH + 0.3)
ax.set_ylim(-0.3, COURT_LENGTH + 0.3)

ax.set_xlabel("Court X (meters)")
ax.set_ylabel("Court Y (meters)")
ax.set_title("Badminton Player Trajectories - V6 Full")

ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

print()
print("Saved:", OUTPUT_FILE)

plt.show()
