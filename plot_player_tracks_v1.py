from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("results/player_tracks_v1.csv")
OUTPUT = Path("results/player_tracks_v1_plot.png")

df = pd.read_csv(INPUT)

fig, ax = plt.subplots(figsize=(8, 12))

for player in ["Player_A", "Player_B"]:

    p = df[df["player"] == player].sort_values("frame")

    ax.plot(
        p["court_x_m"],
        p["court_y_m"],
        linewidth=1.5,
        alpha=0.8,
        label="{} ({} points)".format(player, len(p))
    )

    # Start
    ax.scatter(
        p["court_x_m"].iloc[0],
        p["court_y_m"].iloc[0],
        s=70,
        marker="o"
    )

    # End
    ax.scatter(
        p["court_x_m"].iloc[-1],
        p["court_y_m"].iloc[-1],
        s=70,
        marker="x"
    )

# Court midpoint
ax.axhline(
    6.7,
    linewidth=1,
    linestyle="--",
    label="Court midpoint"
)

ax.set_xlim(-0.3, 5.5)
ax.set_ylim(-0.3, 13.7)

ax.set_xlabel("Court X (meters)")
ax.set_ylabel("Court Y (meters)")
ax.set_title("Reconstructed Player Trajectories")

ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

print()
print("Saved:", OUTPUT)

plt.show()
