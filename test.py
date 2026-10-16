import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

d = np.load("data/robot_demos/naive/WIN_20261016_19_07_57_Pro_naive.npz")
cube = d["cube_start_robot"]
goal = d["goal_robot"]

Path("assets").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(4,4))
ax.set_xlim(-0.05, 0.40); ax.set_ylim(-0.25, 0.30)
ax.add_patch(plt.Rectangle((cube[0]-0.045, cube[1]-0.035), 0.09, 0.07, color="saddlebrown", label="object"))
ax.add_patch(plt.Circle((goal[0], goal[1]), 0.095, color="royalblue", fill=False, lw=3, label="goal"))
ax.plot(0, 0, "k^", ms=10, label="robot base")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Sim scene — top-down view")
ax.legend(fontsize=8); ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
fig.savefig("assets/sim_frame.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved")