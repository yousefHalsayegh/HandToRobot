"""Visualise processed HumanDemo .npz files without Isaac Lab.

Shows:
  - Hand openness over time with grasp/release markers
  - Cube trajectory in the table frame (top-down view)
  - Retargeted joint targets over time (if robot demo .npz also provided)

Usage:
    python visualise_demos.py                          # pick from data/processed_human/
    python visualise_demos.py demo_01.npz              # specific processed demo
    python visualise_demos.py demo_01.npz robot.npz    # also show joint targets
"""

import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def load_human(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_robot(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def pick_file(folder, pattern):
    files = sorted(glob.glob(str(Path(folder) / pattern)))
    if not files:
        return None
    print("Available demos:")
    for i, f in enumerate(files):
        print(f"  [{i}] {Path(f).name}")
    idx = input("Pick number (enter for 0): ").strip()
    idx = int(idx) if idx else 0
    return files[idx]


def visualise(human_path, robot_path=None):
    d = load_human(human_path)
    name = Path(human_path).stem

    has_robot = robot_path is not None
    r = load_robot(robot_path) if has_robot else None

    T = len(d["openness"])
    t = np.arange(T)
    gi = int(d["grasp_idx"])
    ri = int(d["release_idx"])

    n_rows = 3 if has_robot else 2
    fig = plt.figure(figsize=(14, 4 * n_rows))
    fig.suptitle(name, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(n_rows, 2, figure=fig)

    # ── 1. Hand openness ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, d["openness"], color="#2196F3", lw=1.5, label="openness")
    ax1.axvline(gi, color="red",   lw=1.5, ls="--", label=f"grasp={gi}")
    ax1.axvline(ri, color="green", lw=1.5, ls="--", label=f"release={ri}")
    ax1.fill_between(t, 0, d["openness"], alpha=0.15, color="#2196F3")
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_ylabel("Openness  (0=pinched, 1=open)")
    ax1.set_xlabel("Frame")
    ax1.legend(loc="upper right")
    ax1.set_title("Hand openness")
    ax1.grid(True, alpha=0.3)

    # ── 2. Cube top-down trajectory (table frame) ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    cube = d["cube_table"]
    valid = ~np.isnan(cube[:, 0])
    if valid.any():
        sc = ax2.scatter(cube[valid, 0], cube[valid, 1],
                         c=np.flatnonzero(valid), cmap="viridis",
                         s=8, label="cube track")
        plt.colorbar(sc, ax=ax2, label="frame")
        if valid[gi]:
            ax2.scatter(*cube[gi, :2], color="red",   s=80, zorder=5, label="grasp")
        if valid[ri]:
            ax2.scatter(*cube[ri, :2], color="green", s=80, zorder=5, label="release")
    ax2.set_xlabel("x (m, table frame)")
    ax2.set_ylabel("y (m, table frame)")
    ax2.set_title("Cube trajectory — top-down")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal", "box")
    ax2.grid(True, alpha=0.3)

    # ── 3. Grasp point trajectory (top-down) ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    grasp = d["grasp_table"]
    gvalid = ~np.isnan(grasp[:, 0])
    if gvalid.any():
        sc2 = ax3.scatter(grasp[gvalid, 0], grasp[gvalid, 1],
                          c=np.flatnonzero(gvalid), cmap="plasma",
                          s=8, label="hand track")
        plt.colorbar(sc2, ax=ax3, label="frame")
    ax3.set_xlabel("x (m, table frame)")
    ax3.set_ylabel("y (m, table frame)")
    ax3.set_title("Hand (grasp point) trajectory — top-down")
    ax3.set_aspect("equal", "box")
    ax3.grid(True, alpha=0.3)

    # ── 4. Robot joint targets (if available) ────────────────────────────────
    if has_robot:
        ax4 = fig.add_subplot(gs[2, :])
        joints = r["joint_targets_rad"]
        names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                 "wrist_flex", "wrist_roll", "gripper"]
        Tr = len(joints)
        for j, name_j in enumerate(names):
            ax4.plot(np.arange(Tr), np.rad2deg(joints[:, j]),
                     lw=1.2, label=name_j)
        ax4.set_xlabel("Frame")
        ax4.set_ylabel("Joint target (degrees)")
        ax4.set_title("Retargeted joint targets")
        ax4.legend(fontsize=8, ncol=3)
        ax4.grid(True, alpha=0.3)

        # print summary
        cs = r["cube_start_robot"]
        g  = r["goal_robot"]
        print(f"\nRobot demo summary:")
        print(f"  cube start (robot frame): {np.round(cs, 3)}")
        print(f"  goal       (robot frame): {np.round(g,  3)}")
        print(f"  separation: {np.linalg.norm(cs[:2] - g[:2]):.3f} m")
        print(f"  method: {r.get('method', b'').decode() if isinstance(r.get('method',''), bytes) else r.get('method','?')}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) >= 1:
        human_path = args[0]
    else:
        human_path = pick_file("data/processed_human", "*.npz")
        if not human_path:
            print("No demos found in data/processed_human/")
            sys.exit(1)

    robot_path = args[1] if len(args) >= 2 else None

    # auto-find matching robot demo if not given
    if robot_path is None:
        stem = Path(human_path).stem
        candidates = sorted(glob.glob(f"data/robot_demos/**/{stem}*.npz", recursive=True))
        if candidates:
            print(f"\nFound robot demo: {candidates[0]}")
            robot_path = candidates[0]

    visualise(human_path, robot_path)