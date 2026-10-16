"""Convert processed HumanDemo .npz files into RobotTrajectory .npz files.

Usage:
    python -m human_data.scripts.make_robot_demos \
        --demos_dir data/processed_human \
        --out_dir   data/robot_demos \
        [--method   object_relative]   # or naive \
        [--augment  8]                 # extra augmented per demo
"""
import argparse, glob, numpy as np
from pathlib import Path
from human_data.trajectory import HumanDemo
from human_data.calibration import TableToRobot
from human_data.retarget import retarget_object_relative, retarget_naive, augment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos_dir", default="data/processed_human")
    ap.add_argument("--out_dir",   default="data/robot_demos")
    ap.add_argument("--method",    default="object_relative", choices=["object_relative","naive"])
    ap.add_argument("--augment",   type=int, default=8)
    args = ap.parse_args()

    cal = TableToRobot(x=-0.10, y=0.0, yaw=3.14159, scale=0.8)   # identity — marker placed at robot base
    fn  = retarget_object_relative if args.method=="object_relative" else retarget_naive

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for path in sorted(glob.glob(f"{args.demos_dir}/*.npz")):
        demo = HumanDemo.load(path)
        traj = fn(demo, cal)
        traj.save(out/f"{demo.name}_{args.method}.npz")
        print(f"  {demo.name} → {traj.method}")
        for i, aug in enumerate(augment(traj, demo, cal, args.augment)):
            aug.save(out/f"{demo.name}_{args.method}_aug{i:02d}.npz")
        print(f"    + {args.augment} augmented")


if __name__ == "__main__":
    main()
