"""Replay retargeted human demos in Isaac Lab, keep successes, export LeRobot dataset.

Usage (after `isaaclab -p`):
    python src/sim/scripts/replay_demos.py \
        --task SOArm-PickPlace-v0 \
        --demos_dir data/robot_demos \
        --repo_id local/hand_to_robot_demos \
        [--headless] [--enable_cameras]
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Replay human demos in Isaac Lab")
parser.add_argument("--task",       type=str, default="SOArm-PickPlace-v0")
parser.add_argument("--demos_dir",  type=str, default="data/robot_demos",
                    help="folder of .npz files from human_data retargeting")
parser.add_argument("--repo_id",    type=str, default="local/hand_to_robot_demos")
parser.add_argument("--num_envs",   type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
args_cli.kit_app = "isaaclab.python.headless.rendering.kit"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob, numpy as np, torch, gymnasium as gym
from pathlib import Path
from tqdm import tqdm
from isaaclab_tasks.utils import parse_env_cfg
import sim.tasks  # registers SOArm-PickPlace-v0
from sim.utils.sim_helpers import sim_step, set_arm_joints, make_lerobot_dataset, add_frame_to_dataset
from sim.utils.robot_sim import JOINT_NAMES, ARRIVAL_THRESHOLD, POSITIONS
from sim.tasks.pick_place.mdp.terminations import object_in_goal

TASK_STR = "pick up the blue cube and place it on the green goal"


def settle(base_env, sa, n=80):
    for _ in range(n):
        sa.set_joint_position_target(torch.tensor(POSITIONS["home"], device=str(base_env.device)).unsqueeze(0))
        sa.write_data_to_sim()
        sim_step(base_env, simulation_app)


def replay_one(base_env, sa, demo: dict, device: str) -> list[dict]:
    
    """Replay one .npz demo. Returns list of frame dicts (empty = failure)."""
   
    q_targets = demo["joint_targets_rad"]  # [T, 6]
    cube_pos  = demo["cube_start_robot"]   # [3]
    goal_pos  = demo["goal_robot"]         # [3]
    T = len(q_targets)

    # teleport cube and goal to their demo positions
    obj  = base_env.scene["object"]
    goal = base_env.scene["goal_marker"]
    env0 = torch.tensor([0], device=device)
    cp = torch.zeros(7, device=device); cp[:3] = torch.tensor(cube_pos); cp[3] = 1.0; cp[2] = 0.02
    gp = torch.zeros(7, device=device); gp[:3] = torch.tensor(goal_pos); gp[3] = 1.0
    obj.write_root_pose_to_sim(cp.unsqueeze(0), env_ids=env0)
    goal.write_root_pose_to_sim(gp.unsqueeze(0), env_ids=env0)

    settle(base_env, sa, n=40)

    frames = []
    for t in range(T):
        
        limits = torch.tensor([-1.57, -1.57, -1.52, -1.49, -2.46, -0.1], device=device)
        limits_hi = torch.tensor([1.57, 1.57, 1.52, 1.49, 2.56, 1.0], device=device)
        q = torch.tensor(q_targets[t], dtype=torch.float32, device=device)
        q = torch.clamp(q, limits, limits_hi).unsqueeze(0)
        sa.write_data_to_sim()
        sim_step(base_env, simulation_app)

        rgb = np.zeros((84, 84, 3), dtype=np.uint8)   # (H,W,3) uint8
        qpos = sa.data.joint_pos[0].cpu().numpy()
        qact = q_targets[t]
        frames.append({"rgb": rgb, "state": np.rad2deg(qpos), "action": np.rad2deg(qact)})
        

    success = bool(object_in_goal(base_env)[0].item())
    return frames if success else []


def main():
    env_cfg  = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env      = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    device   = str(base_env.device)
    sa       = base_env.scene["robot"]

    env.reset()
    sim_step(base_env, simulation_app)

    demo_paths = sorted(glob.glob(f"{args_cli.demos_dir}/*.npz"))
    if not demo_paths:
        raise SystemExit(f"no .npz files found in {args_cli.demos_dir}")

    dataset = make_lerobot_dataset(args_cli.repo_id)
    n_ok, n_fail = 0, 0

    for path in tqdm(demo_paths, desc="demos"):
        demo = dict(np.load(path))
        frames = replay_one(base_env, sa, demo, device)
        if not frames:
            n_fail += 1
            tqdm.write(f"  FAIL  {Path(path).name}")
            continue
         # save sim diagram on first success
        if n_ok == 0:
            _d = dict(np.load(path))
            _cube = _d["cube_start_robot"]
            _goal = _d["goal_robot"]
            if _cube[0] > 0.03:  # only save if cube is actually in workspace
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                Path("assets").mkdir(exist_ok=True)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_xlim(-0.05, 0.40); ax.set_ylim(-0.25, 0.30)
                ax.add_patch(plt.Rectangle((_cube[0]-0.045, _cube[1]-0.035), 0.09, 0.07, color="saddlebrown", label="object"))
                ax.add_patch(plt.Circle((_goal[0], _goal[1]), 0.095, color="royalblue", fill=False, lw=3, label="goal"))
                ax.plot(0, 0, "k^", ms=10, label="robot base")
                ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
                ax.set_title("Sim scene — top-down view")
                ax.legend(fontsize=8); ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)
                fig.savefig("assets/sim_frame.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
        for f in frames:
            add_frame_to_dataset(dataset, f["state"], f["action"], f["rgb"], TASK_STR)
        dataset.save_episode()
        n_ok += 1
        tqdm.write(f"  OK    {Path(path).name}  ({len(frames)} frames)")

    dataset.finalize()
    print(f"\nDone: {n_ok} kept, {n_fail} failed → dataset at {args_cli.repo_id}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
