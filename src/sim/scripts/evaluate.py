"""Evaluate a trained LeRobot checkpoint (SmolVLA or ACT) on the pick-and-place task.

Usage (after `isaaclab -p`):
    python src/sim/scripts/evaluate.py \
        --task SOArm-PickPlace-v0 \
        --checkpoint path/to/checkpoint \
        --policy smolvla          # or act \
        --episodes 50 \
        [--headless] [--enable_cameras]
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Evaluate LeRobot policy on pick-and-place")
parser.add_argument("--task",       type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--policy",     type=str, default="smolvla", choices=["smolvla","act"])
parser.add_argument("--episodes",   type=int, default=50)
parser.add_argument("--output_csv", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv, numpy as np, torch, gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from tqdm import tqdm
import sim.tasks  # noqa
from sim.utils.sim_helpers import sim_step
from sim.tasks.pick_place.mdp.terminations import object_in_goal

TASK_STR = "pick up the blue cube and place it on the green goal"


def load_policy(checkpoint: str, policy_type: str, device: str):
    import torch
    from pathlib import Path
    from lerobot.policies.act.modeling_act import ACTPolicy
    
    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.eval().to(device)
    return policy


def obs_to_dict(base_env, device: str) -> dict:
    """Build the observation dict expected by lerobot predict_action."""
    qpos = base_env.scene["robot"].data.joint_pos[0].cpu().numpy()           # (6,)
    return {
        "observation.state":        np.rad2deg(qpos).astype(np.float32),
    }


def main():
    env_cfg  = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env      = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    device   = str(base_env.device)

    env.reset()
    sim_step(base_env, simulation_app)

    policy = load_policy(args_cli.checkpoint, args_cli.policy, device)

    successes, attempts = 0, 0
    episode_lengths = []

    with tqdm(total=args_cli.episodes, desc="eval") as pbar:
        while attempts < args_cli.episodes:
            obs_dict = obs_to_dict(base_env, device)
           
            state = torch.tensor(obs_dict["observation.state"], dtype=torch.float32).unsqueeze(0).to(device)
            dummy_image = torch.zeros(1, 3, 84, 84, dtype=torch.float32).to(device)
            obs_t = {
                "observation.state": state,
                "observation.images.front": dummy_image,
            }
            with torch.inference_mode():
                action = policy.select_action(obs_t)  # (1, 6) degrees
           
            action_rad = torch.deg2rad(action.float()).to(device)
            if action_rad.dim() == 1:
                action_rad = action_rad.unsqueeze(0)
            _, _, terminated, truncated, _ = env.step(action_rad)
            done = bool((terminated | truncated)[0].item())
            if done:
                success = bool(object_in_goal(base_env)[0].item())
                successes += int(success)
                attempts  += 1
                pbar.update(1)
                pbar.set_postfix({"success_rate": f"{successes/attempts:.2f}"})
                env.reset()
                sim_step(base_env, simulation_app)

    rate = successes / attempts
    print(f"\nSuccess rate: {successes}/{attempts} = {rate:.3f}")

    if args_cli.output_csv:
        with open(args_cli.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["policy", "checkpoint", "episodes", "successes", "success_rate"])
            w.writerow([args_cli.policy, args_cli.checkpoint, attempts, successes, rate])
        print(f"results -> {args_cli.output_csv}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
