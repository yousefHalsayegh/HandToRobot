from __future__ import annotations
import torch


def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length - 1


def object_in_goal(env, xy_tolerance: float = 0.09, z_tolerance: float = 0.08) -> torch.Tensor:
    obj = env.scene["object"].data.root_pos_w
    goal = env.scene["goal_marker"].data.root_pos_w
    xy_ok = torch.linalg.norm(obj[:, :2] - goal[:, :2], dim=-1) < xy_tolerance
    z_ok = torch.abs(obj[:, 2] - goal[:, 2]) < z_tolerance
    return xy_ok & z_ok
