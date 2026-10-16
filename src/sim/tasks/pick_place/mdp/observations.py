from __future__ import annotations
import torch
import numpy as np


def joint_positions(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_pos.clone()

def object_position_w(env) -> torch.Tensor:
    return env.scene["object"].data.root_pos_w.clone()

def ee_position_w(env) -> torch.Tensor:
    return env.scene["ee_frame"].data.target_pos_w[..., 0, :].clone()

def object_to_goal(env) -> torch.Tensor:
    return env.scene["goal_marker"].data.root_pos_w - env.scene["object"].data.root_pos_w

# def front_camera_rgb(env) -> torch.Tensor:
#     """Returns (num_envs, H, W, 3) uint8 RGB from the front camera."""
#     return env.scene["front_cam"].data.output["rgb"].clone()
