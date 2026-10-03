from __future__ import annotations
import torch


def joint_positions(env) -> torch.Tensor:
    """Robot joint positions in simulator joint order."""
    return env.scene["robot"].data.joint_pos.clone()


def object_position_w(env) -> torch.Tensor:
    """World-frame object position."""
    return env.scene["object"].data.root_pos_w


def ee_position_w(env) -> torch.Tensor:
    """World-frame end-effector position from the frame transformer."""
    return env.scene["ee_frame"].data.target_pos_w[..., 0, :]


def object_to_goal(env) -> torch.Tensor:
    """Vector from object centre to goal centre."""
    return env.scene["goal"].data.root_pos_w - env.scene["object"].data.root_pos_w
