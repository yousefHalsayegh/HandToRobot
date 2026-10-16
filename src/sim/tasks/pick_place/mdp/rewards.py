from __future__ import annotations
import torch

def reaching_reward(env, std: float = 0.10) -> torch.Tensor:
    ee = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    obj = env.scene["object"].data.root_pos_w
    d = torch.linalg.norm(ee - obj, dim=-1)
    return 1.0 - torch.tanh(d / std)

def lift_reward(env, min_height: float = 0.05) -> torch.Tensor:
    object_z = env.scene["object"].data.root_pos_w[:, 2]
    table_z = env.scene.env_origins[:, 2]
    return (object_z > table_z + min_height).float()

def place_reward(env, std: float = 0.08) -> torch.Tensor:
    obj = env.scene["object"].data.root_pos_w
    goal = env.scene["goal_marker"].data.root_pos_w
    d = torch.linalg.norm(obj - goal, dim=-1)
    return 1.0 - torch.tanh(d / std)
