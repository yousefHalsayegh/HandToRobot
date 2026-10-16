import numpy as np
import torch

def sim_step(base_env, simulation_app):
    base_env.sim.step()
    base_env.scene.update(base_env.sim.get_physics_dt())
    simulation_app.update()
 
 

def set_arm_joints(base_env, joint_positions: np.ndarray, device: str):
    """Teleport arm to exact joint positions instantly."""
    robot = base_env.scene["robot"]
    target = torch.tensor(
        joint_positions, dtype=torch.float32, device=device
    ).unsqueeze(0)
    zeros = torch.zeros_like(target)
    env_ids = torch.tensor([0], device=device)
    robot.write_joint_state_to_sim(target, zeros, env_ids=env_ids)
 
 

LEROBOT_REPO_ID = "local/hand_to_robot_demos"
LEROBOT_FEATURES = {
    "observation.state": {
        "dtype": "float32", "shape": (6,),
        "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                  "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
    },
    "action": {
        "dtype": "float32", "shape": (6,),
        "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                  "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
    },
    "observation.images.front": {
    "dtype": "video", "shape": (84, 84, 3),
    "names": ["height", "width", "channels"],
},
}




def make_lerobot_dataset(repo_id: str = LEROBOT_REPO_ID, fps: int = 30):
    """Create a fresh LeRobot dataset for the pick-and-place demos."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="so101_follower",
        features=LEROBOT_FEATURES,
    )


def add_frame_to_dataset(dataset, joint_pos_deg, action_deg, rgb_hwc, task: str):
    """Add one timestep. joint_pos_deg and action_deg are numpy arrays of 6 values in degrees."""
    import numpy as np
    frame = {
        "observation.state": np.asarray(joint_pos_deg, dtype=np.float32),
        "action":            np.asarray(action_deg,    dtype=np.float32),
        "task":              task,
    }
    if rgb_hwc is not None:
        frame["observation.images.front"] = np.asarray(rgb_hwc, dtype=np.uint8)
    dataset.add_frame(frame)
