from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

@dataclass
class HumanDemo:
    """Output of process_video: all tracks in the table frame (metres)."""
    name: str
    fps: float
    grasp_px: np.ndarray        # [T,2]  thumb+index midpoint in pixels
    grasp_table: np.ndarray     # [T,3]  grasp point, table frame
    openness: np.ndarray        # [T]    0=closed, 1=open
    cube_table: np.ndarray      # [T,3]  cube centre, table frame (NaN when lost)
    cube_start: np.ndarray      # [3]
    cube_end: np.ndarray        # [3]
    grasp_idx: int
    release_idx: int

    def __len__(self):
        return len(self.openness)

    def save(self, path):
        np.savez_compressed(path, **{k: v for k, v in self.__dict__.items()})

    @classmethod
    def load(cls, path) -> "HumanDemo":
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in z.files}
        kw["name"] = str(kw["name"]); kw["fps"] = float(kw["fps"])
        kw["grasp_idx"] = int(kw["grasp_idx"]); kw["release_idx"] = int(kw["release_idx"])
        return cls(**kw)


@dataclass
class RobotTrajectory:
    """Output of retarget: joint targets ready for Isaac Lab."""
    fps: float
    joint_targets_rad: np.ndarray   # [T,6]
    cube_start_robot: np.ndarray    # [3]
    goal_robot: np.ndarray          # [3]
    method: str = ""

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)
