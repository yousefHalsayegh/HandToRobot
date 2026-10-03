from dataclasses import dataclass
import numpy as np

@dataclass
class WorkspaceCalibration:
    human_min: np.ndarray
    human_max: np.ndarray
    robot_min: np.ndarray
    robot_max: np.ndarray

    def map_position(self, position) -> np.ndarray:
        p = np.asarray(position, dtype=np.float32)
        span = np.maximum(self.human_max - self.human_min, 1e-6)
        unit = np.clip((p - self.human_min) / span, 0.0, 1.0)
        return self.robot_min + unit * (self.robot_max - self.robot_min)
