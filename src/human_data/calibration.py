"""Robot-base frame relative to the table (ArUco marker) frame."""
from dataclasses import dataclass
import numpy as np

@dataclass
class TableToRobot:
    """Uniform-scale + yaw transform from table frame to robot base frame.
    Default: identity (put the marker where the robot base sits, +x pointing
    toward the workspace)."""
    x: float = 0.0; y: float = 0.0; yaw: float = 0.0; scale: float = 1.0

    @property
    def R(self):
        c,s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

    def map(self, p_table) -> np.ndarray:
        p = np.asarray(p_table, float)
        return self.scale*(self.R @ p) + np.array([self.x, self.y, 0.])
