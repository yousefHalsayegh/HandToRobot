from dataclasses import dataclass
from typing import Optional

@dataclass
class HumanFrame:
    timestamp: float
    wrist_position: tuple[float, float, float]
    wrist_quaternion: tuple[float, float, float, float]
    hand_openness: float
    object_position: Optional[tuple[float, float, float]] = None

@dataclass
class RobotTarget:
    timestamp: float
    ee_position: tuple[float, float, float]
    ee_quaternion: tuple[float, float, float, float]
    gripper: float
