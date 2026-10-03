"""Hand-tracking adapter.

Implement the chosen hand tracker here. Keep tracker output independent of
Isaac Lab by emitting HumanFrame records from trajectory.py.
"""
from pathlib import Path

def extract_hand_trajectory(video_path: str | Path):
    raise NotImplementedError("Connect the selected hand-pose estimator here.")
