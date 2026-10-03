from .trajectory import HumanFrame, RobotTarget
from .calibration import WorkspaceCalibration

def retarget_frame(frame: HumanFrame, calibration: WorkspaceCalibration) -> RobotTarget:
    """Baseline wrist-to-end-effector retargeting.

    Keep this baseline simple. The research extensions can compare it with
    object-relative or phase/subgoal retargeting later.
    """
    ee = calibration.map_position(frame.wrist_position)
    gripper = 1.0 - max(0.0, min(1.0, frame.hand_openness))
    return RobotTarget(
        timestamp=frame.timestamp,
        ee_position=tuple(float(x) for x in ee),
        ee_quaternion=frame.wrist_quaternion,
        gripper=gripper,
    )
