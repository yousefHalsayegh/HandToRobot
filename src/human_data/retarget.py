"""Retarget a HumanDemo into a RobotTrajectory using object-relative coordinates.

Two methods are exported so you can compare them:
  - retarget_naive:          raw wrist position → workspace-scaled IK
  - retarget_object_relative: hand offset from cube/goal → scaled IK  (main contribution)
"""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from .trajectory import HumanDemo, RobotTrajectory
from .calibration import TableToRobot

# lazy import so the module loads without Isaac Lab
def _ik():
    from sim.utils.robot_sim import JOINT_NAMES
    # IK is in so101_kinematics, shared between pipeline and sim
    try:
        from human_data.so101_kinematics import SO101Kinematics, gripper_from_openness
        return SO101Kinematics(), gripper_from_openness
    except ImportError:
        raise ImportError("scipy is required for IK: pip install scipy")


GRASP_Z   = 0.018   # tool-point height when grasping on the table surface
LIFT_Z    = 0.075   # lift height
PLACE_Z   = 0.022   # tool-point height when releasing on goal


def _smooth(arr, fps, window_s=0.10):
    w = max(int(window_s * fps) | 1, 3)
    return savgol_filter(arr, w, 2, axis=0)


def _fill_gaps(arr):
    """Linear interpolation over NaN rows."""
    a = arr.copy()
    valid = ~np.isnan(a[:,0])
    idx   = np.where(valid)[0]
    for d in range(a.shape[1]):
        a[:,d] = np.interp(np.arange(len(a)), idx, a[idx,d])
    return a


def _cube_track_robot(demo: HumanDemo, cal: TableToRobot) -> np.ndarray:
    filled = _fill_gaps(demo.cube_table)
    return np.stack([cal.map(filled[t]) for t in range(len(demo))])


def _solve_trajectory(ee_robot: np.ndarray, openness: np.ndarray, method: str) -> np.ndarray:
    kin, g_from_o = _ik()
    T = len(ee_robot)
    Q = np.zeros((T, 6))
    q_prev = None
    for t in range(T):
        q, _ = kin.ik(ee_robot[t], q_seed=q_prev)
        o = openness[t] if not np.isnan(openness[t]) else 1.0  # default open
        g = g_from_o(o)
        Q[t, :5] = q
        Q[t, 5]  = g
        q_prev = q
    return Q


def retarget_naive(demo: HumanDemo, cal: TableToRobot,
                   cube_robot=None, goal_robot=None) -> RobotTrajectory:
    """Baseline: map hand position directly (workspace-scaled)."""
    cube_r = _cube_track_robot(demo, cal) if cube_robot is None else np.broadcast_to(cube_robot, (len(demo),3)).copy()
    cube_start = cube_r[demo.grasp_idx]
    goal       = goal_robot if goal_robot is not None else cal.map(demo.cube_end)

    grasp_r = np.stack([cal.map(demo.grasp_table[t]) for t in range(len(demo))])
    grasp_r = np.clip(_smooth(grasp_r, demo.fps), None, None)  # smooth only
    grasp_r[:, 2] = np.clip(grasp_r[:, 2], 0.012, LIFT_Z)

    Q = _solve_trajectory(grasp_r, demo.openness, "naive")
    return RobotTrajectory(fps=demo.fps, joint_targets_rad=Q,
                           cube_start_robot=cube_start, goal_robot=goal, method="naive")


def retarget_object_relative(demo: HumanDemo, cal: TableToRobot,
                              cube_robot=None, goal_robot=None) -> RobotTrajectory:
    """Main contribution: express hand offset relative to cube (pre-grasp) and
    goal (post-release), then re-anchor to target sim positions.
    Depth errors in the hand track mostly cancel out."""
    cube_r     = _cube_track_robot(demo, cal) if cube_robot is None else np.broadcast_to(cube_robot, (len(demo),3)).copy()
    cube_start = cube_r[demo.grasp_idx].copy()
    goal       = goal_robot if goal_robot is not None else cal.map(demo.cube_end)

    # human hand offset from the relevant anchor
    grasp_r    = np.stack([cal.map(demo.grasp_table[t]) for t in range(len(demo))])
    offset     = grasp_r - cube_r              # [T,3] offset from cube

    gi, ri = demo.grasp_idx, demo.release_idx
    T      = len(demo)
    ee     = np.zeros((T, 3))

    # approach phase: hand offset from cube start
    for t in range(gi+1):
        ee[t] = cube_start + offset[t]
        ee[t, 2] = GRASP_Z if t == gi else max(offset[t,2], GRASP_Z)

    # transport phase: linearly blend from cube_start to goal
    n_trans = max(ri - gi, 1)
    for t in range(gi, ri+1):
        alpha = (t - gi) / n_trans
        anchor = (1-alpha)*cube_start + alpha*goal
        ee[t]    = anchor
        ee[t, 2] = LIFT_Z   # carry high

    # place phase: descend to goal
    for t in range(ri, T):
        ee[t]    = goal
        ee[t, 2] = PLACE_Z

    ee = np.clip(_smooth(ee, demo.fps), None, None)
    ee[:, 2] = np.clip(ee[:, 2], 0.012, LIFT_Z)

    Q = _solve_trajectory(ee, demo.openness, "object_relative")
    return RobotTrajectory(fps=demo.fps, joint_targets_rad=Q,
                           cube_start_robot=cube_start, goal_robot=goal,
                           method="object_relative")


def augment(traj: RobotTrajectory, demo: HumanDemo, cal: TableToRobot,
            n_augmented: int = 8, seed: int = 0) -> list[RobotTrajectory]:
    """Re-anchor the same human motion to randomly sampled cube/goal positions."""
    from .calibration import TableToRobot
    rng    = np.random.default_rng(seed)
    out    = []
    for _ in range(n_augmented):
        cx = rng.uniform(0.15, 0.24); cy = rng.uniform(-0.12, 0.00)
        gx = rng.uniform(0.15, 0.24); gy = rng.uniform(0.04, 0.14)
        if abs(np.hypot(gx-cx, gy-cy)) < 0.08: continue  # too close
        t = retarget_object_relative(demo, cal,
                                     cube_robot=np.array([cx, cy, 0.02]),
                                     goal_robot=np.array([gx, gy, 0.022]))
        out.append(t)
    return out
