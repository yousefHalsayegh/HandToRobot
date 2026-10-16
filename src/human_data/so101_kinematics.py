"""Pure-NumPy forward/inverse kinematics for the SO-ARM101, read from the same
URDF that Isaac Lab imports. No Isaac dependency, so it runs anywhere.

Frames:
  - robot base frame = URDF ``base_link`` (Isaac: the robot root, z up).
  - tool point       = URDF ``gripper_frame_link`` (between the jaw tips).

IK solves position + "approach axis points down" for the 5 arm joints with
bounded least squares, respecting Isaac's soft joint limits.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

URDF_PATH = Path(__file__).resolve().parents[1] / "sim/robots/trs_so101/urdf/so_arm101.urdf"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINT_NAMES[:5]
TOOL_LINK = "gripper_frame_link"
# Must match ``soft_joint_pos_limit_factor`` in sim/robots/trs_so101/so_arm101.py
SOFT_LIMIT_FACTOR = 0.9


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle_R(axis, a):
    x, y, z = axis / np.linalg.norm(axis)
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


@dataclass
class _Joint:
    name: str
    type: str
    parent: str
    child: str
    T_origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class SO101Kinematics:
    def __init__(self, urdf_path: str | Path = URDF_PATH, tool_link: str = TOOL_LINK):
        root = ET.parse(urdf_path).getroot()
        joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy_to_R(*rpy), xyz
            a = j.find("axis")
            axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.array([0, 0, 1.0])
            lim = j.find("limit")
            lo = float(lim.get("lower")) if lim is not None else 0.0
            hi = float(lim.get("upper")) if lim is not None else 0.0
            joints[j.find("child").get("link")] = _Joint(
                j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"), T, axis, lo, hi
            )
        # chain base -> tool
        chain, link = [], tool_link
        while link in joints:
            chain.append(joints[link])
            link = joints[link].parent
        self.chain = chain[::-1]
        self.all_joints = {j.name: j for j in joints.values()}
        mid = {n: 0.5 * (self.all_joints[n].lower + self.all_joints[n].upper) for n in JOINT_NAMES}
        half = {n: 0.5 * (self.all_joints[n].upper - self.all_joints[n].lower) * SOFT_LIMIT_FACTOR for n in JOINT_NAMES}
        self.lower = np.array([mid[n] - half[n] for n in JOINT_NAMES])
        self.upper = np.array([mid[n] + half[n] for n in JOINT_NAMES])

    # ------------------------------------------------------------------ FK
    def fk(self, q: np.ndarray) -> np.ndarray:
        """4x4 pose of the tool frame in the robot base frame. q: 5 or 6 joints (rad)."""
        qd = dict(zip(JOINT_NAMES, np.asarray(q, dtype=float)))
        T = np.eye(4)
        for j in self.chain:
            T = T @ j.T_origin
            if j.type == "revolute":
                R = np.eye(4)
                R[:3, :3] = _axis_angle_R(j.axis, qd.get(j.name, 0.0))
                T = T @ R
        return T

    def fk_frames(self, q) -> dict[str, np.ndarray]:
        """Pose of every link on the chain (base frame), keyed by child link name."""
        qd = dict(zip(JOINT_NAMES, np.asarray(q, dtype=float)))
        T, out = np.eye(4), {}
        for j in self.chain:
            T = T @ j.T_origin
            if j.type == "revolute":
                R = np.eye(4)
                R[:3, :3] = _axis_angle_R(j.axis, qd.get(j.name, 0.0))
                T = T @ R
            out[j.child] = T.copy()
        return out

    def closing_axis(self, q) -> np.ndarray:
        """Direction in which the jaws close (base frame): perpendicular to the jaw
        hinge and to the approach axis. Used to align the jaws with the cube faces."""
        Tg = self.fk_frames(q)["gripper_link"]
        g = self.all_joints["gripper"]
        hinge = Tg[:3, :3] @ g.T_origin[:3, :3] @ g.axis
        c = np.cross(hinge, self.approach_axis(q))
        return c / (np.linalg.norm(c) + 1e-9)

    def roll_aligned_to_yaw(self, q, target_pos, cube_yaw: float = 0.0):
        """Choose wrist_roll so the jaws close across two parallel cube faces, then
        re-solve IK with that roll fixed. Returns (q, err)."""
        c = self.closing_axis(q)
        yaw = np.arctan2(c[1], c[0])
        rel = (yaw - cube_yaw) % (np.pi / 2)
        delta = -rel if rel < np.pi / 4 else (np.pi / 2 - rel)
        best = None
        for d in (delta, -delta, delta + np.pi / 2, -delta - np.pi / 2, delta - np.pi / 2, -delta + np.pi / 2):
            roll = q[4] + d
            if not (self.lower[4] <= roll <= self.upper[4]):
                continue
            qq, err = self.ik(target_pos, q_seed=np.r_[q[:4], roll], wrist_roll=roll)
            cc = self.closing_axis(qq)
            mis = abs(((np.arctan2(cc[1], cc[0]) - cube_yaw + np.pi / 4) % (np.pi / 2)) - np.pi / 4)
            score = err + 0.01 * mis
            if best is None or score < best[2]:
                best = (qq, err, score)
        return (best[0], best[1]) if best else (q, np.linalg.norm(self.tool_position(q) - target_pos))

    def tool_position(self, q):
        return self.fk(q)[:3, 3]

    # ------------------------------------------------------------------ IK
    def _seed_table(self):
        """Random joint samples with their tool position/approach, used to
        cold-start IK near the right solution branch (built once, ~2 s)."""
        if not hasattr(self, "_table"):
            rng = np.random.default_rng(0)
            Q = rng.uniform(self.lower[:5], self.upper[:5], size=(20000, 5))
            P = np.array([self.tool_position(q) for q in Q])
            A = np.array([self.approach_axis(q) for q in Q])
            self._table = (Q, P, A)
        return self._table

    def _nearest_seeds(self, target, wrist_roll, n=4):
        Q, P, A = self._seed_table()
        score = np.linalg.norm(P - target, axis=1) + 0.05 * np.linalg.norm(A - [0, 0, -1.0], axis=1)
        seeds = Q[np.argsort(score)[:n]].copy()
        if wrist_roll is not None:
            seeds[:, 4] = wrist_roll
        lo, hi = self.lower[:5], self.upper[:5]
        return [np.clip(q, lo + 1e-4, hi - 1e-4) for q in seeds]

    def approach_axis(self, q) -> np.ndarray:
        """Unit vector, in base frame, pointing from the wrist towards the jaw tips."""
        T = self.fk(q)
        Tw = self._wrist_pose(q)
        v = T[:3, 3] - Tw[:3, 3]
        return v / np.linalg.norm(v)

    def _wrist_pose(self, q):
        qd = dict(zip(JOINT_NAMES, np.asarray(q, dtype=float)))
        T = np.eye(4)
        for j in self.chain:
            T = T @ j.T_origin
            if j.name == "wrist_roll":
                return T
            if j.type == "revolute":
                R = np.eye(4)
                R[:3, :3] = _axis_angle_R(j.axis, qd.get(j.name, 0.0))
                T = T @ R
        return T

    def ik(self, target_pos, q_seed=None, wrist_roll=None, tol: float = 0.002):
        """Position is the priority; the jaws point down as much as the joint
        limits allow (near the workspace edge the wrist tilts). Returns (q[5], err_m)."""
        best = None
        for w in (0.3, 0.1, 0.03, 0.0):
            q, err = self._ik_weighted(target_pos, q_seed, w, wrist_roll)
            if best is None or err < best[1]:
                best = (q, err)
            if err < tol:
                return q, err
        return best

    def _ik_weighted(
        self,
        target_pos,
        q_seed=None,
        top_down_weight: float = 0.3,
        wrist_roll: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Return (q[5], position_error_m). Gripper joint is not touched.

        ``top_down_weight`` > 0 asks the jaws to point straight down (-z);
        ``wrist_roll`` fixes the roll joint (free if None).
        """
        target = np.asarray(target_pos, dtype=float)
        lo, hi = self.lower[:5].copy(), self.upper[:5].copy()
        seed = np.clip(np.zeros(5) if q_seed is None else np.asarray(q_seed[:5], float), lo + 1e-4, hi - 1e-4)
        if wrist_roll is not None:
            lo[4], hi[4] = wrist_roll - 1e-6, wrist_roll + 1e-6
            seed[4] = wrist_roll

        def residual(q):
            r = [self.tool_position(q) - target]
            if top_down_weight > 0:
                r.append(top_down_weight * (self.approach_axis(q) - np.array([0, 0, -1.0])))
            return np.concatenate(r)

        best = None
        seeds = [seed] if q_seed is not None else self._nearest_seeds(target, wrist_roll)
        for s in seeds:
            sol = least_squares(residual, s, bounds=(lo, hi), xtol=1e-10, ftol=1e-10, max_nfev=300)
            err = np.linalg.norm(self.tool_position(sol.x) - target)
            score = err + 0.01 * np.linalg.norm(sol.fun[3:]) if top_down_weight > 0 else err
            if best is None or score < best[2]:
                best = (sol.x, err, score)
        return best[0], best[1]


def gripper_from_openness(openness: float, closed: float = -0.10, open_: float = 1.0) -> float:
    """Map human openness (0 = closed, 1 = open) to the SO-101 gripper joint (rad).
    ``closed`` slightly below 0 so the position controller squeezes the cube."""
    o = float(np.clip(openness, 0.0, 1.0))
    return closed + o * (open_ - closed)
