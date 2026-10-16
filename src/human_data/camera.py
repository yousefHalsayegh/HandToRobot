"""Camera model: intrinsics + ArUco-based pose estimation.

The 'table frame' is defined by the printed ArUco marker:
  origin = marker centre, x/y along the marker axes, z up out of the table.
"""
from __future__ import annotations
import json, cv2, numpy as np
from dataclasses import dataclass
from pathlib import Path

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID  = 0


@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int

    @classmethod
    def approximate_phone(cls, width, height, hfov_deg=69.0):
        """Good enough for a first pass — run cv2.calibrateCamera for real numbers."""
        f = 0.5 * max(width, height) / np.tan(np.radians(hfov_deg) / 2)
        K = np.array([[f,0,width/2],[0,f,height/2],[0,0,1]], dtype=np.float64)
        return cls(K, np.zeros(5), width, height)

    def save(self, path):
        Path(path).write_text(json.dumps({"K":self.K.tolist(),"dist":self.dist.tolist(),
                                           "width":self.width,"height":self.height}, indent=2))
    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["K"]), np.array(d["dist"]), d["width"], d["height"])

    def scaled_to(self, w, h):
        if (w,h)==(self.width,self.height): return self
        K = self.K.copy(); K[0]*=w/self.width; K[1]*=h/self.height
        return Intrinsics(K, self.dist, w, h)


@dataclass
class CameraPose:
    R_ct: np.ndarray   # camera→table rotation
    t_ct: np.ndarray   # camera→table translation

    @property
    def center_table(self):
        return -self.R_ct.T @ self.t_ct

    def cam_to_table(self, p_cam):
        return self.R_ct.T @ (np.asarray(p_cam) - self.t_ct)


def detect_camera_pose(frames_bgr, intr: Intrinsics, marker_size_m: float) -> CameraPose:
    """Average ArUco pose over all frames where the marker is visible."""
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
    h = marker_size_m / 2
    obj_pts = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], dtype=np.float64)
    rvecs, tvecs = [], []
    for f in frames_bgr:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or MARKER_ID not in ids.ravel(): continue
        c = corners[list(ids.ravel()).index(MARKER_ID)].reshape(4,2).astype(np.float64)
        ok, rv, tv = cv2.solvePnP(obj_pts, c, intr.K, intr.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok: rvecs.append(rv.ravel()); tvecs.append(tv.ravel())
    if not rvecs:
        raise RuntimeError("ArUco marker not detected — is it in view and printed correctly?")
    rv = np.median(rvecs, axis=0); tv = np.median(tvecs, axis=0)
    return CameraPose(cv2.Rodrigues(rv)[0], tv)


def pixel_to_table_plane(px, intr: Intrinsics, pose: CameraPose, z=0.0) -> np.ndarray:
    px = np.array(px, dtype=np.float64).reshape(-1, 2)
    fx, fy = intr.K[0,0], intr.K[1,1]
    cx, cy = intr.K[0,2], intr.K[1,2]
    n = np.stack([(px[:,0] - cx) / fx, (px[:,1] - cy) / fy], axis=1)
    d_cam = np.c_[n, np.ones(len(n))]
    d_tab = d_cam @ pose.R_ct
    d_tab /= np.linalg.norm(d_tab, axis=1, keepdims=True)
    c = pose.center_table
    s = (z - c[2]) / d_tab[:,2]
    return c + s[:,None] * d_tab
