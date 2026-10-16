"""Process a raw demo video into a HumanDemo .npz.

Usage:
    python -m human_data.scripts.process_video \
        data/raw_human/demo_01.mp4 \
        --marker_size 0.10 \
        --cube_color blue \
        [--intrinsics data/camera_intrinsics.json] \
        [--debug_video data/processed_human/demo_01_debug.mp4]
"""
import argparse, cv2, numpy as np
from pathlib import Path
from scipy.signal import savgol_filter


def read_video(path, resize_w=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        if resize_w and f.shape[1] != resize_w:
            h = int(round(f.shape[0]*resize_w/f.shape[1]))
            f = cv2.resize(f,(resize_w,h),interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames, float(fps)


def _segment(cube_xy, openness, fps, move_thresh=0.015, open_thresh=0.5):
    valid = ~np.isnan(cube_xy[:,0])
    vidx  = np.flatnonzero(valid)
    start = np.median(cube_xy[vidx[:15]], axis=0)
    end   = np.median(cube_xy[vidx[-15:]], axis=0)
    if np.linalg.norm(end-start) < 2*move_thresh:
        raise RuntimeError("cube barely moved — check colour preset or video")
    d_s = np.where(valid, np.linalg.norm(cube_xy-start,axis=1), np.inf)
    T = len(openness); w = int(1.5*fps)
    m0 = next((i for i in range(T) if d_s[i]>move_thresh), 0)
    d_e = np.where(valid, np.linalg.norm(cube_xy-end,axis=1), np.inf)
    m1 = T-1-next((i for i,v in enumerate(d_e[::-1]) if v>move_thresh), 0)
    o  = np.nan_to_num(openness, nan=1.0)
    g  = next((i for i in range(m0,max(m0-w,1),-1) if o[i]<=open_thresh<o[i-1]), max(m0-3,0))
    r  = next((i+1 for i in range(m1,min(m1+w,T-1)) if o[i]<=open_thresh<o[i+1]), min(m1+3,T-1))
    return g, max(r,g+1), start, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--marker_size", type=float, default=0.10, help="printed ArUco side (m)")
    ap.add_argument("--cube_color",  default="blue")
    ap.add_argument("--intrinsics",  default=None)
    ap.add_argument("--debug_video", default=None)
    ap.add_argument("--out_dir",     default="data/processed_human")
    ap.add_argument("--resize_w",    type=int, default=1280)
    args = ap.parse_args()

    frames, fps = read_video(args.video, args.resize_w)
    print(f"{len(frames)} frames at {fps:.1f} fps")
    h, w = frames[0].shape[:2]

    from human_data.camera import Intrinsics, detect_camera_pose, pixel_to_table_plane
    intr = Intrinsics.load(args.intrinsics) if args.intrinsics else \
           Intrinsics.approximate_phone(w, h)

    pose = detect_camera_pose(frames, intr, args.marker_size)
    print("Camera pose estimated")

    from human_data.object_tracking import track_cube
    cube_tr = track_cube(frames, args.cube_color)
    cube_px = cube_tr["px"]
    # lift cube px to 3-D table coords (cube sits at z = 0.02)
    valid_idx = np.flatnonzero(cube_tr["valid"])
    cube_table = np.full((len(frames),3), np.nan)
    cube_table[valid_idx] = pixel_to_table_plane(cube_px[valid_idx], intr, pose, z=0.02)

    from human_data.hand_tracking import track_hand
    hand_tr = track_hand(frames, fps)
    grasp_px = hand_tr["grasp_px"]
    # grasp point on table — project to z=0.02 (approximate; retarget.py refines)
    g_valid = np.flatnonzero(hand_tr["valid"])
    grasp_table = np.full((len(frames),3), np.nan)
    grasp_table[g_valid] = pixel_to_table_plane(grasp_px[g_valid], intr, pose, z=0.04)

    # gap-fill and smooth grasp
    from scipy.signal import savgol_filter
    def fill(a):
        out=a.copy(); idx=np.flatnonzero(~np.isnan(a[:,0]))
        for d in range(3): out[:,d]=np.interp(np.arange(len(a)),idx,a[idx,d])
        return out
    grasp_table = fill(grasp_table)
    w2 = max(int(0.10*fps)|1, 3)
    grasp_table = savgol_filter(grasp_table, w2, 2, axis=0)

    gi, ri, cube_start, cube_end = _segment(cube_table[:,:2], hand_tr["openness"], fps)
    print(f"grasp_idx={gi}  release_idx={ri}")

    from human_data.trajectory import HumanDemo
    demo = HumanDemo(
        name=Path(args.video).stem, fps=fps,
        grasp_px=grasp_px, grasp_table=grasp_table,
        openness=hand_tr["openness"],
        cube_table=cube_table, cube_start=np.append(cube_start,0.02),
        cube_end=np.append(cube_end,0.022),
        grasp_idx=gi, release_idx=ri,
    )
    out = Path(args.out_dir)/f"{demo.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    demo.save(out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
