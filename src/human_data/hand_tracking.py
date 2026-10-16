"""MediaPipe Hand Landmarker (Tasks API, mediapipe>=0.10.14) wrapper."""
from __future__ import annotations
import urllib.request, cv2, numpy as np
from pathlib import Path

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = Path.home()/".cache"/"handtorobot"/"hand_landmarker.task"
THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9


def _ensure_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe model → {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def track_hand(frames_bgr, fps, handedness=None):
    """Returns dict: img [T,21,2] px, world [T,21,3] m,
       grasp_px [T,2], openness [T] 0=closed 1=open, valid [T]."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2 if handedness else 1,
    )
    T = len(frames_bgr)
    img   = np.full((T,21,2), np.nan)
    world = np.full((T,21,3), np.nan)

    with vision.HandLandmarker.create_from_options(opts) as lm:
        for i, f in enumerate(frames_bgr):
            h,w = f.shape[:2]
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                      int(1000*i/fps))
            if not res.hand_landmarks: continue
            k = 0
            if handedness:
                labels=[hd[0].category_name.lower() for hd in res.handedness]
                if handedness.lower() not in labels: continue
                k = labels.index(handedness.lower())
            img[i]   = [(p.x*w, p.y*h) for p in res.hand_landmarks[k]]
            world[i] = [(p.x, p.y, p.z) for p in res.hand_world_landmarks[k]]

    valid      = ~np.isnan(img[:,0,0])
    grasp_px   = 0.5*(img[:,THUMB_TIP]+img[:,INDEX_TIP])
    pinch      = np.where(valid, np.linalg.norm(world[:,THUMB_TIP]-world[:,INDEX_TIP],axis=1), np.nan)
    palm       = np.where(valid, np.linalg.norm(world[:,WRIST]-world[:,MIDDLE_MCP],axis=1)+1e-6, np.nan)
    raw        = pinch/palm
    v          = raw[~np.isnan(raw)]
    lo,hi      = (np.percentile(v,5), np.percentile(v,95)) if len(v)>5 else (0,1)
    openness   = np.clip((raw-lo)/max(hi-lo,1e-6), 0, 1)
    return {"img":img, "world":world, "grasp_px":grasp_px, "openness":openness, "valid":valid}
