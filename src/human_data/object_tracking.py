"""HSV colour-blob tracking of the demonstrated cube."""
from __future__ import annotations
import cv2, numpy as np

HSV_PRESETS = {
    "blue":   [((95,120,50),(130,255,255))],
    "green":  [((40,80,50),(85,255,255))],
    "yellow": [((20,120,100),(35,255,255))],
    "red":    [((0,140,70),(8,255,255)),((170,140,70),(180,255,255))],
    "dark": [((0, 0, 0), (180, 100, 80))],
}


def track_cube(frames_bgr, color="blue", min_area_frac=2e-4):
    """Returns dict: px [T,2] (NaN when lost), valid [T]."""
    ranges = HSV_PRESETS[color] if isinstance(color, str) else color
    T  = len(frames_bgr); px = np.full((T,2), np.nan); prev = None
    for i, f in enumerate(frames_bgr):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges: mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_OPEN,k),cv2.MORPH_CLOSE,k)
        n,_,stats,cents = cv2.connectedComponentsWithStats(mask)
        cand = [(stats[j,cv2.CC_STAT_AREA],cents[j]) for j in range(1,n)
                if stats[j,cv2.CC_STAT_AREA] >= min_area_frac*mask.size]
        if not cand: continue
        _,c = (max(cand,key=lambda x:x[0]) if prev is None
               else min(cand,key=lambda x:np.linalg.norm(x[1]-prev)-1e-3*x[0]))
        px[i]=c; prev=c
    return {"px": px, "valid": ~np.isnan(px[:,0])}
