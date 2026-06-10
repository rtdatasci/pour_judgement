"""
scoring.py — Objective latte art metrics via classical CV (no ML, instant, CPU-free-tier friendly).

Pipeline:
  1. Find the cup (Hough circle, with a center-crop fallback).
  2. Isolate the crema surface, segment milk foam vs crema (Otsu on lightness).
  3. Compute four sub-scores in [0, 100]:
       contrast    — separation between foam and crema tones
       symmetry    — best mirror-axis IoU of the foam pattern (rotation-tolerant)
       centering   — how close the pattern centroid sits to the cup center
       definition  — edge sharpness along the foam/crema boundary
  4. Weighted total in [0, 100].

All functions are pure; main entry point is `score_image(path) -> dict`.
"""

from __future__ import annotations

import cv2
import numpy as np

WEIGHTS = {
    "contrast": 0.25,
    "symmetry": 0.30,
    "centering": 0.15,
    "definition": 0.30,
}

MAX_SIDE = 720  # downscale large photos for speed


# ---------------------------------------------------------------- utilities

def _load(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _find_cup(img: np.ndarray) -> tuple[int, int, int]:
    """Return (cx, cy, r) of the cup interior. Falls back to a centered circle."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    h, w = gray.shape
    min_r, max_r = int(min(h, w) * 0.20), int(min(h, w) * 0.55)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w),
        param1=120, param2=40, minRadius=min_r, maxRadius=max_r,
    )
    if circles is not None and len(circles[0]) > 0:
        cx, cy, r = circles[0][0]
        return int(cx), int(cy), int(r)
    # Fallback: assume the cup roughly fills the frame center.
    return w // 2, h // 2, int(min(h, w) * 0.42)


def _crema_mask(img: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    """Binary mask of the drink surface (slightly inset from the rim)."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(r * 0.86), 255, -1)
    return mask


def _foam_mask(img: np.ndarray, surface: np.ndarray) -> np.ndarray:
    """Segment milk foam (light) from crema (dark) inside the surface mask."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    vals = L[surface > 0]
    if vals.size == 0:
        return np.zeros_like(surface)
    thresh, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foam = ((L > thresh) & (surface > 0)).astype(np.uint8) * 255
    # Clean speckle so symmetry/edges measure the pattern, not noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foam = cv2.morphologyEx(foam, cv2.MORPH_OPEN, kernel)
    foam = cv2.morphologyEx(foam, cv2.MORPH_CLOSE, kernel)
    return foam


def _clamp(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


# ---------------------------------------------------------------- sub-scores

def _contrast_score(img: np.ndarray, surface: np.ndarray, foam: np.ndarray) -> float:
    """Tonal separation between foam and crema. ~35+ L* gap is championship white-on-brown."""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    crema = (surface > 0) & (foam == 0)
    fm = (foam > 0)
    if fm.sum() < 50 or crema.sum() < 50:
        return 5.0  # essentially no pattern poured
    gap = float(L[fm].mean() - L[crema].mean())  # 0..255 scale
    return _clamp(gap / 90.0 * 100.0)


def _symmetry_score(foam: np.ndarray, cx: int, cy: int) -> float:
    """Mirror IoU of the foam pattern about its best vertical-ish axis (tilt-tolerant)."""
    if foam.sum() == 0:
        return 0.0
    ys, xs = np.nonzero(foam)
    pcx, pcy = float(xs.mean()), float(ys.mean())
    best = 0.0
    for angle in (-30, -15, 0, 15, 30):
        M = cv2.getRotationMatrix2D((pcx, pcy), angle, 1.0)
        rot = cv2.warpAffine(foam, M, (foam.shape[1], foam.shape[0]))
        # Mirror about the vertical line through the pattern centroid.
        shift = int(round(2 * pcx)) - rot.shape[1]
        flipped = cv2.flip(rot, 1)
        Mt = np.float32([[1, 0, shift], [0, 1, 0]])
        flipped = cv2.warpAffine(flipped, Mt, (rot.shape[1], rot.shape[0]))
        inter = np.logical_and(rot > 0, flipped > 0).sum()
        union = np.logical_or(rot > 0, flipped > 0).sum()
        if union > 0:
            best = max(best, inter / union)
    return _clamp(best * 100.0)


def _centering_score(foam: np.ndarray, cx: int, cy: int, r: int) -> float:
    if foam.sum() == 0:
        return 0.0
    ys, xs = np.nonzero(foam)
    d = np.hypot(xs.mean() - cx, ys.mean() - cy) / max(r, 1)
    return _clamp((1.0 - d / 0.5) * 100.0)  # centroid > half-radius off-center -> 0


def _definition_score(img: np.ndarray, foam: np.ndarray) -> float:
    """Edge crispness along the foam boundary — wiggle lines should cut, not blur."""
    if foam.sum() == 0:
        return 0.0
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    contours, _ = cv2.findContours(foam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(foam)
    cv2.drawContours(boundary, contours, -1, 255, 3)
    edge_vals = grad[boundary > 0]
    if edge_vals.size == 0:
        return 0.0
    return _clamp(float(edge_vals.mean()) / 80.0 * 100.0)


# ---------------------------------------------------------------- entry point

def score_image(path: str) -> dict:
    img = _load(path)
    cx, cy, r = _find_cup(img)
    surface = _crema_mask(img, cx, cy, r)
    foam = _foam_mask(img, surface)

    foam_frac = foam.sum() / max(surface.sum(), 1)  # both are 0/255 masks -> ratio holds

    scores = {
        "contrast": round(_contrast_score(img, surface, foam), 1),
        "symmetry": round(_symmetry_score(foam, cx, cy), 1),
        "centering": round(_centering_score(foam, cx, cy, r), 1),
        "definition": round(_definition_score(img, foam), 1),
    }
    total = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)

    return {
        "total": round(total, 1),
        "subscores": scores,
        "foam_fraction": round(float(foam_frac), 3),
        "cup": {"cx": cx, "cy": cy, "r": r},
        "weakest": min(scores, key=scores.get),
    }


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(score_image(sys.argv[1]), indent=2))
