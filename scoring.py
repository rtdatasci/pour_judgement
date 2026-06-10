"""
scoring.py — Objective latte art metrics via classical CV (no ML, instant, CPU-friendly).

Pipeline:
  1. Find the cup (Hough circle, with a center-crop fallback).
  2. Isolate the crema surface, segment milk foam vs crema (Otsu on lightness).
  3. Compute five sub-scores in [0, 100]:
       contrast    — separation between foam and crema tones (requires genuinely
                     light foam, not just "lighter than the rest")
       symmetry    — best mirror-axis IoU of the foam pattern (rotation-tolerant)
       centering   — pattern centroid vs cup center
       definition  — edge sharpness along the foam/crema boundary
       texture     — milk quality: glossy microfoam scores high, visible
                     bubbles / rough speckled foam scores low
  4. A "presence" gate multiplies the total: almost-no-pattern or
     flooded-white cups can't ride individual metrics to a high score.
  5. A power curve pushes mediocre pours down — championship scores must be earned.

Main entry point: `score_image(path) -> dict`.
"""

from __future__ import annotations

import cv2
import numpy as np

WEIGHTS = {
    "contrast": 0.20,
    "symmetry": 0.25,
    "centering": 0.10,
    "definition": 0.25,
    "texture": 0.20,
}

MAX_SIDE = 720
CURVE = 1.35  # >1 = harsher; total = 100 * (raw/100)^CURVE


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
    return w // 2, h // 2, int(min(h, w) * 0.42)


def _crema_mask(img: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(r * 0.86), 255, -1)
    return mask


def _foam_masks(img: np.ndarray, surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw, cleaned) foam masks. The raw-vs-clean difference measures speckle."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    vals = L[surface > 0]
    if vals.size == 0:
        z = np.zeros_like(surface)
        return z, z
    thresh, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw = ((L > thresh) & (surface > 0)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    return raw, clean


def _clamp(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


# ---------------------------------------------------------------- sub-scores

def _contrast_score(img: np.ndarray, surface: np.ndarray, foam: np.ndarray) -> float:
    """Tonal separation, gated by absolute foam lightness.

    Otsu will always find a split, so the L* gap alone over-rewards muddy cups.
    Real white-on-brown needs foam that is *actually light* (L >~ 150/255), so the
    gap is scaled by a lightness factor.
    """
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    crema = (surface > 0) & (foam == 0)
    fm = (foam > 0)
    if fm.sum() < 200 or crema.sum() < 200:
        return 5.0
    foam_mean = float(L[fm].mean())
    gap = foam_mean - float(L[crema].mean())              # 0..255
    lightness = max(0.0, min(1.0, (foam_mean - 120.0) / 80.0))  # 1.0 at L>=200
    return _clamp((gap / 110.0) * lightness * 100.0)


def _symmetry_score(foam: np.ndarray) -> float:
    if foam.sum() == 0:
        return 0.0
    ys, xs = np.nonzero(foam)
    pcx, pcy = float(xs.mean()), float(ys.mean())
    best = 0.0
    for angle in (-30, -15, 0, 15, 30):
        M = cv2.getRotationMatrix2D((pcx, pcy), angle, 1.0)
        rot = cv2.warpAffine(foam, M, (foam.shape[1], foam.shape[0]))
        shift = int(round(2 * pcx)) - rot.shape[1]
        flipped = cv2.flip(rot, 1)
        Mt = np.float32([[1, 0, shift], [0, 1, 0]])
        flipped = cv2.warpAffine(flipped, Mt, (rot.shape[1], rot.shape[0]))
        inter = np.logical_and(rot > 0, flipped > 0).sum()
        union = np.logical_or(rot > 0, flipped > 0).sum()
        if union > 0:
            best = max(best, inter / union)
    # Mirror IoU is naturally generous (blobs self-mirror well) — re-map so that
    # only genuinely tight symmetry reaches the top band.
    return _clamp(((best - 0.35) / 0.6) * 100.0)


def _centering_score(foam: np.ndarray, cx: int, cy: int, r: int) -> float:
    if foam.sum() == 0:
        return 0.0
    ys, xs = np.nonzero(foam)
    d = np.hypot(xs.mean() - cx, ys.mean() - cy) / max(r, 1)
    return _clamp((1.0 - d / 0.45) * 100.0)


def _definition_score(img: np.ndarray, foam: np.ndarray) -> float:
    """Edge crispness along the foam boundary. Blurry, bleeding edges score low."""
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
    return _clamp(float(edge_vals.mean()) / 130.0 * 100.0)


def _texture_score(img: np.ndarray, foam_raw: np.ndarray, foam_clean: np.ndarray) -> float:
    """Milk texture: glossy microfoam vs visible bubbles.

    Two signals, both inside the foam:
      roughness — Laplacian energy in the foam interior (away from the pattern
                  edge). Smooth paint-like microfoam is flat; bubbles are busy.
      speckle   — how much of the raw Otsu mask was noise removed by the
                  morphological clean-up. Bubbly foam segments as confetti.
    """
    if foam_clean.sum() == 0:
        return 0.0
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    interior = cv2.erode(foam_clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    rough_score = 50.0  # neutral if pattern too thin to have an interior
    if interior.sum() > 200 * 255:
        lap = cv2.Laplacian(L, cv2.CV_32F, ksize=3)
        rough = float(np.abs(lap[interior > 0]).mean())
        # rough ~<4 = glossy, ~15+ = clearly bubbly
        rough_score = _clamp((1.0 - (rough - 4.0) / 14.0) * 100.0)

    diff = cv2.bitwise_xor(foam_raw, foam_clean)
    speckle = diff.sum() / max(foam_clean.sum(), 1)
    speckle_score = _clamp((1.0 - speckle / 0.18) * 100.0)

    return round(0.65 * rough_score + 0.35 * speckle_score, 1)


def _presence_factor(foam_frac: float) -> float:
    """Gate the total: a cup needs a real pattern.

    Sweet spot ~8–45% foam coverage. Below 5% there's essentially nothing;
    above 60% the cup is flooded white.
    """
    if foam_frac < 0.02:
        return 0.15
    if foam_frac < 0.08:
        return 0.15 + 0.85 * (foam_frac - 0.02) / 0.06
    if foam_frac <= 0.45:
        return 1.0
    if foam_frac <= 0.65:
        return 1.0 - 0.6 * (foam_frac - 0.45) / 0.20
    return 0.4


# ---------------------------------------------------------------- entry point

def score_image(path: str) -> dict:
    img = _load(path)
    cx, cy, r = _find_cup(img)
    surface = _crema_mask(img, cx, cy, r)
    foam_raw, foam = _foam_masks(img, surface)

    foam_frac = foam.sum() / max(surface.sum(), 1)

    scores = {
        "contrast": round(_contrast_score(img, surface, foam), 1),
        "symmetry": round(_symmetry_score(foam), 1),
        "centering": round(_centering_score(foam, cx, cy, r), 1),
        "definition": round(_definition_score(img, foam), 1),
        "texture": round(_texture_score(img, foam_raw, foam), 1),
    }
    raw_total = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    gated = raw_total * _presence_factor(float(foam_frac))
    total = 100.0 * (gated / 100.0) ** CURVE  # harshness curve

    return {
        "total": round(total, 1),
        "subscores": scores,
        "foam_fraction": round(float(foam_frac), 3),
        "cup": {"cx": cx, "cy": cy, "r": r},
        "weakest": min(scores, key=scores.get),
        "bubbly": scores["texture"] < 40.0 and float(foam_frac) >= 0.04,
    }


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(score_image(sys.argv[1]), indent=2))