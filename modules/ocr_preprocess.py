"""Optional OCR preprocessing (Phase 5).

Plate crops off road footage are small, low-contrast, and sometimes skewed.
These functions are *optional* enhancement steps whose value must be **measured**
(with `evaluate_ocr.py --compare-preprocess`) rather than assumed — some help,
some hurt, and every one costs time, so none of the expensive ones run per-frame
by default in the runtime.

Each function takes and returns a BGR image (OpenCV order) so they compose. A
named **pipeline** is an ordered list of steps; :func:`apply_pipeline` runs one.
:data:`PIPELINES` holds the presets the evaluator compares; `"none"` is the
identity baseline the others are scored against.

Kept intentionally dependency-light (numpy + cv2, both CI deps) so the whole
thing is importable and testable without the model stack.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray_bgr(img: np.ndarray) -> np.ndarray:
    """Grayscale, returned as 3-channel BGR so steps stay composable."""
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def upscale(img: np.ndarray, *, min_height: int = 64, max_scale: float = 4.0) -> np.ndarray:
    """Upscale a small crop so characters clear EasyOCR's minimum stroke width.
    No-op if the crop is already tall enough."""
    h, w = img.shape[:2]
    if h == 0 or h >= min_height:
        return img
    scale = min(max_scale, min_height / h)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_CUBIC)


def enhance_contrast(img: np.ndarray, *, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE (adaptive histogram equalisation) on the luminance channel — lifts
    low-contrast plates without blowing out already-bright ones."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lch, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lch = clahe.apply(lch)
    return cv2.cvtColor(cv2.merge((lch, a, b)), cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray, *, d: int = 5, sigma: float = 40.0) -> np.ndarray:
    """Edge-preserving bilateral filter — removes sensor/compression noise while
    keeping character edges crisp (cheaper than fastNlMeans)."""
    return cv2.bilateralFilter(img, d, sigma, sigma)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp-mask style sharpening to recover blurred character edges."""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def adaptive_threshold(img: np.ndarray, *, block: int = 21, c: int = 9) -> np.ndarray:
    """Binarise with a local threshold (handles uneven lighting across a plate),
    returned as 3-channel BGR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    block = block if block % 2 == 1 else block + 1  # must be odd
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def pad(img: np.ndarray, *, border: int = 10) -> np.ndarray:
    """Add a white quiet-zone around the crop; EasyOCR reads better with margin."""
    return cv2.copyMakeBorder(
        img, border, border, border, border, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


def perspective_correct(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp a skewed plate flat given its four corner points (TL, TR, BR, BL).

    Corners come from a caller that has them (e.g. an OCR detector's polygon);
    there is no reliable auto-corner-finder for arbitrary crops, so this is an
    opt-in step used only when corners are available.
    """
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError("corners must be a 4x2 array of (x, y) points")
    (tl, tr, br, bl) = corners
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width, height = max(1, width), max(1, height)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


# Named steps usable in a pipeline (perspective_correct needs corners, so it is
# not part of the auto presets).
STEPS = {
    "gray": to_gray_bgr,
    "upscale": upscale,
    "contrast": enhance_contrast,
    "denoise": denoise,
    "sharpen": sharpen,
    "threshold": adaptive_threshold,
    "pad": pad,
}

# Presets the evaluator compares. "none" is the identity baseline.
PIPELINES = {
    "none": [],
    "upscale": ["upscale"],
    "upscale_pad": ["upscale", "pad"],
    "contrast": ["contrast"],
    "gray_contrast": ["gray", "contrast"],
    "clean": ["upscale", "denoise", "contrast", "pad"],
    "binary": ["upscale", "contrast", "threshold", "pad"],
    "sharp": ["upscale", "sharpen", "pad"],
}


def apply_pipeline(img: np.ndarray, steps) -> np.ndarray:
    """Run an ordered list of step names (or a preset name) over an image."""
    if isinstance(steps, str):
        steps = PIPELINES[steps]
    out = img
    for name in steps:
        out = STEPS[name](out)
    return out
