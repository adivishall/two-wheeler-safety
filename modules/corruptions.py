"""Synthetic image corruptions for detector robustness testing.

There is no night / rain / glare / motion subset in the dataset, so the only way
to measure how the detector degrades under those conditions is to transform the
images it has. Every corruption here is:

* **label-preserving** — photometric, blur, resolution or occlusion changes
  only; nothing moves an object, so the ground-truth boxes stay correct and a
  drop in AP is the model's, not a label artifact. (Geometric changes such as
  oblique viewpoints would move boxes; they are deliberately not faked here.)
* **deterministic** — seeded from the image name, so every model sees the same
  corrupted image and runs are reproducible;
* **labelled as synthetic** — results carry ``"synthetic_transform": true``.

What a result means: "under this transform of held-out images, AP@50 fell by
X". What it does not mean: performance on real night/rain footage, whose
statistics a transform only approximates.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _rng(key: str, salt: str) -> np.random.Generator:
    seed = int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def gaussian_blur(img, key, *, sigma: float):
    import cv2

    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def motion_blur(img, key, *, length: int):
    import cv2

    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0 / length  # horizontal: traffic moves across frame
    return cv2.filter2D(img, -1, k)


def low_light(img, key, *, gain: float, noise: float):
    """Darken (gamma + gain) and add sensor noise, the way a dusk frame looks."""
    rng = _rng(key, "lowlight")
    x = img.astype(np.float32) / 255.0
    x = np.power(x, 1.6) * gain
    x = x * 255.0 + rng.normal(0.0, noise, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def glare(img, key, *, strength: float):
    """A bright elliptical bloom (sun / headlight) at a seeded position."""
    rng = _rng(key, "glare")
    h, w = img.shape[:2]
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    r = 0.22 * min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2 * r * r))
    out = img.astype(np.float32) + strength * 255.0 * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg(img, key, *, quality: int):
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def low_resolution(img, key, *, factor: float):
    """Down- then up-sample: a distant subject / cheap camera. Boxes unchanged."""
    import cv2

    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def occlusion(img, key, *, area: float, patches: int = 4):
    """Grey rectangles covering ~``area`` of the frame at seeded positions
    (other vehicles, poles). They may cover labelled objects: that is the
    point — the labels stay, so a hidden object correctly counts as missed."""
    rng = _rng(key, "occlusion")
    h, w = img.shape[:2]
    out = img.copy()
    side = np.sqrt(area * h * w / patches)
    for _ in range(patches):
        ph = int(side * rng.uniform(0.6, 1.4))
        pw = int(side * side / max(ph, 1))
        y = int(rng.uniform(0, max(1, h - ph)))
        x = int(rng.uniform(0, max(1, w - pw)))
        out[y:y + ph, x:x + pw] = 127
    return out


# name -> (function, kwargs). Two severities each, so a result is a trend,
# not a single point.
CORRUPTIONS: dict[str, tuple] = {
    "gaussian_blur_s2": (gaussian_blur, {"sigma": 2.0}),
    "gaussian_blur_s4": (gaussian_blur, {"sigma": 4.0}),
    "motion_blur_9": (motion_blur, {"length": 9}),
    "motion_blur_21": (motion_blur, {"length": 21}),
    "low_light_mild": (low_light, {"gain": 0.6, "noise": 6.0}),
    "low_light_severe": (low_light, {"gain": 0.3, "noise": 12.0}),
    "glare_mild": (glare, {"strength": 0.35}),
    "glare_severe": (glare, {"strength": 0.7}),
    "jpeg_q20": (jpeg, {"quality": 20}),
    "jpeg_q8": (jpeg, {"quality": 8}),
    "low_res_x0.5": (low_resolution, {"factor": 0.5}),
    "low_res_x0.25": (low_resolution, {"factor": 0.25}),
    "occlusion_10pct": (occlusion, {"area": 0.10}),
    "occlusion_25pct": (occlusion, {"area": 0.25}),
}


def get(name: str):
    """A ``corrupt(image, key) -> image`` callable for a named corruption."""
    fn, kwargs = CORRUPTIONS[name]
    return lambda img, key: fn(img, key, **kwargs)


def cache_tag(name: str) -> str:
    """Name + parameters, for prediction caches: a retuned transform is a
    different experiment and must not reuse old predictions."""
    import json

    return f"{name}:{json.dumps(CORRUPTIONS[name][1], sort_keys=True)}"
