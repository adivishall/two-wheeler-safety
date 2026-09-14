"""Generate a tiny SYNTHETIC labelled plate set to sanity-check the OCR evaluator.

There is no labelled OCR dataset for this project, so `evaluate_ocr.py` has
never been run end to end on images. That is a gap in the *tooling's* assurance,
not just in the data: a CSV-reading, image-loading, OCR-calling, metric-computing
path that has never executed on a real file is a path that might not work.

This renders plate-like images with known text and applies the degradations the
schema's ``condition`` column names, so the whole path can be exercised:

    python3 make_ocr_sanity_set.py --out data/ocr_sanity
    python3 evaluate_ocr.py --labels data/ocr_sanity/labels.csv

**What these numbers are and are not.** They are a check that the evaluator
runs and that its metrics move in the right direction as images get worse. They
are **not** a benchmark of EasyOCR and **not** field OCR accuracy — these are
clean vector-rendered glyphs on a flat background, which is far easier than a
photographed plate at dusk, and they contain none of the failure modes that
actually matter (specular glare, motion blur from a moving camera, mud,
non-standard fonts, bent plates). Every artifact this writes is stamped
``"synthetic": true`` so a number from it can never be mistaken for a field
measurement.

The conditions mirror the documented schema:

    clean     rendered plate, no degradation
    blur      Gaussian blur (out-of-focus / motion)
    angle     perspective warp (plate seen off-axis)
    lowlight  reduced brightness + sensor noise
    small     downscaled then upscaled (distant plate)
    occluded  a band of the plate covered (mud, bracket, another vehicle)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

CONDITIONS = ("clean", "blur", "angle", "lowlight", "small", "occluded")

# Plate strings spanning the structures modules/plate_info.py recognises.
PLATES = (
    "MH12AB1234", "KA05MN6789", "DL8CAF5031", "TN22BC4567",
    "GJ01AA0001", "UP32DK4321", "RJ14CV0002", "KL07CD8080",
)


def render_plate(text: str, width: int = 440, height: int = 140):
    """A flat white plate with black characters — deliberately idealised."""
    import cv2
    import numpy as np

    img = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (6, 6), (width - 6, height - 6), (25, 25, 25), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0
    thickness = 5
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    # Shrink to fit rather than letting a long plate run off the edge.
    while tw > width - 40 and scale > 0.5:
        scale -= 0.1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    org = ((width - tw) // 2, (height + th) // 2)
    cv2.putText(img, text, org, font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return img


def degrade(img, condition: str, seed: int = 0):
    """Apply one named degradation. ``clean`` returns the image untouched."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    if condition == "clean":
        return img
    if condition == "blur":
        return cv2.GaussianBlur(img, (9, 9), 3.0)
    if condition == "angle":
        src = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = np.asarray([[w * 0.14, h * 0.06], [w * 0.96, 0],
                          [w * 0.88, h * 0.95], [0, h]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, m, (w, h), borderValue=(120, 120, 120))
    if condition == "lowlight":
        dark = (img.astype(np.float32) * 0.32)
        noise = rng.normal(0, 9, dark.shape)
        return np.clip(dark + noise, 0, 255).astype(np.uint8)
    if condition == "small":
        tiny = cv2.resize(img, (w // 6, h // 6), interpolation=cv2.INTER_AREA)
        return cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)
    if condition == "occluded":
        out = img.copy()
        x0 = int(w * 0.42)
        cv2.rectangle(out, (x0, 0), (x0 + int(w * 0.2), h), (70, 60, 55), -1)
        return out
    raise ValueError(f"unknown condition: {condition}")


def build(out_dir: str, plates=PLATES, conditions=CONDITIONS) -> dict:
    import cv2

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for i, plate in enumerate(plates):
        base = render_plate(plate)
        for condition in conditions:
            img = degrade(base, condition, seed=i)
            name = f"{plate}_{condition}.jpg"
            cv2.imwrite(os.path.join(img_dir, name), img)
            rows.append({
                "image_path": os.path.join("images", name),
                "plate": plate,
                "condition": condition,
            })

    csv_path = os.path.join(out_dir, "labels.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "plate", "condition"])
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "synthetic": True,
        "purpose": (
            "Sanity-check that the OCR evaluation path runs end to end and that "
            "its metrics degrade as images degrade."
        ),
        "not_a_benchmark": (
            "Vector-rendered glyphs on a flat background. NOT field OCR "
            "accuracy, NOT a measurement of EasyOCR. Contains none of glare, "
            "motion blur, mud, bent plates or non-standard fonts."
        ),
        "plates": list(plates),
        "conditions": list(conditions),
        "images": len(rows),
        "schema": "image_path,plate,condition",
    }
    with open(os.path.join(out_dir, "dataset.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    meta["csv"] = csv_path
    return meta


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate a synthetic labelled plate set for evaluator sanity checks.",
    )
    ap.add_argument("--out", default="data/ocr_sanity", help="output directory")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS),
                    choices=list(CONDITIONS))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    meta = build(args.out, conditions=tuple(args.conditions))
    print(f"Wrote {meta['images']} synthetic plate images to {args.out}/images")
    print(f"Labels: {meta['csv']}")
    print("\n*** SYNTHETIC — not a benchmark. ***")
    print(meta["not_a_benchmark"])
    print(f"\nRun the evaluator with:\n  python3 evaluate_ocr.py --labels {meta['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
