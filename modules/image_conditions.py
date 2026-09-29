"""Detector accuracy by *measured* image condition, on real held-out images.

The field-condition question ("does it still work at night, in glare, on small
distant plates, in dense traffic?") needs condition labels the dataset doesn't
have. What it does have is pixels and boxes, from which some conditions can be
*measured* rather than labelled:

* **lighting** — mean luminance. ``dark`` (< 60 / 255) is a night proxy: by eye,
  11 of the 12 darkest validation images are road scenes after dusk (several
  carry burned-in 18:48–20:24 timestamps); one is a studio portrait. It is a
  proxy, not a label.
* **sharpness** — variance of the Laplacian at 640 px width (focus/motion blur
  proxy), cut into terciles fitted on the validation split.
* **glare** — share of near-saturated pixels (≥ 250); ``glare`` when ≥ 5%.
* **rider boxes** — labelled rider boxes in the image: 0, 1, 2–3, 4+ (a dense
  traffic proxy; one ``TripleRiding`` box covers three people, so this counts
  vehicles-with-riders more than people).
* **source** — the upstream dataset (``ds1`` / ``dst``), whose label sets differ.
* **object size** — per class, the box's relative size (√area / √image area),
  terciles fitted on validation ground truth; small objects are the "far away"
  and "small plate" proxy. Evaluated with the COCO range convention.

Not measurable here: **resolution** (every image was pre-resized to 640 or 512
px by the dataset export), **camera angle**, **weather** and true **time of
day** — those need the field dataset (``modules.field_data``).

All thresholds are fitted on validation and applied unchanged to test.
"""

from __future__ import annotations

import math

import numpy as np

from modules.detection_stats import (
    bootstrap_ap,
    match_for_ap,
    match_for_ap_in_range,
    paired_bootstrap,
    recall_at,
    unpaired_gap,
)

DARK_LUMINANCE = 60.0
BRIGHT_LUMINANCE = 150.0
GLARE_PIXEL = 250
GLARE_SHARE = 0.05
RIDER_CLASSES = ("WithHelmet", "WithoutHelmet", "TripleRiding")
MIN_INSTANCES = 10  # below this a per-class stratum AP is shown but flagged


def image_properties(image) -> dict:
    """Luminance, sharpness and glare of one BGR image."""
    import cv2

    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape[:2]
    g640 = cv2.resize(grey, (640, max(1, round(640 * h / w)))) if w != 640 else grey
    return {
        "luminance": round(float(grey.mean()), 2),
        "sharpness": round(float(cv2.Laplacian(g640, cv2.CV_64F).var()), 1),
        "glare_share": round(float((grey >= GLARE_PIXEL).mean()), 4),
        "width": int(w), "height": int(h),
    }


def relative_size(box, w: int, h: int) -> float:
    x1, y1, x2, y2 = box
    return math.sqrt(max(0.0, (x2 - x1) * (y2 - y1)) / float(w * h))


def fit_thresholds(rows: list[dict], props: list[dict], class_names: list[str]) -> dict:
    """Cut points fitted on one split (validation) and reused on the others."""
    sharp = np.array([p["sharpness"] for p in props])
    sizes: dict[str, list[float]] = {n: [] for n in class_names}
    for r in rows:
        for c, *box in r["gt"]:
            sizes[class_names[c]].append(relative_size(box, r["w"], r["h"]))
    return {
        "luminance": {"dark_below": DARK_LUMINANCE, "bright_above": BRIGHT_LUMINANCE},
        "glare": {"pixel": GLARE_PIXEL, "share": GLARE_SHARE},
        "sharpness_terciles": [round(float(x), 1) for x in np.percentile(sharp, [100 / 3,
                                                                              200 / 3])],
        "size_terciles": {n: ([round(float(x), 4) for x in np.percentile(v, [100 / 3, 200 / 3])]
                              if len(v) >= 3 else None) for n, v in sizes.items()},
    }


def image_strata(row: dict, prop: dict, thr: dict, class_names: list[str],
                 source: str) -> dict:
    lum = prop["luminance"]
    lo, hi = thr["sharpness_terciles"]
    riders = sum(1 for c, *_ in row["gt"] if class_names[c] in RIDER_CLASSES)
    return {
        "lighting (luminance proxy)": ("dark" if lum < DARK_LUMINANCE else
                                       "bright" if lum > BRIGHT_LUMINANCE else "normal"),
        "sharpness (Laplacian proxy)": ("low" if prop["sharpness"] < lo else
                                        "high" if prop["sharpness"] > hi else "mid"),
        "glare": "glare" if prop["glare_share"] >= GLARE_SHARE else "no glare",
        "labelled rider boxes": ("0" if riders == 0 else "1" if riders == 1 else
                                 "2-3" if riders <= 3 else "4+"),
        "source": source,
    }


def _records(rows: list[dict], n_classes: int) -> list:
    return [match_for_ap([(c, tuple(b)) for c, *b in r["gt"]],
                         [(c, tuple(p[:4]), p[4]) for c, *p in r["preds"]], n_classes)
            for r in rows]


def _stratum(records_in, records_out, class_names, op_thresholds, n_boot, seed) -> dict:
    ap = bootstrap_ap(records_in, class_names, n_boot=n_boot, seed=seed)
    out = {"images": len(records_in), "ap": ap,
           "at_operating_threshold": recall_at(records_in, class_names, op_thresholds)}
    if records_out:
        out["gap_vs_rest"] = unpaired_gap(records_in, records_out, class_names,
                                          n_boot=n_boot, seed=seed)
    return out


def evaluate(rows: list[dict], props: list[dict], sources: list[str], class_names: list[str],
             thr: dict, *, op_thresholds: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """Per-stratum AP@50 (bootstrap CI), precision/recall at the operating
    thresholds, and each stratum's AP gap against the rest of the split."""
    k = len(class_names)
    records = _records(rows, k)
    strata = [image_strata(r, p, thr, class_names, s) for r, p, s in zip(rows, props, sources)]
    out: dict = {"images": len(rows), "thresholds": thr,
                 "overall": _stratum(records, [], class_names, op_thresholds, n_boot, seed),
                 "image_strata": {}, "object_size": {},
                 "image_sizes": sorted({f"{p['width']}x{p['height']}" for p in props})}
    for attr in strata[0] if strata else []:
        values = sorted({s[attr] for s in strata})
        out["image_strata"][attr] = {}
        for value in values:
            inside = [rec for rec, s in zip(records, strata) if s[attr] == value]
            rest = [rec for rec, s in zip(records, strata) if s[attr] != value]
            out["image_strata"][attr][value] = _stratum(inside, rest, class_names,
                                                        op_thresholds, n_boot, seed)
    for c, name in enumerate(class_names):
        cuts = thr["size_terciles"].get(name)
        if not cuts:
            continue
        bands = {"small": (0.0, cuts[0]), "medium": (cuts[0], cuts[1]),
                 "large": (cuts[1], math.inf)}
        out["object_size"][name] = {}
        band_records = {}
        for band, (lo, hi) in bands.items():
            recs = []
            for r in rows:
                def in_range(cls, box, w=r["w"], h=r["h"], lo=lo, hi=hi, c=c):
                    return cls == c and lo <= relative_size(box, w, h) < hi
                recs.append(match_for_ap_in_range(
                    [(cc, tuple(b)) for cc, *b in r["gt"]],
                    [(cc, tuple(p[:4]), p[4]) for cc, *p in r["preds"]], k, in_range))
            band_records[band] = recs
            ap = bootstrap_ap(recs, class_names, n_boot=n_boot, seed=seed)["per_class"][name]
            out["object_size"][name][band] = {"relative_size": [lo, None if hi == math.inf
                                                                else hi], **ap}
        # Same images, two object ranges: a paired comparison.
        pb = paired_bootstrap(band_records["small"], band_records["large"], class_names,
                              n_boot=n_boot, seed=seed)["per_class"][name]
        out["object_size"][name]["small_minus_large"] = pb
    return out


def notable_gaps(report: dict, class_names: list[str]) -> tuple[list[dict], int]:
    """Per-class strata whose AP differs from the rest of the split with a CI
    excluding zero, and how many comparisons were tested (so the reader can
    see how many would clear 95% by chance).

    Per class only: a stratum's mAP averages just the classes present in it,
    so mAP across strata compares different class mixes. A comparison is only
    made when both the stratum and the rest hold >= MIN_INSTANCES of the class."""
    found, tested = [], 0
    overall = report["overall"]["ap"]["per_class"]
    for attr, values in report["image_strata"].items():
        items = list(values.items())
        if len(items) == 2:  # "glare vs rest" IS "no glare vs rest", mirrored: test once
            items = items[:1]
        for value, st in items:
            gap = st.get("gap_vs_rest")
            if not gap:
                continue
            for name in class_names:
                inst = st["ap"]["per_class"][name]["instances"]
                rest = overall[name]["instances"] - inst
                if inst < MIN_INSTANCES or rest < MIN_INSTANCES:
                    continue
                tested += 1
                g = gap["per_class"][name]
                if g["significant"]:
                    found.append({"attribute": attr, "value": value, "class": name,
                                  "diff": g["diff"], "ci": [g["ci_low"], g["ci_high"]],
                                  "images": st["images"], "instances": inst})
    for name, bands in report["object_size"].items():
        g = bands.get("small_minus_large")
        if not g or min(bands["small"]["instances"], bands["large"]["instances"]) < MIN_INSTANCES:
            continue
        tested += 1
        if g["significant"]:
            found.append({"attribute": "object size", "value": "small vs large",
                          "class": name, "diff": g["diff"], "ci": [g["ci_low"], g["ci_high"]],
                          "images": report["images"], "instances": bands["small"]["instances"]})
    return sorted(found, key=lambda f: f["diff"]), tested
