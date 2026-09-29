"""Detector metrics with uncertainty: AP@50 per class, bootstrap CIs, paired
model comparisons, and confidence-threshold selection.

Why this exists: the test split has **27** ``WithHelmet`` instances. At that
size a single AP number says little on its own — the previous model decisions
("v2 is worse", "probe wins") compared point estimates that could easily be
sampling noise. This module answers the question properly:

* :func:`bootstrap_ap` — resample *images* with replacement (instances inside
  an image are correlated, so images are the unit) and report a percentile
  confidence interval for each class's AP@50 and for mAP@50.
* :func:`paired_bootstrap` — resample the SAME images for two models and
  report the CI of the AP difference and how often model A beats model B. A
  difference is only called real when that CI excludes zero.
* :func:`f1_optimal_thresholds` — per-class confidence thresholds chosen on one
  split (validation) and then *applied* to another (test), so a threshold is a
  decision made before looking at the reported data.

AP is computed the way Ultralytics does it (greedy class-aware matching at IoU
0.5 in score order, monotone precision envelope, 101-point interpolation), so
the point estimates track ``model.val()`` closely; small residual differences
come from NMS/letterbox details and are reported next to the official number,
not hidden. Everything here is pure numpy on cached predictions — no model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.evaluation import iou


@dataclass
class ImageRecord:
    """One image's scored predictions, flattened for AP.

    ``pred_cls``/``pred_score``/``pred_tp`` are parallel arrays; ``gt_counts``
    is the number of ground-truth boxes per class."""

    pred_cls: np.ndarray
    pred_score: np.ndarray
    pred_tp: np.ndarray
    gt_counts: np.ndarray


def match_for_ap(gt, preds, n_classes: int, iou_threshold: float = 0.5) -> ImageRecord:
    """Class-aware greedy matching (the COCO/Ultralytics convention): within
    each class, predictions in descending score order claim the unmatched
    same-class GT box they overlap most, if IoU >= threshold.

    ``gt``: ``[(cls, box)]``; ``preds``: ``[(cls, box, score)]``."""
    gt_counts = np.zeros(n_classes, dtype=np.int64)
    for c, _ in gt:
        if 0 <= c < n_classes:
            gt_counts[c] += 1
    cls_out, score_out, tp_out = [], [], []
    for c in range(n_classes):
        g = [b for k, b in gt if k == c]
        used = [False] * len(g)
        ps = sorted((p for p in preds if p[0] == c), key=lambda p: -p[2])
        for _, box, score in ps:
            best, best_j = iou_threshold, -1
            for j, gb in enumerate(g):
                if used[j]:
                    continue
                v = iou(box, gb)
                if v >= best:
                    best, best_j = v, j
            tp = best_j >= 0
            if tp:
                used[best_j] = True
            cls_out.append(c)
            score_out.append(score)
            tp_out.append(1.0 if tp else 0.0)
    return ImageRecord(
        np.asarray(cls_out, dtype=np.int64),
        np.asarray(score_out, dtype=np.float64),
        np.asarray(tp_out, dtype=np.float64),
        gt_counts,
    )


def match_for_ap_in_range(gt, preds, n_classes: int, in_range,
                          iou_threshold: float = 0.5) -> ImageRecord:
    """:func:`match_for_ap` restricted to boxes ``in_range(cls, box)`` accepts,
    by the COCO area-range convention: ground truth outside the range is
    *ignored* (a prediction matched to it is dropped, neither TP nor FP), and an
    unmatched prediction counts as a false positive only if it is itself in
    range. Otherwise "small objects" AP would charge the detector for every
    large object it found."""
    gt_counts = np.zeros(n_classes, dtype=np.int64)
    for c, box in gt:
        if 0 <= c < n_classes and in_range(c, box):
            gt_counts[c] += 1
    cls_out, score_out, tp_out = [], [], []
    for c in range(n_classes):
        g = [b for k, b in gt if k == c]
        g_in = [in_range(c, b) for b in g]
        used = [False] * len(g)
        for _, box, score in sorted((p for p in preds if p[0] == c), key=lambda p: -p[2]):
            # COCO: prefer an in-range object; fall back to an ignored one only
            # if no in-range object matches (a prediction overlapping both a
            # small and a large object counts for the small-object AP).
            best_j = -1
            for want_in in (True, False):
                best = iou_threshold
                for j, gb in enumerate(g):
                    if used[j] or g_in[j] != want_in:
                        continue
                    v = iou(box, gb)
                    if v >= best:
                        best, best_j = v, j
                if best_j >= 0:
                    break
            if best_j >= 0:
                used[best_j] = True
                if not g_in[best_j]:
                    continue  # matched an ignored (out-of-range) object
                tp = 1.0
            elif in_range(c, box):
                tp = 0.0
            else:
                continue
            cls_out.append(c)
            score_out.append(score)
            tp_out.append(tp)
    return ImageRecord(np.asarray(cls_out, dtype=np.int64),
                       np.asarray(score_out, dtype=np.float64),
                       np.asarray(tp_out, dtype=np.float64), gt_counts)


def _ap_from_curve(recall: np.ndarray, precision: np.ndarray) -> float:
    """Ultralytics ``compute_ap`` (8.4): precision envelope + 101-point
    interpolation, with precision dropping to 0 right after the last achieved
    recall — without that sentinel a detector that tops out at 50% recall is
    credited a spurious triangle up to recall 1.0 (0.75 instead of 0.495)."""
    last = recall[-1] if len(recall) else 1.0
    mrec = np.concatenate(([0.0], recall, [last], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


class _ClassIndex:
    """All predictions of one class across the split, pre-sorted by score, with
    the image each came from — so any image re-weighting is one cumsum."""

    def __init__(self, records: list[ImageRecord], c: int):
        img, score, tp = [], [], []
        for i, r in enumerate(records):
            m = r.pred_cls == c
            if m.any():
                img.append(np.full(int(m.sum()), i))
                score.append(r.pred_score[m])
                tp.append(r.pred_tp[m])
        self.img = np.concatenate(img) if img else np.zeros(0, dtype=np.int64)
        s = np.concatenate(score) if score else np.zeros(0)
        t = np.concatenate(tp) if tp else np.zeros(0)
        order = np.argsort(-s, kind="stable")
        self.img, self.score, self.tp = self.img[order], s[order], t[order]
        self.gt = np.asarray([r.gt_counts[c] for r in records], dtype=np.float64)

    def ap(self, weights: np.ndarray | None = None) -> float:
        w_img = np.ones(len(self.gt)) if weights is None else weights
        n_gt = float((w_img * self.gt).sum())
        if n_gt <= 0:
            return float("nan")
        if self.score.size == 0:
            return 0.0
        w = w_img[self.img]
        tpc = np.cumsum(w * self.tp)
        fpc = np.cumsum(w * (1.0 - self.tp))
        recall = tpc / n_gt
        precision = tpc / np.maximum(tpc + fpc, 1e-16)
        return _ap_from_curve(recall, precision)

    def pr_at(self, threshold: float, weights: np.ndarray | None = None) -> dict:
        w_img = np.ones(len(self.gt)) if weights is None else weights
        keep = self.score >= threshold
        w = w_img[self.img][keep]
        tp = float((w * self.tp[keep]).sum())
        fp = float((w * (1.0 - self.tp[keep])).sum())
        n_gt = float((w_img * self.gt).sum())
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / n_gt if n_gt else float("nan")
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"threshold": round(threshold, 4), "precision": round(p, 4),
                "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp,
                "gt": n_gt}


def _nanmean(x, axis=None):
    """nanmean that returns NaN quietly when every entry is NaN (a stratum with
    no instances of any class) instead of warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _std(samples: np.ndarray) -> float | None:
    """Std of the non-NaN samples; None for a class with no GT (all NaN)."""
    samples = samples[~np.isnan(samples)]
    return _r(np.std(samples)) if samples.size else None


def _percentile_ci(samples: np.ndarray, level: float) -> tuple[float, float]:
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    lo = (1 - level) / 2 * 100
    return float(np.percentile(samples, lo)), float(np.percentile(samples, 100 - lo))


def _r(x, nd: int = 4) -> float | None:
    """Round a numpy/python float for JSON; NaN (no GT for a class) -> None."""
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, nd)


def bootstrap_ap(records: list[ImageRecord], class_names: list[str], *,
                 n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Point AP@50 per class + mAP@50, with image-bootstrap percentile CIs."""
    k = len(class_names)
    idx = [_ClassIndex(records, c) for c in range(k)]
    point = np.array([ix.ap() for ix in idx])
    rng = np.random.default_rng(seed)
    n = len(records)
    boots = np.empty((n_boot, k))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        boots[b] = [ix.ap(w) for ix in idx]
    per_class: dict[str, dict] = {}
    out: dict = {"images": n, "n_boot": n_boot, "seed": seed, "level": level,
                 "per_class": per_class}
    for c, name in enumerate(class_names):
        lo, hi = _percentile_ci(boots[:, c], level)
        per_class[name] = {
            "ap50": _r(point[c]), "ci_low": _r(lo), "ci_high": _r(hi),
            "std": _std(boots[:, c]), "instances": int(idx[c].gt.sum()),
        }
    with np.errstate(all="ignore"):
        m_boot = _nanmean(boots, axis=1)
    lo, hi = _percentile_ci(m_boot, level)
    out["map50"] = {"value": _r(_nanmean(point)), "ci_low": _r(lo), "ci_high": _r(hi),
                    "std": _std(m_boot)}
    return out


def unpaired_gap(rec_a: list[ImageRecord], rec_b: list[ImageRecord], class_names: list[str],
                 *, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """AP(A) - AP(B) for two DISJOINT image sets (e.g. dark images vs the
    rest), each resampled independently. ``significant`` when the CI excludes
    zero. For the same images under two models use :func:`paired_bootstrap`."""
    k = len(class_names)
    ia = [_ClassIndex(rec_a, c) for c in range(k)]
    ib = [_ClassIndex(rec_b, c) for c in range(k)]
    point = np.array([ia[c].ap() - ib[c].ap() for c in range(k)])
    rng = np.random.default_rng(seed)
    na, nb = len(rec_a), len(rec_b)
    diffs = np.empty((n_boot, k))
    for b in range(n_boot):
        wa = np.bincount(rng.integers(0, na, na), minlength=na).astype(np.float64)
        wb = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(np.float64)
        diffs[b] = [ia[c].ap(wa) - ib[c].ap(wb) for c in range(k)]
    out: dict = {"n_a": na, "n_b": nb, "per_class": {}}
    for c, name in enumerate(class_names):
        lo, hi = _percentile_ci(diffs[:, c], level)
        out["per_class"][name] = {"diff": _r(point[c]), "ci_low": _r(lo), "ci_high": _r(hi),
                                  "significant": bool(lo > 0 or hi < 0)}
    with np.errstate(all="ignore"):
        m = _nanmean(diffs, axis=1)
    lo, hi = _percentile_ci(m, level)
    out["map50"] = {"diff": _r(_nanmean(point)), "ci_low": _r(lo), "ci_high": _r(hi),
                    "significant": bool(lo > 0 or hi < 0)}
    return out


MIN_SEEDS = 3


def _t_quantile(level: float, df: float) -> float:
    """Two-sided Student-t quantile (no scipy): exact for df 1-2 at 95%,
    Cornish-Fisher beyond (error < 0.01 for df >= 3)."""
    z = {0.9: 1.644854, 0.95: 1.959964, 0.99: 2.575829}.get(round(level, 2), 1.959964)
    if not np.isfinite(df):
        return z
    if round(level, 2) == 0.95 and df < 3:
        return 12.706 if df < 2 else 4.303
    return float(z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2))


def recipe_comparison(recipes: dict[str, list[list[ImageRecord]]], baseline: str,
                      class_names: list[str], *, n_boot: int = 2000, seed: int = 0,
                      level: float = 0.95, min_seeds: int = MIN_SEEDS) -> dict:
    """Compare training RECIPES, each trained with several seeds, on one split.

    ``recipes[name]`` is one record list per seed (same images, same order).
    A recipe's AP is the mean over its seeds. The difference's uncertainty has
    two parts: which images were drawn (a paired image bootstrap of the
    difference of seed means) and which seeds were drawn (the between-seed
    variance of each mean, s^2 / n). They are added and turned into a Welch-t
    interval, whose degrees of freedom come from the seed term — with three
    seeds a recipe mean is uncertain, and the t quantile says so (4.30 at 2 df).
    Resampling three seeds with replacement instead understated that spread and
    promoted identical recipes about twice as often as nominal. Fewer than
    ``min_seeds`` seeds on either side: ``insufficient seeds``, never a verdict.
    A candidate is ``promote`` only if its mAP@50 gain's CI excludes zero and
    no class is significantly worse; any class significantly worse ``reject``s.

    Simulated identical recipes (3 seeds each, seed SD 0.06, 200 trials): false
    promote 2.0% (nominal 2.5%); false reject 9.5% — the price of letting any
    one of four classes veto, accepted because missing a real regression in a
    class is the worse error."""
    k = len(class_names)
    names = list(recipes)
    n_img = {len(r) for runs in recipes.values() for r in runs}
    if len(n_img) != 1:
        raise ValueError("every checkpoint must be scored on the same images")
    n = n_img.pop()
    idx = {r: [[_ClassIndex(run, c) for c in range(k)] for run in recipes[r]] for r in names}
    per_seed = {r: np.array([[ix.ap() for ix in run] for run in idx[r]]) for r in names}
    rng = np.random.default_rng(seed)
    boot = {r: np.empty((n_boot, len(idx[r]), k)) for r in names}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        for r in names:
            boot[r][b] = [[ix.ap(w) for ix in run] for run in idx[r]]
    out: dict = {"images": n, "baseline": baseline, "min_seeds": min_seeds, "level": level,
                 "interval": "Welch-t: image bootstrap + between-seed variance",
                 "recipes": {}, "comparisons": {}}
    for r in names:
        ps = per_seed[r]
        out["recipes"][r] = {
            "seeds": len(ps),
            "per_class": {name: {"mean_ap50": _r(_nanmean(ps[:, c])),
                                 "between_seed_sd": (_r(np.std(ps[:, c], ddof=1))
                                                     if len(ps) > 1 else None),
                                 "per_seed": [_r(x) for x in ps[:, c]]}
                          for c, name in enumerate(class_names)},
            "map50": _r(_nanmean(_nanmean(ps, axis=1))),
        }

    def interval(cand_seeds, base_seeds, cand_boot, base_boot):
        """cand/base_seeds: per-seed values; *_boot: (n_boot, seeds) arrays."""
        point = _nanmean(cand_seeds) - _nanmean(base_seeds)
        d = _nanmean(cand_boot, axis=1) - _nanmean(base_boot, axis=1)
        d = d[~np.isnan(d)]
        var_img = float(np.var(d, ddof=1)) if d.size > 1 else 0.0
        nc, nb = len(cand_seeds), len(base_seeds)
        a = float(np.var(cand_seeds, ddof=1)) / nc if nc > 1 else 0.0
        bb = float(np.var(base_seeds, ddof=1)) / nb if nb > 1 else 0.0
        denom = (a * a / (nc - 1) if nc > 1 else 0.0) + (bb * bb / (nb - 1) if nb > 1 else 0.0)
        df = (var_img + a + bb) ** 2 / denom if denom > 0 else float("inf")
        half = _t_quantile(level, df) * float(np.sqrt(var_img + a + bb))
        return point, point - half, point + half

    for r in names:
        if r == baseline:
            continue
        cmp: dict = {"per_class": {}}
        worse = []
        for c, name in enumerate(class_names):
            pt, lo, hi = interval(per_seed[r][:, c], per_seed[baseline][:, c],
                                  boot[r][:, :, c], boot[baseline][:, :, c])
            cmp["per_class"][name] = {"diff": _r(pt), "ci_low": _r(lo), "ci_high": _r(hi)}
            if hi < 0:
                worse.append(name)
        pt, lo, hi = interval(_nanmean(per_seed[r], axis=1), _nanmean(per_seed[baseline], axis=1),
                              _nanmean(boot[r], axis=2), _nanmean(boot[baseline], axis=2))
        cmp["map50"] = {"diff": _r(pt), "ci_low": _r(lo), "ci_high": _r(hi)}
        seeds_ok = min(len(per_seed[r]), len(per_seed[baseline])) >= min_seeds
        if not seeds_ok:
            verdict = (f"insufficient seeds ({len(per_seed[baseline])} vs {len(per_seed[r])}; "
                       f"need {min_seeds} each)")
        elif worse:
            verdict = "reject: significantly worse on " + ", ".join(worse)
        elif lo > 0:
            verdict = "promote"
        else:
            verdict = "not distinguishable"
        cmp["verdict"] = verdict
        cmp["regresses"] = worse
        out["comparisons"][r] = cmp
    return out


def recall_at(records: list[ImageRecord], class_names: list[str], thresholds: dict) -> dict:
    """Precision/recall per class at the operating threshold(s) — what a user of
    the running system experiences, as opposed to AP's whole curve."""
    return {name: _ClassIndex(records, c).pr_at(thresholds.get(name, 0.25))
            for c, name in enumerate(class_names)}


def paired_bootstrap(rec_a: list[ImageRecord], rec_b: list[ImageRecord],
                     class_names: list[str], *, n_boot: int = 2000, seed: int = 0,
                     level: float = 0.95) -> dict:
    """AP(A) - AP(B) per class and for mAP, on the SAME resampled images.

    ``p_a_better`` is the fraction of resamples where A's AP exceeds B's. The
    difference is called ``significant`` only when the CI excludes zero."""
    if len(rec_a) != len(rec_b):
        raise ValueError("paired bootstrap needs predictions on the same images")
    k = len(class_names)
    ia = [_ClassIndex(rec_a, c) for c in range(k)]
    ib = [_ClassIndex(rec_b, c) for c in range(k)]
    rng = np.random.default_rng(seed)
    n = len(rec_a)
    diffs = np.empty((n_boot, k))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        diffs[b] = [ia[c].ap(w) - ib[c].ap(w) for c in range(k)]
    point = np.array([ia[c].ap() - ib[c].ap() for c in range(k)])
    out: dict = {"n_boot": n_boot, "seed": seed, "level": level, "per_class": {}}
    for c, name in enumerate(class_names):
        lo, hi = _percentile_ci(diffs[:, c], level)
        out["per_class"][name] = {
            "diff": _r(point[c]), "ci_low": _r(lo), "ci_high": _r(hi),
            "p_a_better": _r(float(np.mean(diffs[:, c] > 0)), 3),
            "significant": bool(lo > 0 or hi < 0),
        }
    md = _nanmean(diffs, axis=1)
    lo, hi = _percentile_ci(md, level)
    out["map50"] = {"diff": _r(_nanmean(point)), "ci_low": _r(lo), "ci_high": _r(hi),
                    "p_a_better": _r(float(np.mean(md > 0)), 3),
                    "significant": bool(lo > 0 or hi < 0)}
    return out


def f1_optimal_thresholds(records: list[ImageRecord], class_names: list[str],
                          *, grid=None) -> dict:
    """Per-class confidence threshold that maximises F1 on ``records``."""
    grid = np.round(np.arange(0.05, 0.951, 0.025), 4) if grid is None else grid
    out = {}
    for c, name in enumerate(class_names):
        ix = _ClassIndex(records, c)
        rows = [ix.pr_at(float(t)) for t in grid]
        best = max(rows, key=lambda r: (r["f1"], -r["threshold"]))
        out[name] = best
    return out


def apply_thresholds(records: list[ImageRecord], class_names: list[str],
                     thresholds: dict[str, float]) -> dict:
    """Precision/recall/F1 per class at FIXED thresholds (chosen elsewhere)."""
    out = {}
    for c, name in enumerate(class_names):
        out[name] = _ClassIndex(records, c).pr_at(float(thresholds[name]))
    return out


def pr_curve(records: list[ImageRecord], class_names: list[str], *, points: int = 19) -> dict:
    """Precision/recall at evenly spaced confidence thresholds, per class."""
    grid = np.linspace(0.05, 0.95, points)
    return {name: [_ClassIndex(records, c).pr_at(float(t)) for t in grid]
            for c, name in enumerate(class_names)}


# -- (de)serialisation of cached predictions ---------------------------------

def records_to_json(records: list[ImageRecord]) -> list[dict]:
    return [{"c": r.pred_cls.tolist(), "s": [round(x, 5) for x in r.pred_score.tolist()],
             "t": r.pred_tp.astype(int).tolist(), "g": r.gt_counts.tolist()}
            for r in records]


def records_from_json(rows: list[dict]) -> list[ImageRecord]:
    return [ImageRecord(np.asarray(r["c"], dtype=np.int64),
                        np.asarray(r["s"], dtype=np.float64),
                        np.asarray(r["t"], dtype=np.float64),
                        np.asarray(r["g"], dtype=np.int64)) for r in rows]
