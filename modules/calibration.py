"""Confidence calibration experiments (Phase 10).

The pipeline treats its violation ``confidence`` as an **ordering score, not a
probability** — we have never validated that "0.8" means "80% correct", so the
code never calls it a probability. This module provides the *optional* machinery
to check whether a score could be turned into a calibrated probability, using
labelled (score, correct?) data:

* :func:`reliability_curve` / :func:`expected_calibration_error` /
  :func:`brier_score` — measure how far a score is from being a probability.
* :class:`PlattCalibrator` (1-D logistic) and :class:`IsotonicCalibrator`
  (pool-adjacent-violators) — learn a monotonic score→probability mapping.
* :func:`fit_and_evaluate` — fit both, report before/after ECE + Brier, and pick
  the best; :func:`save_calibrator` / :func:`load_calibrator` persist it.

The contract the rest of the system honours: a calibrated probability is only
ever exposed **separately** from, and never overwrites, the raw score — and only
after a calibrator has been fitted and validated on real labelled data. Until
someone runs this on such data (e.g. exported from the human-review workflow),
no probability is claimed. Pure Python, no numpy/sklearn, so it is CI-testable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def brier_score(probs, labels) -> float:
    """Mean squared error between predicted probability and 0/1 outcome.
    Lower is better; 0 is perfect."""
    n = len(probs)
    if n == 0:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / n


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_curve(probs, labels, n_bins: int = 10) -> list[ReliabilityBin]:
    """Bin predictions by confidence and report, per bin, the mean confidence vs
    the empirical accuracy. A well-calibrated model has accuracy ≈ confidence in
    every bin."""
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # last bin is closed on the right so p==1.0 lands somewhere
        members = [
            (p, y) for p, y in zip(probs, labels)
            if (lo <= p < hi) or (b == n_bins - 1 and p == 1.0)
        ]
        if members:
            mc = sum(p for p, _ in members) / len(members)
            acc = sum(y for _, y in members) / len(members)
        else:
            mc = acc = 0.0
        bins.append(ReliabilityBin(lo, hi, len(members), mc, acc))
    return bins


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin size."""
    n = len(probs)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in reliability_curve(probs, labels, n_bins):
        if b.count:
            ece += (b.count / n) * abs(b.accuracy - b.mean_confidence)
    return ece


def maximum_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """MCE: the worst per-bin confidence/accuracy gap."""
    gaps = [
        abs(b.accuracy - b.mean_confidence)
        for b in reliability_curve(probs, labels, n_bins) if b.count
    ]
    return max(gaps) if gaps else 0.0


class PlattCalibrator:
    """1-D logistic calibration: ``p = sigmoid(a * score + b)``, fit by gradient
    descent on log-loss. Simple, monotonic, two parameters."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = a
        self.b = b

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def fit(self, scores, labels, *, lr: float = 0.1, iters: int = 2000) -> "PlattCalibrator":
        n = len(scores)
        if n == 0:
            return self
        a, b = self.a, self.b
        for _ in range(iters):
            ga = gb = 0.0
            for s, y in zip(scores, labels):
                p = self._sigmoid(a * s + b)
                err = p - y
                ga += err * s
                gb += err
            a -= lr * ga / n
            b -= lr * gb / n
        self.a, self.b = a, b
        return self

    def predict(self, score: float) -> float:
        return _clamp01(self._sigmoid(self.a * score + self.b))

    def to_dict(self) -> dict:
        return {"method": "platt", "a": self.a, "b": self.b}


class IsotonicCalibrator:
    """Isotonic (monotonic, non-parametric) calibration via pool-adjacent-
    violators. More flexible than Platt but needs more data."""

    def __init__(self, xs=None, ys=None):
        self.xs = list(xs) if xs else []
        self.ys = list(ys) if ys else []

    def fit(self, scores, labels) -> "IsotonicCalibrator":
        paired = sorted(zip(scores, labels))
        if not paired:
            return self
        # Pool-adjacent-violators: each block holds (sum_y, weight, x_right).
        blocks = [[float(y), 1.0, s] for s, y in paired]
        i = 0
        while i < len(blocks) - 1:
            mean_i = blocks[i][0] / blocks[i][1]
            mean_next = blocks[i + 1][0] / blocks[i + 1][1]
            if mean_i > mean_next:  # violation -> merge and back up
                blocks[i][0] += blocks[i + 1][0]
                blocks[i][1] += blocks[i + 1][1]
                blocks[i][2] = blocks[i + 1][2]
                del blocks[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1
        self.xs = [blk[2] for blk in blocks]
        self.ys = [blk[0] / blk[1] for blk in blocks]
        return self

    def predict(self, score: float) -> float:
        if not self.xs:
            return _clamp01(score)
        if score <= self.xs[0]:
            return _clamp01(self.ys[0])
        if score >= self.xs[-1]:
            return _clamp01(self.ys[-1])
        # linear interpolation between the two surrounding block right-edges
        for i in range(1, len(self.xs)):
            if score <= self.xs[i]:
                x0, x1 = self.xs[i - 1], self.xs[i]
                y0, y1 = self.ys[i - 1], self.ys[i]
                t = (score - x0) / (x1 - x0) if x1 != x0 else 0.0
                return _clamp01(y0 + t * (y1 - y0))
        return _clamp01(self.ys[-1])

    def to_dict(self) -> dict:
        return {"method": "isotonic", "xs": self.xs, "ys": self.ys}


def _metrics(probs, labels, n_bins: int) -> dict:
    return {
        "ece": round(expected_calibration_error(probs, labels, n_bins), 4),
        "mce": round(maximum_calibration_error(probs, labels, n_bins), 4),
        "brier": round(brier_score(probs, labels), 4),
    }


MIN_PER_CLASS = 30


def _fold_ids(n: int, folds: int, groups, seed: int) -> list[int]:
    """Fold index per sample. With ``groups`` (e.g. the session/video a review
    came from) whole groups share a fold, so a calibrator is never evaluated on
    samples from a video it was fitted on."""
    import random

    rng = random.Random(seed)
    if groups is None:
        order = list(range(n))
        rng.shuffle(order)
        ids = [0] * n
        for rank, i in enumerate(order):
            ids[i] = rank % folds
        return ids
    keys = sorted(set(groups), key=str)
    rng.shuffle(keys)
    fold_of = {k: r % folds for r, k in enumerate(keys)}
    return [fold_of[g] for g in groups]


def _out_of_fold(make, scores, labels, fold_ids, folds: int) -> list[float]:
    probs = [0.0] * len(scores)
    for f in range(folds):
        train = [i for i, k in enumerate(fold_ids) if k != f]
        test = [i for i, k in enumerate(fold_ids) if k == f]
        if not test:
            continue
        cal = make().fit([scores[i] for i in train], [labels[i] for i in train])
        for i in test:
            probs[i] = cal.predict(scores[i])
    return probs


def _brier_gain(raw, cal, labels, *, n_boot: int, seed: int, level: float = 0.95) -> dict:
    """Mean per-sample Brier improvement of ``cal`` over ``raw`` (positive =
    better) with a bootstrap CI over samples."""
    import random

    d = [(r - y) ** 2 - (c - y) ** 2 for r, c, y in zip(raw, cal, labels)]
    n = len(d)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = boots[int((1 - level) / 2 * n_boot)]
    hi = boots[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
    return {"mean": round(sum(d) / n, 5), "ci_low": round(lo, 5), "ci_high": round(hi, 5)}


def threshold_table(scores, labels, thresholds=None) -> list[dict]:
    """If only violations scoring >= t were auto-accepted: how many, what share
    of them are correct (precision), and what share of all correct ones they
    keep (recall). The operating question a score must answer, calibrated or
    not."""
    thresholds = thresholds or [round(0.05 * i, 2) for i in range(1, 20)]
    positives = sum(labels)
    rows = []
    for t in thresholds:
        kept = [y for s, y in zip(scores, labels) if s >= t]
        tp = sum(kept)
        rows.append({
            "threshold": t, "kept": len(kept),
            "precision": round(tp / len(kept), 4) if kept else None,
            "recall": round(tp / positives, 4) if positives else None,
        })
    return rows


def fit_and_evaluate(scores, labels, *, n_bins: int = 10, folds: int = 5, groups=None,
                     seed: int = 0, n_boot: int = 1000,
                     min_per_class: int = MIN_PER_CLASS) -> dict:
    """Is the raw score calibrated, and would Platt or isotonic calibration help
    on data it was NOT fitted on?

    Every calibrator is scored **out of fold** (k-fold; grouped by ``groups``
    when given). Scoring a calibrator on the rows it was fitted to rewards
    memorisation — isotonic regression can nearly reproduce the per-bin
    accuracies in-sample — so the in-sample numbers are reported only to show
    that optimism. A calibrator is adopted (``best``) only if its out-of-fold
    Brier score beats the raw score's with a bootstrap CI excluding zero; Brier
    is the selection metric because, unlike ECE, it is a proper scoring rule and
    doesn't depend on binning. With fewer than ``min_per_class`` correct or
    incorrect outcomes, the verdict is ``insufficient_data`` and nothing is
    adopted."""
    n = len(scores)
    pos = sum(1 for y in labels if y)
    neg = n - pos
    raw = list(scores)
    out: dict = {
        "n": n, "correct": pos, "incorrect": neg, "folds": folds,
        "grouped": groups is not None, "selection": "out-of-fold Brier, bootstrap CI",
        "raw": _metrics(raw, labels, n_bins),
        "reliability_raw": [b.__dict__ for b in reliability_curve(raw, labels, n_bins)],
        "threshold_table": threshold_table(raw, labels),
    }
    if min(pos, neg) < min_per_class:
        out.update(verdict="insufficient_data", best="raw",
                   reason=f"need >= {min_per_class} correct and incorrect outcomes; "
                          f"have {pos} / {neg}")
        return out

    fold_ids = _fold_ids(n, folds, groups, seed)
    effective_folds = len(set(fold_ids))
    candidates: dict[str, type[PlattCalibrator] | type[IsotonicCalibrator]] = {
        "platt": PlattCalibrator, "isotonic": IsotonicCalibrator}
    gains = {}
    for name, make in candidates.items():
        oof = _out_of_fold(make, raw, labels, fold_ids, folds)
        fitted = make().fit(raw, labels)
        in_sample = [fitted.predict(x) for x in raw]
        out[name] = {**_metrics(oof, labels, n_bins),
                     "in_sample_ece": round(expected_calibration_error(in_sample, labels,
                                                                       n_bins), 4),
                     "params": fitted.to_dict()}
        gains[name] = _brier_gain(raw, oof, labels, n_boot=n_boot, seed=seed)
        out[name]["brier_gain_vs_raw"] = gains[name]
    significant = {k: g for k, g in gains.items() if g["ci_low"] > 0}
    if effective_folds < 2:
        out.update(verdict="insufficient_data", best="raw",
                   reason="fewer than two folds (too few groups)")
    elif significant:
        best = max(significant, key=lambda k: significant[k]["mean"])
        out.update(verdict="calibration_helps", best=best)
    else:
        out.update(verdict="raw_is_as_good", best="raw",
                   reason="no calibrator improved out-of-fold Brier with a CI "
                          "excluding zero")
    return out


def save_calibrator(calibrator, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(calibrator.to_dict(), fh, indent=2, sort_keys=True)


def load_calibrator(path: str):
    """Load a saved calibrator; returns a Platt/Isotonic instance or None."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if data.get("method") == "platt":
        return PlattCalibrator(a=data["a"], b=data["b"])
    if data.get("method") == "isotonic":
        return IsotonicCalibrator(xs=data["xs"], ys=data["ys"])
    return None
