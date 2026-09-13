"""Does a higher YOLO score actually mean a more likely-correct detection?

`modules/evaluation.ConfidenceStats` already answers a coarse version of this
(mean score when right vs wrong). That single number can hide the shape of the
relationship: a model can have a healthy mean separation while its 0.9 bucket is
no better than its 0.5 bucket, which is precisely the property that decides
whether a confidence threshold is a useful review-queue knob.

So this module bins predictions by score and reports, per bin and per class, the
fraction that were correct. That is a **score-vs-accuracy curve**, not a
calibration curve: YOLO's objectness*class score is not a probability and this
module never claims it is. What it does claim, and what it measures, is
*monotonicity* — whether accuracy rises with the score — summarised as:

* :attr:`ConfidenceCurve.spearman` — rank correlation between bin score and bin
  accuracy (+1 = perfectly ordered, 0 = the score carries no ranking signal),
* :attr:`ConfidenceCurve.monotonic_violations` — adjacent bin pairs where
  accuracy *falls* as score rises,
* ``gap`` — accuracy in the top bin minus the bottom bin.

If you want an actual probability, that is `modules/calibration.py` (Platt /
isotonic against labelled review outcomes) — a different, unshipped step.

Pure Python: no numpy, no torch, deterministic, unit-tested in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_EDGES = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0001)


@dataclass
class Bin:
    low: float
    high: float
    n: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def mean_score(self) -> float:
        """Bin midpoint — the representative score for the bucket."""
        return (self.low + self.high) / 2

    def as_dict(self) -> dict:
        return {
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "n": self.n,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
        }


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation, ties averaged. 0.0 for degenerate input."""
    n = len(xs)
    if n < 2:
        return 0.0

    def ranks(vs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(num / (dx * dy), 4) if dx and dy else 0.0


@dataclass
class ConfidenceCurve:
    """Binned score-vs-accuracy for one class (or for all classes pooled)."""

    label: str
    bins: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return sum(b.n for b in self.bins)

    @property
    def populated(self) -> list:
        """Bins with at least one prediction — the only ones worth reading."""
        return [b for b in self.bins if b.n]

    @property
    def spearman(self) -> float:
        pop = self.populated
        return _spearman([b.mean_score for b in pop], [b.accuracy for b in pop])

    @property
    def monotonic_violations(self) -> int:
        pop = self.populated
        return sum(
            1 for a, b in zip(pop, pop[1:]) if b.accuracy < a.accuracy
        )

    @property
    def gap(self) -> float:
        """Top populated bin's accuracy minus the bottom's. >0 = the score helps."""
        pop = self.populated
        return round(pop[-1].accuracy - pop[0].accuracy, 4) if len(pop) >= 2 else 0.0

    def verdict(self) -> str:
        """A one-line, honest reading of the curve."""
        if self.n < 20:
            return "too few predictions to judge"
        if self.gap > 0.1 and self.spearman > 0.5:
            return "higher score => more likely correct (useful ranking signal)"
        if self.gap <= 0.0:
            return "score does NOT separate correct from wrong (do not threshold on it)"
        return "weak/non-monotonic relationship; threshold with care"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "bins": [b.as_dict() for b in self.bins],
            "spearman": self.spearman,
            "monotonic_violations": self.monotonic_violations,
            "top_minus_bottom_bin_accuracy": self.gap,
            "verdict": self.verdict(),
        }


@dataclass
class ScoredPrediction:
    """One prediction: its score, whether it was correct, and its class name."""

    score: float
    correct: bool
    class_name: str = "all"


def build_curve(
    preds: list[ScoredPrediction], *, label: str = "all", edges=DEFAULT_EDGES
) -> ConfidenceCurve:
    """Bin predictions by score into a :class:`ConfidenceCurve`."""
    bounds = list(edges)
    bins = [Bin(low=bounds[i], high=bounds[i + 1]) for i in range(len(bounds) - 1)]
    for p in preds:
        for b in bins:
            # Half-open bins; the final edge is >1 so score 1.0 lands in the top.
            if b.low <= p.score < b.high:
                b.n += 1
                b.correct += 1 if p.correct else 0
                break
    return ConfidenceCurve(label=label, bins=bins)


def analyse(preds: list[ScoredPrediction], *, edges=DEFAULT_EDGES) -> dict:
    """Overall curve plus one curve per class name present."""
    overall = build_curve(preds, label="all", edges=edges)
    grouped: dict[str, list] = {}
    for p in preds:
        grouped.setdefault(p.class_name, []).append(p)
    by_class = {
        name: build_curve(items, label=name, edges=edges)
        for name, items in sorted(grouped.items())
    }
    return {
        "note": (
            "YOLO confidence is a model score, not a calibrated probability. "
            "This measures whether accuracy increases with the score "
            "(ranking usefulness), not whether 0.8 means 80% correct."
        ),
        "edges": list(edges),
        "overall": overall.as_dict(),
        "per_class": {k: v.as_dict() for k, v in by_class.items()},
    }


def render_histogram(curve: ConfidenceCurve, width: int = 40) -> list[str]:
    """A text histogram of bin counts with the accuracy beside each bar.

    Text, not a PNG, on purpose: it lands in the Markdown report, survives in a
    terminal and a diff, and needs no plotting dependency in CI.
    """
    lines = [f"{'score bin':>12}  {'n':>5}  {'acc':>6}  histogram"]
    peak = max((b.n for b in curve.bins), default=0)
    for b in curve.bins:
        bar = "#" * int(width * b.n / peak) if peak else ""
        acc = f"{b.accuracy:.3f}" if b.n else "  -  "
        lines.append(f"{b.low:5.2f}-{b.high:<5.2f}  {b.n:>5}  {acc:>6}  {bar}")
    return lines
