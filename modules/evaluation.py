"""Detection-evaluation primitives (model-free, unit-testable).

The heavy, authoritative metrics (mAP, per-class precision/recall) come from
Ultralytics ``model.val()`` in ``evaluate_model.py``. This module holds the
*pure* logic that ``val()`` does not expose per image and that the error-analysis
report needs: greedy IoU matching of predictions to ground truth, a confusion
matrix with a background row/column, and precision/recall/F1 from counts.

Everything here is plain Python (no torch, no ultralytics, no file I/O), so it
runs in CI without the model or dataset and is covered by ``tests/test_evaluation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def iou(box_a, box_b) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Match:
    """One matched (or unmatched) pair from :func:`match_image`."""

    gt_class: int | None  # None => a false positive (prediction with no GT)
    pred_class: int | None  # None => a false negative (GT with no prediction)
    score: float | None  # prediction confidence, if any
    iou: float


def match_image(
    gt: list[tuple[int, tuple]],
    preds: list[tuple[int, tuple, float]],
    iou_threshold: float = 0.5,
) -> list[Match]:
    """Greedily match one image's predictions to its ground-truth boxes.

    Args:
        gt: ``[(class_id, (x1,y1,x2,y2)), ...]`` ground-truth boxes.
        preds: ``[(class_id, (x1,y1,x2,y2), score), ...]`` predictions.
        iou_threshold: minimum IoU for a prediction to match a GT box.

    Matching is class-agnostic on the IoU (so a WithHelmet predicted over a
    WithoutHelmet GT is reported as a *class confusion*, not a separate FP+FN);
    each GT and each prediction is used at most once, highest-confidence
    predictions first. Returns a list of :class:`Match`:

    * ``gt_class`` and ``pred_class`` both set  -> a matched pair (correct if
      the classes are equal, a class confusion otherwise),
    * ``pred_class`` set, ``gt_class`` None     -> false positive,
    * ``gt_class`` set, ``pred_class`` None      -> false negative (missed).
    """
    matches: list[Match] = []
    used_gt: set[int] = set()

    # Highest-confidence predictions claim their best GT first.
    order = sorted(range(len(preds)), key=lambda i: preds[i][2], reverse=True)
    for pi in order:
        p_cls, p_box, p_score = preds[pi]
        best_iou = iou_threshold
        best_gi = -1
        for gi, (g_cls, g_box) in enumerate(gt):
            if gi in used_gt:
                continue
            ov = iou(p_box, g_box)
            if ov >= best_iou:
                best_iou = ov
                best_gi = gi
        if best_gi >= 0:
            used_gt.add(best_gi)
            matches.append(
                Match(gt_class=gt[best_gi][0], pred_class=p_cls,
                      score=p_score, iou=best_iou)
            )
        else:
            matches.append(Match(gt_class=None, pred_class=p_cls,
                                 score=p_score, iou=0.0))

    for gi, (g_cls, _g_box) in enumerate(gt):
        if gi not in used_gt:
            matches.append(Match(gt_class=g_cls, pred_class=None,
                                 score=None, iou=0.0))

    return matches


class ConfusionMatrix:
    """Square confusion matrix over ``num_classes`` plus a background bucket.

    Row = ground-truth class, column = predicted class. The extra last index
    (``num_classes``) is *background*: a false negative lands in
    ``[gt_class][background]`` (GT predicted as nothing) and a false positive in
    ``[background][pred_class]`` (nothing predicted as a class).
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.bg = num_classes
        size = num_classes + 1
        self.matrix = [[0] * size for _ in range(size)]

    def add_matches(self, matches: list[Match]) -> None:
        for m in matches:
            row = m.gt_class if m.gt_class is not None else self.bg
            col = m.pred_class if m.pred_class is not None else self.bg
            self.matrix[row][col] += 1

    def per_class_counts(self) -> dict[int, dict[str, int]]:
        """TP/FP/FN per class derived from the matrix."""
        out: dict[int, dict[str, int]] = {}
        for c in range(self.num_classes):
            tp = self.matrix[c][c]
            fn = sum(self.matrix[c]) - tp  # GT c predicted as anything else
            fp = sum(self.matrix[r][c] for r in range(self.num_classes + 1)) - tp
            out[c] = {"tp": tp, "fp": fp, "fn": fn}
        return out

    def as_list(self) -> list[list[int]]:
        return [row[:] for row in self.matrix]


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


@dataclass
class ClassReport:
    name: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def build_class_reports(
    cm: ConfusionMatrix, class_names: list[str]
) -> list[ClassReport]:
    counts = cm.per_class_counts()
    reports = []
    for c, name in enumerate(class_names):
        cnt = counts[c]
        prf = precision_recall_f1(cnt["tp"], cnt["fp"], cnt["fn"])
        reports.append(
            ClassReport(
                name=name, tp=cnt["tp"], fp=cnt["fp"], fn=cnt["fn"],
                precision=prf["precision"], recall=prf["recall"], f1=prf["f1"],
            )
        )
    return reports


@dataclass
class ConfidenceStats:
    """Confidence-vs-correctness summary, to answer 'does the model *know* when
    it is right?'. A well-behaved model has higher mean confidence on correct
    predictions than on wrong ones."""

    n_correct: int = 0
    n_wrong: int = 0
    correct_scores: list[float] = field(default_factory=list)
    wrong_scores: list[float] = field(default_factory=list)

    def add(self, correct: bool, score: float) -> None:
        if correct:
            self.n_correct += 1
            self.correct_scores.append(score)
        else:
            self.n_wrong += 1
            self.wrong_scores.append(score)

    @staticmethod
    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    def summary(self) -> dict:
        return {
            "n_correct": self.n_correct,
            "n_wrong": self.n_wrong,
            "mean_conf_correct": self._mean(self.correct_scores),
            "mean_conf_wrong": self._mean(self.wrong_scores),
            # >0 means the model is more confident when right (the desirable
            # direction); <=0 is a red flag for confidently-wrong predictions.
            "separation": round(
                self._mean(self.correct_scores) - self._mean(self.wrong_scores), 4
            ),
        }
