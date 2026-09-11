"""OCR evaluation metrics (pure, model-free).

The scoring half of `evaluate_ocr.py` lives here so it can be unit-tested and run
in CI without EasyOCR/torch. Given a list of (ground-truth, prediction,
confidence) observations it computes the plate-OCR metrics that actually matter:

* **exact-match accuracy** — prediction equals truth character-for-character.
* **normalized-match accuracy** — equal after `normalize_plate` (strip
  non-alphanumerics, upper-case); the metric the pipeline really cares about,
  since that is how plates are stored and looked up.
* **character accuracy** — ``1 - edit_distance/len(truth)`` averaged; partial
  credit for a mostly-right read.
* **mean edit distance** — Levenshtein between normalized truth and prediction.
* **invalid-output rate** — fraction of predictions that don't match a valid
  Indian plate structure (junk the runtime would refuse to fine on).
* **confidence vs correctness** — mean OCR confidence when exactly right vs
  wrong; a positive separation means confidence is a useful review-ordering
  signal.

Everything is deterministic and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modules.plate_info import matches_structure, normalize_plate


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insertions/deletions/substitutions), O(len_a*len_b)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def char_accuracy(truth: str, pred: str) -> float:
    """1 - normalized edit distance, clamped to [0, 1]. Empty truth: 1.0 if the
    prediction is also empty, else 0.0."""
    if not truth:
        return 1.0 if not pred else 0.0
    dist = levenshtein(truth, pred)
    return max(0.0, 1.0 - dist / len(truth))


@dataclass
class OcrObservation:
    truth: str
    pred: str
    confidence: float = 0.0
    condition: str = "all"


@dataclass
class OcrMetrics:
    n: int = 0
    exact_match: float = 0.0
    normalized_match: float = 0.0
    char_accuracy: float = 0.0
    mean_edit_distance: float = 0.0
    invalid_rate: float = 0.0
    mean_conf_correct: float = 0.0
    mean_conf_wrong: float = 0.0
    confidence_separation: float = 0.0

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def evaluate(observations: list[OcrObservation]) -> OcrMetrics:
    """Aggregate a list of observations into :class:`OcrMetrics`."""
    n = len(observations)
    if n == 0:
        return OcrMetrics()

    exact = norm_match = char_acc = edit_sum = invalid = 0
    conf_correct: list[float] = []
    conf_wrong: list[float] = []

    for o in observations:
        t_norm = normalize_plate(o.truth)
        p_norm = normalize_plate(o.pred)
        is_exact = o.pred == o.truth
        is_norm = t_norm == p_norm and t_norm != ""
        exact += 1 if is_exact else 0
        norm_match += 1 if is_norm else 0
        char_acc += char_accuracy(t_norm, p_norm)
        edit_sum += levenshtein(t_norm, p_norm)
        if not matches_structure(p_norm):
            invalid += 1
        (conf_correct if is_norm else conf_wrong).append(o.confidence)

    mcc = sum(conf_correct) / len(conf_correct) if conf_correct else 0.0
    mcw = sum(conf_wrong) / len(conf_wrong) if conf_wrong else 0.0
    return OcrMetrics(
        n=n,
        exact_match=exact / n,
        normalized_match=norm_match / n,
        char_accuracy=char_acc / n,
        mean_edit_distance=edit_sum / n,
        invalid_rate=invalid / n,
        mean_conf_correct=mcc,
        mean_conf_wrong=mcw,
        confidence_separation=mcc - mcw,
    )


@dataclass
class OcrReport:
    overall: OcrMetrics
    by_condition: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "overall": self.overall.as_dict(),
            "by_condition": {k: v.as_dict() for k, v in self.by_condition.items()},
        }


def evaluate_with_conditions(observations: list[OcrObservation]) -> OcrReport:
    """Overall metrics plus a breakdown by each observation's ``condition``
    (clean/blur/angle/lowlight/small/occluded/…). Conditions are whatever labels
    appear in the data; a set with no condition column is all ``"all"``."""
    overall = evaluate(observations)
    conditions: dict[str, list[OcrObservation]] = {}
    for o in observations:
        conditions.setdefault(o.condition, []).append(o)
    by_condition = {c: evaluate(obs) for c, obs in sorted(conditions.items())}
    return OcrReport(overall=overall, by_condition=by_condition)
