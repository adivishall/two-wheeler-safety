"""Choose what a human labels next — for information, not volume.

A labelling hour is the scarcest input to the improvement loop, so candidates
are ranked by how much a label would *change what we know*, and the queue is
kept diverse so an hour doesn't go on twenty copies of one failure.

Signals (each in [0, 1], named in the queue so a reviewer sees why):

From detector output on an image pool (predictions vs whatever labels exist):

* ``helmet_contradiction`` — a WithHelmet and a WithoutHelmet box on the same
  rider (IoU >= 0.5), both above threshold: the model can't decide.
* ``near_threshold`` — a no-helmet/triple score within 0.1 of its operating
  threshold: the decision would flip with small changes.
* ``possible_missing_label`` — a confident prediction (>= 0.6) with no label
  of any class under it. On validation, 11 of the 16 most confident "false
  positives" were real, unlabelled objects (``manual_error_review.md``).
* ``class_disagreement`` — a confident prediction on a labelled rider of a
  different class: a model error or a label error, informative either way.
* ``label_gap`` — a confident Plate/helmet prediction on an image from a source
  whose labels never include that class (the disjoint-label audit): the
  highest-value data fix found (``docs/ERROR_ANALYSIS.md`` F5).
* ``known_weak_stratum`` — the image is in a condition the stratified
  evaluation confirmed as weak (multi-rider scenes for WithoutHelmet,
  ``conditions.md``); a bonus, never a reason on its own.

From the review queue (pipeline output awaiting a human): low final score,
a plate vote with a runner-up or small margin, weak association.

Priority is ``1 - prod(1 - s)`` over the signals (any strong signal suffices;
several add up; bounded in [0, 1]) — a ranking, not a probability. Selection
is greedy with diminishing returns per failure *pattern* (``decay ** n`` for
the n-th pick of a pattern) and skips perceptual near-duplicates of anything
already picked. Model signals select from TRAINING data only: relabelling the
held-out images the model disagrees with would bias the evaluation toward the
model, so evaluation labels are audited on a uniform-random, model-blind sample
instead (``select_for_labeling.blind_audit``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modules.evaluation import iou

NAMES = ("Plate", "WithHelmet", "WithoutHelmet", "TripleRiding")
DECISION_CLASSES = {"WithoutHelmet", "TripleRiding"}
CONFIDENT = 0.6
NEAR = 0.1
DEFAULT_DECAY = 0.6


@dataclass
class Candidate:
    item_id: str
    source: str  # detector_pool | review_queue | withheld
    purpose: str  # training_relabel | review (evaluation images are never model-selected)
    signals: dict[str, float]
    pattern: str
    image: str | None = None
    dhash: int | None = None
    detail: dict = field(default_factory=dict)

    @property
    def priority(self) -> float:
        p = 1.0
        for s in self.signals.values():
            p *= 1.0 - max(0.0, min(1.0, s))
        return round(1.0 - p, 4)

    def as_dict(self) -> dict:
        return {"item_id": self.item_id, "source": self.source, "purpose": self.purpose,
                "priority": self.priority, "pattern": self.pattern,
                "signals": {k: round(v, 3) for k, v in sorted(self.signals.items())},
                "image": self.image, "detail": self.detail}


def image_signals(gt, preds, *, source_classes: set[str] | None,
                  op_thresholds: dict[str, float], weak_riders: bool = False) -> dict:
    """Signals for one image. ``gt``: ``[(cls_name, box)]``; ``preds``:
    ``[(cls_name, box, score)]``; ``source_classes``: the classes this image's
    source ever labels (None = unknown)."""
    sig: dict[str, float] = {}
    rider_preds = [p for p in preds if p[0] in ("WithHelmet", "WithoutHelmet")]
    for a in rider_preds:
        for b in rider_preds:
            if a[0] == "WithHelmet" and b[0] == "WithoutHelmet" and iou(a[1], b[1]) >= 0.5:
                lo = min(a[2], b[2])
                if lo >= op_thresholds.get("WithHelmet", 0.25):
                    sig["helmet_contradiction"] = max(sig.get("helmet_contradiction", 0.0),
                                                      lo / max(a[2], b[2]))
    for name, _box, score in preds:
        t = op_thresholds.get(name)
        if name in DECISION_CLASSES and t is not None and abs(score - t) <= NEAR:
            sig["near_threshold"] = max(sig.get("near_threshold", 0.0),
                                        1.0 - abs(score - t) / NEAR)
    for name, box, score in preds:
        if score < CONFIDENT:
            continue
        overlapping = [(g, iou(box, gb)) for g, gb in gt]
        if source_classes is not None and name not in source_classes:
            sig["label_gap"] = max(sig.get("label_gap", 0.0), score)
            continue
        if not any(v >= 0.5 for _, v in overlapping):
            sig["possible_missing_label"] = max(sig.get("possible_missing_label", 0.0), score)
        elif name in NAMES[1:] and not any(g == name and v >= 0.5 for g, v in overlapping) \
                and any(g in NAMES[1:] and v >= 0.5 for g, v in overlapping):
            sig["class_disagreement"] = max(sig.get("class_disagreement", 0.0), score)
    if sig and weak_riders:
        sig["known_weak_stratum"] = 0.3
    return sig


def pattern_of(signals: dict) -> str:
    """The failure pattern an item exemplifies: its strongest signal (the
    stratum bonus never defines a pattern)."""
    core = {k: v for k, v in signals.items() if k != "known_weak_stratum"}
    return max(core, key=lambda k: core[k]) if core else "none"


def review_signals(violation: dict, sidecar: dict | None) -> dict:
    """Signals for a pipeline decision awaiting review."""
    sig: dict[str, float] = {}
    conf = violation.get("confidence")
    if conf is not None:
        sig["low_score"] = max(0.0, min(1.0, (0.9 - float(conf)) / 0.5))
    if sidecar:
        votes = sidecar.get("plate_votes") or {}
        if votes.get("runner_up"):
            sig["plate_runner_up"] = 1.0
        margin = votes.get("margin")
        if margin is not None:
            sig["plate_vote_margin"] = max(0.0, 1.0 - float(margin))
        assoc = (sidecar.get("confidence") or {}).get("association")
        if assoc is not None:
            sig["weak_association"] = max(0.0, 1.0 - float(assoc))
    return {k: v for k, v in sig.items() if v > 0}


WITHHELD_SIGNAL = {"contested": 1.0, "low_agreement": 0.8, "low_margin": 0.8,
                   "too_few_supporting_reads": 0.5, "no_plate": 0.3}


def select(candidates: list[Candidate], budget: int, *, decay: float = DEFAULT_DECAY,
           near_duplicate_bits: int = 5) -> list[Candidate]:
    """Greedy, diversity-aware pick of ``budget`` items: each round takes the
    highest ``priority * decay ** picked_in_pattern``, skipping near-duplicates
    (dHash within ``near_duplicate_bits``) of an item already picked."""
    pool = [c for c in candidates if c.signals]
    picked: list[Candidate] = []
    per_pattern: dict[str, int] = {}
    picked_hashes: list[int] = []
    while pool and len(picked) < budget:
        best_i = max(range(len(pool)),
                     key=lambda i: pool[i].priority * decay ** per_pattern.get(pool[i].pattern, 0))
        c = pool.pop(best_i)
        if c.dhash is not None and any(bin(c.dhash ^ h).count("1") <= near_duplicate_bits
                                       for h in picked_hashes):
            continue
        picked.append(c)
        per_pattern[c.pattern] = per_pattern.get(c.pattern, 0) + 1
        if c.dhash is not None:
            picked_hashes.append(c.dhash)
    return picked


# What a label on each pattern would teach, and the measurement that tests it.
RESOLVES = {
    "label_gap": ("Plates/heads are unlabelled on the triple-riding source, so the detector "
                  "learned them as background there.",
                  "Plate recall on the dst source; share of triple riding that becomes fineable."),
    "possible_missing_label": ("Confident detections with no label are mostly real, unlabelled "
                               "objects on validation.",
                               "Measured precision after relabelling; label-noise rate per class."),
    "class_disagreement": ("A confident class that disagrees with the label is a model error or "
                           "a label error — the label decides which.",
                           "WithHelmet/WithoutHelmet confusion rate on corrected labels."),
    "helmet_contradiction": ("The model asserts both helmet and no helmet on one rider.",
                             "Contradiction rate; no-helmet precision at the operating point."),
    "near_threshold": ("Decisions at the operating threshold flip with small changes.",
                       "Precision/recall of the 0.375 no-helmet floor on the labelled slice."),
    "plate_runner_up": ("The plate vote had a competing reading.",
                        "OCR wrong-plate rate on contested votes (field OCR evaluation)."),
    "plate_vote_margin": ("The plate vote was narrow.",
                          "OCR wrong-plate rate by vote margin."),
    "low_score": ("The pipeline's own score was low.",
                  "Precision by score band — the input calibration needs."),
    "weak_association": ("The plate-to-rider link was weak.",
                         "Association accuracy on reviewed fines."),
    "withheld": ("A confirmed violation was withheld for want of a plate.",
                 "Share of withheld violations with a readable plate (OCR coverage cost)."),
}
