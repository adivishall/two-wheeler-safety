"""Unified violation confidence model.

The raw YOLO box confidence is **not** the truth about a violation: a model can
be confidently wrong (the README documents a helmeted rider called
``WithoutHelmet`` at 0.887), and a fine also depends on whether the violation
held over time, whether it was associated to the right vehicle, and whether the
plate was read reliably. This module combines those four signals into one
number so every recorded violation carries an auditable score.

These are **confidence scores in [0, 1]**, not calibrated probabilities — we
have not measured them against ground truth, so we deliberately avoid calling
them probabilities.

## Components

* **detection** — how strongly the model asserted the violation this frame
  (the box confidence; for overspeed, the margin over the speed limit).
* **temporal** — how well the violation persisted, ``min(1, streak / needed)``.
* **association** — how confidently the violation was tied to *this* vehicle's
  plate (horizontal overlap of plate under rider; 1.0 when association is not
  applicable, e.g. overspeed measured on the plate itself).
* **ocr** — the plate stabilizer's agreement score for the elected plate.

## Formula (configurable)

``final = sum(w_i * c_i) / sum(w_i)`` over the four components, each weight in
:class:`ConfidenceConfig`. A weighted arithmetic mean is chosen for
transparency (each component contributes proportionally to its weight); set a
weight to 0 to drop a component. All inputs and the result are clamped to
[0, 1].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


@dataclass(frozen=True)
class ConfidenceConfig:
    w_detection: float = 1.0
    w_temporal: float = 1.0
    w_association: float = 1.0
    w_ocr: float = 1.0


DEFAULT_CONFIDENCE_CONFIG = ConfidenceConfig()


@dataclass
class ViolationConfidence:
    violation: str
    detection: float
    temporal: float
    association: float
    ocr: float
    final: float
    supporting_frames: int

    def as_dict(self) -> dict:
        return asdict(self)


def temporal_confidence(streak: int, needed: int) -> float:
    """``min(1, streak / needed)`` -- 1.0 once the violation has held for the
    required number of consecutive frames."""
    if needed <= 0:
        return 1.0
    return _clamp01(streak / needed)


def compute_confidence(
    violation: str,
    *,
    detection: float,
    temporal: float,
    association: float,
    ocr: float,
    supporting_frames: int,
    config: ConfidenceConfig = DEFAULT_CONFIDENCE_CONFIG,
) -> ViolationConfidence:
    """Combine the four component scores into a final confidence score."""
    det = _clamp01(detection)
    tem = _clamp01(temporal)
    ass = _clamp01(association)
    ocr_c = _clamp01(ocr)

    weighted = (
        config.w_detection * det
        + config.w_temporal * tem
        + config.w_association * ass
        + config.w_ocr * ocr_c
    )
    total = (
        config.w_detection
        + config.w_temporal
        + config.w_association
        + config.w_ocr
    )
    final = _clamp01(weighted / total) if total else 0.0

    return ViolationConfidence(
        violation=violation,
        detection=round(det, 4),
        temporal=round(tem, 4),
        association=round(ass, 4),
        ocr=round(ocr_c, 4),
        final=round(final, 4),
        supporting_frames=supporting_frames,
    )
