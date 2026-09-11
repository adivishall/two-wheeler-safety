"""Structured evidence packages.

The old pipeline saved a single cropped JPEG named ``{plate}_{violation}_{sec}``.
Two problems: seconds-resolution timestamps collide when two fines land in the
same second, and a lone crop is weak evidence.

:func:`build_evidence` writes a full package per confirmed violation:

* the **original** frame (unannotated),
* the **annotated** frame (boxes + labels),
* a **plate** crop and a **violation** crop,
* a **JSON sidecar** with the timestamp, frame index, track id, plate,
  violation, confidence breakdown, supporting-frame count, and any speed /
  calibration data.

Filenames are made unique with a random token (not ``time.time()`` alone), and
all paths stored are basenames so an evidence directory stays portable/servable
via ``/evidence/<name>``.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import cv2


def _safe_token(text: str, fallback: str = "UNKNOWN") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text or "")
    return cleaned or fallback


def _safe_crop(img, box):
    """Return a clamped crop, or None if the box is empty/degenerate."""
    if img is None or box is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


@dataclass
class EvidencePackage:
    evidence_id: str
    base: str
    original_path: str | None
    annotated_path: str | None
    plate_crop_path: str | None
    violation_crop_path: str | None
    metadata_path: str
    files: dict = field(default_factory=dict)

    @property
    def primary_path(self) -> str | None:
        """The image a fine record points at (annotated frame if we have it)."""
        return self.annotated_path or self.violation_crop_path or self.original_path


def build_evidence(
    evidence_root: str,
    *,
    plate: str,
    violation: str,
    original,
    annotated=None,
    plate_box=None,
    violation_box=None,
    frame_index: int = 0,
    track_id: int | None = None,
    confidence: dict | None = None,
    speed: dict | None = None,
    timestamp: datetime | None = None,
    model_version: str | None = None,
) -> EvidencePackage:
    """Write an evidence package and return the paths (as basenames)."""
    os.makedirs(evidence_root, exist_ok=True)
    ts = timestamp or datetime.now(timezone.utc)
    evidence_id = uuid.uuid4().hex[:12]
    base = f"{_safe_token(plate)}_{violation}_f{frame_index}_{evidence_id}"

    def _write(suffix, img):
        if img is None or getattr(img, "size", 0) == 0:
            return None
        name = f"{base}_{suffix}.jpg"
        cv2.imwrite(os.path.join(evidence_root, name), img)
        return name

    original_name = _write("original", original)
    annotated_name = _write("annotated", annotated) if annotated is not None else None
    plate_name = _write("plate", _safe_crop(original, plate_box))
    violation_name = _write("violation", _safe_crop(original, violation_box))

    files = {
        k: v
        for k, v in {
            "original": original_name,
            "annotated": annotated_name,
            "plate_crop": plate_name,
            "violation_crop": violation_name,
        }.items()
        if v is not None
    }

    metadata = {
        "evidence_id": evidence_id,
        "plate": plate,
        "violation": violation,
        "track_id": track_id,
        "frame_index": frame_index,
        "timestamp": ts.isoformat(),
        "model_version": model_version,
        "confidence": confidence,
        "speed": speed,
        "files": files,
    }
    metadata_name = f"{base}.json"
    with open(os.path.join(evidence_root, metadata_name), "w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    return EvidencePackage(
        evidence_id=evidence_id,
        base=base,
        original_path=original_name,
        annotated_path=annotated_name,
        plate_crop_path=plate_name,
        violation_crop_path=violation_name,
        metadata_path=metadata_name,
        files=files,
    )


def load_metadata(evidence_root: str, metadata_name: str) -> dict:
    with open(os.path.join(evidence_root, metadata_name)) as fh:
        return json.load(fh)


# Re-export for callers that want to serialize a confidence dataclass cheaply.
def to_plain(obj) -> dict:
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else dict(obj)
