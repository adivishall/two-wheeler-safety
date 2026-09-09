"""Evidence package tests (deterministic; small synthetic frames)."""

import json

import numpy as np

from modules.evidence import build_evidence, load_metadata


def _frame(w=200, h=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_build_evidence_writes_all_parts(tmp_path):
    pkg = build_evidence(
        str(tmp_path),
        plate="MH02DL4596",
        violation="no_helmet",
        original=_frame(),
        annotated=_frame(),
        plate_box=(10, 150, 90, 190),
        violation_box=(0, 0, 100, 100),
        frame_index=42,
        track_id=7,
        confidence={"final": 0.97},
    )
    for name in (pkg.original_path, pkg.annotated_path, pkg.plate_crop_path,
                 pkg.violation_crop_path, pkg.metadata_path):
        assert name is not None
        assert (tmp_path / name).exists()

    meta = load_metadata(str(tmp_path), pkg.metadata_path)
    assert meta["plate"] == "MH02DL4596"
    assert meta["violation"] == "no_helmet"
    assert meta["frame_index"] == 42
    assert meta["track_id"] == 7
    assert meta["confidence"] == {"final": 0.97}
    assert "T" in meta["timestamp"]  # ISO-8601
    assert set(meta["files"]) >= {"original", "annotated", "plate_crop", "violation_crop"}


def test_evidence_ids_are_unique_within_the_same_second(tmp_path):
    # Two fines for the same plate/violation/frame must not collide (the old
    # time.time()-second scheme could).
    a = build_evidence(str(tmp_path), plate="MH02DL4596", violation="no_helmet",
                       original=_frame(), annotated=_frame(), frame_index=1)
    b = build_evidence(str(tmp_path), plate="MH02DL4596", violation="no_helmet",
                       original=_frame(), annotated=_frame(), frame_index=1)
    assert a.evidence_id != b.evidence_id
    assert a.base != b.base
    assert a.metadata_path != b.metadata_path


def test_paths_are_basenames_for_portability(tmp_path):
    pkg = build_evidence(str(tmp_path), plate="KA05AB1", violation="triple_riding",
                         original=_frame(), annotated=_frame(), frame_index=3)
    assert "/" not in pkg.annotated_path
    assert pkg.primary_path == pkg.annotated_path  # annotated is primary


def test_missing_crops_are_skipped_not_crashed(tmp_path):
    # Degenerate/empty violation box -> no violation crop, but the rest is fine.
    pkg = build_evidence(str(tmp_path), plate="MH02DL4596", violation="no_helmet",
                         original=_frame(), annotated=_frame(),
                         violation_box=(50, 50, 50, 50), frame_index=1)
    assert pkg.violation_crop_path is None
    assert pkg.original_path is not None
    meta = json.loads((tmp_path / pkg.metadata_path).read_text())
    assert "violation_crop" not in meta["files"]


def test_unsafe_plate_is_sanitized_in_filenames(tmp_path):
    pkg = build_evidence(str(tmp_path), plate="../../etc/passwd", violation="no_helmet",
                         original=_frame(), annotated=_frame(), frame_index=1)
    assert "/" not in pkg.base and ".." not in pkg.base
