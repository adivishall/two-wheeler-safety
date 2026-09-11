"""Evidence package tests (deterministic; small synthetic frames)."""

import json

import numpy as np

from modules.evidence import (
    build_evidence,
    load_metadata,
    sha256_file,
    verify_evidence,
)


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


def test_model_version_is_recorded_in_metadata(tmp_path):
    pkg = build_evidence(str(tmp_path), plate="MH02DL4596", violation="no_helmet",
                         original=_frame(), annotated=_frame(), frame_index=1,
                         model_version="traffic-4class@1.0.0")
    meta = load_metadata(str(tmp_path), pkg.metadata_path)
    assert meta["model_version"] == "traffic-4class@1.0.0"


def test_model_version_defaults_to_none(tmp_path):
    pkg = build_evidence(str(tmp_path), plate="MH02DL4596", violation="no_helmet",
                         original=_frame(), annotated=_frame(), frame_index=1)
    meta = load_metadata(str(tmp_path), pkg.metadata_path)
    assert meta["model_version"] is None


# -- integrity (Phase 13) ---------------------------------------------------

def test_integrity_fields_and_hashes_are_recorded(tmp_path):
    pkg = build_evidence(
        str(tmp_path), plate="MH02DL4596", violation="no_helmet",
        original=_frame(), annotated=_frame(), frame_index=5,
        model_version="traffic-4class@1.0.0", pipeline_version="1.0.0",
        source_id="clip.mp4", config_snapshot={"conf_threshold": 0.25},
    )
    meta = load_metadata(str(tmp_path), pkg.metadata_path)
    assert meta["pipeline_version"] == "1.0.0"
    assert meta["source_id"] == "clip.mp4"
    assert meta["config_snapshot"] == {"conf_threshold": 0.25}
    # a hash for every written artifact, matching the file on disk
    assert set(meta["hashes"]) == set(meta["files"])
    for key, name in meta["files"].items():
        assert meta["hashes"][key] == sha256_file(str(tmp_path / name))


def test_verify_evidence_passes_on_untampered_package(tmp_path):
    pkg = build_evidence(str(tmp_path), plate="KA05AB1", violation="triple_riding",
                         original=_frame(), annotated=_frame(), frame_index=1)
    result = verify_evidence(str(tmp_path), pkg.metadata_path)
    assert result.ok
    assert result.checked >= 2
    assert result.mismatched == []


def test_verify_evidence_detects_a_modified_artifact(tmp_path):
    pkg = build_evidence(str(tmp_path), plate="KA05AB1", violation="no_helmet",
                         original=_frame(), annotated=_frame(), frame_index=1)
    # tamper: overwrite the annotated image with different pixels
    import cv2
    tampered = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / pkg.annotated_path), tampered)
    result = verify_evidence(str(tmp_path), pkg.metadata_path)
    assert not result.ok
    assert any(key == "annotated" for key, _e, _a in result.mismatched)


def test_verify_evidence_detects_a_deleted_artifact(tmp_path):
    pkg = build_evidence(str(tmp_path), plate="KA05AB1", violation="no_helmet",
                         original=_frame(), annotated=_frame(), frame_index=1)
    (tmp_path / pkg.original_path).unlink()
    result = verify_evidence(str(tmp_path), pkg.metadata_path)
    assert not result.ok
    assert any(actual == "MISSING" for _k, _e, actual in result.mismatched)
