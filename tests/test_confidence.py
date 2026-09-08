"""Violation confidence engine tests (pure, deterministic)."""

from modules.confidence import (
    ConfidenceConfig,
    compute_confidence,
    temporal_confidence,
)


def test_temporal_confidence_ramps_and_saturates():
    assert temporal_confidence(0, 5) == 0.0
    assert temporal_confidence(3, 5) == 0.6
    assert temporal_confidence(5, 5) == 1.0
    assert temporal_confidence(9, 5) == 1.0  # saturates at 1
    assert temporal_confidence(1, 0) == 1.0  # no requirement -> full


def test_all_ones_gives_one():
    c = compute_confidence(
        "no_helmet", detection=1, temporal=1, association=1, ocr=1,
        supporting_frames=5,
    )
    assert c.final == 1.0


def test_equal_weights_is_the_mean():
    c = compute_confidence(
        "no_helmet", detection=0.8, temporal=1.0, association=0.6, ocr=0.4,
        supporting_frames=5,
    )
    assert c.final == round((0.8 + 1.0 + 0.6 + 0.4) / 4, 4)


def test_weights_are_configurable():
    # Zero every weight but detection -> final equals detection.
    cfg = ConfidenceConfig(w_detection=1, w_temporal=0, w_association=0, w_ocr=0)
    c = compute_confidence(
        "no_helmet", detection=0.73, temporal=0.1, association=0.1, ocr=0.1,
        supporting_frames=3, config=cfg,
    )
    assert c.final == 0.73


def test_inputs_are_clamped():
    c = compute_confidence(
        "triple_riding", detection=1.5, temporal=-0.2, association=2, ocr=0.5,
        supporting_frames=7,
    )
    assert 0.0 <= c.final <= 1.0
    assert c.detection == 1.0
    assert c.temporal == 0.0


def test_carries_type_and_supporting_frames():
    c = compute_confidence(
        "overspeed", detection=0.5, temporal=1.0, association=1.0, ocr=0.9,
        supporting_frames=6,
    )
    assert c.violation == "overspeed"
    assert c.supporting_frames == 6
    assert set(c.as_dict()) == {
        "violation", "detection", "temporal", "association", "ocr",
        "final", "supporting_frames",
    }
