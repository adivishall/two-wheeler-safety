"""Failure-artifact writing: the crops that let a metric be inspected."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from modules.eval_artifacts import (
    CATEGORIES,
    ArtifactWriter,
    FailureCase,
    _crop_with_boxes,
    categorise,
)

cv2 = pytest.importorskip("cv2")


def image(h=400, w=400):
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------

def test_categorise_false_negative():
    assert categorise(None, 2, None) == "false_negatives"


def test_categorise_false_positive():
    assert categorise(2, None, 0.9) == "false_positives"


def test_categorise_class_confusion():
    assert categorise(1, 2, 0.9) == "class_confusions"


def test_categorise_low_confidence_only_when_correct():
    assert categorise(2, 2, 0.30) == "low_confidence"
    # A *wrong* low-confidence prediction is a confusion, not a low-conf case.
    assert categorise(1, 2, 0.30) == "class_confusions"


def test_categorise_correct_and_confident_is_not_saved():
    assert categorise(2, 2, 0.95) is None


def test_categorise_threshold_is_configurable():
    assert categorise(2, 2, 0.6) is None
    assert categorise(2, 2, 0.6, low_conf_threshold=0.8) == "low_confidence"


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def test_crop_includes_context_around_the_box():
    crop = _crop_with_boxes(image(), [((100, 100, 200, 200), (0, 0, 255))])
    # padded by 60% each side of a 100px box => ~220px
    assert crop.shape[0] > 200 and crop.shape[1] > 200


def test_crop_is_clamped_to_image_bounds():
    crop = _crop_with_boxes(image(100, 100), [((0, 0, 100, 100), (0, 0, 255))])
    assert crop.shape[0] <= 100 and crop.shape[1] <= 100


def test_tiny_box_is_grown_to_stay_legible():
    """A 6px plate crop is useless to look at; min_size must enlarge it."""
    crop = _crop_with_boxes(image(), [((200, 200, 206, 206), (0, 0, 255))])
    assert crop.shape[0] >= 90 and crop.shape[1] >= 90


def test_crop_spans_both_boxes_when_prediction_and_gt_differ():
    crop = _crop_with_boxes(image(), [
        ((50, 50, 100, 100), (0, 0, 255)),
        ((250, 250, 300, 300), (0, 200, 0)),
    ])
    assert crop.shape[0] >= 250 and crop.shape[1] >= 250


def test_box_entirely_outside_the_image_returns_none():
    assert _crop_with_boxes(image(), [((500, 500, 600, 600), (0, 0, 255))]) is None


def test_inverted_coordinates_are_normalised_not_rejected():
    """min/max on the corner lists means a box given as (x2,y2,x1,y1) still
    crops, rather than silently dropping a failure case."""
    crop = _crop_with_boxes(image(), [((200, 200, 100, 100), (0, 0, 255))])
    assert crop is not None
    assert crop.size > 0


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _case(**kw):
    base = dict(category="false_positives", source_image="a/b/img.jpg",
                predicted_class="WithHelmet", score=0.9, box=(10, 10, 60, 60))
    base.update(kw)
    return FailureCase(**base)


def test_add_writes_a_crop_and_records_the_case(tmp_path):
    w = ArtifactWriter(root=str(tmp_path), model_version="m@1.0.0")
    saved = w.add("false_positives", image(), _case())
    assert saved is not None
    assert os.path.exists(os.path.join(str(tmp_path), saved.artifact))
    assert saved.model_version == "m@1.0.0"
    assert len(w.cases) == 1


def test_cap_is_enforced_and_reported(tmp_path):
    w = ArtifactWriter(root=str(tmp_path), per_category=2)
    for _ in range(5):
        w.add("false_positives", image(), _case())
    assert w.summary()["counts"]["false_positives"] == 2
    assert "false_positives" in w.summary()["truncated"]
    assert w.room_for("false_positives") is False
    assert w.room_for("false_negatives") is True


def test_unknown_category_raises(tmp_path):
    w = ArtifactWriter(root=str(tmp_path))
    with pytest.raises(ValueError):
        w.add("nonsense", image(), _case())


def test_case_with_no_boxes_is_skipped(tmp_path):
    w = ArtifactWriter(root=str(tmp_path))
    assert w.add("false_positives", image(), _case(box=None, gt_box=None)) is None
    assert w.cases == []


def test_missing_image_is_skipped(tmp_path):
    w = ArtifactWriter(root=str(tmp_path))
    assert w.add("false_positives", None, _case()) is None


def test_index_is_written_with_summary_and_legend(tmp_path):
    w = ArtifactWriter(root=str(tmp_path), model_version="m@2")
    w.add("false_negatives", image(), _case(category="false_negatives",
                                            predicted_class=None,
                                            expected_class="Plate", score=None,
                                            box=None, gt_box=(10, 10, 60, 60)))
    path = w.write_index()
    data = json.loads(open(path).read())
    assert data["model_version"] == "m@2"
    assert data["legend"]["red_box"] == "model prediction"
    assert set(data["summary"]["counts"]) == set(CATEGORIES)
    assert data["cases"][0]["expected_class"] == "Plate"
    assert data["cases"][0]["gt_box"] == [10.0, 10.0, 60.0, 60.0]


def test_index_on_an_empty_writer_is_still_valid(tmp_path):
    w = ArtifactWriter(root=str(tmp_path))
    data = json.loads(open(w.write_index()).read())
    assert data["cases"] == []
    assert data["summary"]["truncated"] == []


def test_filenames_are_unique_within_a_category(tmp_path):
    """Two failures in the same source image must not overwrite each other."""
    w = ArtifactWriter(root=str(tmp_path))
    a = w.add("false_positives", image(), _case())
    b = w.add("false_positives", image(), _case(box=(70, 70, 120, 120)))
    assert a.artifact != b.artifact
    assert len(os.listdir(os.path.join(str(tmp_path), "false_positives"))) == 2
