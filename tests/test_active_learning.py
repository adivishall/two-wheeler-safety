"""Active-learning selection: signals fire on the configurations they name,
selection prefers new information over repeats, and evaluation images are
routed to evaluation relabelling, never training."""

import json

import pytest

from modules.active_learning import (
    Candidate,
    image_signals,
    pattern_of,
    review_signals,
    select,
)

OP = {"Plate": 0.25, "WithHelmet": 0.25, "WithoutHelmet": 0.375, "TripleRiding": 0.3}
RIDER = (100, 100, 180, 300)
ALL = {"Plate", "WithHelmet", "WithoutHelmet"}


@pytest.mark.parametrize("gt, preds, source_classes, expect", [
    ([("WithHelmet", RIDER)], [("WithHelmet", RIDER, 0.7), ("WithoutHelmet", RIDER, 0.6)],
     ALL, "helmet_contradiction"),
    ([("WithoutHelmet", RIDER)], [("WithoutHelmet", RIDER, 0.40)], ALL, "near_threshold"),
    ([], [("Plate", (10, 10, 60, 30), 0.8)], ALL, "possible_missing_label"),
    ([("WithHelmet", RIDER)], [("WithoutHelmet", RIDER, 0.9)], ALL, "class_disagreement"),
    ([("TripleRiding", RIDER)], [("Plate", (110, 300, 170, 320), 0.8)], {"TripleRiding"},
     "label_gap"),
])
def test_each_signal_fires_on_its_configuration(gt, preds, source_classes, expect):
    sig = image_signals(gt, preds, source_classes=source_classes, op_thresholds=OP)
    assert expect in sig and pattern_of(sig) == expect


def test_an_agreeing_confident_detection_is_not_informative():
    sig = image_signals([("WithoutHelmet", RIDER)], [("WithoutHelmet", RIDER, 0.95)],
                        source_classes=ALL, op_thresholds=OP)
    assert sig == {}


def test_the_weak_stratum_bonus_never_selects_an_image_on_its_own():
    agreeing = image_signals([("WithoutHelmet", RIDER)], [("WithoutHelmet", RIDER, 0.95)],
                             source_classes=ALL, op_thresholds=OP, weak_riders=True)
    assert agreeing == {}
    uncertain = image_signals([("WithoutHelmet", RIDER)], [("WithoutHelmet", RIDER, 0.40)],
                              source_classes=ALL, op_thresholds=OP, weak_riders=True)
    assert uncertain["known_weak_stratum"] > 0 and pattern_of(uncertain) == "near_threshold"


def test_priority_rewards_several_signals_but_stays_bounded():
    one = Candidate("a", "detector_pool", "training_relabel", {"x": 0.5}, "p")
    two = Candidate("b", "detector_pool", "training_relabel", {"x": 0.5, "y": 0.5}, "p")
    assert one.priority == 0.5 and two.priority == 0.75
    assert Candidate("c", "s", "p", {"x": 1.0, "y": 1.0}, "p").priority == 1.0


def test_selection_prefers_a_new_pattern_over_the_tenth_copy_of_one():
    same = [Candidate(f"s{i}", "detector_pool", "training_relabel", {"x": 0.9},
                      "training_relabel:x") for i in range(10)]
    other = Candidate("o", "detector_pool", "training_relabel", {"y": 0.6},
                      "training_relabel:y")
    picked = select(same + [other], 3, decay=0.6)
    assert "o" in [c.item_id for c in picked]
    greedy = sorted(same + [other], key=lambda c: -c.priority)[:3]
    assert "o" not in [c.item_id for c in greedy]  # a plain top-k would miss it


def test_selection_skips_near_duplicate_images():
    a = Candidate("a", "detector_pool", "training_relabel", {"x": 0.9}, "p:x", dhash=0b1010)
    b = Candidate("b", "detector_pool", "training_relabel", {"y": 0.8}, "p:y", dhash=0b1011)
    c = Candidate("c", "detector_pool", "training_relabel", {"z": 0.7}, "p:z",
                  dhash=(1 << 40) - 1)
    assert [x.item_id for x in select([a, b, c], 3)] == ["a", "c"]


def test_review_signals_read_the_plate_vote_and_association():
    sig = review_signals({"confidence": 0.55},
                         {"plate_votes": {"runner_up": "MH12AB1284", "margin": 0.35},
                          "confidence": {"association": 0.6}})
    assert sig["plate_runner_up"] == 1.0 and sig["plate_vote_margin"] == pytest.approx(0.65)
    assert sig["weak_association"] == pytest.approx(0.4) and sig["low_score"] > 0.5
    assert review_signals({"confidence": 0.99}, {"plate_votes": {"margin": 1.0},
                                                 "confidence": {"association": 1.0}}) == {}


def test_review_queue_uses_pipeline_output_only_and_picks_up_withheld(tmp_path):
    from modules.db import Database
    from select_for_labeling import review_candidates

    ev = tmp_path / "evidence"
    ev.mkdir()
    (ev / "m1.json").write_text(json.dumps(
        {"plate_votes": {"runner_up": "MH12AB1284", "margin": 0.3},
         "confidence": {"association": 0.9}}))
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.record_fine("MH12AB1234", "no_helmet", None, confidence=0.6,
                   evidence={"metadata_path": "m1.json"})
    db.record_fine("KA05CD1111", "no_helmet", None, confidence=0.4)  # no sidecar
    db.log_event("violation_withheld", record_type="session", record_id="run1",
                 actor="pipeline", metadata={"track_id": 7, "violation": "no_helmet",
                                             "reason": "contested"})
    cands, pending = review_candidates(str(tmp_path / "t.db"), str(ev))
    assert pending == 2
    ids = sorted(c.item_id for c in cands)
    assert ids == ["violation:1", "withheld:run1:7"]
    withheld = next(c for c in cands if c.source == "withheld")
    assert withheld.signals["withheld"] == 1.0 and withheld.purpose == "review"


def test_evaluation_images_are_routed_to_evaluation_relabelling(tmp_path):
    import cv2
    import numpy as np

    from select_for_labeling import detector_candidates

    img = tmp_path / "ds1_000001.jpg"
    cv2.imwrite(str(img), np.zeros((64, 64, 3), dtype=np.uint8))
    rows = [{"image": str(img), "w": 64, "h": 64, "gt": [],
             "preds": [[0, 1, 1, 30, 10, 0.9]]}]
    names = ["Plate", "WithHelmet", "WithoutHelmet", "TripleRiding"]
    out = detector_candidates(rows, "val", "evaluation_relabel", names,
                              {"ds1": {"Plate"}}, OP, lambda p: p)
    assert out[0].purpose == "evaluation_relabel"
    assert out[0].pattern == "evaluation_relabel:possible_missing_label"
