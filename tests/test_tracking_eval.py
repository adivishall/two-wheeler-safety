"""Crossing-vehicle tracking: the diagnosis, the fix, and its regressions.

The claim these tests protect is specific: the `crossing` weakness was an
*association* fault, not a tracking fault, and raising the body-merge thresholds
fixes it without breaking the merges the association layer exists to perform.
"""

from __future__ import annotations

import pytest

from modules.association import AssociationConfig, DetBox, merge_bodies
from modules.geometry import containment, iou
from modules.system_eval import (
    _flatten,
    builtin_scenarios,
    evaluate_association,
    evaluate_system,
    evaluate_tracking,
)
from modules.tracking_eval import (
    NEW_CONFIG,
    OLD_CONFIG,
    check_merge_behaviour,
    compare_configs,
    measure_tracking,
    scenarios,
    two_crossing,
)


def by_name(items):
    return {s.name: s for s in items}


# ---------------------------------------------------------------------------
# The diagnosis: the merge happens before the tracker runs
# ---------------------------------------------------------------------------

def test_crossing_bodies_actually_reach_the_old_merge_threshold():
    """The premise of the whole diagnosis: at the crossing frame the two
    vehicles' bodies overlap by more than the old 0.3 IoU, so the old config
    *had* to merge them."""
    sc = two_crossing(speed=20)
    overlaps = []
    for frame in sc.frames:
        boxes = [d.box for vf in frame for d in vf.dets if d.label != "Plate"]
        if len(boxes) == 2:
            overlaps.append(iou(boxes[0], boxes[1]))
    peak = max(overlaps)
    assert peak > OLD_CONFIG.body_merge_iou, "scenario no longer stresses the merge"
    assert peak < NEW_CONFIG.body_merge_iou, "new threshold would not fix it"


def test_old_config_collapses_two_vehicles_into_one_body():
    """The tracker is handed one anchor where there were two — which is why no
    motion model could have fixed this."""
    sc = two_crossing(speed=20)
    collapsed_old = collapsed_new = 0
    for frame in sc.frames:
        dets, _ = _flatten(frame)
        present = len(frame)
        if len(merge_bodies(dets, OLD_CONFIG)) < present:
            collapsed_old += 1
        if len(merge_bodies(dets, NEW_CONFIG)) < present:
            collapsed_new += 1
    assert collapsed_old > 0
    assert collapsed_new == 0


def test_merged_frames_and_id_switches_coincide():
    """Attribution: every ID switch under the old config happens in a scenario
    that also had a spurious merge. If switches occurred without merges, the
    tracker would be implicated too."""
    for sc in scenarios():
        m = measure_tracking(sc, OLD_CONFIG)
        if m.id_switches > 0:
            assert m.merged_frames > 0, f"{sc.name}: switch with no merge"


# ---------------------------------------------------------------------------
# Merge behaviour: the fix must not break legitimate merges
# ---------------------------------------------------------------------------

def test_helmet_inside_a_triple_body_still_merges():
    """The case the merge rule exists for: a sub-part box inside a rider body."""
    dets = [DetBox("TripleRiding", (100, 100, 200, 260), 0.9),
            DetBox("WithHelmet", (130, 105, 170, 150), 0.9)]
    bodies = merge_bodies(dets, NEW_CONFIG)
    assert len(bodies) == 1
    assert bodies[0].has_triple and bodies[0].has_helmet


def test_helmet_contradiction_still_merges():
    """Helmet and no-helmet asserted on one rider must stay one body, or the
    contradiction safeguard never fires and a wrong fine is issued."""
    dets = [DetBox("WithoutHelmet", (100, 100, 160, 200), 0.9),
            DetBox("WithHelmet", (102, 101, 158, 198), 0.85)]
    bodies = merge_bodies(dets, NEW_CONFIG)
    assert len(bodies) == 1
    assert bodies[0].ambiguous_helmet


def test_two_crossing_riders_no_longer_merge():
    dets = [DetBox("WithoutHelmet", (240, 100, 290, 200), 0.9),
            DetBox("TripleRiding", (260, 100, 310, 200), 0.9)]
    assert len(merge_bodies(dets, OLD_CONFIG)) == 1  # the old bug
    assert len(merge_bodies(dets, NEW_CONFIG)) == 2  # fixed


def test_merge_regression_suite_improves_and_never_regresses():
    old = check_merge_behaviour(OLD_CONFIG)
    new = check_merge_behaviour(NEW_CONFIG)
    assert new["_summary"]["correct"] > old["_summary"]["correct"]
    assert new["_summary"]["correct"] == new["_summary"]["total"]
    # No case that the old config got right may be broken by the new one.
    for name, row in old.items():
        if name.startswith("_"):
            continue
        if row["correct"]:
            assert new[name]["correct"], f"{name} regressed"


def test_thresholds_are_principled_not_fitted():
    """The new values must be the standard ones, not numbers tuned to squeeze
    past this particular scenario's 0.429 / 0.600 geometry."""
    assert NEW_CONFIG.body_merge_iou == 0.5  # canonical same-object IoU
    assert NEW_CONFIG.body_merge_containment == 0.8  # "mostly inside"
    # Sanity: the spurious case sits below both, with real margin.
    a, b = (240, 100, 290, 200), (260, 100, 310, 200)
    assert iou(a, b) < NEW_CONFIG.body_merge_iou - 0.05
    assert containment(a, b) < NEW_CONFIG.body_merge_containment - 0.1


# ---------------------------------------------------------------------------
# End-to-end effect
# ---------------------------------------------------------------------------

def test_new_config_removes_every_id_switch_in_the_suite():
    report = compare_configs()
    assert report["totals"]["old"]["id_switches"] > 0
    assert report["totals"]["new"]["id_switches"] == 0


def test_new_config_removes_the_wrong_fines():
    """The metric that actually matters: the old thresholds issued false
    positives and missed real violations on overlapping traffic."""
    t = compare_configs()["totals"]
    assert t["old"]["fine_fp"] > 0 and t["old"]["fine_fn"] > 0
    assert t["new"]["fine_fp"] == 0
    assert t["new"]["fine_fn"] == 0
    assert t["new"]["fine_tp"] > t["old"]["fine_tp"]


def test_association_accuracy_improves():
    t = compare_configs()["totals"]
    assert t["new"]["assoc_accuracy"] > t["old"]["assoc_accuracy"]
    assert t["new"]["assoc_accuracy"] > 0.95


def test_existing_scenarios_do_not_regress():
    """The shipped suite must be unchanged or better — a fix that trades one
    scenario for another is not a fix."""
    for sc in builtin_scenarios():
        old = evaluate_system(sc, config=OLD_CONFIG)
        new = evaluate_system(sc, config=NEW_CONFIG)
        assert new.true_positives >= old.true_positives, sc.name
        assert new.false_positives <= old.false_positives, sc.name
        assert new.false_negatives <= old.false_negatives, sc.name


def test_occlusion_recovery_keeps_the_same_identity():
    sc = by_name(scenarios())["occlusion_recovery"]
    m = measure_tracking(sc, NEW_CONFIG)
    assert m.id_switches == 0
    assert m.new_ids == 1  # one vehicle, one identity, despite a 3-frame gap
    assert m.recovered_after_gap >= 1


def test_three_way_crossing_keeps_three_identities():
    sc = by_name(scenarios())["three_crossing"]
    m = measure_tracking(sc, NEW_CONFIG)
    assert m.new_ids == 3
    assert m.id_switches == 0


def test_long_overlap_is_the_case_the_old_config_lost():
    sc = by_name(scenarios())["long_overlap"]
    old = evaluate_system(sc, config=OLD_CONFIG)
    new = evaluate_system(sc, config=NEW_CONFIG)
    assert old.false_positives > 0  # fined the wrong vehicle
    assert new.false_positives == 0
    assert new.true_positives == 2


def test_config_defaults_now_carry_the_fix():
    """The shipped default must be the measured-better one, or the A/B was
    academic."""
    shipped = AssociationConfig()
    assert shipped.body_merge_iou == NEW_CONFIG.body_merge_iou
    assert shipped.body_merge_containment == NEW_CONFIG.body_merge_containment


def test_old_config_is_pinned_to_history_not_to_the_defaults():
    """Once the new values became the defaults, an OLD_CONFIG built from
    AssociationConfig() would silently make the A/B new-vs-new."""
    assert OLD_CONFIG.body_merge_iou == 0.3
    assert OLD_CONFIG.body_merge_containment == 0.6


def test_crossing_scenario_in_the_shipped_suite_is_now_clean():
    sc = by_name(builtin_scenarios())["crossing"]
    assert evaluate_tracking(sc).id_switches == 0
    assert evaluate_association(sc).accuracy == pytest.approx(1.0)


def test_comparison_is_deterministic():
    assert compare_configs() == compare_configs()
