"""End-to-end system-evaluation tests (Phases 6/7/9).

These drive the real tracker/association/state-machine/OCR-stabilizer logic over
deterministic synthetic scenarios and assert on the resulting system, tracking,
and association metrics. Model-free."""

from modules.association import DetBox
from modules.system_eval import (
    GTVehicle,
    Scenario,
    VehicleFrame,
    builtin_scenarios,
    evaluate_all,
    evaluate_association,
    evaluate_system,
    evaluate_tracking,
)


def _by_name():
    return {s.name: s for s in builtin_scenarios()}


# -- system-level fines -----------------------------------------------------

def test_single_no_helmet_produces_one_correct_fine():
    m = evaluate_system(_by_name()["single_no_helmet"])
    assert m.true_positives == 1
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.duplicates == 0


def test_clean_helmet_is_never_fined():
    m = evaluate_system(_by_name()["clean_helmet"])
    assert m.emitted_fines == 0
    assert m.true_positives == 0
    assert m.false_positives == 0


def test_late_plate_still_lands_the_fine():
    # Plate only becomes readable at frame 4, but the violation must still be
    # fined once it is (pipeline records on 'confirmed', not the transition).
    m = evaluate_system(_by_name()["late_plate"])
    assert m.true_positives == 1
    assert m.false_negatives == 0


def test_two_adjacent_bikes_do_not_swap_plates():
    m = evaluate_system(_by_name()["two_adjacent"])
    # bike 1 (no_helmet) fined with ITS plate; bike 2 (clean) not fined.
    assert m.true_positives == 1
    assert m.wrong_plate == 0
    assert m.wrong_vehicle == 0
    assert m.false_positives == 0


def test_three_bikes_only_middle_is_fined():
    m = evaluate_system(_by_name()["three_bikes"])
    assert m.true_positives == 1
    assert m.false_positives == 0
    assert m.false_negatives == 0


def test_occlusion_recovers_and_fines():
    m = evaluate_system(_by_name()["occlusion"])
    assert m.true_positives == 1
    assert m.false_negatives == 0


def test_crossing_bikes_are_still_attributed_correctly():
    sc = _by_name()["crossing"]
    m = evaluate_system(sc)
    # Both violations fined to the right vehicle+plate despite the crossing.
    assert m.true_positives == 2
    assert m.wrong_plate == 0
    assert m.false_negatives == 0


# -- tracking metrics -------------------------------------------------------

def test_tracking_clean_scenario_has_no_switches():
    t = evaluate_tracking(_by_name()["single_no_helmet"])
    assert t.id_switches == 0
    assert t.false_tracks == 0
    assert t.coverage == 1.0


def test_crossing_stresses_the_tracker():
    # The crossing scenario is the deliberate stress case: heavy overlap mid
    # cross causes measurable ID switches — this documents the tracker's limit.
    t = evaluate_tracking(_by_name()["crossing"])
    assert t.id_switches >= 1


def test_occlusion_is_survived_without_an_id_switch():
    # gt disappears for 2 frames then returns; with max_age=15 the track is kept
    # LOST and re-CONFIRMED to the SAME id, so there is no switch and every
    # frame the gt is actually present is covered. This documents that the
    # tracker recovers from a short occlusion rather than fragmenting.
    t = evaluate_tracking(_by_name()["occlusion"])
    assert t.id_switches == 0
    assert t.false_tracks == 0
    assert t.coverage == 1.0  # coverage is measured over frames the gt is present


# -- association metrics ----------------------------------------------------

def test_association_perfect_when_bikes_are_separated():
    a = evaluate_association(_by_name()["three_bikes"])
    assert a.wrong == 0
    assert a.accuracy == 1.0


def test_association_degrades_when_bikes_overlap():
    a = evaluate_association(_by_name()["crossing"])
    # overlap during the cross causes at least one ambiguous/wrong association.
    assert a.ambiguous_bodies >= 1 or a.wrong >= 1


# -- a false-positive guard: fabricate a wrong-plate scenario ---------------

def test_wrong_plate_is_detected_as_false_positive():
    # OCR consistently reads the wrong (but valid) plate -> the emitted fine's
    # plate won't match GT, so it must score as wrong_plate + false_positive,
    # never as a true positive.
    frames = []
    for f in range(8):
        body = (100 + 5 * f, 100, 150 + 5 * f, 200)
        plate = (105 + 5 * f, 205, 140 + 5 * f, 235)
        frames.append([VehicleFrame(1, [DetBox("WithoutHelmet", body, 0.9),
                                        DetBox("Plate", plate, 0.9)], "KA05MN6789")])
    sc = Scenario("wrong_plate", [GTVehicle(1, "MH12AB1234", frozenset({"no_helmet"}))], frames)
    m = evaluate_system(sc)
    assert m.true_positives == 0
    assert m.wrong_plate == 1
    assert m.false_positives == 1
    assert m.false_negatives == 1


# -- aggregate --------------------------------------------------------------

def test_evaluate_all_totals_are_consistent():
    out = evaluate_all()
    tot = out["totals"]
    # sum of per-scenario TP equals the total TP
    tp = sum(r["system"]["true_positives"] for r in out["scenarios"].values())
    assert tp == tot["true_positives"]
    assert tot["true_positives"] >= 8  # every violation scenario fined correctly
    assert tot["false_positives"] == 0
