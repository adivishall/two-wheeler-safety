"""Frame-level association tests.

These use synthetic boxes (no model, fully deterministic) and specifically
target the cases where the old ``nearest_plate_id`` heuristic mis-assigns a
violation to the wrong vehicle.
"""

import main  # legacy nearest-centroid helper, kept for comparison
from modules.association import (
    DetBox,
    associate,
    gated_min_cost_matching,
    hungarian,
    merge_bodies,
)

# --- Hungarian assignment ---------------------------------------------------

def test_hungarian_simple_diagonal():
    assert hungarian([[1, 2], [2, 1]]) == [0, 1]


def test_hungarian_beats_greedy_nearest():
    # Greedy "take the smallest cost first" grabs (0,0)=1, forcing row1 onto
    # col1=5 for a total of 6. The optimal one-to-one is (0,1)+(1,0)=5.
    cost = [[1, 3], [2, 5]]
    assign = hungarian(cost)
    assert assign == [1, 0]
    total = sum(cost[r][c] for r, c in enumerate(assign))
    assert total == 5


def test_hungarian_rectangular_more_rows_than_cols():
    # 3 rows, 1 col: only the cheapest row can be matched.
    assert hungarian([[1], [2], [3]]) == [0, -1, -1]


def test_hungarian_rectangular_more_cols_than_rows():
    assert hungarian([[3, 1, 2]]) == [1]


def test_gated_matching_drops_forbidden_pairs():
    cost = [[1e9, 0.2], [0.3, 1e9]]
    assert gated_min_cost_matching(cost, gate=1e9) == [(0, 1), (1, 0)]


# --- body merging -----------------------------------------------------------

def test_contradicting_helmet_pair_merges_into_one_ambiguous_body():
    # WithHelmet and WithoutHelmet on nearly the same box = the model
    # contradicting itself on one rider. Must become ONE ambiguous body, not
    # two vehicles, and must NOT count as a no-helmet violation.
    dets = [
        DetBox("WithoutHelmet", (240, 274, 880, 1242), conf=0.60),
        DetBox("WithHelmet", (229, 263, 890, 1235), conf=0.39),
    ]
    bodies = merge_bodies(dets)
    assert len(bodies) == 1
    assert bodies[0].ambiguous_helmet is True
    assert bodies[0].no_helmet_violation is False


def test_helmet_inside_triple_body_merges():
    # A helmet head detected inside a large triple-riding body box is the same
    # vehicle (containment), so they merge rather than competing for a plate.
    dets = [
        DetBox("TripleRiding", (0, 0, 300, 400), conf=0.8),
        DetBox("WithoutHelmet", (40, 20, 110, 90), conf=0.7),
    ]
    bodies = merge_bodies(dets)
    assert len(bodies) == 1
    assert bodies[0].has_triple and bodies[0].has_no_helmet


def test_two_distant_riders_stay_separate():
    dets = [
        DetBox("WithoutHelmet", (0, 0, 100, 100)),
        DetBox("WithHelmet", (500, 0, 600, 100)),
    ]
    assert len(merge_bodies(dets)) == 2


# --- the headline case: global assignment beats nearest-plate ---------------

def _pair_plate_index_for_body(result, predicate):
    """Return the plate index associated with the (single) body matching
    ``predicate``, or None."""
    body_idx = next(
        i for i, b in enumerate(result.bodies) if predicate(b)
    )
    for b_i, p_i in result.pairs:
        if b_i == body_idx:
            return p_i
    return None


def test_nearest_plate_mis_assigns_but_association_is_correct():
    # Two bikes. Bike A's rider is high in frame with its plate far below;
    # bike B is nearer the camera so *B's* plate sits close to A's rider.
    a_rider = (0, 0, 100, 100)       # A: no helmet
    a_plate = (20, 300, 80, 340)     # A's plate, far below A's rider
    b_rider = (200, 0, 300, 100)     # B: helmet
    b_plate = (60, 90, 140, 130)     # B's plate, right under A's rider

    # Old heuristic: A's no-helmet box is nearest to B's plate -> WRONG.
    tracked_plates = {0: a_plate, 1: b_plate}
    assert main.nearest_plate_id(a_rider, tracked_plates) == 1  # picks B (wrong)

    # New association: global one-to-one forces A's rider onto A's plate,
    # because B's rider can only plausibly take B's plate.
    result = associate([
        DetBox("WithoutHelmet", a_rider, conf=0.9),
        DetBox("Plate", a_plate, conf=0.9),
        DetBox("WithHelmet", b_rider, conf=0.9),
        DetBox("Plate", b_plate, conf=0.9),
    ])
    plates = result.plates
    a_plate_idx = plates.index(a_plate)
    assigned = _pair_plate_index_for_body(result, lambda b: b.no_helmet_violation)
    assert assigned == a_plate_idx  # A's no-helmet body -> A's plate (correct)


def test_one_violation_never_assigned_to_two_vehicles():
    # Two riders, one shared-looking region, two plates: each plate goes to at
    # most one body and vice versa.
    result = associate([
        DetBox("WithoutHelmet", (0, 0, 100, 100)),
        DetBox("WithoutHelmet", (120, 0, 220, 100)),
        DetBox("Plate", (20, 150, 80, 190)),
        DetBox("Plate", (140, 150, 200, 190)),
    ])
    plate_assignments = [p for _, p in result.pairs if p is not None]
    assert len(plate_assignments) == len(set(plate_assignments))  # no plate twice
    body_assignments = [b for b, p in result.pairs if b is not None and p is not None]
    assert len(body_assignments) == len(set(body_assignments))  # no body twice


def test_plate_with_no_plausible_body_stays_unmatched():
    # A lone plate far from any rider becomes a plate-only vehicle instance.
    result = associate([
        DetBox("WithoutHelmet", (0, 0, 100, 100)),
        DetBox("Plate", (20, 120, 80, 160)),          # belongs to the rider
        DetBox("Plate", (2000, 2000, 2060, 2040)),    # orphan plate
    ])
    orphan_idx = result.plates.index((2000, 2000, 2060, 2040))
    assert (None, orphan_idx) in result.pairs
