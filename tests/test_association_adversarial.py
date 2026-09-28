"""Adversarial checks of the association layer.

Two kinds:

* **Algorithmic**: `hungarian` is compared against brute force on hundreds of
  random cost matrices (square and rectangular, with ties and forbidden
  pairs), and gated matching is checked never to give up a feasible pair to
  avoid a forbidden one.
* **Geometric**: the rider<->plate layouts that break naive association —
  a neighbour's plate occluded, inaccurate boxes, overlapping riders,
  perspective scale differences, bikes one behind the other, a plate
  straddling two riders. Each asserts the pairing a human would make, or —
  where the geometry is genuinely ambiguous — only the invariants that must
  hold regardless (one-to-one, deterministic).
"""

import itertools
import random

import pytest

from modules.association import (
    _BIG,
    DetBox,
    associate,
    gated_min_cost_matching,
    hungarian,
)


def _brute_force(cost):
    n, m = len(cost), len(cost[0])
    best = None
    if n <= m:
        for cols in itertools.permutations(range(m), n):
            total = sum(cost[r][c] for r, c in enumerate(cols))
            best = total if best is None else min(best, total)
    else:
        for rows in itertools.permutations(range(n), m):
            total = sum(cost[r][c] for c, r in enumerate(rows))
            best = total if best is None else min(best, total)
    return best


def test_hungarian_is_optimal_on_300_random_matrices():
    """One test over 300 random cases (not 300 tests: the count stays honest)."""
    for seed in range(300):
        rng = random.Random(seed)
        n, m = rng.randint(1, 6), rng.randint(1, 6)
        ties = rng.random() < 0.3
        cost = [[float(rng.randint(0, 3)) if ties else rng.random() * 10 for _ in range(m)]
                for _ in range(n)]
        assign = hungarian(cost)
        used = [c for c in assign if c != -1]
        assert len(used) == len(set(used)) == min(n, m), seed  # one-to-one, maximal
        total = sum(cost[r][c] for r, c in enumerate(assign) if c != -1)
        assert total == pytest.approx(_brute_force(cost)), seed


def _max_feasible(cost):
    n, m = len(cost), len(cost[0])
    best = 0
    for k in range(min(n, m), 0, -1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                if all(cost[r][c] < _BIG for r, c in zip(rows, cols)):
                    return k
    return best


def test_gating_never_sacrifices_a_feasible_pair():
    """Forbidden pairs are _BIG. Over 200 random cases the matching must still
    pair as many feasible (row, col) as any assignment could — a solver that
    dodged one forbidden pair by dropping two feasible ones would lose a real
    association."""
    for seed in range(200):
        rng = random.Random(1000 + seed)
        n, m = rng.randint(1, 5), rng.randint(1, 5)
        cost = [[_BIG if rng.random() < 0.45 else rng.random() for _ in range(m)]
                for _ in range(n)]
        pairs = gated_min_cost_matching(cost, gate=_BIG)
        assert all(cost[r][c] < _BIG for r, c in pairs), seed
        assert len(pairs) == _max_feasible(cost), seed


# -- geometry -----------------------------------------------------------------

def _rider(x, y=100, w=60, h=110, label="WithoutHelmet", conf=0.9):
    return DetBox(label, (x, y, x + w, y + h), conf)


def _plate(x, y, w=40, h=14):
    return DetBox("Plate", (x, y, x + w, y + h), 0.9)


def _pairs(dets):
    """{body_x1: plate_x1 or None} for readability."""
    r = associate(dets)
    out = {}
    for bi, pi in r.pairs:
        if bi is None:
            continue
        out[r.bodies[bi].box[0]] = r.plates[pi][0] if pi is not None else None
    return out


def test_occluded_neighbour_plate_is_not_stolen():
    # Two adjacent bikes; B's plate is hidden. A keeps its own plate, B gets none.
    dets = [_rider(100), _rider(175), _plate(110, 212)]
    assert _pairs(dets) == {100: 110, 175: None}


def test_jittered_boxes_still_pair_correctly():
    rng = random.Random(3)
    for _ in range(50):
        j = lambda: rng.randint(-8, 8)  # noqa: E731 - ~15% of rider width
        dets = [_rider(100 + j()), _rider(220 + j()),
                _plate(110 + j(), 212 + j()), _plate(230 + j(), 212 + j())]
        pairs = _pairs(dets)
        (a, pa), (b, pb) = sorted(pairs.items())
        assert pa is not None and pb is not None
        assert abs(pa - a) < abs(pa - b) and abs(pb - b) < abs(pb - a)


def test_overlapping_riders_below_the_merge_threshold_pair_correctly():
    # Riders overlap (IoU ~0.2) but are separate bodies; each plate sits under
    # its own rider.
    dets = [_rider(100), _rider(135), _plate(105, 212), _plate(150, 212)]
    r = associate(dets)
    assert len(r.bodies) == 2
    assert _pairs(dets) == {100: 105, 135: 150}


def test_perspective_far_small_bike_and_near_large_bike():
    near_rider = DetBox("WithoutHelmet", (300, 300, 420, 600), 0.9)
    near_plate = DetBox("Plate", (330, 580, 390, 620), 0.9)
    far_rider = DetBox("WithHelmet", (100, 50, 130, 110), 0.9)
    far_plate = DetBox("Plate", (105, 108, 125, 116), 0.9)
    assert _pairs([near_rider, near_plate, far_rider, far_plate]) == {300: 330, 100: 105}


def test_bikes_one_behind_the_other_in_the_same_lane():
    # Rear view, same column: the far bike (higher in the image) has its plate
    # just above the near rider's head. The vertical penalty (a plate is below
    # its rider) must keep it with the far bike.
    far_rider = _rider(200, y=40, w=50, h=90)
    far_plate = _plate(207, 128, w=36, h=12)
    near_rider = _rider(195, y=150, w=70, h=130)
    near_plate = _plate(210, 278, w=40, h=14)
    assert _pairs([far_rider, far_plate, near_rider, near_plate]) == {200: 207, 195: 210}


def test_plate_straddling_two_riders_is_given_to_at_most_one():
    # Genuinely ambiguous: the invariants must hold whatever is chosen.
    dets = [_rider(100), _rider(170), _plate(145, 212)]
    r = associate(dets)
    plates_used = [pi for bi, pi in r.pairs if bi is not None and pi is not None]
    assert len(plates_used) <= 1
    assert r.pairs == associate(list(dets)).pairs  # deterministic


def test_a_far_away_plate_is_not_forced_onto_a_rider():
    dets = [_rider(100), _plate(600, 400)]
    assert _pairs(dets) == {100: None}
