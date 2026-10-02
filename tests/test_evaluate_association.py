"""The Hungarian-vs-greedy comparison behind docs/DESIGN_DECISIONS.md #2."""

from evaluate_association import builtin_suite_disagreements, compare, greedy_matching
from modules.association import _BIG, gated_min_cost_matching


def test_greedy_takes_the_cheapest_pair_first_and_hungarian_does_not():
    # Greedy grabs (0,0)=1 and is forced into (1,1)=10: total 11.
    # Hungarian pays (0,1)+(1,0) = 2+2 = 4.
    cost = [[1.0, 2.0], [2.0, 10.0]]
    assert greedy_matching(cost) == [(0, 0), (1, 1)]
    assert sorted(gated_min_cost_matching(cost, _BIG)) == [(0, 1), (1, 0)]


def test_greedy_skips_forbidden_pairs():
    assert greedy_matching([[_BIG, 1.0], [_BIG, _BIG]]) == [(0, 1)]


def test_hungarian_attributes_fewer_plates_wrongly_than_greedy():
    result = compare(seeds=range(2), frames_per_seed=500)
    c = result["counts"]
    assert c["frames"] > 500
    assert c["hungarian_wrong_pairs"] < c["greedy_wrong_pairs"]
    assert compare(seeds=range(2), frames_per_seed=500) == result  # deterministic


def test_the_shipped_synthetic_suites_do_not_separate_the_two_matchers():
    # Documents why the comparison needs its own layouts: on every built-in
    # scenario frame greedy and Hungarian agree.
    out = builtin_suite_disagreements()
    assert out["frames_with_2plus_bodies_and_plates"] > 50
    assert out["frames_where_matchers_differ"] == 0
