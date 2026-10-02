"""Hungarian vs greedy plate-to-rider matching, on the shipped cost.

The design claim (docs/DECISIONS.md #2) is that a global one-to-one assignment
attributes plates better than taking the cheapest rider/plate pair first. The
built-in synthetic scenarios cannot show it: on every frame of
``modules/system_eval.builtin_scenarios`` and ``modules/tracking_eval.scenarios``
the two matchers produce the same pairs. This script measures the difference on
random multi-rider layouts with known ownership.

Both matchers see the identical cost matrix from
``modules.association._plate_body_cost`` (default weights and gate). Riders are
used directly as bodies (no ``merge_bodies``), so only the matching step
differs. Layouts are synthetic: what is measured is the matching rule under
this generator, not accuracy on real footage.

    python3 evaluate_association.py               # eval/results/association_baseline.*
    python3 evaluate_association.py --out /tmp/x  # anywhere else
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

from modules.association import _BIG, DEFAULT_CONFIG, _plate_body_cost, gated_min_cost_matching

GENERATOR = (
    "2-5 riders per frame, width 40-90 px, height 1.8x width, top-left uniform in "
    "x 0-600, y 50-300; each rider's own plate present with p=0.85, width 0.6x the "
    "rider, offset x -15..+25 px and y -20..+15 px from the rider's bottom edge. "
    "Frames with fewer than 2 plates are skipped."
)


def greedy_matching(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Cheapest remaining (row, col) pair first, skipping forbidden pairs."""
    pairs, used_r, used_c = [], set(), set()
    cells = sorted((cost[r][c], r, c) for r in range(len(cost)) for c in range(len(cost[0])))
    for value, r, c in cells:
        if value >= _BIG or r in used_r or c in used_c:
            continue
        pairs.append((r, c))
        used_r.add(r)
        used_c.add(c)
    return sorted(pairs)


def random_frame(rng: random.Random):
    """Riders, plates, and the true rider index of each plate."""
    riders, plates, owner = [], [], {}
    for i in range(rng.randint(2, 5)):
        x, y, w = rng.randint(0, 600), rng.randint(50, 300), rng.randint(40, 90)
        h = int(w * 1.8)
        riders.append((x, y, x + w, y + h))
        if rng.random() < 0.85:
            px, py = x + rng.randint(-15, 25), y + h + rng.randint(-20, 15)
            owner[len(plates)] = i
            plates.append((px, py, px + int(w * 0.6), py + 14))
    return riders, plates, owner


def compare(seeds=range(5), frames_per_seed: int = 2000) -> dict:
    t = dict(frames=0, differ=0, greedy_wrong_frames=0, hungarian_wrong_frames=0,
             greedy_wrong_pairs=0, hungarian_wrong_pairs=0, greedy_pairs=0, hungarian_pairs=0)
    for seed in seeds:
        rng = random.Random(seed)
        for _ in range(frames_per_seed):
            riders, plates, owner = random_frame(rng)
            if len(plates) < 2:
                continue
            cost = []
            for r in riders:
                row = []
                for p in plates:
                    c, gated = _plate_body_cost(p, r, DEFAULT_CONFIG)
                    row.append(_BIG if gated else c)
                cost.append(row)
            g = greedy_matching(cost)
            h = sorted(gated_min_cost_matching(cost, _BIG))
            gw = sum(1 for r, c in g if owner.get(c) != r)
            hw = sum(1 for r, c in h if owner.get(c) != r)
            t["frames"] += 1
            t["differ"] += g != h
            t["greedy_wrong_frames"] += gw > 0
            t["hungarian_wrong_frames"] += hw > 0
            t["greedy_wrong_pairs"] += gw
            t["hungarian_wrong_pairs"] += hw
            t["greedy_pairs"] += len(g)
            t["hungarian_pairs"] += len(h)
    f = t["frames"] or 1
    return {
        "generator": GENERATOR,
        "seeds": list(seeds),
        "frames_per_seed": frames_per_seed,
        "counts": t,
        "rates": {
            "frames_where_matchers_differ": round(t["differ"] / f, 4),
            "greedy_frames_with_a_wrong_pair": round(t["greedy_wrong_frames"] / f, 4),
            "hungarian_frames_with_a_wrong_pair": round(t["hungarian_wrong_frames"] / f, 4),
            "greedy_wrong_pair_rate": round(t["greedy_wrong_pairs"] / max(1, t["greedy_pairs"]), 4),
            "hungarian_wrong_pair_rate": round(
                t["hungarian_wrong_pairs"] / max(1, t["hungarian_pairs"]), 4),
        },
        "caveat": "Synthetic layouts and a hand-weighted cost; not a field measurement.",
    }


def builtin_suite_disagreements() -> dict:
    """Frames of the shipped synthetic suites where the two matchers differ,
    using the full association path (``merge_bodies`` first, as ``associate``
    does)."""
    from modules.association import merge_bodies
    from modules.system_eval import _flatten, builtin_scenarios
    from modules.tracking_eval import scenarios as crossing_scenarios

    frames = multi = differ = 0
    for scenario in builtin_scenarios() + crossing_scenarios():
        for frame in scenario.frames:
            dets, _owners = _flatten(frame)
            bodies = merge_bodies(dets)
            plates = [d.box for d in dets if d.label == "Plate"]
            if not bodies or not plates:
                continue
            cost = []
            for b in bodies:
                row = []
                for p in plates:
                    c, gated = _plate_body_cost(p, b.box, DEFAULT_CONFIG)
                    row.append(_BIG if gated else c)
                cost.append(row)
            frames += 1
            multi += len(bodies) > 1 and len(plates) > 1
            differ += greedy_matching(cost) != sorted(gated_min_cost_matching(cost, _BIG))
    return {"frames": frames, "frames_with_2plus_bodies_and_plates": multi,
            "frames_where_matchers_differ": differ}


def render_md(result: dict) -> str:
    c, r = result["counts"], result["rates"]
    return "\n".join([
        "# Plate-to-rider matching: Hungarian vs greedy",
        "",
        f"- Generated by `python3 evaluate_association.py` (seeds {result['seeds']}, "
        f"{result['frames_per_seed']} frames each).",
        f"- Layouts: {result['generator']}",
        "- Same cost matrix for both (`modules/association.py:_plate_body_cost`, default config).",
        f"- **{result['caveat']}**",
        "",
        "| | greedy | Hungarian |",
        "|---|---:|---:|",
        f"| frames with at least one wrong pair | {c['greedy_wrong_frames']} "
        f"({r['greedy_frames_with_a_wrong_pair']:.1%}) | {c['hungarian_wrong_frames']} "
        f"({r['hungarian_frames_with_a_wrong_pair']:.1%}) |",
        f"| wrong pairs / pairs emitted | {c['greedy_wrong_pairs']} / {c['greedy_pairs']} "
        f"({r['greedy_wrong_pair_rate']:.2%}) | {c['hungarian_wrong_pairs']} / "
        f"{c['hungarian_pairs']} ({r['hungarian_wrong_pair_rate']:.2%}) |",
        "",
        f"Frames scored: {c['frames']}; the two matchers differ on {c['differ']} "
        f"({r['frames_where_matchers_differ']:.1%}).",
        "",
        "On the shipped synthetic suites (`builtin_scenarios` + crossing scenarios) "
        f"the matchers differ on {result['builtin_suites']['frames_where_matchers_differ']} "
        f"of {result['builtin_suites']['frames_with_2plus_bodies_and_plates']} frames "
        "with two or more riders and plates, so those suites do not exercise this choice.",
        "",
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="eval/results", help="output directory")
    ap.add_argument("--frames", type=int, default=2000, help="frames per seed (default 2000)")
    args = ap.parse_args(argv)
    result = compare(frames_per_seed=args.frames)
    result["builtin_suites"] = builtin_suite_disagreements()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "association_baseline.json"), "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    md = render_md(result)
    with open(os.path.join(args.out, "association_baseline.md"), "w") as fh:
        fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
