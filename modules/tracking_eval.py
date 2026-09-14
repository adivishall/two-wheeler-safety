"""Targeted tracking / association scenarios for the crossing-vehicle problem.

`modules/system_eval.py` flagged `crossing` as the weak point: association
accuracy 0.85 and 4 ID switches. This module exists to answer *why*, and to
prove which layer is responsible before anything is changed.

**The diagnosis, measured frame by frame.** In the crossing scenario the two
vehicles' body boxes reach IoU 0.43 at frame 7. `AssociationConfig`'s
``body_merge_iou`` is 0.3, so ``merge_bodies`` unions two *different vehicles*
into a single body, and two vehicle instances collapse into one **before the
tracker is ever called**. One track then has no anchor to match, and when the
bikes separate the mapping flips back — 4 ID switches, none of them the
tracker's doing.

That rules out the obvious "fix": a Kalman filter (or any better motion model)
cannot recover an object that was merged away upstream. The tracker already
carries a constant-velocity prediction and matches correctly on every frame
where it is given two anchors. **The association layer is the bottleneck, not
the tracker.**

The geometry separates cleanly, and not by a threshold fitted to one scenario:

    case                              IoU     containment
    two crossing riders (spurious)    0.429   0.600
    helmet inside a triple body       0.113   1.000
    helmet/no-helmet contradiction    0.905   1.000

Neither number alone separates them. What does is the pair of *principled*
rules: merge when IoU > 0.5 (the standard "same object" criterion in detection)
**or** containment > 0.8 ("one box is mostly inside the other", which is what a
sub-part like a helmet within a rider actually looks like). Both legitimate
merges satisfy one of those; the spurious merge satisfies neither.

:func:`compare_configs` measures the old and new settings over the whole
scenario suite so the change is accepted on evidence across many cases rather
than on the one it was diagnosed from.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.association import AssociationConfig, DetBox, associate, merge_bodies
from modules.system_eval import (
    GTVehicle,
    Scenario,
    VehicleFrame,
    _flatten,
    _gt_boxes,
    _owner_by_overlap,
    evaluate_association,
    evaluate_system,
)
from modules.vehicle_tracker import VehicleTracker

# The two configurations under test. OLD_CONFIG is pinned to the *literal*
# historical values rather than to `AssociationConfig()`, so that once the new
# values become the defaults this A/B still compares what it says it compares
# instead of silently becoming new-vs-new.
OLD_CONFIG = AssociationConfig(
    body_merge_iou=0.3,
    body_merge_containment=0.6,
)
NEW_CONFIG = AssociationConfig(
    body_merge_iou=0.5,  # canonical "same object" IoU
    body_merge_containment=0.8,  # "mostly inside" rather than "somewhat overlapping"
)


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def _shift(box, dx: int, dy: int, f: int):
    x1, y1, x2, y2 = box
    return (x1 + dx * f, y1 + dy * f, x2 + dx * f, y2 + dy * f)


def _bike(gt_id: int, label: str, plate: str, x0: int, dx: int, f: int,
          *, y0: int = 100, w: int = 50, h: int = 100, plate_text=True):
    """One vehicle's detections at frame ``f``: a body box plus its plate."""
    body = _shift((x0, y0, x0 + w, y0 + h), dx, 0, f)
    plate_box = _shift((x0 + 5, y0 + h + 5, x0 + w - 10, y0 + h + 35), dx, 0, f)
    return VehicleFrame(
        gt_id,
        [DetBox(label, body, 0.9), DetBox("Plate", plate_box, 0.9)],
        plate if plate_text else None,
    )


def two_crossing(*, n: int = 10, speed: int = 20) -> Scenario:
    """Two bikes moving toward each other; they overlap mid-clip.

    ``speed`` controls how fast they pass through each other, which controls how
    many frames the overlap lasts — the variable that decides whether a merge is
    a blip or a sustained loss of identity.
    """
    frames = [[
        _bike(1, "WithoutHelmet", "MH12AB1234", 100, speed, f),
        _bike(2, "TripleRiding", "KA05MN6789", 400, -speed, f),
    ] for f in range(n)]
    return Scenario(
        f"two_crossing_speed{speed}",
        [GTVehicle(1, "MH12AB1234", frozenset({"no_helmet"})),
         GTVehicle(2, "KA05MN6789", frozenset({"triple_riding"}))],
        frames,
    )


def three_crossing(*, n: int = 12) -> Scenario:
    """Three bikes: two converging, one passing steadily through the middle.

    The hardest association case in the suite — at the crossing frames all three
    bodies are mutually close, so a merge can swallow more than two identities.
    """
    frames = [[
        _bike(1, "WithoutHelmet", "MH12AB1234", 60, 22, f),
        _bike(2, "WithHelmet", "KA05MN6789", 420, -22, f),
        _bike(3, "TripleRiding", "DL8CAF5031", 240, 0, f, y0=140),
    ] for f in range(n)]
    return Scenario(
        "three_crossing",
        [GTVehicle(1, "MH12AB1234", frozenset({"no_helmet"})),
         GTVehicle(2, "KA05MN6789", frozenset()),
         GTVehicle(3, "DL8CAF5031", frozenset({"triple_riding"}))],
        frames,
    )


def partial_overlap(*, n: int = 10, gap: int = 30) -> Scenario:
    """Two bikes riding side by side with a constant partial overlap.

    Never separates, so a wrong merge is never repaired by the geometry — the
    case where an over-eager merge costs a whole vehicle rather than a few
    frames of confusion.
    """
    frames = [[
        _bike(1, "WithoutHelmet", "MH12AB1234", 100, 5, f),
        _bike(2, "WithHelmet", "KA05MN6789", 100 + gap, 5, f),
    ] for f in range(n)]
    return Scenario(
        f"partial_overlap_gap{gap}",
        [GTVehicle(1, "MH12AB1234", frozenset({"no_helmet"})),
         GTVehicle(2, "KA05MN6789", frozenset())],
        frames,
    )


def long_overlap(*, n: int = 14, overlap_from: int = 4, overlap_to: int = 11) -> Scenario:
    """Two bikes that overlap heavily for seven consecutive frames.

    A brief merge can be absorbed by temporal voting; a long one is where a
    track is genuinely lost and a new identity is created.
    """
    frames = []
    for f in range(n):
        if overlap_from <= f < overlap_to:
            x2 = 118  # nearly on top of bike 1
        else:
            x2 = 100 + 60 + max(0, (f - overlap_to) * 25) + max(0, (overlap_from - f) * 25)
        frames.append([
            _bike(1, "WithoutHelmet", "MH12AB1234", 100, 0, f),
            _bike(2, "TripleRiding", "KA05MN6789", x2, 0, f),
        ])
    return Scenario(
        "long_overlap",
        [GTVehicle(1, "MH12AB1234", frozenset({"no_helmet"})),
         GTVehicle(2, "KA05MN6789", frozenset({"triple_riding"}))],
        frames,
    )


def occlusion_recovery(*, n: int = 14, hidden=(5, 6, 7)) -> Scenario:
    """One bike disappears entirely for three frames and returns.

    Tests whether the same identity is recovered (``max_age`` holding the track
    LOST) rather than a new one being created.
    """
    frames = []
    for f in range(n):
        row = [_bike(1, "WithoutHelmet", "MH12AB1234", 100, 8, f)]
        if f in hidden:
            row = []
        frames.append(row)
    return Scenario(
        "occlusion_recovery",
        [GTVehicle(1, "MH12AB1234", frozenset({"no_helmet"}))],
        frames,
    )


def late_plate_while_crossing(*, n: int = 12) -> Scenario:
    """Two bikes cross, and the violating bike's plate only becomes readable
    after the crossing — so identity must survive the merge to be fined."""
    frames = []
    for f in range(n):
        frames.append([
            _bike(1, "WithoutHelmet", "MH12AB1234", 100, 20, f, plate_text=f >= 8),
            _bike(2, "WithHelmet", "KA05MN6789", 400, -20, f),
        ])
    return Scenario(
        "late_plate_while_crossing",
        [GTVehicle(1, "MH12AB1234", frozenset({"no_helmet"})),
         GTVehicle(2, "KA05MN6789", frozenset())],
        frames,
    )


def scenarios() -> list:
    """The full targeted suite."""
    return [
        two_crossing(speed=20),
        two_crossing(speed=10),  # slower pass = longer overlap
        three_crossing(),
        partial_overlap(gap=30),
        partial_overlap(gap=20),  # tighter: harder
        long_overlap(),
        occlusion_recovery(),
        late_plate_while_crossing(),
    ]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class TrackQuality:
    """Per-scenario tracking outcome under one association config."""

    scenario: str
    frames: int = 0
    gt_frames: int = 0
    covered_gt_frames: int = 0
    id_switches: int = 0
    fragmentations: int = 0
    new_ids: int = 0  # distinct track ids created; ideally == number of vehicles
    merged_frames: int = 0  # frames where instances < GT vehicles present
    gaps: int = 0  # times a vehicle reappeared after >=1 frame away
    recovered_after_gap: int = 0  # of those, how many kept the SAME track id

    @property
    def coverage(self) -> float:
        return self.covered_gt_frames / self.gt_frames if self.gt_frames else 1.0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["coverage"] = round(self.coverage, 4)
        return d


def measure_tracking(scenario: Scenario, config: AssociationConfig) -> TrackQuality:
    """Run the real tracker over a scenario and count identity failures.

    ``merged_frames`` is the diagnostic that attributes blame: it counts frames
    where ``associate()`` returned fewer vehicle instances than there were
    ground-truth vehicles on screen. Those are frames where the tracker was
    handed an impossible input, so any ID switch on them belongs to the
    association layer rather than to tracking.
    """
    m = TrackQuality(scenario=scenario.name)
    tracker = VehicleTracker(config=config)
    last_id: dict[int, int] = {}
    covered_prev: dict[int, bool] = {}
    # Last frame index on which each vehicle was covered by a track. A gap is
    # counted whenever a vehicle is covered again after being uncovered for one
    # or more frames — whether it was missed by the tracker or genuinely absent
    # from the scene (a full occlusion). Counting only tracker misses would
    # report zero recoveries for the occlusion scenario, which is precisely the
    # case the metric exists to measure.
    last_covered_frame: dict[int, int] = {}
    ids_seen: set[int] = set()

    for f, frame in enumerate(scenario.frames):
        m.frames += 1
        dets, _ = _flatten(frame)
        gt_boxes = _gt_boxes(frame)
        present = set(gt_boxes)
        m.gt_frames += len(present)

        result = associate(dets, config)
        if present and len(result.bodies) < len(present):
            m.merged_frames += 1

        tracks = tracker.update(dets, f)
        covered: dict[int, int] = {}
        for t in tracks:
            ids_seen.add(t.track_id)
            g = _owner_by_overlap(t.box, gt_boxes)
            if g is not None:
                covered[g] = t.track_id

        for g in present:
            if g in covered:
                m.covered_gt_frames += 1
                tid = covered[g]
                if g in last_id and last_id[g] != tid:
                    m.id_switches += 1
                if covered_prev.get(g) is False:
                    m.fragmentations += 1
                if g in last_covered_frame and f - last_covered_frame[g] > 1:
                    m.gaps += 1
                    if last_id.get(g) == tid:
                        m.recovered_after_gap += 1
                last_id[g] = tid
                covered_prev[g] = True
                last_covered_frame[g] = f
            elif g in covered_prev:
                covered_prev[g] = False

    m.new_ids = len(ids_seen)
    return m


def compare_configs(
    old: AssociationConfig = OLD_CONFIG,
    new: AssociationConfig = NEW_CONFIG,
    scenario_list=None,
) -> dict:
    """Measure both configs over the whole suite and total the differences.

    Reports association accuracy, ID switches, fragmentation, spurious merges
    and end-to-end fine correctness for each, so a change cannot be accepted on
    a tracking metric while quietly breaking the fines.
    """
    scenario_list = scenario_list or scenarios()
    out: dict = {"scenarios": {}, "totals": {}}
    totals = {
        k: {"id_switches": 0, "merged_frames": 0, "fragmentations": 0,
            "assoc_correct": 0, "assoc_expected": 0, "assoc_wrong": 0,
            "extra_ids": 0, "fine_tp": 0, "fine_fp": 0, "fine_fn": 0}
        for k in ("old", "new")
    }

    for sc in scenario_list:
        row: dict = {}
        for key, cfg in (("old", old), ("new", new)):
            tq = measure_tracking(sc, cfg)
            assoc = evaluate_association(sc, cfg)
            sysm = evaluate_system(sc, config=cfg)
            row[key] = {
                "tracking": tq.as_dict(),
                "association": assoc.as_dict(),
                "system": sysm.as_dict(),
            }
            t = totals[key]
            t["id_switches"] += tq.id_switches
            t["merged_frames"] += tq.merged_frames
            t["fragmentations"] += tq.fragmentations
            t["extra_ids"] += max(0, tq.new_ids - len(sc.vehicles))
            t["assoc_correct"] += assoc.correct
            t["assoc_expected"] += assoc.expected_pairs
            t["assoc_wrong"] += assoc.wrong
            t["fine_tp"] += sysm.true_positives
            t["fine_fp"] += sysm.false_positives
            t["fine_fn"] += sysm.false_negatives
        out["scenarios"][sc.name] = row

    for key, t in totals.items():
        t["assoc_accuracy"] = (
            round(t["assoc_correct"] / t["assoc_expected"], 4)
            if t["assoc_expected"] else 1.0
        )
    out["totals"] = totals
    out["config"] = {
        "old": {"body_merge_iou": old.body_merge_iou,
                "body_merge_containment": old.body_merge_containment},
        "new": {"body_merge_iou": new.body_merge_iou,
                "body_merge_containment": new.body_merge_containment},
    }
    return out


def merge_regression_cases() -> list:
    """Merges that MUST keep happening, as (name, box_a, box_b, should_merge).

    Raising the merge thresholds is only safe if the legitimate merges survive.
    These are the two the association layer exists for: a sub-part box inside a
    rider body, and the helmet/no-helmet contradiction on one rider.
    """
    return [
        ("helmet_inside_triple", (100, 100, 200, 260), (130, 105, 170, 150), True),
        ("contradiction_same_rider", (100, 100, 160, 200), (102, 101, 158, 198), True),
        ("plate_sized_part_in_body", (100, 100, 180, 240), (120, 200, 160, 230), True),
        ("two_riders_crossing", (240, 100, 290, 200), (260, 100, 310, 200), False),
        ("two_riders_side_by_side", (100, 100, 150, 200), (130, 100, 180, 200), False),
        ("far_apart", (100, 100, 150, 200), (400, 100, 450, 200), False),
    ]


def check_merge_behaviour(config: AssociationConfig) -> dict:
    """Run every regression case through ``merge_bodies`` under one config."""
    results = {}
    for name, a, b, should_merge in merge_regression_cases():
        bodies = merge_bodies(
            [DetBox("WithoutHelmet", a, 0.9), DetBox("WithHelmet", b, 0.9)], config
        )
        merged = len(bodies) == 1
        results[name] = {
            "merged": merged,
            "expected": should_merge,
            "correct": merged == should_merge,
        }
    results["_summary"] = {
        "correct": sum(1 for k, v in results.items()
                       if not k.startswith("_") and v["correct"]),
        "total": sum(1 for k in results if not k.startswith("_")),
    }
    return results
