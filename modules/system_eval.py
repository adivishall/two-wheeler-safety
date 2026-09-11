"""End-to-end system evaluation on deterministic synthetic scenarios.

Raw YOLO mAP answers "how good is the detector?". This module answers a
different, harder question the spec insists on separating:

    How often does the COMPLETE pipeline — track -> associate -> OCR-vote ->
    temporal confirm -> confidence -> violation -> attribution — turn detections
    into a correct, correctly-attributed fine?

To isolate *pipeline logic* from *detector quality*, the input is synthetic:
each scenario is a list of frames, each frame a list of :class:`VehicleFrame`
(the detections belonging to one ground-truth vehicle that frame, plus the OCR
text of its plate). The detections are fed through the **real** runtime code —
`VehicleTracker`, `associate`, `PlateStabilizer`, the helmet/triple state
machines, and `compute_confidence` — so what is measured is exactly the logic
that runs in production, not a re-implementation.

Three evaluations, deliberately distinct from YOLO metrics:

* :func:`evaluate_tracking` (Phase 6) — ID switches, fragmentation, false
  tracks, and GT coverage.
* :func:`evaluate_association` (Phase 7) — correct / wrong / missed / ambiguous
  rider<->plate associations.
* :func:`evaluate_system` (Phase 9) — system TP / FP / FN plus wrong-vehicle and
  wrong-plate attribution and duplicate fines.

Everything is deterministic (fixed synthetic inputs, ordered iteration), so the
numbers are reproducible and unit-testable without the model stack.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.association import DetBox, associate
from modules.geometry import Box, intersection_area, union_box
from modules.plate_recognizer import PlateStabilizer
from modules.vehicle_tracker import VehicleTracker
from modules.violation_state import (
    HelmetConfig,
    HelmetStateMachine,
    TripleConfig,
    TripleRidingStateMachine,
)

BODY_LABELS = ("WithHelmet", "WithoutHelmet", "TripleRiding")


# ---------------------------------------------------------------------------
# Scenario representation
# ---------------------------------------------------------------------------

@dataclass
class GTVehicle:
    """A ground-truth vehicle: its plate and the violations it should be fined
    for (subset of {"no_helmet", "triple_riding"})."""

    gt_id: int
    plate: str
    violations: frozenset = frozenset()


@dataclass
class VehicleFrame:
    """One GT vehicle's detections in one frame, plus the OCR text of its plate
    that frame (None if the plate was unreadable / not detected)."""

    gt_id: int
    dets: list  # list[DetBox] — the body/plate boxes emitted for this vehicle
    plate_text: str | None = None


@dataclass
class Scenario:
    name: str
    vehicles: list  # list[GTVehicle]
    frames: list  # list[list[VehicleFrame]]

    @property
    def gt_by_id(self) -> dict:
        return {v.gt_id: v for v in self.vehicles}


# ---------------------------------------------------------------------------
# Owner-mapping helpers (map a merged body / a track back to a GT vehicle)
# ---------------------------------------------------------------------------

def _flatten(frame) -> tuple[list, list]:
    """Flatten a frame into (dets, owners) where owners[i] is the gt_id that
    produced dets[i], in a stable order."""
    dets: list = []
    owners: list = []
    for vf in frame:
        for d in vf.dets:
            dets.append(d)
            owners.append(vf.gt_id)
    return dets, owners


def _gt_boxes(frame, *, bodies_only: bool = False) -> dict:
    """Union box per gt_id present this frame (optionally bodies only)."""
    boxes: dict = {}
    for vf in frame:
        for d in vf.dets:
            if bodies_only and d.label not in BODY_LABELS:
                continue
            boxes[vf.gt_id] = (
                d.box if vf.gt_id not in boxes else union_box(boxes[vf.gt_id], d.box)
            )
    return boxes


def _owner_by_overlap(box: Box, gt_boxes: dict) -> int | None:
    """gt_id whose box overlaps ``box`` most (by intersection area), or None."""
    best_gt = None
    best_overlap = 0.0
    for gt_id, gt_box in sorted(gt_boxes.items()):
        overlap = intersection_area(box, gt_box)
        if overlap > best_overlap:
            best_overlap = overlap
            best_gt = gt_id
    return best_gt


# ---------------------------------------------------------------------------
# Phase 7 — association quality
# ---------------------------------------------------------------------------

@dataclass
class AssociationMetrics:
    frames: int = 0
    expected_pairs: int = 0  # GT vehicles with both a body and a plate present
    correct: int = 0
    wrong: int = 0  # body of one gt paired with plate of another
    missed: int = 0  # expected pair not formed
    ambiguous_bodies: int = 0  # a merged body spanning >1 gt

    @property
    def accuracy(self) -> float:
        return self.correct / self.expected_pairs if self.expected_pairs else 1.0

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["accuracy"] = round(self.accuracy, 4)
        return d


def evaluate_association(scenario: Scenario) -> AssociationMetrics:
    m = AssociationMetrics()
    for frame in scenario.frames:
        m.frames += 1
        dets, owners = _flatten(frame)
        # plate owner list, in the same order associate() builds its plates list.
        plate_owners = [owners[i] for i, d in enumerate(dets) if d.label == "Plate"]
        body_gt_boxes = _gt_boxes(frame, bodies_only=True)
        plate_gt_boxes = {
            vf.gt_id: d.box
            for vf in frame for d in vf.dets if d.label == "Plate"
        }

        # An expected pair exists for each gt that has BOTH a body and a plate.
        expected = set(body_gt_boxes) & set(plate_gt_boxes)
        m.expected_pairs += len(expected)

        result = associate(dets)
        matched_gt: set = set()
        for body_i, plate_i in result.pairs:
            if body_i is None or plate_i is None:
                continue
            body_box = result.bodies[body_i].box
            body_owner = _owner_by_overlap(body_box, body_gt_boxes)
            plate_owner = plate_owners[plate_i]
            # ambiguous: does this merged body overlap >1 gt's body region?
            overlapped = [
                gt for gt, gb in body_gt_boxes.items()
                if intersection_area(body_box, gb) > 0
            ]
            if len(overlapped) > 1:
                m.ambiguous_bodies += 1
            if body_owner == plate_owner and body_owner is not None:
                m.correct += 1
                matched_gt.add(body_owner)
            else:
                m.wrong += 1
        m.missed += len(expected - matched_gt)
    return m


# ---------------------------------------------------------------------------
# Phase 6 — tracking quality
# ---------------------------------------------------------------------------

@dataclass
class TrackingMetrics:
    frames: int = 0
    gt_frames: int = 0  # (gt, frame) cells where the gt was present
    matched_gt_frames: int = 0  # of those, covered by a track
    id_switches: int = 0
    fragmentations: int = 0
    false_tracks: int = 0

    @property
    def coverage(self) -> float:
        return self.matched_gt_frames / self.gt_frames if self.gt_frames else 1.0

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["coverage"] = round(self.coverage, 4)
        return d


def evaluate_tracking(scenario: Scenario) -> TrackingMetrics:
    m = TrackingMetrics()
    tracker = VehicleTracker()
    # gt_id -> assigned track_id last frame it was matched (for ID switches)
    last_track_for_gt: dict = {}
    # gt_id -> was it covered by a track the previous frame (for fragmentation)
    prev_covered: dict = {}
    track_gt_votes: dict = {}  # track_id -> {gt_id: count} for false-track calc

    for frame_idx, frame in enumerate(scenario.frames):
        m.frames += 1
        dets, _owners = _flatten(frame)
        gt_boxes = _gt_boxes(frame)
        present = set(gt_boxes)
        m.gt_frames += len(present)

        tracks = tracker.update(dets, frame_idx)

        # Map each track seen this frame to a gt.
        covered: dict = {}  # gt_id -> track_id
        for tr in tracks:
            gt = _owner_by_overlap(tr.box, gt_boxes)
            track_gt_votes.setdefault(tr.track_id, {})
            track_gt_votes[tr.track_id][gt] = (
                track_gt_votes[tr.track_id].get(gt, 0) + 1
            )
            if gt is not None:
                covered[gt] = tr.track_id  # last writer wins; scenarios avoid ties

        for gt in present:
            if gt in covered:
                m.matched_gt_frames += 1
                tid = covered[gt]
                if gt in last_track_for_gt and last_track_for_gt[gt] != tid:
                    m.id_switches += 1
                last_track_for_gt[gt] = tid
                # fragmentation: gt reappears after a gap
                if prev_covered.get(gt) is False:
                    m.fragmentations += 1
                prev_covered[gt] = True
            elif gt in prev_covered:
                prev_covered[gt] = False

    # A track whose majority vote is "no gt" (None) is a false/spurious track.
    for tid, votes in track_gt_votes.items():
        majority = max(votes, key=lambda g: (votes[g], g if g is not None else -1))
        if majority is None:
            m.false_tracks += 1
    return m


# ---------------------------------------------------------------------------
# Phase 9 — whole-system fines
# ---------------------------------------------------------------------------

@dataclass
class SystemMetrics:
    scenario: str = ""
    expected_fines: int = 0
    emitted_fines: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    wrong_vehicle: int = 0  # violation attributed to a gt that doesn't have it
    wrong_plate: int = 0    # right gt+violation, wrong plate string
    duplicates: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 1.0

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["precision"] = round(self.precision, 4)
        d["recall"] = round(self.recall, 4)
        return d


# body-label -> violation key mapping
def _violation_signals(body) -> dict:
    return {
        "has_helmet": body.has_helmet,
        "has_no_helmet": body.has_no_helmet,
        "no_helmet_conf": body.no_helmet_conf,
        "has_triple": body.has_triple,
        "triple_conf": body.triple_conf,
        "ambiguous": body.ambiguous_helmet,
    }


def evaluate_system(
    scenario: Scenario,
    *,
    confirm_window: int = 3,
) -> SystemMetrics:
    """Drive the full pipeline logic and score emitted fines against GT."""
    m = SystemMetrics(scenario=scenario.name)
    gt_by_id = scenario.gt_by_id
    for v in scenario.vehicles:
        m.expected_fines += len(v.violations)

    tracker = VehicleTracker()
    helmet_sms: dict = {}
    triple_sms: dict = {}
    stabilizers: dict = {}
    track_gt_votes: dict = {}
    emitted: set = set()  # (track_id, violation) already fined
    fines: list = []  # (track_id, violation, plate_string)

    for frame_idx, frame in enumerate(scenario.frames):
        dets, _ = _flatten(frame)
        gt_boxes = _gt_boxes(frame)
        # gt_id -> its plate OCR text this frame
        plate_texts = {vf.gt_id: vf.plate_text for vf in frame}

        tracks = tracker.update(dets, frame_idx)
        for tr in tracks:
            tid = tr.track_id
            gt = _owner_by_overlap(tr.box, gt_boxes)
            track_gt_votes.setdefault(tid, {})
            track_gt_votes[tid][gt] = track_gt_votes[tid].get(gt, 0) + 1

            hsm = helmet_sms.setdefault(
                tid, HelmetStateMachine(HelmetConfig(confirm_window=confirm_window)))
            tsm = triple_sms.setdefault(
                tid, TripleRidingStateMachine(TripleConfig(confirm_window=confirm_window)))
            stab = stabilizers.setdefault(tid, PlateStabilizer())

            if gt is not None and plate_texts.get(gt):
                stab.add(plate_texts[gt], conf=0.9)

            if tr.body is not None:
                sig = _violation_signals(tr.body)
                hsm.update(
                    has_helmet=sig["has_helmet"], has_no_helmet=sig["has_no_helmet"],
                    no_helmet_conf=sig["no_helmet_conf"], ambiguous=sig["ambiguous"],
                    frame_idx=frame_idx,
                )
                tsm.update(
                    has_triple=sig["has_triple"], conf=sig["triple_conf"],
                    frame_idx=frame_idx,
                )

            stable = stab.result().stable
            if stable:
                if hsm.confirmed and (tid, "no_helmet") not in emitted:
                    emitted.add((tid, "no_helmet"))
                    fines.append((tid, "no_helmet", stable))
                if tsm.confirmed and (tid, "triple_riding") not in emitted:
                    emitted.add((tid, "triple_riding"))
                    fines.append((tid, "triple_riding", stable))

    # Score fines against GT.
    m.emitted_fines = len(fines)
    matched_expected: set = set()  # (gt_id, violation)
    seen: set = set()
    for tid, violation, plate in fines:
        if (tid, violation) in seen:
            m.duplicates += 1
        seen.add((tid, violation))
        votes = track_gt_votes.get(tid, {})
        gt_id = max(votes, key=lambda g: (votes[g], g if g is not None else -1)) \
            if votes else None
        gt = gt_by_id.get(gt_id)
        if gt is None or violation not in gt.violations:
            # attributed to a vehicle that shouldn't have this violation
            m.false_positives += 1
            if gt is not None:
                m.wrong_vehicle += 1
            continue
        if plate != gt.plate:
            m.wrong_plate += 1
            m.false_positives += 1
            continue
        m.true_positives += 1
        matched_expected.add((gt_id, violation))

    expected_set = {
        (v.gt_id, viol) for v in scenario.vehicles for viol in v.violations
    }
    m.false_negatives = len(expected_set - matched_expected)
    return m


# ---------------------------------------------------------------------------
# Built-in scenarios (the multi-bike cases the spec asks for)
# ---------------------------------------------------------------------------

def _linear(box0: Box, dx: int, dy: int, frame: int) -> Box:
    x1, y1, x2, y2 = box0
    return (x1 + dx * frame, y1 + dy * frame, x2 + dx * frame, y2 + dy * frame)


def _single_vehicle_scenario(
    name, gt_id, plate, body_label, violations, *, n=8, box0=(100, 100, 160, 200),
    dx=6, plate_from=0, plate_offset=(10, 205, 50, 235),
) -> Scenario:
    frames = []
    for f in range(n):
        body_box = _linear(box0, dx, 0, f)
        dets = [DetBox(body_label, body_box, 0.9)]
        text = None
        if f >= plate_from:
            px1, py1, px2, py2 = _linear(plate_offset, dx, 0, f)
            dets.append(DetBox("Plate", (px1, py1, px2, py2), 0.9))
            text = plate
        frames.append([VehicleFrame(gt_id, dets, text)])
    return Scenario(name, [GTVehicle(gt_id, plate, frozenset(violations))], frames)


def _two_vehicle_frame(f, a, b) -> list:
    return [a(f), b(f)]


def builtin_scenarios() -> list:
    """The deterministic multi-bike scenarios the spec asks for."""
    scenarios: list = []

    # 1. no-helmet rider, plate readable throughout.
    scenarios.append(_single_vehicle_scenario(
        "single_no_helmet", 1, "MH12AB1234", "WithoutHelmet", {"no_helmet"}))

    # 2. triple riding.
    scenarios.append(_single_vehicle_scenario(
        "single_triple", 1, "KA05MN6789", "TripleRiding", {"triple_riding"}))

    # 3. plate appears late (frame 4) — fine must still land once readable.
    scenarios.append(_single_vehicle_scenario(
        "late_plate", 1, "MH12AB1234", "WithoutHelmet", {"no_helmet"}, plate_from=4))

    # 4. clean helmeted rider — must NOT be fined.
    scenarios.append(_single_vehicle_scenario(
        "clean_helmet", 1, "MH12AB1234", "WithHelmet", set()))

    # 5. two adjacent bikes, one no-helmet one clean; plates must not swap.
    def bikeA(f):
        body = _linear((100, 100, 150, 200), 5, 0, f)
        plate = _linear((105, 205, 140, 235), 5, 0, f)
        return VehicleFrame(1, [DetBox("WithoutHelmet", body, 0.9),
                                DetBox("Plate", plate, 0.9)], "MH12AB1234")

    def bikeB(f):
        body = _linear((300, 100, 350, 200), 5, 0, f)
        plate = _linear((305, 205, 340, 235), 5, 0, f)
        return VehicleFrame(2, [DetBox("WithHelmet", body, 0.9),
                                DetBox("Plate", plate, 0.9)], "KA05MN6789")

    scenarios.append(Scenario(
        "two_adjacent",
        [GTVehicle(1, "MH12AB1234", frozenset({"no_helmet"})),
         GTVehicle(2, "KA05MN6789", frozenset())],
        [_two_vehicle_frame(f, bikeA, bikeB) for f in range(8)],
    ))

    # 6. crossing bikes: A moves right, B moves left, they cross mid-scene.
    def crossA(f):
        body = _linear((100, 100, 150, 200), 20, 0, f)
        plate = _linear((105, 205, 140, 235), 20, 0, f)
        return VehicleFrame(1, [DetBox("WithoutHelmet", body, 0.9),
                                DetBox("Plate", plate, 0.9)], "MH12AB1234")

    def crossB(f):
        body = _linear((400, 100, 450, 200), -20, 0, f)
        plate = _linear((405, 205, 440, 235), -20, 0, f)
        return VehicleFrame(2, [DetBox("TripleRiding", body, 0.9),
                                DetBox("Plate", plate, 0.9)], "KA05MN6789")

    scenarios.append(Scenario(
        "crossing",
        [GTVehicle(1, "MH12AB1234", frozenset({"no_helmet"})),
         GTVehicle(2, "KA05MN6789", frozenset({"triple_riding"}))],
        [_two_vehicle_frame(f, crossA, crossB) for f in range(10)],
    ))

    # 7. three bikes side by side, middle one no-helmet.
    def three(f):
        out = []
        specs = [
            (1, 100, "WithHelmet", "MH12AB1234", set()),
            (2, 260, "WithoutHelmet", "KA05MN6789", {"no_helmet"}),
            (3, 420, "WithHelmet", "DL8CAF5031", set()),
        ]
        for gt_id, x, label, plate, _v in specs:
            body = _linear((x, 100, x + 50, 200), 4, 0, f)
            pl = _linear((x + 5, 205, x + 40, 235), 4, 0, f)
            out.append(VehicleFrame(gt_id, [DetBox(label, body, 0.9),
                                            DetBox("Plate", pl, 0.9)], plate))
        return out

    scenarios.append(Scenario(
        "three_bikes",
        [GTVehicle(1, "MH12AB1234", frozenset()),
         GTVehicle(2, "KA05MN6789", frozenset({"no_helmet"})),
         GTVehicle(3, "DL8CAF5031", frozenset())],
        [three(f) for f in range(8)],
    ))

    # 8. occlusion: a no-helmet rider vanishes for 2 frames then returns.
    def occ(f):
        if f in (4, 5):  # occluded — no detections
            return []
        body = _linear((100, 100, 150, 200), 6, 0, f)
        plate = _linear((105, 205, 140, 235), 6, 0, f)
        return [VehicleFrame(1, [DetBox("WithoutHelmet", body, 0.9),
                                 DetBox("Plate", plate, 0.9)], "MH12AB1234")]

    scenarios.append(Scenario(
        "occlusion",
        [GTVehicle(1, "MH12AB1234", frozenset({"no_helmet"}))],
        [occ(f) for f in range(10)],
    ))

    return scenarios


def evaluate_all(scenarios=None) -> dict:
    """Run every evaluation over every scenario; return a JSON-able summary."""
    scenarios = scenarios or builtin_scenarios()
    out: dict = {"scenarios": {}, "totals": {}}
    tot = SystemMetrics(scenario="ALL")
    for sc in scenarios:
        sysm = evaluate_system(sc)
        out["scenarios"][sc.name] = {
            "system": sysm.as_dict(),
            "tracking": evaluate_tracking(sc).as_dict(),
            "association": evaluate_association(sc).as_dict(),
        }
        for f in ("expected_fines", "emitted_fines", "true_positives",
                  "false_positives", "false_negatives", "wrong_vehicle",
                  "wrong_plate", "duplicates"):
            setattr(tot, f, getattr(tot, f) + getattr(sysm, f))
    out["totals"] = tot.as_dict()
    return out
