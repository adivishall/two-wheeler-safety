"""The per-frame violation decision core: model-free, I/O-free, shared.

Everything between "the detector emitted these boxes" and "record this fine"
lives here, in one class:

    detections -> VehicleTracker (association + identity)
               -> PlateStabilizer (temporal OCR vote, only on a fresh plate box)
               -> helmet / triple-riding state machines (temporal confirmation)
               -> SpeedEstimator (video time, only on a fresh plate box)
               -> compute_confidence
               -> at most one ViolationDecision per (vehicle, violation)

Why it exists: ``process_video`` used to own this loop inline, and the two
model-free evaluators (``system_eval.evaluate_system`` and
``pipeline_eval.run_pipeline_decisions``) each carried their *own copy* of it.
The copies had drifted from the shipped loop in ways that changed results — a
confirm window of 3 instead of the shipped 5, a frame with no rider box skipped
instead of counted as a miss, OCR text handed over whether or not a plate box
existed. So the headline "pipeline metrics" measured a sibling of the shipped
code, not the shipped code. Now the web app, the CLI, the benchmark and every
evaluator drive this one class; the only things that differ are where the boxes
come from (YOLO vs a synthetic scenario) and how a plate crop becomes text
(EasyOCR vs a lookup).

The class never touches pixels. OCR is injected as ``read_plate(plate_box)``,
so this module imports no torch/cv2/easyocr and runs in CI.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from modules.association import DEFAULT_CONFIG, AssociationConfig, DetBox
from modules.confidence import (
    ViolationConfidence,
    compute_confidence,
    temporal_confidence,
)
from modules.geometry import Box, horizontal_overlap_ratio
from modules.plate_recognizer import (
    DEFAULT_PLATE_CONFIG,
    PlateConfig,
    PlateResult,
    PlateStabilizer,
)
from modules.speed import SpeedEstimate, SpeedEstimator
from modules.vehicle import TrackState, VehicleTrack
from modules.vehicle_tracker import (
    DEFAULT_TRACKER_CONFIG,
    TrackerConfig,
    VehicleTracker,
)
from modules.violation_state import (
    HelmetConfig,
    HelmetStateMachine,
    TripleConfig,
    TripleRidingStateMachine,
)

# ``read_plate(plate_box) -> (text, confidence)`` or None when nothing was read.
PlateReader = Callable[[Box], "tuple[str, float] | None"]

VIOLATION_TYPES = ("no_helmet", "triple_riding", "overspeed")


@dataclass(frozen=True)
class PipelineConfig:
    """Every decision threshold the pipeline applies, in one place.

    Built from :class:`modules.config.DetectionConfig` by
    :func:`pipeline_config_from_detection` so the env vars documented in the
    README are the values actually applied (and the values stamped into
    evidence), not just the values reported.
    """

    confirm_window: int = 5  # supporting frames to confirm (STREAK_THRESHOLD)
    # k-of-n rule: confirm once `confirm_window` of the last `vote_window`
    # observed frames support the violation. None = strictly consecutive.
    vote_window: int | None = None
    # The helmet decision's own streak length (None = confirm_window). Kept
    # separate so tuning helmet confirmation can't silently change triple-riding
    # or overspeed, which were not part of that experiment.
    helmet_confirm_window: int | None = None
    # Helmet track-level gate (see HelmetConfig): at least `helmet_min_observed`
    # frames with a rider box, `helmet_min_fraction` of them no-helmet. Chosen
    # by evaluate_temporal.py (eval/results/temporal_confirmation.md): at the
    # detector's measured val error rates it cut the worst-case share of
    # helmeted riders flagged from 35% to 2% (held-out seed), at the cost of
    # never fining a rider seen on fewer than 12 frames.
    helmet_min_observed: int = 12
    helmet_min_fraction: float = 0.7
    helmet_min_conf: float = 0.3  # HELMET_MIN_CONF
    triple_min_conf: float = 0.3  # TRIPLE_MIN_CONF
    speed_limit_kmh: float = 40.0  # SPEED_LIMIT_KMH
    ocr_lock_confidence: float = 0.90  # stop re-OCRing a locked plate
    ocr_lock_min_observations: int = 5
    trace_max_frames: int = 20
    association: AssociationConfig = DEFAULT_CONFIG
    tracker: TrackerConfig = DEFAULT_TRACKER_CONFIG
    plate: PlateConfig = DEFAULT_PLATE_CONFIG

    def snapshot(self) -> dict:
        """The thresholds that decide a fine, for the evidence sidecar."""
        return {
            "confirm_window": self.confirm_window,
            "helmet_confirm_window": self.helmet_confirm_window or self.confirm_window,
            "vote_window": self.vote_window,
            "helmet_min_observed": self.helmet_min_observed,
            "helmet_min_fraction": self.helmet_min_fraction,
            "helmet_min_conf": self.helmet_min_conf,
            "triple_min_conf": self.triple_min_conf,
            "speed_limit_kmh": self.speed_limit_kmh,
            "plate_min_observations": self.plate.min_observations,
            "plate_min_agreement": self.plate.min_confidence,
            "plate_min_margin": self.plate.min_margin,
        }


DEFAULT_PIPELINE_CONFIG = PipelineConfig()


def pipeline_config_from_detection(detection, **overrides) -> PipelineConfig:
    """Map a :class:`modules.config.DetectionConfig` onto a PipelineConfig."""
    base = dict(
        confirm_window=detection.streak_threshold,
        helmet_min_conf=detection.helmet_min_conf,
        helmet_min_observed=detection.helmet_min_observed,
        helmet_min_fraction=detection.helmet_min_fraction,
        triple_min_conf=detection.triple_min_conf,
        speed_limit_kmh=detection.speed_limit_kmh,
        ocr_lock_confidence=detection.ocr_lock_confidence,
        ocr_lock_min_observations=detection.ocr_lock_min_observations,
        trace_max_frames=detection.trace_max_frames,
    )
    base.update(overrides)
    return PipelineConfig(**base)


@dataclass
class ViolationDecision:
    """A (vehicle, violation) the pipeline has decided to record.

    Emitted exactly once per (track, violation), and only once the vehicle has
    a temporally-stable plate — a confirmed violation with no trustworthy plate
    is held (and retried on later frames), never fined against a guess.
    """

    track: VehicleTrack
    violation: str
    plate: str
    violation_box: Box
    confidence: ViolationConfidence
    frame_index: int
    timestamp: float
    # True if the plate box was detected in THIS frame, i.e. a plate crop taken
    # from this frame shows the plate. False means the plate was read earlier.
    plate_visible: bool
    plate_votes: dict = field(default_factory=dict)
    speed: dict | None = None
    trace: list | None = None


@dataclass
class TrackFrame:
    """What the pipeline concluded about one vehicle this frame (for drawing
    and for evaluators that need to map tracks back to ground truth)."""

    track: VehicleTrack
    plate_visible: bool
    ocr_ran: bool
    plate_label: str | None  # text worth drawing under the plate, if any
    speed: SpeedEstimate | None = None
    over_limit: bool = False


@dataclass
class FrameResult:
    frame_index: int
    tracks: list[TrackFrame]
    decisions: list[ViolationDecision]


class ViolationPipeline:
    """Stateful per-video decision engine. One instance per video/scenario."""

    def __init__(
        self,
        config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
        *,
        speed_estimator: SpeedEstimator | None = None,
        trace: bool = False,
    ):
        self.config = config
        self.tracker = VehicleTracker(config.association, config.tracker)
        self.speed_estimator = speed_estimator
        self.trace_enabled = trace
        self._helmet_window = config.helmet_confirm_window or config.confirm_window
        self._helmet_cfg = HelmetConfig(
            confirm_window=self._helmet_window, min_conf=config.helmet_min_conf,
            vote_window=config.vote_window,
            min_observed=config.helmet_min_observed,
            min_fraction=config.helmet_min_fraction)
        self._triple_cfg = TripleConfig(
            confirm_window=config.confirm_window, min_conf=config.triple_min_conf,
            vote_window=config.vote_window)
        self.stabilizers: dict[int, PlateStabilizer] = {}
        self.helmet: dict[int, HelmetStateMachine] = {}
        self.triple: dict[int, TripleRidingStateMachine] = {}
        self._overspeed_streak: dict[int, int] = {}
        self._reported: set[tuple[int, str]] = set()
        # (plate, violation) already fined this run. Track ids alone can't stop
        # a duplicate: a rider who leaves view for longer than the tracker's
        # max_age comes back as a NEW track and used to be fined again.
        self._fined: set[tuple[str, str]] = set()
        self.suppressed_duplicates: list[dict] = []
        self._traces: dict[int, deque] = {}
        self.confirmed_track_ids: set[int] = set()
        self.ocr_calls = 0
        # Confirmed violations still waiting for a trustworthy plate. Reported
        # at the end so "confirmed but never fined" is visible, not silent.
        self._held: dict[tuple[int, str], str] = {}

    # -- queries used by evaluators and the summary ---------------------------

    def confirmed_violations(self, track_id: int) -> set[str]:
        """Violations whose state machine confirmed for this track (whether or
        not a plate was ever stable enough to fine)."""
        out = set()
        if self.helmet.get(track_id) and self.helmet[track_id].confirmed:
            out.add("no_helmet")
        if self.triple.get(track_id) and self.triple[track_id].confirmed:
            out.add("triple_riding")
        return out

    def unfined_confirmations(self) -> list[dict]:
        """Confirmed (track, violation) pairs that never got a stable plate —
        the pipeline abstained rather than fine a guessed plate."""
        return [
            {"track_id": tid, "violation": v, "reason": reason}
            for (tid, v), reason in sorted(self._held.items())
            if (tid, v) not in self._reported
        ]

    # -- the per-frame step ----------------------------------------------------

    def step(
        self,
        frame_index: int,
        dets: list[DetBox],
        *,
        timestamp: float,
        read_plate: PlateReader | None = None,
    ) -> FrameResult:
        """Advance one frame. ``timestamp`` is VIDEO time in seconds
        (``frame_index / fps``), never wall-clock time."""
        cfg = self.config
        tracks = self.tracker.update(dets, frame_index)
        frames: list[TrackFrame] = []
        decisions: list[ViolationDecision] = []
        keep_streak: set[int] = set()  # tracks whose overspeed streak survives

        for track in tracks:
            tid = track.track_id
            if track.state is TrackState.CONFIRMED:
                self.confirmed_track_ids.add(tid)
            visible = track.plate_visible
            body = track.body

            # ---- OCR -> temporal vote, only on a plate box from THIS frame.
            # A stale box would crop whatever is now where the plate used to be.
            stab = self.stabilizers.get(tid)
            locked = (
                track.stable_plate is not None
                and track.plate_confidence >= cfg.ocr_lock_confidence
                and stab is not None
                and stab.num_observations >= cfg.ocr_lock_min_observations
            )
            ocr_ran = False
            label = track.stable_plate if visible else None
            if visible and read_plate is not None and not locked:
                assert track.plate_box is not None  # implied by plate_visible
                ocr_ran = True
                self.ocr_calls += 1
                reading = read_plate(track.plate_box)
                if reading:
                    text, conf = reading
                    stab = self.stabilizers.setdefault(tid, PlateStabilizer(cfg.plate))
                    stab.add(text, conf)
                    res = stab.result()
                    track.stable_plate = res.stable
                    track.plate_confidence = res.confidence
                    track.plate_observations = stab.observations
                    label = res.stable or res.normalized

            # ---- association score: plate-under-rider overlap, remembered from
            # the last frame where both boxes were detected.
            if body is not None and visible:
                assert track.plate_box is not None
                track.association_score = horizontal_overlap_ratio(
                    track.plate_box, body.box)
            assoc = track.association_score if track.association_score is not None else 0.5

            # ---- detection trace: the frame's violation signal for this track.
            if self.trace_enabled and body is not None:
                self._append_trace(track, frame_index, timestamp)

            # ---- speed / overspeed: video time, fresh plate box only.
            est: SpeedEstimate | None = None
            if self.speed_estimator is not None and visible:
                assert track.plate_box is not None
                est = self.speed_estimator.estimate(tid, track.plate_box, timestamp)
                if est.valid and est.kmh > cfg.speed_limit_kmh:
                    keep_streak.add(tid)
                    self._overspeed_streak[tid] = self._overspeed_streak.get(tid, 0) + 1
                    track.overspeed_state = "candidate"
                    streak = self._overspeed_streak[tid]
                    if streak >= cfg.confirm_window:
                        track.overspeed_state = "confirmed"
                        margin = (est.kmh - cfg.speed_limit_kmh) / max(1.0, cfg.speed_limit_kmh)
                        self._maybe_emit(
                            decisions, track, "overspeed", track.plate_box,
                            frame_index, timestamp,
                            detection=margin, association=1.0,
                            supporting_frames=streak,
                            speed={"kmh": est.kmh, "uncertainty": est.uncertainty,
                                   "samples": est.samples,
                                   "limit": cfg.speed_limit_kmh},
                        )
                elif est.valid:
                    self._overspeed_streak[tid] = 0  # measured, and under the limit
            elif self.speed_estimator is not None:
                keep_streak.add(tid)  # no measurement this frame: hold the streak

            # ---- helmet: temporal state machine. A frame with no rider box is
            # "nothing observed" (a miss), exactly as in production.
            hsm = self.helmet.setdefault(tid, HelmetStateMachine(self._helmet_cfg))
            hstate = hsm.update(
                has_helmet=bool(body and body.has_helmet),
                has_no_helmet=bool(body and body.no_helmet_violation),
                no_helmet_conf=body.no_helmet_conf if body else 0.0,
                ambiguous=bool(body and body.ambiguous_helmet),
                frame_idx=frame_index,
            )
            track.helmet_state = hstate.value
            if hsm.confirmed and body is not None:
                self._maybe_emit(
                    decisions, track, "no_helmet", body.box, frame_index, timestamp,
                    detection=hsm.best_conf, association=assoc,
                    supporting_frames=hsm.supporting_frames,
                )

            # ---- triple riding.
            tsm = self.triple.setdefault(tid, TripleRidingStateMachine(self._triple_cfg))
            tstate = tsm.update(
                has_triple=bool(body and body.has_triple),
                conf=body.triple_conf if body else 0.0,
                frame_idx=frame_index,
                observed=body is not None,
            )
            track.triple_state = tstate.value
            if tsm.confirmed and body is not None:
                self._maybe_emit(
                    decisions, track, "triple_riding", body.box, frame_index, timestamp,
                    detection=tsm.best_conf, association=assoc,
                    supporting_frames=tsm.frames_observed,
                )

            frames.append(TrackFrame(
                track=track, plate_visible=visible, ocr_ran=ocr_ran,
                plate_label=label, speed=est,
                over_limit=bool(est and est.valid and est.kmh > cfg.speed_limit_kmh),
            ))

        # Overspeed must be consecutive: a vehicle not seen this frame, or
        # measured under the limit, starts over.
        for tid in list(self._overspeed_streak):
            if tid not in keep_streak:
                self._overspeed_streak[tid] = 0

        return FrameResult(frame_index, frames, decisions)

    # -- internals -------------------------------------------------------------

    def _plate_result(self, tid: int) -> PlateResult | None:
        stab = self.stabilizers.get(tid)
        return stab.result() if stab is not None else None

    def _maybe_emit(
        self, decisions, track: VehicleTrack, violation: str, box: Box,
        frame_index: int, timestamp: float, *, detection: float,
        association: float, supporting_frames: int, speed: dict | None = None,
    ) -> None:
        key = (track.track_id, violation)
        if key in self._reported:
            return
        plate = track.stable_plate
        if not plate:
            res = self._plate_result(track.track_id)
            self._held[key] = (res.abstain_reason if res else None) or "no_plate_read"
            return
        self._reported.add(key)
        self._held.pop(key, None)
        if (plate, violation) in self._fined:
            # Same plate, same violation, same run: one fine. If two genuinely
            # different vehicles share an OCR'd plate this under-fines — the
            # safe direction.
            self.suppressed_duplicates.append({
                "track_id": track.track_id, "plate": plate, "violation": violation,
                "frame_index": frame_index,
            })
            return
        self._fined.add((plate, violation))
        res = self._plate_result(track.track_id)
        window = self._helmet_window if violation == "no_helmet" else self.config.confirm_window
        conf = compute_confidence(
            violation,
            detection=detection,
            temporal=temporal_confidence(supporting_frames, window),
            association=association,
            ocr=track.plate_confidence,
            supporting_frames=supporting_frames,
        )
        track.violation_history.append(conf)
        decisions.append(ViolationDecision(
            track=track, violation=violation, plate=plate, violation_box=box,
            confidence=conf, frame_index=frame_index, timestamp=timestamp,
            plate_visible=track.plate_visible,
            plate_votes=res.votes_dict() if res else {},
            speed=speed,
            trace=list(self._traces.get(track.track_id, [])) if self.trace_enabled else None,
        ))

    def _append_trace(self, track: VehicleTrack, frame_index: int, timestamp: float) -> None:
        body = track.body
        assert body is not None
        if body.no_helmet_violation:
            label, conf = "WithoutHelmet", body.no_helmet_conf
        elif body.has_triple:
            label, conf = "TripleRiding", body.triple_conf
        elif body.has_helmet:
            label, conf = "WithHelmet", 0.0
        else:
            return
        self._traces.setdefault(
            track.track_id, deque(maxlen=self.config.trace_max_frames)
        ).append({
            "track_id": track.track_id,
            "label": label,
            "confidence": round(float(conf), 4),
            "box": list(track.box),
            "frame_index": frame_index,
            "timestamp": round(timestamp, 3),
        })


# ---------------------------------------------------------------------------
# Single photo: the same association and plate rules, with no time axis
# ---------------------------------------------------------------------------

@dataclass
class ImageVehicle:
    """One vehicle found in a single photo, and what may be recorded for it."""

    body_box: Box | None
    plate_box: Box | None
    plate: str | None  # structure-valid (possibly look-alike-corrected) plate
    plate_raw: str | None  # what OCR actually returned
    plate_conf: float
    association: float
    violations: list[dict] = field(default_factory=list)  # [{type, detection}]
    abstained: list[dict] = field(default_factory=list)  # [{type, reason}]


def single_frame_decisions(
    dets: list[DetBox],
    read_plate: PlateReader | None,
    *,
    config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    contradiction_iou: float = 0.1,
) -> list[ImageVehicle]:
    """Per-vehicle decisions for ONE photo.

    A photo has no time axis, so there is no temporal confirmation and no
    cross-frame OCR vote. What it does get is the same association layer the
    video path uses, so each violation is attributed to the plate of *its*
    vehicle (the old photo path fined every violation in the frame against
    whichever plate was OCR'd last). To compensate for the missing temporal
    evidence it is stricter, not looser:

    * a plate must read as a structurally valid Indian plate (after at most
      ``config.plate.max_edits`` look-alike corrections) or nothing is recorded;
    * a no-helmet box that overlaps ANY helmet box by IoU > ``contradiction_iou``
      is treated as the model contradicting itself and abstains — even across
      two merged bodies, which the video path would let time resolve.
    """
    from modules.association import associate
    from modules.geometry import iou
    from modules.plate_recognizer import correct_plate

    result = associate(dets, config.association)
    helmet_boxes = [d.box for d in dets if d.label == "WithHelmet"]
    out: list[ImageVehicle] = []
    for body_i, plate_i in result.pairs:
        body = result.bodies[body_i] if body_i is not None else None
        plate_box = result.plates[plate_i] if plate_i is not None else None

        plate = plate_raw = None
        plate_conf = 0.0
        if plate_box is not None and read_plate is not None:
            reading = read_plate(plate_box)
            if reading:
                plate_raw, plate_conf = reading
                plate, _ = correct_plate(plate_raw, config.plate)
        assoc = (
            horizontal_overlap_ratio(plate_box, body.box)
            if body is not None and plate_box is not None else 0.0
        )
        vehicle = ImageVehicle(
            body_box=body.box if body else None, plate_box=plate_box, plate=plate,
            plate_raw=plate_raw, plate_conf=float(plate_conf), association=assoc,
        )

        candidates: list[tuple[str, float]] = []
        if body is not None:
            if body.ambiguous_helmet:
                vehicle.abstained.append({"type": "no_helmet", "reason": "contradiction"})
            elif body.no_helmet_violation:
                contested = any(
                    iou(body.box, hb) > contradiction_iou for hb in helmet_boxes
                )
                if contested:
                    vehicle.abstained.append(
                        {"type": "no_helmet", "reason": "overlaps_helmet_box"})
                elif body.no_helmet_conf < config.helmet_min_conf:
                    vehicle.abstained.append({"type": "no_helmet", "reason": "low_confidence"})
                else:
                    candidates.append(("no_helmet", body.no_helmet_conf))
            if body.has_triple:
                if body.triple_conf < config.triple_min_conf:
                    vehicle.abstained.append(
                        {"type": "triple_riding", "reason": "low_confidence"})
                else:
                    candidates.append(("triple_riding", body.triple_conf))

        for vtype, det_conf in candidates:
            if plate is None:
                reason = (
                    "no_plate_associated" if plate_box is None
                    else "plate_unreadable" if not plate_raw
                    else "plate_not_valid_format"
                )
                vehicle.abstained.append({"type": vtype, "reason": reason})
            else:
                vehicle.violations.append({"type": vtype, "detection": det_conf})
        out.append(vehicle)
    return out
