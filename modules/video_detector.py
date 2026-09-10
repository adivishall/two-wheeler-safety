"""Video violation pipeline for the web app's video-upload route.

Detect + track vehicles across frames, then confirm a violation over a short
streak of frames before recording it. Packaged as a single reusable
``process_video()`` with **per-call state** (its own tracker, streak counters,
reported set) so it's safe to call from a web request, records fines through a
callback instead of POSTing to itself over HTTP, and reports progress so the
browser can show a bar.

Association is done by :class:`~modules.vehicle_tracker.VehicleTracker`, which
groups each frame's raw boxes into per-vehicle instances (a rider/body region
plus its plate, matched one-to-one) and follows them across frames with stable
IDs. This replaces the old "attribute each violation box to the nearest tracked
plate" heuristic, which mis-assigns when bikes are close together.

``main.py`` remains the standalone CLI; this is the library the server uses.
"""

import os
import time

import cv2

from modules.association import DetBox
from modules.confidence import compute_confidence, temporal_confidence
from modules.evidence import build_evidence
from modules.geometry import horizontal_overlap_ratio
from modules.logging_setup import get_logger
from modules.plate_recognizer import PlateStabilizer
from modules.vehicle import TrackState
from modules.vehicle_tracker import VehicleTracker
from modules.violation_state import (
    HelmetConfig,
    HelmetStateMachine,
    TripleConfig,
    TripleRidingStateMachine,
)

# Draw colors (BGR) — green for plates/helmet-on, red for violations.
_GREEN = (0, 200, 0)
_RED = (0, 0, 230)
_AMBER = (0, 200, 255)

_LABEL_COLORS = {
    "Plate": _GREEN,
    "WithHelmet": _GREEN,
    "WithoutHelmet": _RED,
    "TripleRiding": _RED,
}

log = get_logger("video")


def _put_label(frame, text, org, color):
    x, y = org
    cv2.putText(
        frame, text, (int(x), int(max(0, y))),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
    )


def process_video(
    video_path,
    model,
    reader,
    output_path,
    record_fn,
    progress_cb=None,
    pixels_per_meter=None,
    max_width=1280,
    speed_limit_kmh=40,
    streak_threshold=5,
    max_frames=None,
    cancel_check=None,
    max_seconds=None,
):
    """Detect two-wheeler violations across a video.

    Args:
        video_path:   input video file.
        model, reader: loaded YOLO model + EasyOCR reader.
        output_path:  where to write the annotated (H.264/avc1) video.
        record_fn:    callable(plate, violation, evidence_path) -> amount.
                      Called once per confirmed (vehicle, violation); this is
                      how a fine gets persisted.
        progress_cb:  callable(frames_done, total_frames) for a progress bar.
        pixels_per_meter: speed calibration; speed/overspeed is skipped if None.
        max_width:    frames wider than this are downscaled before processing.
        streak_threshold: consecutive frames a violation must hold before it's
                      recorded (filters single-frame model flicker).
        max_frames:   optional cap for a quick run.

    Returns a summary dict: frames processed, vehicle count, output path, fps,
    and the list of recorded violations (plate, violation, amount, evidence).
    """
    from modules.speed import SpeedEstimator

    tracker = VehicleTracker()
    speed_estimator = (
        SpeedEstimator(pixels_per_meter=pixels_per_meter) if pixels_per_meter else None
    )

    helmet_cfg = HelmetConfig(confirm_window=streak_threshold)
    triple_cfg = TripleConfig(confirm_window=streak_threshold)

    reported = set()  # (track_id, violation) already fined this run
    streak = {}  # (track_id, violation) -> consecutive-frame count (overspeed only)
    recorded = []  # summaries of the fines we recorded
    stabilizers = {}  # track_id -> PlateStabilizer (temporal OCR voting)
    helmet_sms = {}  # track_id -> HelmetStateMachine (Phase 4)
    triple_sms = {}  # track_id -> TripleRidingStateMachine (Phase 5)
    confirmed_ids = set()  # distinct vehicles that reached CONFIRMED

    evidence_dir = os.path.dirname(output_path) or "evidence"
    os.makedirs(evidence_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("could not open video: %s", video_path)
        raise ValueError(f"could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total_frames = min(total_frames, max_frames) if total_frames else max_frames
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    frame_idx = 0
    start_wall = time.monotonic()

    def confirm(track_id, violation, seen):
        """Bump the streak for (track, violation); True once it's held for
        streak_threshold consecutive frames."""
        key = (track_id, violation)
        seen.add(key)
        streak[key] = streak.get(key, 0) + 1
        return streak[key] >= streak_threshold

    def maybe_record(track, violation, plate, original, annotated, violation_box,
                     *, detection, association, supporting_frames, speed=None):
        """Record a confirmed (vehicle, violation) once, with a full confidence
        breakdown and a structured evidence package."""
        tid = track.track_id
        key = (tid, violation)
        if key in reported or not plate:
            return
        reported.add(key)

        conf = compute_confidence(
            violation,
            detection=detection,
            temporal=temporal_confidence(supporting_frames, streak_threshold),
            association=association,
            ocr=track.plate_confidence,
            supporting_frames=supporting_frames,
        )
        track.violation_history.append(conf)

        pkg = build_evidence(
            evidence_dir,
            plate=plate,
            violation=violation,
            original=original,
            annotated=annotated,
            plate_box=track.plate_box,
            violation_box=violation_box,
            frame_index=frame_idx,
            track_id=tid,
            confidence=conf.as_dict(),
            speed=speed,
        )
        primary = os.path.join(evidence_dir, pkg.primary_path) if pkg.primary_path else ""
        amount = record_fn(plate, violation, primary)
        track.evidence_frames[violation] = pkg.metadata_path
        log.info(
            "confirmed %s for track %s (plate=%s, conf=%.2f)",
            violation, tid, plate, conf.final,
        )
        recorded.append({
            "plate": plate,
            "violation": violation,
            "amount": amount,
            "evidence": "/evidence/" + os.path.basename(primary) if primary else None,
            "evidence_id": pkg.evidence_id,
            "metadata": "/evidence/" + pkg.metadata_path,
            "confidence": conf.final,
            "confidence_breakdown": conf.as_dict(),
            "plate_observations": len(track.plate_observations),
        })

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            frame_idx -= 1
            break

        # Cooperative cancellation and a processing-time ceiling: stop cleanly
        # and return what was found so far rather than running unbounded.
        if cancel_check is not None and cancel_check():
            break
        if max_seconds is not None and (time.monotonic() - start_wall) > max_seconds:
            break

        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(
                frame, (max_width, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"avc1"), src_fps, (w, h)
            )

        # Keep the original frame untouched (clean OCR + real "original"
        # evidence); draw boxes/labels onto a copy that gets written out.
        annotated = frame.copy()

        results = model(frame, verbose=False)[0]

        dets = []
        for box in results.boxes:
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            dets.append(DetBox(label, (x1, y1, x2, y2), conf))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _LABEL_COLORS.get(label, _GREEN), 2)

        tracks = tracker.update(dets, frame_idx)
        seen = set()

        for track in tracks:
            tid = track.track_id
            if track.state is TrackState.CONFIRMED:
                confirmed_ids.add(tid)

            # ---- OCR the plate and feed the temporal stabilizer ----
            # detail=1 gives per-box confidence; the weakest group gates the
            # reading's confidence. The stabilizer votes across frames, so a
            # single noisy frame can't set the plate we fine on.
            if track.plate_box is not None:
                px1, py1, px2, py2 = track.plate_box
                crop = frame[max(0, py1):py2, max(0, px1):px2]
                ocr = reader.readtext(crop, detail=1) if crop.size else []
                if ocr:
                    joined = "".join(text for _, text, _ in ocr)
                    conf = min(float(c) for _, _, c in ocr)
                    stab = stabilizers.setdefault(tid, PlateStabilizer())
                    stab.add(joined, conf)
                    res = stab.result()
                    track.stable_plate = res.stable
                    track.plate_confidence = res.confidence
                    track.plate_observations = stab.observations
                    display = res.stable or res.normalized
                    if display:
                        _put_label(annotated, display, (px1, py2 + 20), (255, 255, 0))

            # Fine only on the temporally-voted stable plate, never a raw frame.
            plate = track.stable_plate

            body = track.body
            # Association confidence: how well the plate sits under the rider.
            assoc = (
                horizontal_overlap_ratio(track.plate_box, body.box)
                if body is not None and track.plate_box is not None
                else 0.5
            )

            # ---- speed / overspeed (only when calibrated) ----
            # Video time (frame_idx / fps), so a slow machine can't change the
            # estimated speed. The estimate carries an uncertainty and is only
            # valid after enough samples; overspeed is still temporally confirmed.
            if speed_estimator is not None and track.plate_box is not None:
                est = speed_estimator.estimate(tid, track.plate_box, frame_idx / src_fps)
                if est.valid and est.kmh > speed_limit_kmh:
                    _put_label(
                        annotated, f"{est.kmh:.0f}+-{est.uncertainty:.0f} km/h",
                        (track.plate_box[0], track.plate_box[1] - 10), _RED,
                    )
                    track.overspeed_state = "candidate"
                    if confirm(tid, "overspeed", seen):
                        track.overspeed_state = "confirmed"
                        # margin over the limit as the detection score; speed is
                        # measured on the plate itself, so association is 1.0.
                        margin = (est.kmh - speed_limit_kmh) / max(1, speed_limit_kmh)
                        maybe_record(track, "overspeed", plate, frame, annotated,
                                     track.plate_box, detection=margin, association=1.0,
                                     supporting_frames=streak.get((tid, "overspeed"), 0),
                                     speed={"kmh": est.kmh, "uncertainty": est.uncertainty,
                                            "limit": speed_limit_kmh})

            # ---- helmet: temporal state machine (Phase 4) ----
            hsm = helmet_sms.setdefault(tid, HelmetStateMachine(helmet_cfg))
            hstate = hsm.update(
                has_helmet=bool(body and body.has_helmet),
                has_no_helmet=bool(body and body.no_helmet_violation),
                no_helmet_conf=body.no_helmet_conf if body else 0.0,
                ambiguous=bool(body and body.ambiguous_helmet),
                frame_idx=frame_idx,
            )
            track.helmet_state = hstate.value
            if body is not None:
                bx1, by1 = body.box[0], body.box[1]
                if body.ambiguous_helmet:
                    _put_label(annotated, "Ambiguous helmet", (bx1, by1 - 10), _AMBER)
                elif body.no_helmet_violation:
                    _put_label(annotated, "No Helmet!", (bx1, by1 - 10), _RED)
            if hsm.confirmed and body is not None:
                # Record once the state machine has confirmed; retries each frame
                # until the plate is readable, then maybe_record dedups.
                maybe_record(track, "no_helmet", plate, frame, annotated, body.box,
                             detection=hsm.best_conf, association=assoc,
                             supporting_frames=hsm.supporting_frames)

            # ---- triple riding: temporal state machine (Phase 5) ----
            tsm = triple_sms.setdefault(tid, TripleRidingStateMachine(triple_cfg))
            tstate = tsm.update(
                has_triple=bool(body and body.has_triple),
                conf=body.triple_conf if body else 0.0,
                frame_idx=frame_idx,
            )
            track.triple_state = tstate.value
            if body is not None and body.has_triple:
                _put_label(annotated, "Triple Riding!", (body.box[0], body.box[1] - 28), _RED)
            if tsm.confirmed and body is not None:
                maybe_record(track, "triple_riding", plate, frame, annotated, body.box,
                             detection=tsm.best_conf, association=assoc,
                             supporting_frames=tsm.frames_observed)

        # reset streaks for (track, violation) pairs not seen this frame
        for key in list(streak):
            if key not in seen:
                streak[key] = 0

        writer.write(annotated)

        if progress_cb and frame_idx % 5 == 0:
            progress_cb(frame_idx, total_frames)

    cap.release()
    if writer:
        writer.release()
    if progress_cb:
        progress_cb(frame_idx, total_frames or frame_idx)

    log.info(
        "video done: %d frames, %d vehicle(s) confirmed, %d violation(s) recorded",
        frame_idx, len(confirmed_ids), len(recorded),
    )
    return {
        "frames": frame_idx,
        "plates_tracked": len(confirmed_ids),
        "fps": round(src_fps, 1),
        "output": "/evidence/" + os.path.basename(output_path),
        "violations": recorded,
    }
