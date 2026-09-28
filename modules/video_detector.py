"""Video violation pipeline: the I/O shell around :class:`ViolationPipeline`.

``process_video`` reads frames, runs the detector, and hands the raw boxes to
:class:`modules.pipeline.ViolationPipeline`, which owns every decision
(association, identity, temporal OCR voting, temporal confirmation, speed,
confidence, de-duplication). This module only does the things that need pixels
or disk: resizing, cropping plates for OCR, drawing, writing the annotated
video, building evidence packages, and calling ``record_fn``.

Keeping the decisions out of this file is deliberate: the model-free evaluators
drive the *same* ``ViolationPipeline`` with synthetic detections, so the
pipeline metrics describe the code that actually runs here.

State is per call (a fresh pipeline per video), so it is safe to call from a
web request; fines are recorded through a callback, and progress is reported
so the browser can show a bar. ``main.py`` is the standalone CLI around it.
"""

import os
import time

import cv2

from modules.association import DetBox
from modules.evidence import build_evidence
from modules.logging_setup import get_logger
from modules.pipeline import PipelineConfig, ViolationPipeline

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


def _read_plate_text(reader, crop):
    """EasyOCR a plate crop -> (joined text, weakest box confidence) or None.

    detail=1 gives per-box confidence; the weakest group gates the reading, so
    one confidently-read half of a plate can't vouch for an unreadable half."""
    if crop is None or crop.size == 0:
        return None
    ocr = reader.readtext(crop, detail=1)
    if not ocr:
        return None
    joined = "".join(text for _, text, _ in ocr)
    conf = min(float(c) for _, _, c in ocr)
    return joined, conf


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
    model_version=None,
    pipeline_version=None,
    source_id=None,
    config_snapshot=None,
    session_id=None,
    trace_enabled=False,
    trace_max_frames=20,
    profiler=None,
    ocr_lock_confidence=0.90,
    ocr_lock_min_observations=5,
    conf_threshold=None,
    helmet_min_conf=0.375,
    triple_min_conf=0.3,
    pipeline_config=None,
    device=None,
):
    """Detect two-wheeler violations across a video.

    Args:
        video_path:   input video file.
        model, reader: loaded YOLO model + EasyOCR reader.
        output_path:  where to write the annotated (H.264/avc1) video.
        record_fn:    callable(plate, violation, evidence_path, **extra) -> amount.
                      Called once per confirmed (vehicle, violation); this is how
                      a fine gets persisted. ``extra`` carries keyword-only
                      metadata (confidence, track_id, session_id,
                      detection_trace, evidence); a recorder may ignore any it
                      doesn't use.
        progress_cb:  callable(frames_done, total_frames) for a progress bar.
        pixels_per_meter: speed calibration **in source-video pixels** (measured
                      on an original frame). Frames wider than ``max_width`` are
                      downscaled before processing, so the calibration is scaled
                      by the same factor. Speed/overspeed is skipped if None.
        max_width:    frames wider than this are downscaled before processing.
        streak_threshold: frames a violation must persist before it's recorded
                      (filters single-frame model flicker).
        conf_threshold: detector box-confidence floor passed to YOLO (None =
                      the library default, 0.25).
        helmet_min_conf / triple_min_conf: per-frame confidence a violation box
                      must clear to count toward confirmation.
        pipeline_config: a full :class:`PipelineConfig`; overrides the
                      individual threshold arguments above when given.
        device:       inference device passed to YOLO (None = library default,
                      which on Apple silicon means the CPU).
        max_frames:   optional cap for a quick run.

    Returns a summary dict: frames processed, vehicles confirmed, output path,
    source fps, *processing* fps, why processing stopped, and the list of
    recorded violations.
    """
    from modules.profiling import NULL
    from modules.speed import SpeedEstimator

    prof = profiler or NULL
    cfg = pipeline_config or PipelineConfig(
        confirm_window=streak_threshold,
        helmet_min_conf=helmet_min_conf,
        triple_min_conf=triple_min_conf,
        speed_limit_kmh=speed_limit_kmh,
        ocr_lock_confidence=ocr_lock_confidence,
        ocr_lock_min_observations=ocr_lock_min_observations,
        trace_max_frames=trace_max_frames,
    )
    pipeline = ViolationPipeline(cfg, trace=trace_enabled)
    recorded = []

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

    # What was actually applied — stamped into every evidence package. Built
    # from the live arguments, so evidence can never claim a threshold the run
    # did not use (the old snapshot echoed env config the web job ignored).
    applied = {
        **(config_snapshot or {}),
        **cfg.snapshot(),
        "conf_threshold": conf_threshold if conf_threshold is not None else 0.25,
        "max_width": max_width,
        "pixels_per_meter_source": pixels_per_meter,
        "source_fps": round(src_fps, 3),
    }
    model_kwargs = {"verbose": False}
    if conf_threshold is not None:
        model_kwargs["conf"] = conf_threshold
    if device:
        model_kwargs["device"] = device
        applied["device"] = device

    writer = None
    frame_idx = 0
    stopped = "end_of_video"
    start_wall = time.monotonic()

    def record(decision, original, annotated):
        """Persist one decision: evidence package, then ``record_fn``."""
        track = decision.track
        with prof.stage("evidence"):
            pkg = build_evidence(
                evidence_dir,
                plate=decision.plate,
                violation=decision.violation,
                original=original,
                annotated=annotated,
                # A plate crop only when the plate is in THIS frame; otherwise
                # the crop would show whatever now sits at the old location.
                plate_box=track.plate_box if decision.plate_visible else None,
                # Likewise the violation crop: only if the rider is in this frame.
                violation_box=decision.violation_box if decision.violation_visible else None,
                frame_index=decision.frame_index,
                track_id=track.track_id,
                confidence=decision.confidence.as_dict(),
                speed=decision.speed,
                model_version=model_version,
                pipeline_version=pipeline_version,
                source_id=source_id,
                config_snapshot=applied,
                plate_votes=decision.plate_votes,
                video_time_s=round(decision.timestamp, 3),
            )
        primary = os.path.join(evidence_dir, pkg.primary_path) if pkg.primary_path else ""
        with prof.stage("db"):
            amount = record_fn(
                decision.plate, decision.violation, primary,
                confidence=decision.confidence.final, track_id=track.track_id,
                session_id=session_id, detection_trace=decision.trace,
                evidence=pkg.db_paths(),
            )
        track.evidence_frames[decision.violation] = pkg.metadata_path
        log.info(
            "confirmed %s for track %s (plate=%s, conf=%.2f)",
            decision.violation, track.track_id, decision.plate, decision.confidence.final,
        )
        recorded.append({
            "plate": decision.plate,
            "violation": decision.violation,
            "amount": amount,
            "track_id": track.track_id,
            "frame_index": decision.frame_index,
            "evidence": "/evidence/" + os.path.basename(primary) if primary else None,
            "evidence_id": pkg.evidence_id,
            "metadata": "/evidence/" + pkg.metadata_path,
            "confidence": decision.confidence.final,
            "confidence_breakdown": decision.confidence.as_dict(),
            "plate_observations": len(track.plate_observations),
            "plate_votes": decision.plate_votes,
        })

    while True:
        with prof.stage("read"):
            ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            frame_idx -= 1
            stopped = "max_frames"
            break

        # Cooperative cancellation and a processing-time ceiling: stop cleanly
        # and return what was found so far rather than running unbounded.
        if cancel_check is not None and cancel_check():
            frame_idx -= 1
            stopped = "cancelled"
            break
        if max_seconds is not None and (time.monotonic() - start_wall) > max_seconds:
            frame_idx -= 1
            stopped = "time_limit"
            break

        scale = 1.0
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
            if pixels_per_meter:
                # Calibration is measured on source frames; the estimator sees
                # resized ones. Without this a 1920-px video processed at 1280
                # reports every speed at 2/3 of its true value.
                pipeline.speed_estimator = SpeedEstimator(
                    pixels_per_meter=pixels_per_meter * scale)
                applied["pixels_per_meter_processed"] = round(pixels_per_meter * scale, 4)

        # Keep the original frame untouched (clean OCR + real "original"
        # evidence); draw boxes/labels onto a copy that gets written out.
        annotated = frame.copy()

        with prof.stage("yolo"):
            results = model(frame, **model_kwargs)[0]

        dets = []
        for box in results.boxes:
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append(DetBox(label, (x1, y1, x2, y2), float(box.conf[0])))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _LABEL_COLORS.get(label, _GREEN), 2)

        ocr_seconds = 0.0

        def read_plate(plate_box, _frame=frame):
            nonlocal ocr_seconds
            px1, py1, px2, py2 = plate_box
            crop = _frame[max(0, py1):py2, max(0, px1):px2]
            t0 = time.perf_counter()
            try:
                return _read_plate_text(reader, crop)
            finally:
                dt = time.perf_counter() - t0
                ocr_seconds += dt
                if profiler is not None:
                    profiler.add("ocr", dt)  # one entry per OCR call, not per frame

        # Tracking + association + OCR voting + state machines, in one call.
        # OCR time is carved out so the stages stay non-overlapping.
        t0 = time.perf_counter()
        step = pipeline.step(frame_idx, dets, timestamp=frame_idx / src_fps,
                             read_plate=read_plate)
        step_seconds = time.perf_counter() - t0
        if profiler is not None:
            profiler.add("track", max(0.0, step_seconds - ocr_seconds))

        for tf in step.tracks:
            track, body = tf.track, tf.track.body
            if tf.plate_visible and tf.plate_label:
                px1, _, _, py2 = track.plate_box
                _put_label(annotated, tf.plate_label, (px1, py2 + 20), (255, 255, 0))
            if tf.over_limit and tf.speed is not None:
                _put_label(
                    annotated, f"{tf.speed.kmh:.0f}+-{tf.speed.uncertainty:.0f} km/h",
                    (track.plate_box[0], track.plate_box[1] - 10), _RED,
                )
            if body is not None:
                bx1, by1 = body.box[0], body.box[1]
                if body.ambiguous_helmet:
                    _put_label(annotated, "Ambiguous helmet", (bx1, by1 - 10), _AMBER)
                elif body.no_helmet_violation:
                    _put_label(annotated, "No Helmet!", (bx1, by1 - 10), _RED)
                if body.has_triple:
                    _put_label(annotated, "Triple Riding!", (bx1, by1 - 28), _RED)

        for decision in step.decisions:
            record(decision, frame, annotated)

        with prof.stage("encode"):
            writer.write(annotated)

        if progress_cb and frame_idx % 5 == 0:
            progress_cb(frame_idx, total_frames)

    wall = time.monotonic() - start_wall
    prof.set_wall(wall)
    cap.release()
    if writer:
        writer.release()
    if progress_cb:
        progress_cb(frame_idx, total_frames or frame_idx)

    held = pipeline.unfined_confirmations()
    log.info(
        "video done (%s): %d frames, %d vehicle(s) confirmed, %d violation(s) "
        "recorded, %d confirmed but not fined (no stable plate)",
        stopped, frame_idx, len(pipeline.confirmed_track_ids), len(recorded), len(held),
    )
    summary = {
        "frames": frame_idx,
        "plates_tracked": len(pipeline.confirmed_track_ids),
        # Source video frame rate (kept under the historical key). NOT how fast
        # this run processed — that is processing_fps.
        "fps": round(src_fps, 1),
        "source_fps": round(src_fps, 3),
        "processing_fps": round(frame_idx / wall, 2) if wall > 0 else None,
        "wall_seconds": round(wall, 3),
        "stopped_reason": stopped,
        "output": "/evidence/" + os.path.basename(output_path),
        "violations": recorded,
        "unfined_confirmations": held,
        "suppressed_duplicates": pipeline.suppressed_duplicates,
        "ocr_calls": pipeline.ocr_calls,
    }
    profile = prof.summary()
    if profile:  # only present when a real profiler was passed (benchmark)
        summary["profile"] = profile
    return summary
