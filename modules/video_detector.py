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
from modules.geometry import horizontal_overlap_ratio
from modules.plate_recognizer import PlateStabilizer
from modules.vehicle import TrackState
from modules.vehicle_tracker import VehicleTracker

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

    reported = set()  # (track_id, violation) already fined this run
    streak = {}  # (track_id, violation) -> consecutive-frame count
    recorded = []  # summaries of the fines we recorded
    stabilizers = {}  # track_id -> PlateStabilizer (temporal OCR voting)
    confirmed_ids = set()  # distinct vehicles that reached CONFIRMED

    evidence_dir = os.path.dirname(output_path) or "evidence"
    os.makedirs(evidence_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total_frames = min(total_frames, max_frames) if total_frames else max_frames
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    frame_idx = 0

    def confirm(track_id, violation, seen):
        """Bump the streak for (track, violation); True once it's held for
        streak_threshold consecutive frames."""
        key = (track_id, violation)
        seen.add(key)
        streak[key] = streak.get(key, 0) + 1
        return streak[key] >= streak_threshold

    def maybe_record(track, violation, plate, frame, box, detection, association):
        """Record a confirmed (vehicle, violation) once, with a full confidence
        breakdown. ``detection``/``association`` are the per-violation component
        scores; temporal and OCR come from the streak and stabilizer."""
        tid = track.track_id
        key = (tid, violation)
        if key in reported or not plate:
            return
        reported.add(key)
        x1, y1, x2, y2 = box
        crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        evidence_file = os.path.join(
            evidence_dir, f"{plate}_{violation}_{int(time.time() * 1000)}.jpg"
        )
        if crop.size:
            cv2.imwrite(evidence_file, crop)
        amount = record_fn(plate, violation, evidence_file)

        supporting = streak.get(key, 0)
        conf = compute_confidence(
            violation,
            detection=detection,
            temporal=temporal_confidence(supporting, streak_threshold),
            association=association,
            ocr=track.plate_confidence,
            supporting_frames=supporting,
        )
        track.violation_history.append(conf)
        recorded.append({
            "plate": plate,
            "violation": violation,
            "amount": amount,
            "evidence": "/evidence/" + os.path.basename(evidence_file),
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

        results = model(frame, verbose=False)[0]

        dets = []
        for box in results.boxes:
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            dets.append(DetBox(label, (x1, y1, x2, y2), conf))
            cv2.rectangle(frame, (x1, y1), (x2, y2), _LABEL_COLORS.get(label, _GREEN), 2)

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
                        _put_label(frame, display, (px1, py2 + 20), (255, 255, 0))

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
            if speed_estimator is not None and track.plate_box is not None:
                speed = speed_estimator.calculate_speed(tid, track.plate_box)
                if speed > speed_limit_kmh:
                    _put_label(frame, f"{speed} km/h", (track.plate_box[0], track.plate_box[1] - 10), _RED)
                    if confirm(tid, "overspeed", seen):
                        # margin over the limit as the detection score; speed is
                        # measured on the plate itself, so association is 1.0.
                        margin = (speed - speed_limit_kmh) / max(1, speed_limit_kmh)
                        maybe_record(track, "overspeed", plate, frame, track.plate_box,
                                     detection=margin, association=1.0)

            # ---- helmet / triple-riding from the vehicle's body ----
            if body is not None:
                bx1, by1 = body.box[0], body.box[1]
                if body.ambiguous_helmet:
                    _put_label(frame, "Ambiguous helmet", (bx1, by1 - 10), _AMBER)
                elif body.no_helmet_violation:
                    _put_label(frame, "No Helmet!", (bx1, by1 - 10), _RED)
                    if confirm(tid, "no_helmet", seen):
                        maybe_record(track, "no_helmet", plate, frame, body.box,
                                     detection=body.no_helmet_conf, association=assoc)
                if body.has_triple:
                    _put_label(frame, "Triple Riding!", (bx1, by1 - 28), _RED)
                    if confirm(tid, "triple_riding", seen):
                        maybe_record(track, "triple_riding", plate, frame, body.box,
                                     detection=body.triple_conf, association=assoc)

        # reset streaks for (track, violation) pairs not seen this frame
        for key in list(streak):
            if key not in seen:
                streak[key] = 0

        writer.write(frame)

        if progress_cb and frame_idx % 5 == 0:
            progress_cb(frame_idx, total_frames)

    cap.release()
    if writer:
        writer.release()
    if progress_cb:
        progress_cb(frame_idx, total_frames or frame_idx)

    return {
        "frames": frame_idx,
        "plates_tracked": len(confirmed_ids),
        "fps": round(src_fps, 1),
        "output": "/evidence/" + os.path.basename(output_path),
        "violations": recorded,
    }
