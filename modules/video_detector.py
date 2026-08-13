"""Video violation pipeline for the web app's video-upload route.

This mirrors what `main.py` does at the command line — detect + track plates
across frames, associate no-helmet / triple-riding to the nearest tracked
plate, estimate speed (when calibrated), and confirm a violation over a short
streak of frames before recording it — but packaged as a single reusable
`process_video()` with **per-call state** (its own tracker, streak counters,
and reported set) so it's safe to call from a web request, records fines
through a callback instead of POSTing to itself over HTTP, and reports
progress so the browser can show a bar.

`main.py` is left as the standalone CLI; this is the library the server uses.
"""

import math
import os
import time

import cv2

from modules.detector import iou, clean_plate

# Draw colors (BGR) — green for plates/helmet-on, red for violations.
_GREEN = (0, 200, 0)
_RED = (0, 0, 230)
_AMBER = (0, 200, 255)


def _centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _nearest_plate_id(box, tracked_plates):
    """Attribute a violation box to the nearest tracked plate (best-effort;
    can mis-attribute when several vehicles are close together)."""
    if not tracked_plates:
        return None
    cx, cy = _centroid(box)
    return min(
        tracked_plates,
        key=lambda tid: math.hypot(
            cx - _centroid(tracked_plates[tid])[0],
            cy - _centroid(tracked_plates[tid])[1],
        ),
    )


def _is_contradicted(box, other_boxes, iou_threshold=0.1):
    """A WithoutHelmet box overlapping a WithHelmet box is the model
    contradicting itself on one rider — trust neither."""
    return any(iou(box, other) > iou_threshold for other in other_boxes)


def _put_label(frame, text, org, color):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


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
        max_width:    frames wider than this are downscaled before processing
                      (4K footage is needlessly slow and huge to write back).
        max_frames:   optional cap for a quick run.

    Returns a summary dict: frames processed, plate count, output path, fps,
    and the list of recorded violations (plate, violation, amount, evidence).
    """
    from utils.tracker import CentroidTracker
    from modules.speed import SpeedEstimator

    tracker = CentroidTracker()
    speed_estimator = (
        SpeedEstimator(pixels_per_meter=pixels_per_meter) if pixels_per_meter else None
    )

    reported = set()          # (track_id, violation) already fined this run
    streak = {}               # (track_id, violation) -> consecutive-frame count
    recorded = []             # summaries of the fines we recorded

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
    scale = 1.0
    frame_idx = 0

    def confirm(track_id, violation, seen):
        """Bump the streak for (track, violation); True once it's held for
        streak_threshold consecutive frames."""
        key = (track_id, violation)
        seen.add(key)
        streak[key] = streak.get(key, 0) + 1
        return streak[key] >= streak_threshold

    def maybe_record(track_id, violation, plate, frame, box):
        key = (track_id, violation)
        if key in reported or not plate:
            return
        reported.add(key)
        x1, y1, x2, y2 = box
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        evidence_file = os.path.join(
            evidence_dir, f"{plate}_{violation}_{int(time.time()*1000)}.jpg"
        )
        if crop.size:
            cv2.imwrite(evidence_file, crop)
        amount = record_fn(plate, violation, evidence_file)
        recorded.append({
            "plate": plate,
            "violation": violation,
            "amount": amount,
            "evidence": "/evidence/" + os.path.basename(evidence_file),
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

        plates, without_helmet, with_helmet, triple_riding = [], [], [], []
        for box in results.boxes:
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if label == "Plate":
                plates.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 2)
            elif label == "WithoutHelmet":
                without_helmet.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), _RED, 2)
            elif label == "TripleRiding":
                triple_riding.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), _RED, 2)
            else:  # WithHelmet
                with_helmet.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 2)

        tracked_plates = tracker.update(plates)
        plate_text_by_id = {}
        seen = set()

        for track_id, plate_box in tracked_plates.items():
            px1, py1, px2, py2 = plate_box
            crop = frame[max(0, py1):py2, max(0, px1):px2]
            texts = reader.readtext(crop, detail=0) if crop.size else []
            plate_text = clean_plate("".join(texts)) if texts else None
            plate_text_by_id[track_id] = plate_text

            if plate_text:
                _put_label(frame, plate_text, (px1, py2 + 20), (255, 255, 0))

            if speed_estimator is not None:
                speed = speed_estimator.calculate_speed(track_id, plate_box)
                if speed > speed_limit_kmh:
                    _put_label(frame, f"{speed} km/h", (px1, py1 - 10), _RED)
                    if confirm(track_id, "overspeed", seen):
                        maybe_record(track_id, "overspeed", plate_text, frame, plate_box)

        for box in without_helmet:
            if _is_contradicted(box, with_helmet):
                _put_label(frame, "Ambiguous helmet", (box[0], box[1] - 10), _AMBER)
                continue
            _put_label(frame, "No Helmet!", (box[0], box[1] - 10), _RED)
            pid = _nearest_plate_id(box, tracked_plates)
            if pid is not None and confirm(pid, "no_helmet", seen):
                maybe_record(pid, "no_helmet", plate_text_by_id.get(pid), frame, tracked_plates[pid])

        for box in triple_riding:
            _put_label(frame, "Triple Riding!", (box[0], box[1] - 10), _RED)
            pid = _nearest_plate_id(box, tracked_plates)
            if pid is not None and confirm(pid, "triple_riding", seen):
                maybe_record(pid, "triple_riding", plate_text_by_id.get(pid), frame, tracked_plates[pid])

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
        "plates_tracked": len(tracker.objects),
        "fps": round(src_fps, 1),
        "output": "/evidence/" + os.path.basename(output_path),
        "violations": recorded,
    }
