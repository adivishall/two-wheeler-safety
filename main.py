"""Standalone command-line video pipeline.

This is the CLI wrapper around the *same* modern video pipeline the web app
runs (``modules.video_detector.process_video``): per-vehicle tracking with
Hungarian association, temporal OCR stabilization, temporal helmet / triple-
riding confirmation, confidence scoring, and structured evidence. It no longer
carries its own copy of that logic — the old centroid-tracker + nearest-plate
path lived only here and drifted from the library the server used.

Confirmed violations are POSTed to a running fine-checker API (``--api-url``),
matching the original behaviour, unless ``--no-report`` is given (then evidence
is written but nothing is sent).

    python3 main.py --source clip.mp4                      # annotate + report
    python3 main.py --source clip.mp4 --output out.mp4     # choose output path
    python3 main.py --source clip.mp4 --max-frames 300     # quick test
    python3 main.py --source clip.mp4 --pixels-per-meter 68  # enable speed
    python3 main.py --source 0 --no-report                 # webcam, no API

The small pure helpers below (``iou`` / ``is_contradicted`` / ``centroid`` /
``nearest_plate_id`` / ``clean_plate``) are the original association/OCR
utilities. They are retained because the project's regression tests pin the
helmet-contradiction bug through them; importing this module stays dependency-
light (heavy libs load only inside ``main``).
"""

from __future__ import annotations

import argparse
import math
import os
import uuid

from modules.config import load_config
from modules.logging_setup import configure_logging, get_logger

log = get_logger("cli")


# --- pure helpers (kept for regression tests / reference) ------------------

def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def nearest_plate_id(box, tracked_plates):
    """Legacy best-effort association: the nearest tracked plate to a box.
    Superseded in the live pipeline by VehicleTracker's one-to-one association;
    kept here for reference and tests."""
    if not tracked_plates:
        return None
    cx, cy = centroid(box)
    return min(
        tracked_plates,
        key=lambda tid: math.hypot(
            cx - centroid(tracked_plates[tid])[0],
            cy - centroid(tracked_plates[tid])[1],
        ),
    )


def clean_plate(text):
    if not text:
        return None
    return "".join(ch for ch in text if ch.isalnum()).upper()


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(area_a + area_b - inter_area)


def is_contradicted(box, other_boxes, iou_threshold=0.1):
    """A WithoutHelmet box overlapping a WithHelmet box is the model
    contradicting itself on one rider — trust neither."""
    return any(iou(box, other) > iou_threshold for other in other_boxes)


# --- CLI -------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect two-wheeler violations in a video (shared pipeline) "
                    "and report them to the fine-checker API.",
    )
    parser.add_argument("--source", default="test_video.mp4",
                        help="video file path or camera index (default: test_video.mp4)")
    parser.add_argument("--output", default=None,
                        help="path to write the annotated video (default: an "
                             "auto-named file under evidence/)")
    parser.add_argument("--model", default=None,
                        help="YOLO weights (default: MODEL_PATH / config default)")
    parser.add_argument("--api-url", default="http://127.0.0.1:5000/detect",
                        help="fine-checker /detect endpoint to report violations to")
    parser.add_argument("--no-report", action="store_true",
                        help="write evidence but don't POST to the API")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after N frames (quick test run)")
    parser.add_argument("--pixels-per-meter", type=float, default=None,
                        help="camera calibration for speed estimation (pixel distance "
                             "in-frame / real metres). Without it, speed/overspeed is "
                             "skipped rather than reporting an uncalibrated number.")
    return parser.parse_args(argv)


def _make_recorder(api_url, api_key, report):
    """Return a ``record_fn(plate, violation, evidence_path) -> amount`` that
    POSTs each confirmed violation to the API (or is a no-op when reporting is
    off). Errors are caught so one failed POST can't abort the whole run."""
    from modules.config import fine_amount

    if not report:
        def _local(plate, violation, evidence_path, **_extra):
            log.info("recorded (local only): %s for %s", violation, plate)
            return fine_amount(violation)
        return _local

    import requests

    def _post(plate, violation, evidence_path, **_extra):
        image_name = os.path.basename(evidence_path) if evidence_path else None
        try:
            resp = requests.post(
                api_url,
                json={"plate": plate, "violation": violation,
                      "image_path": image_name},
                headers={"X-API-Key": api_key} if api_key else {},
                timeout=5,
            )
            log.info("reported %s for %s -> %s", violation, plate, resp.status_code)
        except requests.exceptions.RequestException as exc:
            log.warning("could not reach %s (%s) — is app.py running?", api_url, exc)
        return fine_amount(violation)

    return _post


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)
    config = load_config()

    import cv2  # noqa: F401  (imported to fail fast with a clear message if absent)

    from modules.detector import load_models
    from modules.video_detector import process_video

    model_path = args.model or config.model_path
    if not os.path.exists(model_path):
        log.error("model weights not found: %s (train first or pass --model)", model_path)
        return 2

    source = int(args.source) if str(args.source).isdigit() else args.source

    output = args.output
    if not output:
        os.makedirs(config.evidence_dir, exist_ok=True)
        output = os.path.join(config.evidence_dir, f"annotated_{uuid.uuid4().hex}.mp4")

    log.info("loading model %s", model_path)
    model, reader = load_models(model_path)

    record_fn = _make_recorder(
        args.api_url, config.server.detect_api_key, report=not args.no_report
    )

    def progress(done, total):
        if total and done % 30 == 0:
            log.info("processed %d/%d frames", done, total)

    try:
        summary = process_video(
            source, model, reader, output, record_fn=record_fn,
            progress_cb=progress, pixels_per_meter=args.pixels_per_meter,
            speed_limit_kmh=config.detection.speed_limit_kmh,
            streak_threshold=config.detection.streak_threshold,
            max_frames=args.max_frames,
            max_width=config.detection.max_video_width,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    print(
        f"\nDone: {summary['frames']} frames, "
        f"{summary['plates_tracked']} vehicle(s) confirmed, "
        f"{len(summary['violations'])} violation(s) recorded."
    )
    print(f"Annotated video: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
