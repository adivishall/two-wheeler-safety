"""Single-image detection + OCR CLI.

Thin wrapper around the shared ``modules.detector.analyze_image`` pipeline (the
same code the web app's /analyze route runs), so the CLI and the server can't
drift. Detects violations, reads the plate, writes an annotated evidence image,
and (unless ``--no-report``) POSTs each violation to a running fine-checker API.

    python3 main_ocr.py --image photo.jpg
    python3 main_ocr.py --image photo.jpg --model path/to/best.pt --no-report
    python3 main_ocr.py --image photo.jpg --api-url http://127.0.0.1:5000/detect

Exit codes: 0 success, 1 image could not be read, 2 model weights missing.
"""

from __future__ import annotations

import argparse
import os
import sys

from modules.config import load_config
from modules.logging_setup import configure_logging, get_logger

log = get_logger("cli")

DEFAULT_IMAGE_PATH = "samples/plate8.jpeg"
DEFAULT_API_URL = "http://127.0.0.1:5000/detect"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect two-wheeler violations in a single image and report "
                    "them to the fine-checker API.",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH,
                        help=f"image file to scan (default: {DEFAULT_IMAGE_PATH})")
    parser.add_argument("--model", default=None,
                        help="YOLO weights (default: MODEL_PATH / config default)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help="fine-checker /detect endpoint to report violations to")
    parser.add_argument("--no-report", action="store_true",
                        help="detect and write evidence but don't POST to the API")
    return parser.parse_args(argv)


def _report(api_url, api_key, plate, violation, image_name):
    import requests

    try:
        resp = requests.post(
            api_url,
            json={"plate": plate, "violation": violation, "image_path": image_name},
            headers={"X-API-Key": api_key} if api_key else {},
            timeout=5,
        )
        log.info("reported %s for %s -> %s", violation, plate, resp.status_code)
    except requests.exceptions.RequestException as exc:
        log.warning("could not reach %s (%s) — is app.py running?", api_url, exc)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)
    config = load_config()

    from modules.detector import analyze_image, load_models

    model_path = args.model or config.model_path
    if not os.path.exists(model_path):
        log.error("model weights not found: %s (train first or pass --model)", model_path)
        return 2

    model, reader = load_models(model_path)
    from modules.yolo_io import resolve_device

    det = config.detection
    result = analyze_image(
        args.image, model, reader, evidence_dir=config.evidence_dir,
        conf=det.conf_threshold, contradiction_iou=det.contradiction_iou,
        helmet_min_conf=det.helmet_min_conf, triple_min_conf=det.triple_min_conf,
        device=resolve_device(det.device),
    )

    if result.get("error"):
        log.error("%s", result["error"])
        return 1

    for det in result["detections"]:
        print("Detected:", det["label"])

    if result["evidence_file"]:
        print("Evidence saved:", result["evidence_file"])

    # Per vehicle: each violation is reported against the plate of ITS rider,
    # and only when that plate reads as a valid plate (see analyze_image).
    reported = 0
    for v in result["vehicles"]:
        if v["plate"]:
            print("Plate:", v["plate"])
        for rec in v["violations"]:
            print("Violation:", rec["type"], "->", v["plate"],
                  f"(score {rec['confidence']:.2f})")
            reported += 1
            if not args.no_report:
                _report(args.api_url, config.server.detect_api_key, v["plate"],
                        rec["type"], rec["evidence"]["annotated_path"])
        for a in v["abstained"]:
            print("Not recorded:", a["type"], "-", a["reason"],
                  f"(plate read: {v['plate_raw']!r})")
    if not reported:
        print("No violation recorded.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
