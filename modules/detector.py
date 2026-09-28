"""Shared single-image detection + OCR + annotation.

This is the one place the photo pipeline lives, so the `main_ocr.py` CLI and
the web app's `/analyze` upload route behave identically instead of drifting
apart. `analyze_image` runs the trained YOLO model on one image and hands the
boxes to :func:`modules.pipeline.single_frame_decisions`, which uses the same
Hungarian rider<->plate association as the video path — so with two bikes in a
photo, each violation is attributed to the plate under *that* rider. (The old
photo path fined every violation in the frame against whichever plate the loop
happened to OCR last.) It then draws labeled boxes onto a copy of the image and
writes an evidence package per recordable violation.
"""

import os
import re

import cv2

from modules.geometry import iou as _geometry_iou

# BGR (OpenCV order) box colors per class, chosen to read clearly on road
# footage: green plate, blue helmet-on, red helmet-off, orange triple-riding.
CLASS_COLORS = {
    "Plate": (0, 200, 0),
    "WithHelmet": (255, 170, 0),
    "WithoutHelmet": (0, 0, 230),
    "TripleRiding": (0, 140, 255),
}
DEFAULT_COLOR = (200, 200, 200)


# Kept importable from here for existing callers; one shared definition.
iou = _geometry_iou


def clean_plate(text):
    """Strip everything but letters/digits and upper-case, matching how
    plates are normalized before they're stored or looked up."""
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def load_models(model_path):
    """Load the YOLO weights and an EasyOCR reader.

    Imported lazily inside the function so simply importing this module
    (e.g. from a test) doesn't pull in ultralytics/easyocr or require the
    weights file to exist.
    """
    import easyocr
    from ultralytics import YOLO

    return YOLO(model_path), easyocr.Reader(["en"])


def _draw_annotations(img, detections, plate_text_by_box):
    """Return a copy of `img` with a labeled box drawn for each detection.

    ``plate_text_by_box`` maps a plate box to the text read from THAT plate,
    so with two bikes in frame each plate box is captioned with its own read."""
    annotated = img.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["label"], DEFAULT_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # For the plate, show the OCR'd number rather than the class name —
        # that's the useful thing to see on the evidence photo.
        plate_text = (plate_text_by_box or {}).get(tuple(det["box"]))
        if det["label"] == "Plate" and plate_text:
            caption = plate_text
        else:
            caption = f"{det['label']} {det['conf']:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_top = max(0, y1 - text_h - baseline - 4)

        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + text_w + 6, label_top + text_h + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            caption,
            (x1 + 3, label_top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def analyze_image(
    image_path, model, reader, evidence_dir="evidence", conf=0.25, *,
    contradiction_iou=0.1, helmet_min_conf=0.375, triple_min_conf=0.3,
    model_version=None, pipeline_version=None, source_id=None, device=None,
):
    """Detect violations and read plates in a single image, per vehicle.

    Returns a dict:
        {
            "vehicles":      [{plate, plate_raw, plate_conf, plate_box, body_box,
                               association, violations: [{type, confidence,
                               confidence_breakdown, evidence: {...}}],
                               abstained: [{type, reason}]}, ...],
            "plate":         the primary vehicle's plate (first vehicle with a
                             recordable violation, else the first valid plate),
            "violations":    that vehicle's violation keys (back-compat shape),
            "evidence_file": annotated image path for display, or None,
            "detections":    [{"label", "conf", "box"}, ...],
            "error":         str, only present if the image couldn't be read,
        }

    Only violations listed under ``vehicles[*].violations`` are recordable: they
    have a structurally valid plate from the same vehicle. Everything the model
    saw but the rules would not stand behind is under ``abstained`` with a
    reason, so the caller can show it without fining anyone.
    """
    import uuid

    from modules.association import DetBox
    from modules.confidence import compute_confidence, temporal_confidence
    from modules.evidence import build_evidence
    from modules.pipeline import PipelineConfig, single_frame_decisions

    img = cv2.imread(image_path)
    if img is None:
        return {
            "plate": None,
            "violations": [],
            "vehicles": [],
            "evidence_file": None,
            "detections": [],
            "error": f"could not read image: {image_path}",
        }

    # Predict on the decoded array we already hold (not the path) so the model
    # and the OCR crops are guaranteed to see the same pixels.
    predict_kwargs = {"conf": conf, "verbose": False}
    if device:
        predict_kwargs["device"] = device
    results = model.predict(source=img, **predict_kwargs)

    detections = []
    dets = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            xyxy = tuple(map(int, box.xyxy[0]))
            detections.append({"label": label, "conf": confidence, "box": xyxy})
            dets.append(DetBox(label, xyxy, confidence))

    def read_plate(plate_box):
        x1, y1, x2, y2 = plate_box
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return None
        ocr = reader.readtext(crop, detail=1)
        if not ocr:
            return None
        return "".join(t for _, t, _ in ocr), min(float(c) for _, _, c in ocr)

    cfg = PipelineConfig(helmet_min_conf=helmet_min_conf, triple_min_conf=triple_min_conf)
    vehicles = single_frame_decisions(
        dets, read_plate, config=cfg, contradiction_iou=contradiction_iou)

    plates_for_drawing = {
        v.plate_box: (v.plate or clean_plate(v.plate_raw or "") or None)
        for v in vehicles if v.plate_box is not None
    }
    annotated = _draw_annotations(img, detections, plates_for_drawing)

    os.makedirs(evidence_dir, exist_ok=True)
    evidence_file = None
    if detections:
        evidence_file = os.path.join(evidence_dir, f"scan_{uuid.uuid4().hex[:12]}.jpg")
        cv2.imwrite(evidence_file, annotated)

    out_vehicles = []
    for v in vehicles:
        recs = []
        for viol in v.violations:
            # temporal = 1/confirm_window: a photo is one frame of evidence,
            # and the score says so instead of pretending otherwise.
            score = compute_confidence(
                viol["type"], detection=viol["detection"],
                temporal=temporal_confidence(1, cfg.confirm_window),
                association=v.association, ocr=v.plate_conf, supporting_frames=1,
            )
            pkg = build_evidence(
                evidence_dir, plate=v.plate, violation=viol["type"],
                original=img, annotated=annotated, plate_box=v.plate_box,
                violation_box=v.body_box, frame_index=0,
                confidence=score.as_dict(), model_version=model_version,
                pipeline_version=pipeline_version, source_id=source_id,
                config_snapshot={
                    "mode": "single_image", "conf_threshold": conf,
                    "helmet_min_conf": helmet_min_conf,
                    "triple_min_conf": triple_min_conf,
                    "contradiction_iou": contradiction_iou,
                },
                plate_votes={"stable": v.plate, "raw": v.plate_raw,
                             "ocr_confidence": round(v.plate_conf, 4),
                             "observations": 1},
            )
            recs.append({
                "type": viol["type"],
                "confidence": score.final,
                "confidence_breakdown": score.as_dict(),
                "evidence": pkg.db_paths(),
            })
        out_vehicles.append({
            "plate": v.plate,
            "plate_raw": v.plate_raw,
            "plate_conf": round(v.plate_conf, 4),
            "plate_box": v.plate_box,
            "body_box": v.body_box,
            "association": round(v.association, 4),
            "violations": recs,
            "abstained": v.abstained,
        })

    primary = next((v for v in out_vehicles if v["violations"]), None) or next(
        (v for v in out_vehicles if v["plate"]), None)
    return {
        "vehicles": out_vehicles,
        "plate": primary["plate"] if primary else None,
        "violations": [r["type"] for r in primary["violations"]] if primary else [],
        "evidence_file": evidence_file,
        "detections": detections,
    }
