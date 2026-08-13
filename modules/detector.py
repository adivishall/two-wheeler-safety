"""Shared single-image detection + OCR + annotation.

This is the one place the detection pipeline lives, so the `main_ocr.py`
CLI and the web app's `/analyze` upload route behave identically instead of
drifting apart. `analyze_image` runs the trained YOLO model on one image,
OCRs the plate crop, decides which violations to report (with the same
helmet-contradiction suppression the CLI always used), draws labeled boxes
onto a copy of the image, and saves that annotated copy as the evidence
photo.
"""

import os
import re
import time

import cv2

# BGR (OpenCV order) box colors per class, chosen to read clearly on road
# footage: green plate, blue helmet-on, red helmet-off, orange triple-riding.
CLASS_COLORS = {
    "Plate": (0, 200, 0),
    "WithHelmet": (255, 170, 0),
    "WithoutHelmet": (0, 0, 230),
    "TripleRiding": (0, 140, 255),
}
DEFAULT_COLOR = (200, 200, 200)


def iou(box_a, box_b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
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
    from ultralytics import YOLO
    import easyocr

    return YOLO(model_path), easyocr.Reader(["en"])


def _draw_annotations(img, detections, plate_number):
    """Return a copy of `img` with a labeled box drawn for each detection."""
    annotated = img.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["label"], DEFAULT_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # For the plate, show the OCR'd number rather than the class name —
        # that's the useful thing to see on the evidence photo.
        if det["label"] == "Plate" and plate_number:
            caption = plate_number
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


def analyze_image(image_path, model, reader, evidence_dir="evidence", conf=0.25):
    """Detect violations and read the plate in a single image.

    Returns a dict:
        {
            "plate":         cleaned plate string, or None if none read,
            "violations":    list of violation keys (no_helmet/triple_riding),
            "evidence_file": path to the saved annotated image, or None,
            "detections":    [{"label", "conf", "box"}, ...],
            "error":         str, only present if the image couldn't be read,
        }

    `evidence_file` is written whenever a plate is read (annotated), even if
    there's no violation — matching the CLI's long-standing behavior.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {
            "plate": None,
            "violations": [],
            "evidence_file": None,
            "detections": [],
            "error": f"could not read image: {image_path}",
        }

    results = model.predict(source=image_path, conf=conf, verbose=False)

    detections = []
    without_helmet_boxes = []
    with_helmet_boxes = []
    plate_text = []

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            xyxy = tuple(map(int, box.xyxy[0]))

            detections.append(
                {"label": label, "conf": confidence, "box": xyxy}
            )

            if label == "WithoutHelmet":
                without_helmet_boxes.append(xyxy)
            elif label == "WithHelmet":
                with_helmet_boxes.append(xyxy)
            elif label == "Plate":
                x1, y1, x2, y2 = xyxy
                plate_crop = img[y1:y2, x1:x2]
                plate_text = reader.readtext(plate_crop, detail=0)

    plate_number = clean_plate("".join(plate_text)) if plate_text else None

    # A WithoutHelmet box overlapping a WithHelmet box is the model
    # contradicting itself on the same rider — tested against a real photo
    # where the higher-confidence box was the wrong one, so trust neither.
    # Only report no_helmet if some WithoutHelmet box has no such overlap.
    violations = []
    no_helmet_confirmed = any(
        not any(iou(whb, wb) > 0.1 for wb in with_helmet_boxes)
        for whb in without_helmet_boxes
    )
    if no_helmet_confirmed:
        violations.append("no_helmet")

    if any(d["label"] == "TripleRiding" for d in detections):
        violations.append("triple_riding")

    evidence_file = None
    if plate_number:
        os.makedirs(evidence_dir, exist_ok=True)
        annotated = _draw_annotations(img, detections, plate_number)
        # timestamped so re-scanning the same plate doesn't overwrite an
        # earlier fine's evidence photo
        evidence_file = f"{evidence_dir}/{plate_number}_{int(time.time())}.jpg"
        cv2.imwrite(evidence_file, annotated)

    return {
        "plate": plate_number,
        "violations": violations,
        "evidence_file": evidence_file,
        "detections": detections,
    }
