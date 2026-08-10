import argparse
import os
import time
import math
import cv2
import requests
from ultralytics import YOLO
from modules.plate_ocr import read_plate
from modules.speed import SpeedEstimator
from utils.tracker import CentroidTracker

# =====================================
# SETTINGS
# =====================================

MODEL_PATH = "runs/detect/traffic_model-2/weights/best.pt"
API_URL = "http://127.0.0.1:5000/detect"
SPEED_LIMIT_KMH = 40

# A violation must be detected on this many consecutive frames before it's
# trusted. Single-frame model misfires are common (e.g. a helmeted rider
# briefly flickering to "WithoutHelmet") — requiring a short streak filters
# most of that out without needing a longer, laggier dwell time.
STREAK_THRESHOLD = 5

# Must match DETECT_API_KEY on the server if it has one set; empty is
# fine when the server has no key configured.
API_KEY = os.environ.get("DETECT_API_KEY", "")

os.makedirs("evidence", exist_ok=True)

# =====================================
# CLI ARGS
# =====================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect two-wheeler violations in a video and report them to the fine checker API."
    )
    parser.add_argument("--source", default="test_video.mp4",
                         help="video file path or camera index (default: test_video.mp4)")
    parser.add_argument("--output", default=None,
                         help="path to save an annotated copy of the video (optional)")
    parser.add_argument("--display", action="store_true",
                         help="show a live preview window (needs a display — off by default so this "
                              "also works on a headless server)")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="stop after N frames, useful for a quick test run")
    return parser.parse_args()

# =====================================
# LOAD MODEL + TRACKING
# =====================================

model = YOLO(MODEL_PATH)

tracker = CentroidTracker()
speed_estimator = SpeedEstimator()

# (track_id, violation) pairs already reported this run, so the same
# vehicle isn't fined again on every single frame it stays in view.
reported = set()

# (track_id, violation) -> consecutive-frame count, for the streak check
violation_streak = {}

# =====================================
# HELPERS
# =====================================

def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def nearest_plate_id(box, tracked_plates):
    """Best-effort association: attributes a violation to the nearest
    tracked plate. Works reliably with one vehicle in frame; with several
    vehicles close together it can mis-attribute a violation, since the
    model detects violation and plate boxes independently rather than as
    a single linked vehicle instance."""

    if not tracked_plates:
        return None

    cx, cy = centroid(box)

    return min(
        tracked_plates,
        key=lambda tid: math.hypot(
            cx - centroid(tracked_plates[tid])[0],
            cy - centroid(tracked_plates[tid])[1]
        )
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
    contradicting itself on the same rider — tested against a real photo
    where the *higher-confidence* box was the wrong one, so the safe move
    is to trust neither rather than guess."""
    return any(iou(box, other) > iou_threshold for other in other_boxes)


def confirmed(track_id, violation):
    """True once a violation has been seen on STREAK_THRESHOLD consecutive
    frames for this track. Caller is responsible for resetting streaks
    that weren't seen in the current frame (see `decay_streaks`)."""
    key = (track_id, violation)
    violation_streak[key] = violation_streak.get(key, 0) + 1
    return violation_streak[key] >= STREAK_THRESHOLD


def decay_streaks(seen_this_frame):
    for key in list(violation_streak):
        if key not in seen_this_frame:
            violation_streak[key] = 0


def report_violation(track_id, violation, plate, frame, box):
    key = (track_id, violation)

    if key in reported or not plate:
        return False

    reported.add(key)

    x1, y1, x2, y2 = box
    # timestamped so re-detecting the same plate/violation later doesn't
    # silently overwrite an earlier fine's evidence photo
    evidence_file = f"evidence/{plate}_{violation}_{int(time.time())}.jpg"
    cv2.imwrite(evidence_file, frame[y1:y2, x1:x2])

    try:
        response = requests.post(
            API_URL,
            json={
                "plate": plate,
                "violation": violation,
                "image_path": evidence_file
            },
            headers={"X-API-Key": API_KEY}
        )
        print("Sent:", violation, "for", plate, "-", response.text)

    except requests.exceptions.ConnectionError:
        print(f"Could not reach {API_URL} — is app.py running?")

    return True

# =====================================
# MAIN LOOP
# =====================================

def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Could not open video source: {args.source}")
        return

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if args.max_frames and frame_count > args.max_frames:
            break

        results = model(frame, verbose=False)[0]

        plates = []
        without_helmet = []
        with_helmet = []
        triple_riding = []

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "Plate":
                plates.append((x1, y1, x2, y2))
                color = (0, 255, 0)
            elif label == "WithoutHelmet":
                without_helmet.append((x1, y1, x2, y2))
                color = (0, 0, 255)
            elif label == "TripleRiding":
                triple_riding.append((x1, y1, x2, y2))
                color = (0, 0, 255)
            else:  # WithHelmet
                with_helmet.append((x1, y1, x2, y2))
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        tracked_plates = tracker.update(plates)
        plate_text_by_id = {}
        seen_this_frame = set()

        # ------------------------
        # PER-PLATE: OCR + SPEED
        # ------------------------
        for track_id, plate_box in tracked_plates.items():
            px1, py1, px2, py2 = plate_box

            crop = frame[py1:py2, px1:px2]
            plate_text = clean_plate(read_plate(crop))
            plate_text_by_id[track_id] = plate_text

            if plate_text:
                cv2.putText(frame, plate_text, (px1, py2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            speed = speed_estimator.calculate_speed(track_id, plate_box)

            if speed > SPEED_LIMIT_KMH:
                cv2.putText(frame, f"Speed: {speed} km/h", (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                seen_this_frame.add((track_id, "overspeed"))
                if confirmed(track_id, "overspeed"):
                    report_violation(track_id, "overspeed", plate_text, frame, plate_box)

        # ------------------------
        # NO HELMET / TRIPLE RIDING: associate to nearest tracked plate
        # ------------------------
        for box in without_helmet:
            if is_contradicted(box, with_helmet):
                cv2.putText(frame, "Ambiguous helmet status", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                continue

            cv2.putText(frame, "No Helmet!", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            pid = nearest_plate_id(box, tracked_plates)

            if pid is not None:
                seen_this_frame.add((pid, "no_helmet"))
                if confirmed(pid, "no_helmet"):
                    report_violation(pid, "no_helmet", plate_text_by_id.get(pid), frame, tracked_plates[pid])

        for box in triple_riding:
            cv2.putText(frame, "Triple Riding!", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            pid = nearest_plate_id(box, tracked_plates)

            if pid is not None:
                seen_this_frame.add((pid, "triple_riding"))
                if confirmed(pid, "triple_riding"):
                    report_violation(pid, "triple_riding", plate_text_by_id.get(pid), frame, tracked_plates[pid])

        decay_streaks(seen_this_frame)

        if writer:
            writer.write(frame)

        if args.display:
            cv2.imshow("Output", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"...processed {frame_count} frames ({frame_count / elapsed:.1f} fps)")

    cap.release()
    if writer:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(
        f"\nDone: {frame_count} frames in {elapsed:.1f}s, "
        f"{len(tracker.objects)} plate(s) tracked, {len(reported)} violation(s) reported."
    )


if __name__ == "__main__":
    main()
