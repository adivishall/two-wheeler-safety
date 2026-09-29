"""Builds a tiny, obviously synthetic field dataset on disk for tests.

Frames are smooth random images (distinct per frame, stable under resize so
perceptual hashes behave as on photos); labels are hand-written. Nothing here is
field data — it exercises the loader, the split lock and the leakage guard.
"""

import json
import os

import cv2
import numpy as np

W, H = 320, 240


def frame_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, size=(6, 8, 3), dtype=np.uint8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def build(root, *, sequences=None, cameras=None, vehicles=None, frames=None,
          write_images=True, schema_version="1.0"):
    """Default: cameras camA (rear) and camB (side); sequences s1..s6 on camA,
    x1 on camB; one no-helmet vehicle per sequence with 3 labelled frames."""
    os.makedirs(root, exist_ok=True)
    cameras = cameras or [
        {"camera_id": "camA", "view": "rear", "resolution": [W, H], "fps": 25},
        {"camera_id": "camB", "view": "side", "resolution": [W, H], "fps": 25},
    ]
    if sequences is None:
        sequences = [
            {"sequence_id": f"s{i}", "camera_id": "camA", "fps": 25,
             "frames_dir": f"frames/s{i}", "lighting": "day" if i % 2 else "night",
             "weather": "clear", "start_time": f"2026-09-0{1 + i % 3}T10:00:00"}
            for i in range(1, 7)
        ] + [{"sequence_id": "x1", "camera_id": "camB", "fps": 25, "frames_dir": "frames/x1",
              "lighting": "day", "start_time": "2026-09-01T12:00:00"}]
    if vehicles is None:
        vehicles = [
            {"sequence_id": s["sequence_id"], "vehicle_id": "v1", "first_frame": 0,
             "last_frame": 2, "plate_text": "MH12AB1234", "plate_visibility": "full",
             "rider_count": 1, "helmet_states": ["no_helmet"], "violations": ["no_helmet"],
             "occlusion": "none", "annotator": "ann1", "reviewer": "rev1",
             "review_decision": "accepted", "label_confidence": "certain"}
            for s in sequences
        ]
    if frames is None:
        frames = []
        for n, s in enumerate(sequences):
            for k in range(3):
                frames.append({
                    "sequence_id": s["sequence_id"], "frame_index": k,
                    "frame_path": f"frames/{s['sequence_id']}/{k:06d}.png",
                    "blur": "sharp", "annotator": "ann1", "review_decision": "accepted",
                    "label_confidence": "certain",
                    "objects": [
                        {"vehicle_id": "v1", "role": "rider", "box": [100, 60, 160, 180],
                         "helmet_state": "no_helmet", "occlusion": "none"},
                        {"vehicle_id": "v1", "role": "plate", "box": [110, 180, 150, 195],
                         "plate_text": "MH12AB1234"},
                    ],
                    "_seed": n * 10 + k,
                })
    with open(os.path.join(root, "dataset.json"), "w") as fh:
        json.dump({"schema_version": schema_version, "name": "fixture", "version": "0",
                   "cameras": cameras, "sequences": sequences}, fh)
    with open(os.path.join(root, "vehicles.jsonl"), "w") as fh:
        fh.writelines(json.dumps(v) + "\n" for v in vehicles)
    with open(os.path.join(root, "frames.jsonl"), "w") as fh:
        for f in frames:
            seed = f.pop("_seed", 0)
            fh.write(json.dumps(f) + "\n")
            if write_images:
                path = os.path.join(root, f["frame_path"])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cv2.imwrite(path, frame_image(seed))
    return root
