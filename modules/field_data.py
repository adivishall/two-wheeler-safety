"""Field dataset: real traffic-camera footage with human labels, and the rules
that keep evaluation data out of training.

Everything the project measures past the detector is synthetic today
(docs/AUDIT.md, docs/FIELD_EVALUATION.md). This module is the interface real
labelled footage comes in through. A dataset is a directory:

``dataset.json``
    ``schema_version``, ``name``, ``version``, ``cameras`` (id, view, geometry,
    resolution, fps, optional calibration and speed limit) and ``sequences``
    (id, camera, video or frame directory, fps, start time, lighting, weather).
``vehicles.jsonl``
    One line per vehicle per sequence — the track-level truth end-to-end
    evaluation needs: plate text, plate visibility, rider count, per-rider
    helmet state, violations, occlusion, optional speed, and label provenance
    (annotator, reviewer, reviewer decision, label confidence).
``frames.jsonl``
    One line per labelled frame: frame path, timestamp, image quality and the
    boxes in it (rider/plate, per vehicle), with the same provenance fields.
``splits.lock.json``
    The frozen split assignment (below). Written by :func:`assign_splits`.

Splits are DEVELOPMENT, VALIDATION, HELD_OUT and EXTERNAL. They are assigned to
*groups*, never to frames: frames of one sequence are near-duplicates, so a
sequence (or, with ``group_by="camera_day"``, a camera's whole day) lands in one
split. EXTERNAL is whole cameras, named up front: footage from a viewpoint the
system was never developed on. A new group is placed by a salted hash (so adding
data never reshuffles old data); once written to the lock, an assignment never
changes. Training data can only be produced by :func:`export_training`, which
takes DEVELOPMENT groups only and then runs :func:`check_no_eval_leakage`: no
evaluation group, no byte-identical evaluation image, no near-duplicate by
perceptual hash — or it raises :class:`LeakageError`.

No field dataset ships with the repository (none exists yet); the loader,
validator and guard are tested on small fixtures built in the tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from modules.plate_info import matches_structure, normalize_plate

SCHEMA_VERSION = "1.0"

DEVELOPMENT, VALIDATION, HELD_OUT, EXTERNAL = "development", "validation", "held_out", "external"
SPLITS = (DEVELOPMENT, VALIDATION, HELD_OUT, EXTERNAL)
EVAL_SPLITS = frozenset({VALIDATION, HELD_OUT, EXTERNAL})
DEFAULT_RATIOS = {DEVELOPMENT: 0.6, VALIDATION: 0.2, HELD_OUT: 0.2}

LIGHTING = {"day", "dusk_dawn", "night", "unknown"}
WEATHER = {"clear", "rain", "fog", "unknown"}
OCCLUSION = {"none", "partial", "heavy", "unknown"}
PLATE_VISIBILITY = {"full", "partial", "none"}
HELMET_STATE = {"helmet", "no_helmet", "unknown"}
LABEL_CONFIDENCE = {"certain", "probable", "uncertain"}
REVIEW_DECISION = {"accepted", "corrected", "rejected", "unreviewed"}
BLUR = {"sharp", "mild", "severe", "unknown"}
VIEW = {"rear", "front", "side", "oblique", "overhead", "unknown"}
VIOLATIONS = {"no_helmet", "triple_riding", "overspeed"}
ROLES = {"rider", "plate"}
GROUP_BY = {"sequence", "camera_day"}
# Ids become group keys and export filenames: keep them boring.
_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class FieldDataError(ValueError):
    """The dataset is malformed; ``issues`` lists every problem found."""

    def __init__(self, issues: list[str]):
        super().__init__(f"{len(issues)} problem(s) in field dataset: " + "; ".join(issues[:5]))
        self.issues = issues


class LeakageError(RuntimeError):
    """Evaluation data would have become training data."""

    def __init__(self, problems: list[str]):
        super().__init__(f"refusing to export training data: {len(problems)} leak(s): "
                         + "; ".join(problems[:5]))
        self.problems = problems


# -- records ------------------------------------------------------------------

@dataclass
class Provenance:
    annotator: str
    reviewer: str | None = None
    review_decision: str = "unreviewed"
    label_confidence: str = "certain"


@dataclass
class Camera:
    camera_id: str
    view: str = "unknown"
    resolution: tuple[int, int] | None = None
    fps: float | None = None
    mounting_height_m: float | None = None
    tilt_deg: float | None = None
    pixels_per_meter: float | None = None
    homography: list | None = None
    speed_limit_kmh: float | None = None


@dataclass
class Sequence:
    sequence_id: str
    camera_id: str
    fps: float
    lighting: str = "unknown"
    weather: str = "unknown"
    start_time: str | None = None
    video_path: str | None = None
    frames_dir: str | None = None

    @property
    def day(self) -> str:
        return (self.start_time or "")[:10] or "unknown-day"


@dataclass
class VehicleLabel:
    sequence_id: str
    vehicle_id: str
    first_frame: int
    last_frame: int
    plate_text: str
    plate_visibility: str
    rider_count: int
    helmet_states: list[str]
    violations: set[str]
    occlusion: str
    provenance: Provenance
    speed_kmh: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.sequence_id, self.vehicle_id)


@dataclass
class FrameObject:
    vehicle_id: str
    role: str
    box: tuple[float, float, float, float]
    helmet_state: str | None = None
    plate_text: str | None = None
    occlusion: str = "unknown"


@dataclass
class FrameLabel:
    sequence_id: str
    frame_index: int
    frame_path: str
    timestamp_s: float
    blur: str
    objects: list[FrameObject]
    provenance: Provenance


@dataclass
class FieldDataset:
    root: str
    name: str
    version: str
    cameras: dict[str, Camera]
    sequences: dict[str, Sequence]
    vehicles: dict[tuple[str, str], VehicleLabel]
    frames: list[FrameLabel]
    warnings: list[str] = field(default_factory=list)

    def group_key(self, sequence_id: str, group_by: str = "sequence") -> str:
        seq = self.sequences[sequence_id]
        if group_by == "camera_day":
            return f"camera_day:{seq.camera_id}:{seq.day}"
        return f"sequence:{sequence_id}"

    def frame_file(self, frame: FrameLabel) -> str:
        return os.path.join(self.root, frame.frame_path)


# -- loading and validation ---------------------------------------------------

def _enum(value, allowed: set, where: str, name: str, issues: list, default=None):
    if value is None and default is not None:
        return default
    if value not in allowed:
        issues.append(f"{where}: {name}={value!r} not in {sorted(allowed)}")
        return default if default is not None else value
    return value


def _provenance(d: dict, where: str, issues: list) -> Provenance:
    annotator = d.get("annotator")
    if not annotator:
        issues.append(f"{where}: annotator is required (who made this label)")
    return Provenance(
        annotator=annotator or "",
        reviewer=d.get("reviewer"),
        review_decision=_enum(d.get("review_decision", "unreviewed"), REVIEW_DECISION,
                              where, "review_decision", issues),
        label_confidence=_enum(d.get("label_confidence", "certain"), LABEL_CONFIDENCE,
                               where, "label_confidence", issues),
    )


def _safe_rel(path: str | None, where: str, issues: list) -> str | None:
    if path is None:
        return None
    norm = os.path.normpath(path)
    if os.path.isabs(path) or norm.startswith(".."):
        issues.append(f"{where}: path {path!r} must be relative and inside the dataset")
    return norm


def _read_jsonl(path: str, issues: list) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                issues.append(f"{os.path.basename(path)}:{n}: invalid JSON ({exc})")
    return rows


def load_dataset(root: str, *, check_files: bool = False) -> FieldDataset:
    """Load and validate a field dataset. Raises :class:`FieldDataError` listing
    every problem; problems that don't make a label unusable (a plate that
    doesn't look like an Indian plate, an unreviewed label) are ``warnings``."""
    issues: list[str] = []
    warnings: list[str] = []
    meta_path = os.path.join(root, "dataset.json")
    if not os.path.exists(meta_path):
        raise FieldDataError([f"no dataset.json in {root}"])
    with open(meta_path) as fh:
        meta = json.load(fh)
    if meta.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"schema_version {meta.get('schema_version')!r} != {SCHEMA_VERSION!r}")

    cameras: dict[str, Camera] = {}
    for c in meta.get("cameras", []):
        cid = c.get("camera_id")
        where = f"camera {cid}"
        if not cid or cid in cameras or not _ID.match(str(cid)):
            issues.append(f"{where}: camera_id missing, duplicate, or not [A-Za-z0-9_.-]")
            continue
        res = c.get("resolution")
        h = c.get("homography")
        if h is not None and not (len(h) == 3 and all(len(r) == 3 for r in h)):
            issues.append(f"{where}: homography must be 3x3")
        cameras[cid] = Camera(
            camera_id=cid, view=_enum(c.get("view", "unknown"), VIEW, where, "view", issues),
            resolution=tuple(res) if res else None, fps=c.get("fps"),
            mounting_height_m=c.get("mounting_height_m"), tilt_deg=c.get("tilt_deg"),
            pixels_per_meter=c.get("pixels_per_meter"), homography=h,
            speed_limit_kmh=c.get("speed_limit_kmh"))

    sequences: dict[str, Sequence] = {}
    for s in meta.get("sequences", []):
        sid = s.get("sequence_id")
        where = f"sequence {sid}"
        if not sid or sid in sequences or not _ID.match(str(sid)):
            issues.append(f"{where}: sequence_id missing, duplicate, or not [A-Za-z0-9_.-]")
            continue
        if s.get("camera_id") not in cameras:
            issues.append(f"{where}: unknown camera_id {s.get('camera_id')!r}")
        if not s.get("fps") or float(s["fps"]) <= 0:
            issues.append(f"{where}: fps must be > 0 (time comes from it)")
        if not (s.get("video_path") or s.get("frames_dir")):
            issues.append(f"{where}: needs video_path or frames_dir")
        sequences[sid] = Sequence(
            sequence_id=sid, camera_id=s.get("camera_id", ""), fps=float(s.get("fps") or 0),
            lighting=_enum(s.get("lighting", "unknown"), LIGHTING, where, "lighting", issues),
            weather=_enum(s.get("weather", "unknown"), WEATHER, where, "weather", issues),
            start_time=s.get("start_time"),
            video_path=_safe_rel(s.get("video_path"), where, issues),
            frames_dir=_safe_rel(s.get("frames_dir"), where, issues))

    vehicles: dict[tuple[str, str], VehicleLabel] = {}
    for v in _read_jsonl(os.path.join(root, "vehicles.jsonl"), issues):
        key = (str(v.get("sequence_id")), str(v.get("vehicle_id")))
        where = f"vehicle {key[0]}/{key[1]}"
        if not v.get("vehicle_id") or not _ID.match(str(v.get("vehicle_id"))):
            issues.append(f"{where}: vehicle_id missing or not [A-Za-z0-9_.-]")
            continue
        if key[0] not in sequences:
            issues.append(f"{where}: unknown sequence")
            continue
        if key in vehicles:
            issues.append(f"{where}: duplicate vehicle")
            continue
        states = [_enum(h, HELMET_STATE, where, "helmet_state", issues)
                  for h in v.get("helmet_states", [])]
        riders = int(v.get("rider_count", len(states)))
        violations = set(v.get("violations", []))
        for bad in violations - VIOLATIONS:
            issues.append(f"{where}: unknown violation {bad!r}")
        if states and len(states) != riders:
            issues.append(f"{where}: {len(states)} helmet states for {riders} riders")
        if "no_helmet" in states and "no_helmet" not in violations:
            issues.append(f"{where}: a rider has no helmet but no_helmet is not labelled")
        if "no_helmet" in violations and states and "no_helmet" not in states:
            issues.append(f"{where}: no_helmet labelled but no rider is bare-headed")
        if (riders >= 3) != ("triple_riding" in violations):
            issues.append(f"{where}: triple_riding must be labelled iff rider_count >= 3")
        plate = normalize_plate(v.get("plate_text", ""))
        visibility = _enum(v.get("plate_visibility"), PLATE_VISIBILITY, where,
                           "plate_visibility", issues)
        if visibility == "none" and plate:
            issues.append(f"{where}: plate_text given for a plate labelled not visible")
        if plate and not matches_structure(plate):
            warnings.append(f"{where}: plate {plate!r} is not shaped like an Indian plate — "
                            "the pipeline's structural check would reject it")
        first, last = int(v.get("first_frame", 0)), int(v.get("last_frame", -1))
        if last < first:
            issues.append(f"{where}: last_frame < first_frame")
        prov = _provenance(v, where, issues)
        if prov.review_decision == "unreviewed":
            warnings.append(f"{where}: label not reviewed")
        vehicles[key] = VehicleLabel(
            sequence_id=key[0], vehicle_id=key[1], first_frame=first, last_frame=last,
            plate_text=plate, plate_visibility=visibility, rider_count=riders,
            helmet_states=states, violations=violations,
            occlusion=_enum(v.get("occlusion", "unknown"), OCCLUSION, where, "occlusion",
                            issues),
            provenance=prov, speed_kmh=v.get("speed_kmh"))

    frames: list[FrameLabel] = []
    seen_frames: set[tuple[str, int]] = set()
    for f in _read_jsonl(os.path.join(root, "frames.jsonl"), issues):
        sid, idx = f.get("sequence_id"), int(f.get("frame_index", -1))
        where = f"frame {sid}#{idx}"
        if sid not in sequences:
            issues.append(f"{where}: unknown sequence")
            continue
        if (sid, idx) in seen_frames:
            issues.append(f"{where}: duplicate frame")
            continue
        seen_frames.add((sid, idx))
        path = _safe_rel(f.get("frame_path"), where, issues)
        if not path:
            issues.append(f"{where}: frame_path is required")
        elif check_files and not os.path.exists(os.path.join(root, path)):
            issues.append(f"{where}: file {path} not found")
        cam = cameras.get(sequences[sid].camera_id)
        objects = []
        for o in f.get("objects", []):
            vid = o.get("vehicle_id")
            owhere = f"{where} object {vid}"
            vehicle = vehicles.get((sid, vid))
            if vehicle is None:
                issues.append(f"{owhere}: no vehicle {vid!r} in sequence {sid}")
                continue
            if not vehicle.first_frame <= idx <= vehicle.last_frame:
                issues.append(f"{owhere}: frame outside the vehicle's first/last frame")
            box = o.get("box") or []
            if len(box) != 4 or not (box[0] < box[2] and box[1] < box[3]):
                issues.append(f"{owhere}: box must be [x1, y1, x2, y2] with x1<x2, y1<y2")
                continue
            if cam and cam.resolution and (box[0] < 0 or box[1] < 0 or box[2] > cam.resolution[0]
                                           or box[3] > cam.resolution[1]):
                issues.append(f"{owhere}: box outside the camera resolution")
            role = _enum(o.get("role"), ROLES, owhere, "role", issues)
            objects.append(FrameObject(
                vehicle_id=vid, role=role,
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                helmet_state=(_enum(o.get("helmet_state", "unknown"), HELMET_STATE, owhere,
                                    "helmet_state", issues) if role == "rider" else None),
                plate_text=(normalize_plate(o.get("plate_text", "")) or None
                            if role == "plate" else None),
                occlusion=_enum(o.get("occlusion", "unknown"), OCCLUSION, owhere,
                                "occlusion", issues)))
        frames.append(FrameLabel(
            sequence_id=sid, frame_index=idx, frame_path=path or "",
            timestamp_s=float(f.get("timestamp_s", idx / (sequences[sid].fps or 1))),
            blur=_enum(f.get("blur", "unknown"), BLUR, where, "blur", issues),
            objects=objects, provenance=_provenance(f, where, issues)))

    if issues:
        raise FieldDataError(issues)
    return FieldDataset(root=root, name=meta.get("name", ""), version=meta.get("version", ""),
                        cameras=cameras, sequences=sequences, vehicles=vehicles,
                        frames=sorted(frames, key=lambda fr: (fr.sequence_id, fr.frame_index)),
                        warnings=warnings)


# -- splits ---------------------------------------------------------------------

LOCK_NAME = "splits.lock.json"


def _hash_unit(salt: str, key: str) -> float:
    digest = hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def _lock_digest(lock: dict) -> str:
    body = {k: lock[k] for k in ("group_by", "salt", "external_cameras", "assignments")}
    return "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def read_lock(root: str) -> dict | None:
    path = os.path.join(root, LOCK_NAME)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        lock = json.load(fh)
    if lock.get("lock_hash") != _lock_digest(lock):
        raise LeakageError([f"{LOCK_NAME} was edited by hand (hash mismatch): split "
                            "assignments can't be trusted"])
    return lock


def assign_splits(ds: FieldDataset, *, salt: str | None = None, ratios: dict | None = None,
                  external_cameras=(), group_by: str = "sequence",
                  write: bool = True) -> dict:
    """Assign every not-yet-assigned group to a split and return the lock.

    Existing assignments are kept verbatim — changing ``ratios`` later only
    affects new groups — and a group can never move. EXTERNAL cameras are fixed
    when the lock is created; naming a camera EXTERNAL after its footage was
    already assigned elsewhere is refused (its sequences may have been used)."""
    if group_by not in GROUP_BY:
        raise ValueError(f"group_by must be one of {sorted(GROUP_BY)}")
    lock = read_lock(ds.root)
    ratios = ratios or DEFAULT_RATIOS
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or set(ratios) - {DEVELOPMENT, VALIDATION,
                                                                   HELD_OUT}:
        raise ValueError("ratios must cover development/validation/held_out and sum to 1")
    if lock is None:
        lock = {"schema_version": SCHEMA_VERSION, "group_by": group_by,
                "salt": salt or hashlib.sha256(ds.name.encode()).hexdigest()[:12],
                "external_cameras": sorted(external_cameras), "assignments": {}}
    else:
        if group_by != lock["group_by"]:
            raise ValueError(f"lock groups by {lock['group_by']!r}; can't switch to "
                             f"{group_by!r} without re-splitting everything")
        new_external = set(external_cameras) - set(lock["external_cameras"])
        for cam in sorted(new_external):
            used = sorted(k for k, a in lock["assignments"].items()
                          if a.get("camera_id") == cam)
            if used:
                raise LeakageError([f"camera {cam} already has groups assigned ({used[0]}, "
                                    "...); it can't become EXTERNAL retroactively — its "
                                    "footage may already have been developed on"])
        lock["external_cameras"] = sorted(set(lock["external_cameras"]) | new_external)

    external = set(lock["external_cameras"])
    order = [(DEVELOPMENT, ratios.get(DEVELOPMENT, 0.0)),
             (VALIDATION, ratios.get(VALIDATION, 0.0)),
             (HELD_OUT, ratios.get(HELD_OUT, 0.0))]
    now = datetime.now(timezone.utc).isoformat()
    for sid, seq in sorted(ds.sequences.items()):
        key = ds.group_key(sid, lock["group_by"])
        if key in lock["assignments"]:
            continue
        if seq.camera_id in external:
            split, rule = EXTERNAL, "external_camera"
        else:
            u, acc, split = _hash_unit(lock["salt"], key), 0.0, HELD_OUT
            for name, share in order:
                acc += share
                if u < acc:
                    split = name
                    break
            rule = "hash"
        lock["assignments"][key] = {"split": split, "rule": rule, "assigned_at": now,
                                    "camera_id": seq.camera_id}
    lock["lock_hash"] = _lock_digest(lock)
    if write:
        with open(os.path.join(ds.root, LOCK_NAME), "w") as fh:
            json.dump(lock, fh, indent=2, sort_keys=True)
    return lock


def split_of(ds: FieldDataset, lock: dict, sequence_id: str) -> str | None:
    a = lock["assignments"].get(ds.group_key(sequence_id, lock["group_by"]))
    return a["split"] if a else None


def frames_in(ds: FieldDataset, lock: dict, split: str) -> list[FrameLabel]:
    return [f for f in ds.frames if split_of(ds, lock, f.sequence_id) == split]


def vehicles_in(ds: FieldDataset, lock: dict, split: str) -> list[VehicleLabel]:
    return [v for v in ds.vehicles.values() if split_of(ds, lock, v.sequence_id) == split]


# -- leakage guard and training export ------------------------------------------

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash_file(path: str) -> int | None:
    import cv2

    from modules.dataset_audit import dhash

    img = cv2.imread(path)
    return dhash(img) if img is not None else None


def check_no_eval_leakage(train_files: list[str], ds: FieldDataset, lock: dict, *,
                          near_duplicate_bits: int = 5, perceptual: bool = True) -> None:
    """Raise :class:`LeakageError` if any training file IS evaluation data.

    Three independent checks, because each alone has a hole: the evaluation
    split's group assignment (catches a frame exported from the wrong split),
    a byte hash (catches the same file copied under another name) and a
    perceptual dHash within ``near_duplicate_bits`` (catches a re-encode or
    resize of an evaluation frame)."""
    eval_frames = [f for f in ds.frames if split_of(ds, lock, f.sequence_id) in EVAL_SPLITS]
    eval_paths = {os.path.normpath(ds.frame_file(f)) for f in eval_frames}
    eval_existing = [p for p in eval_paths if os.path.exists(p)]
    eval_sha = {_sha256(p): p for p in eval_existing}
    eval_dh = ({p: h for p in eval_existing if (h := _dhash_file(p)) is not None}
               if perceptual else {})
    problems = []
    for tf in train_files:
        norm = os.path.normpath(tf)
        if norm in eval_paths:
            problems.append(f"{tf} is an evaluation frame")
            continue
        if not os.path.exists(tf):
            continue
        digest = _sha256(tf)
        if digest in eval_sha:
            problems.append(f"{tf} is byte-identical to evaluation frame {eval_sha[digest]}")
            continue
        if eval_dh:
            h = _dhash_file(tf)
            if h is None:
                continue
            for p, eh in eval_dh.items():
                if bin(h ^ eh).count("1") <= near_duplicate_bits:
                    problems.append(f"{tf} is a near-duplicate of evaluation frame {p}")
                    break
    if problems:
        raise LeakageError(problems)


def guard_training_config(data_yaml: str, field_root: str, *,
                          perceptual: bool = True) -> int:
    """Check a YOLO training config's ``train`` images against a field dataset's
    evaluation splits (see :func:`check_no_eval_leakage`). Returns how many
    images were checked; raises :class:`LeakageError` on any leak, or if the
    field dataset has no split lock (nothing to check against is not a pass)."""
    from modules import yolo_io

    ds = load_dataset(field_root)
    lock = read_lock(field_root)
    if lock is None:
        raise LeakageError([f"{field_root} has no {LOCK_NAME}; assign splits first"])
    data = yolo_io.load_data_yaml(data_yaml)
    train_dir = yolo_io.split_dir(data, "train")
    files = [img for img, _ in yolo_io.iter_image_label_pairs(train_dir)] if train_dir else []
    check_no_eval_leakage(files, ds, lock, perceptual=perceptual)
    return len(files)


def training_frames(ds: FieldDataset, lock: dict, *, min_confidence: str = "probable",
                    reviewed_only: bool = False) -> list[FrameLabel]:
    """DEVELOPMENT frames only — the one source of field training data."""
    order = ["uncertain", "probable", "certain"]
    floor = order.index(min_confidence)
    out = []
    for f in frames_in(ds, lock, DEVELOPMENT):
        if order.index(f.provenance.label_confidence) < floor:
            continue
        if f.provenance.review_decision == "rejected":
            continue
        if reviewed_only and f.provenance.review_decision == "unreviewed":
            continue
        out.append(f)
    return out


def eval_fingerprint(ds: FieldDataset, lock: dict) -> str:
    """Content hash of the evaluation side (group keys + frame paths), recorded
    in every training export so a model can state what it was kept away from."""
    rows = sorted((split_of(ds, lock, f.sequence_id) or "", f.frame_path)
                  for f in ds.frames if split_of(ds, lock, f.sequence_id) in EVAL_SPLITS)
    return "sha256:" + hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def export_training(ds: FieldDataset, lock: dict, out_dir: str, *,
                    class_names=("Plate", "WithHelmet", "WithoutHelmet", "TripleRiding"),
                    min_confidence: str = "probable", perceptual: bool = True) -> dict:
    """Write a YOLO-format training set from DEVELOPMENT frames only, after the
    leakage guard passes. Returns (and writes) the export manifest.

    Classes follow the detector: a plate box is ``Plate``; a rider box is
    ``WithHelmet``/``WithoutHelmet`` by its helmet state, or ``TripleRiding``
    when its vehicle carries three or more riders (the detector's convention).
    Riders with an unknown helmet state are skipped, not guessed."""
    import shutil

    frames = training_frames(ds, lock, min_confidence=min_confidence)
    files = [ds.frame_file(f) for f in frames]
    check_no_eval_leakage(files, ds, lock, perceptual=perceptual)
    idx = {n: i for i, n in enumerate(class_names)}
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    written = skipped = 0
    for f, src in zip(frames, files):
        cam = ds.cameras[ds.sequences[f.sequence_id].camera_id]
        if not cam.resolution or not os.path.exists(src):
            skipped += 1
            continue
        w, h = cam.resolution
        lines = []
        for o in f.objects:
            vehicle = ds.vehicles[(f.sequence_id, o.vehicle_id)]
            if o.role == "plate":
                cls = "Plate"
            elif vehicle.rider_count >= 3:
                cls = "TripleRiding"
            elif o.helmet_state == "helmet":
                cls = "WithHelmet"
            elif o.helmet_state == "no_helmet":
                cls = "WithoutHelmet"
            else:
                continue
            x1, y1, x2, y2 = o.box
            lines.append(f"{idx[cls]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        stem = f"{f.sequence_id}_{f.frame_index:06d}"
        shutil.copyfile(src, os.path.join(img_dir, stem + os.path.splitext(src)[1]))
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        written += 1
    manifest = {
        "dataset": ds.name, "dataset_version": ds.version, "split": DEVELOPMENT,
        "frames": written, "skipped_no_resolution_or_file": skipped,
        "lock_hash": lock["lock_hash"], "eval_fingerprint": eval_fingerprint(ds, lock),
        "leakage_checks": ["group", "sha256"] + (["dhash"] if perceptual else []),
        "min_label_confidence": min_confidence, "classes": list(class_names),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "export_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def coverage(ds: FieldDataset, lock: dict | None = None) -> dict:
    """How many labelled vehicles each split holds per condition value — which
    cells of the field matrix can be measured at all."""
    out: dict = {}
    for v in ds.vehicles.values():
        seq = ds.sequences[v.sequence_id]
        split = split_of(ds, lock, v.sequence_id) if lock else "unassigned"
        cam = ds.cameras.get(seq.camera_id)
        cells = {"lighting": seq.lighting, "weather": seq.weather, "occlusion": v.occlusion,
                 "plate_visibility": v.plate_visibility,
                 "view": cam.view if cam else "unknown", "camera": seq.camera_id}
        for attr, value in cells.items():
            bucket = out.setdefault(split or "unassigned", {}).setdefault(attr, {})
            bucket[value] = bucket.get(value, 0) + 1
    return out
