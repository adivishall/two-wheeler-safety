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
        if visibility == "full" and not plate:
            issues.append(f"{where}: a fully visible plate needs its plate_text (use "
                          "'partial' if a human can't read it) — otherwise a missed fine "
                          "is scored as correctly withheld")
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
        elif (fdir := sequences[sid].frames_dir) and os.path.dirname(path) == os.path.normpath(
                fdir):
            stem = os.path.splitext(os.path.basename(path))[0]
            if not stem.isdigit() or int(stem) != idx:
                issues.append(f"{where}: files in a frames_dir must be named by frame index "
                              f"({path!r} for frame {idx}); the evaluator indexes by name")
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERCEPTUAL_BITS = 5
BOX_MATCH_IOU = 0.7


def _hash_unit(salt: str, key: str) -> float:
    digest = hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def _lock_digest(lock: dict) -> str:
    body = {k: lock.get(k) for k in ("group_by", "salt", "external_cameras", "assignments",
                                     "sequences", "eval_content", "eval_objects")}
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


def verify_lock(ds: FieldDataset, lock: dict) -> list[str]:
    """Problems that make the lock's view of the dataset untrustworthy: a locked
    sequence whose group changed (its camera or date was edited — it would be
    re-hashed into another split) or that disappeared (renamed or removed — its
    frames could come back under a new name in another split)."""
    problems = []
    for sid, rec in sorted(lock.get("sequences", {}).items()):
        if sid not in ds.sequences:
            problems.append(f"sequence {sid} ({rec['split']}) is in the lock but no longer in "
                            "the dataset — renamed or removed after assignment")
        elif ds.group_key(sid, lock["group_by"]) != rec["group"]:
            problems.append(f"sequence {sid} ({rec['split']}) moved from group {rec['group']} "
                            f"to {ds.group_key(sid, lock['group_by'])} after assignment")
    return problems


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_box(box, w: int, h: int) -> list[float]:
    return [round(box[0] / w, 4), round(box[1] / h, 4), round(box[2] / w, 4),
            round(box[3] / h, 4)]


def _crop_dhash(image, nbox) -> int | None:
    """Perceptual hash of one object's crop (normalized box, 10% padding) — a
    re-encoded or resized copy keeps it; another moment of the same fixed
    camera doesn't, unlike a whole-frame hash that matches the background."""
    from modules.dataset_audit import dhash

    h, w = image.shape[:2]
    x1, y1, x2, y2 = nbox
    px, py = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    c = image[max(0, int((y1 - py) * h)):min(h, int((y2 + py) * h)),
              max(0, int((x1 - px) * w)):min(w, int((x2 + px) * w))]
    if c.size == 0 or min(c.shape[:2]) < 4:
        return None
    return dhash(c)


def object_signature(path: str, nboxes: list[list[float]]) -> list[list]:
    """``[[x1, y1, x2, y2, dhash], ...]`` (normalized boxes) for one image."""
    import cv2

    img = cv2.imread(path)
    if img is None:
        return []
    out = []
    for nb in nboxes:
        dh = _crop_dhash(img, nb)
        if dh is not None:
            out.append([*nb, dh])
    return out


def _signatures_match(train_sig: list, eval_sig: list, bits: int) -> bool:
    """Same frame, re-encoded: every object in the training image has an
    evaluation object at the same place (IoU >= 0.7) with a near-identical crop."""
    from modules.evaluation import iou

    if not train_sig or not eval_sig:
        return False
    for tb in train_sig:
        if not any(iou(tb[:4], eb[:4]) >= BOX_MATCH_IOU and bin(tb[4] ^ eb[4]).count("1") <= bits
                   for eb in eval_sig):
            return False
    return True


def _sequence_files(ds: FieldDataset, sid: str) -> list[str]:
    """Every file of a sequence: its labelled frames and, for a frame directory,
    every image in it (unlabelled frames of an evaluation sequence are still
    evaluation footage)."""
    files = {ds.frame_file(f) for f in ds.frames if f.sequence_id == sid}
    seq = ds.sequences[sid]
    if seq.frames_dir:
        d = os.path.join(ds.root, seq.frames_dir)
        if os.path.isdir(d):
            files |= {os.path.join(d, n) for n in os.listdir(d)
                      if os.path.splitext(n)[1].lower() in IMAGE_EXTS}
    return sorted(os.path.normpath(f) for f in files)


def _record_eval_content(ds: FieldDataset, lock: dict) -> None:
    """Append (never remove) the byte hash of every evaluation file and the
    object signature of every labelled evaluation frame, so the guard still
    works if a file is later renamed, moved or deleted."""
    content = lock.setdefault("eval_content", {})
    objects = lock.setdefault("eval_objects", {})
    for sid in ds.sequences:
        split = split_of(ds, lock, sid)
        if split not in EVAL_SPLITS:
            continue
        for path in _sequence_files(ds, sid):
            if os.path.exists(path):
                content.setdefault(_sha256(path),
                                   {"split": split, "path": os.path.relpath(path, ds.root)})
        for f in ds.frames:
            if f.sequence_id != sid or not f.objects:
                continue
            path = ds.frame_file(f)
            rel = os.path.relpath(path, ds.root)
            cam = ds.cameras.get(ds.sequences[sid].camera_id)
            if rel in objects or not os.path.exists(path) or not cam or not cam.resolution:
                continue
            w, h = cam.resolution
            objects[rel] = object_signature(path, [_norm_box(o.box, w, h) for o in f.objects])


def assign_splits(ds: FieldDataset, *, salt: str | None = None, ratios: dict | None = None,
                  external_cameras=(), group_by: str = "sequence",
                  write: bool = True) -> dict:
    """Assign every not-yet-assigned sequence to a split and return the lock.

    The lock records each sequence's split and group; a sequence can never move,
    and if one was renamed, removed or re-grouped (camera or date edited) after
    assignment this refuses to continue (:func:`verify_lock`). Changing
    ``ratios`` only affects new groups. EXTERNAL cameras are fixed when first
    named; naming one after its footage was assigned elsewhere is refused. The
    byte hashes and object signatures of evaluation files are appended to the
    lock, so the guard doesn't depend on file names staying put."""
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
                "external_cameras": sorted(external_cameras), "assignments": {},
                "sequences": {}, "eval_content": {}, "eval_objects": {}}
    else:
        if group_by != lock["group_by"]:
            raise ValueError(f"lock groups by {lock['group_by']!r}; can't switch to "
                             f"{group_by!r} without re-splitting everything")
        problems = verify_lock(ds, lock)
        if problems:
            raise LeakageError(problems)
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
        if sid in lock["sequences"]:
            continue
        key = ds.group_key(sid, lock["group_by"])
        if key not in lock["assignments"]:
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
        lock["sequences"][sid] = {"group": key, "split": lock["assignments"][key]["split"]}
    _record_eval_content(ds, lock)
    lock["lock_hash"] = _lock_digest(lock)
    if write:
        with open(os.path.join(ds.root, LOCK_NAME), "w") as fh:
            json.dump(lock, fh, indent=2, sort_keys=True)
    return lock


def split_of(ds: FieldDataset, lock: dict, sequence_id: str) -> str | None:
    """The locked split of a sequence (None if it was never assigned)."""
    rec = lock.get("sequences", {}).get(sequence_id)
    return rec["split"] if rec else None


def frames_in(ds: FieldDataset, lock: dict, split: str) -> list[FrameLabel]:
    return [f for f in ds.frames if split_of(ds, lock, f.sequence_id) == split]


def vehicles_in(ds: FieldDataset, lock: dict, split: str) -> list[VehicleLabel]:
    return [v for v in ds.vehicles.values() if split_of(ds, lock, v.sequence_id) == split]


# -- leakage guard and training export ------------------------------------------

def check_no_eval_leakage(train_files: list[str], ds: FieldDataset, lock: dict, *,
                          train_boxes: dict | None = None,
                          near_duplicate_bits: int = PERCEPTUAL_BITS,
                          perceptual: bool = True) -> dict:
    """Raise :class:`LeakageError` if any training file IS evaluation data.

    Checks, because each alone has a hole: the lock is still consistent with
    the dataset; the file is not an evaluation file (labelled frame, or any
    image in an evaluation sequence's frame directory); its bytes match no
    evaluation file ever recorded in the lock (a renamed copy; a deleted
    original); and — when its boxes are known (``train_boxes``: path ->
    normalized boxes) — its objects don't all reappear at the same places with
    near-identical crops in one evaluation frame (a re-encode or resize). An
    evaluation frame that is missing from disk and was never hashed makes the
    check fail closed. Returns counts of what was checked."""
    problems = verify_lock(ds, lock)
    eval_seqs = [sid for sid in ds.sequences if split_of(ds, lock, sid) in EVAL_SPLITS]
    eval_paths = {p for sid in eval_seqs for p in _sequence_files(ds, sid)}
    known = dict(lock.get("eval_content", {}))
    known_paths = {os.path.normpath(os.path.join(ds.root, v["path"])) for v in known.values()}
    for p in sorted(eval_paths):
        if os.path.exists(p):
            known.setdefault(_sha256(p), {"path": os.path.relpath(p, ds.root)})
        elif p not in known_paths:
            problems.append(f"evaluation frame {p} is missing and was never hashed into the "
                            "lock; can't verify training data against it")
    eval_paths |= known_paths
    eval_objects = lock.get("eval_objects", {}) if perceptual else {}
    perceptual_checked = 0
    for tf in train_files:
        norm = os.path.normpath(tf)
        if norm in eval_paths:
            problems.append(f"{tf} is an evaluation frame")
            continue
        if not os.path.exists(tf):
            continue
        digest = _sha256(tf)
        if digest in known:
            problems.append(f"{tf} is byte-identical to evaluation file {known[digest]['path']}")
            continue
        if eval_objects and train_boxes and train_boxes.get(tf):
            perceptual_checked += 1
            sig = object_signature(tf, train_boxes[tf])
            hit = next((rel for rel, esig in eval_objects.items()
                        if _signatures_match(sig, esig, near_duplicate_bits)), None)
            if hit:
                problems.append(f"{tf} is a near-duplicate of evaluation frame {hit}")
    if problems:
        raise LeakageError(problems)
    return {"files": len(train_files), "perceptual_checked": perceptual_checked}


def _yolo_sources(data: dict, split: str) -> list[str]:
    """Every image a YOLO config's ``split`` entry names: a directory, a
    ``.txt`` list of image paths, or a list of either."""
    from modules import yolo_io

    entry = data["cfg"].get(split)
    if not entry:
        return []
    root = data["root"]
    entries = entry if isinstance(entry, list) else [entry]
    files: list[str] = []
    for e in entries:
        path = e if os.path.isabs(e) else os.path.join(root, e)
        if os.path.isdir(path):
            files += [img for img, _ in yolo_io.iter_image_label_pairs(path)]
        elif path.endswith(".txt") and os.path.exists(path):
            base = os.path.dirname(path)
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        files.append(line if os.path.isabs(line) else
                                     os.path.normpath(os.path.join(base, line)))
        elif os.path.isfile(path):
            files.append(path)
    return files


def _yolo_boxes(image_path: str) -> list[list[float]]:
    """Normalized xyxy boxes from the YOLO label file beside an image."""
    parts = image_path.replace("\\", "/").split("/")
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
    label = os.path.splitext("/".join(parts))[0] + ".txt"
    boxes = []
    if os.path.exists(label):
        with open(label) as fh:
            for line in fh:
                v = line.split()
                if len(v) >= 5:
                    cx, cy, bw, bh = (float(x) for x in v[1:5])
                    boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return boxes


def guard_training_config(data_yaml: str, field_root: str, *,
                          perceptual: bool = True) -> int:
    """Check every image a YOLO training config trains on (``train``) or selects
    checkpoints with (``val``) against a field dataset's evaluation data. Raises
    :class:`LeakageError` on a leak, on a missing split lock, or when no
    training image can be found (a guard that checked nothing has not passed)."""
    from modules import yolo_io

    ds = load_dataset(field_root)
    lock = read_lock(field_root)
    if lock is None:
        raise LeakageError([f"{field_root} has no {LOCK_NAME}; assign splits first"])
    data = yolo_io.load_data_yaml(data_yaml)
    train = _yolo_sources(data, "train")
    if not train:
        raise LeakageError([f"no training images found from {data_yaml}; nothing was checked"])
    files = sorted(set(train) | set(_yolo_sources(data, "val")))
    boxes = {f: _yolo_boxes(f) for f in files} if perceptual else None
    check_no_eval_leakage(files, ds, lock, train_boxes=boxes, perceptual=perceptual)
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
    """Content hash of the evaluation side (every evaluation file's bytes ever
    recorded), stored in each training export so a model can state what it was
    kept away from."""
    body = sorted(lock.get("eval_content", {}))
    return "sha256:" + hashlib.sha256(json.dumps(body).encode()).hexdigest()[:16]


def _yolo_class(ds: FieldDataset, f: FrameLabel, o: FrameObject) -> str | None:
    vehicle = ds.vehicles[(f.sequence_id, o.vehicle_id)]
    if o.role == "plate":
        return "Plate"
    if vehicle.rider_count >= 3:
        return "TripleRiding"
    return {"helmet": "WithHelmet", "no_helmet": "WithoutHelmet"}.get(o.helmet_state or "")


def export_training(ds: FieldDataset, lock: dict, out_dir: str, *,
                    class_names=("Plate", "WithHelmet", "WithoutHelmet", "TripleRiding"),
                    min_confidence: str = "probable", perceptual: bool = True) -> dict:
    """Write a YOLO-format training set from DEVELOPMENT frames only, after the
    leakage guard passes. Returns (and writes) the export manifest.

    Classes follow the detector: a plate box is ``Plate``; a rider box is
    ``WithHelmet``/``WithoutHelmet`` by its helmet state, or ``TripleRiding``
    when its vehicle carries three or more riders. A frame with a rider whose
    helmet state is unknown is skipped whole: dropping only that box would
    teach the detector the rider is background."""
    import shutil

    frames = training_frames(ds, lock, min_confidence=min_confidence)
    usable, unknown = [], 0
    for f in frames:
        if any(_yolo_class(ds, f, o) is None for o in f.objects):
            unknown += 1
        else:
            usable.append(f)
    files = [ds.frame_file(f) for f in usable]
    boxes = {}
    for f, path in zip(usable, files):
        cam = ds.cameras[ds.sequences[f.sequence_id].camera_id]
        if cam.resolution:
            boxes[path] = [_norm_box(o.box, *cam.resolution) for o in f.objects]
    checked = check_no_eval_leakage(files, ds, lock, train_boxes=boxes, perceptual=perceptual)
    idx = {n: i for i, n in enumerate(class_names)}
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    written = skipped = 0
    for f, src in zip(usable, files):
        cam = ds.cameras[ds.sequences[f.sequence_id].camera_id]
        if not cam.resolution or not os.path.exists(src):
            skipped += 1
            continue
        w, h = cam.resolution
        lines = []
        for o in f.objects:
            cls = _yolo_class(ds, f, o)
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
        "skipped_unknown_helmet_state": unknown,
        "lock_hash": lock["lock_hash"], "eval_fingerprint": eval_fingerprint(ds, lock),
        "leakage_checks": ["lock consistency", "evaluation path", "sha256 (incl. lock history)"]
        + (["object-crop dHash"] if perceptual else []),
        "perceptual_checked": checked["perceptual_checked"],
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
