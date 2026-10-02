"""Field evaluation: the shipped pipeline on labelled field footage.

Two measurements, both model-free at their core (the detector and the OCR
reader are passed in as functions, so tests drive them with fakes):

* :func:`ocr_report` — OCR on real plate crops: last-frame read vs most
  confident read vs the temporal vote, with exact/normalized match, character
  accuracy, edit distance, invalid-format rate, abstention, coverage and
  latency, broken down by lighting, blur, view angle, plate size (distance),
  occlusion and resolution. Reads are taken on the *labelled* plate boxes, so
  this isolates OCR + vote from detection.
* :func:`run_sequence` + :func:`score_sequence` + :func:`condition_matrix` —
  the whole pipeline (detector → association → tracking → OCR → state
  machines → fines) over each labelled sequence, scored against vehicle-level
  truth, per condition. Every missed or wrong fine is attributed to the FIRST
  stage that failed for that vehicle — rider not detected, wrong helmet/triple
  class, track lost, plate not detected, plate not associated, plate misread,
  or the decision rule not confirming — so the bottleneck per condition is a
  count, not an opinion.

Scoring conventions (stated because they decide the numbers):

* A violation is **expected to be fined** only if its plate is visible and
  labelled; a violation with an invisible plate is expected to be *withheld*,
  and a fine on it is counted separately, never as a success.
* Sequences must be **exhaustively labelled** on their labelled frames: a fine
  on a track that matches no labelled vehicle counts as a phantom false fine.
* Matching is IoU ≥ 0.5 between pipeline boxes and labelled boxes.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from modules.association import DetBox
from modules.evaluation import iou  # float boxes (labels), not the int-typed geometry.iou
from modules.field_data import FieldDataset, FrameLabel, VehicleLabel
from modules.ocr_eval import char_accuracy, levenshtein
from modules.plate_info import matches_structure, normalize_plate
from modules.plate_recognizer import correct_plate

RIDER_LABELS = {"WithHelmet", "WithoutHelmet", "TripleRiding"}
VIOLATION_CLASS = {"no_helmet": "WithoutHelmet", "triple_riding": "TripleRiding"}
MATCH_IOU = 0.5
BLUR_ORDER = ["unknown", "sharp", "mild", "severe"]
STAGES = ("detection: rider missed", "detection: wrong class", "tracking: track lost",
          "detection: plate missed", "association: plate mislinked", "ocr: plate misread",
          "decision rule: not confirmed", "decision rule: duplicate")


# -- per-vehicle conditions ---------------------------------------------------------

def _plate_size_bin(width_px: float | None) -> str:
    if width_px is None:
        return "no plate box"
    return "small (<40 px)" if width_px < 40 else "medium (40-80 px)" if width_px < 80 \
        else "large (>=80 px)"


def _resolution_bin(cam) -> str:
    if not cam or not cam.resolution:
        return "unknown"
    h = cam.resolution[1]
    return "<720p" if h < 720 else "720p-1079p" if h < 1080 else ">=1080p"


def frames_by_sequence(ds: FieldDataset) -> dict[str, list[FrameLabel]]:
    out: dict[str, list[FrameLabel]] = {}
    for f in ds.frames:
        out.setdefault(f.sequence_id, []).append(f)
    return out


def vehicle_conditions(ds: FieldDataset, v: VehicleLabel,
                       seq_frames: list[FrameLabel] | None = None) -> dict:
    """The condition cells one labelled vehicle falls in. Pass the sequence's
    frames (``frames_by_sequence``) when calling for many vehicles."""
    seq = ds.sequences[v.sequence_id]
    cam = ds.cameras.get(seq.camera_id)
    if seq_frames is None:
        seq_frames = [f for f in ds.frames if f.sequence_id == v.sequence_id]
    own = [f for f in seq_frames if any(o.vehicle_id == v.vehicle_id for o in f.objects)]
    blur = max((f.blur for f in own), key=BLUR_ORDER.index, default="unknown")
    widths = [o.box[2] - o.box[0] for f in own for o in f.objects
              if o.vehicle_id == v.vehicle_id and o.role == "plate"]
    density = max((len({o.vehicle_id for o in f.objects}) for f in own), default=1)
    crossing = False
    for f in own:
        mine = [o.box for o in f.objects if o.vehicle_id == v.vehicle_id and o.role == "rider"]
        others = [o.box for o in f.objects if o.vehicle_id != v.vehicle_id and o.role == "rider"]
        if any(iou(a, b) > 0.1 for a in mine for b in others):
            crossing = True
            break
    return {
        "lighting": seq.lighting, "weather": seq.weather,
        "view angle": cam.view if cam else "unknown", "occlusion": v.occlusion,
        "plate visibility": v.plate_visibility, "blur (worst frame)": blur,
        "plate size (distance)": _plate_size_bin(statistics.median(widths) if widths
                                                 else None),
        "resolution": _resolution_bin(cam),
        "vehicles in frame": "1" if density <= 1 else "2-3" if density <= 3 else "4+",
        "crossing": "crossing" if crossing else "separate",
        "camera": seq.camera_id,
    }


# -- OCR on labelled plate crops -------------------------------------------------------

@dataclass
class PlateRead:
    text: str | None
    confidence: float
    latency_ms: float


def as_plate(raw: str | None) -> str:
    """What a single read becomes after the same cleaning + look-alike
    correction the stabilizer applies to each observation."""
    norm = normalize_plate(raw or "")
    if not norm:
        return ""
    corrected, _ = correct_plate(norm)
    return corrected or norm


def collect_plate_reads(ds: FieldDataset, vehicles: list[VehicleLabel], read_fn,
                        load_image) -> dict:
    """``read_fn(image, box) -> (text, conf) | None`` on every labelled plate box
    of ``vehicles``, in frame order, timed. ``load_image(frame) -> image``."""
    wanted = {v.key for v in vehicles}
    reads: dict[tuple[str, str], list[PlateRead]] = {k: [] for k in wanted}
    for f in ds.frames:
        plates = [o for o in f.objects if o.role == "plate"
                  and (f.sequence_id, o.vehicle_id) in wanted]
        if not plates:
            continue
        image = load_image(f)
        for o in plates:
            t0 = time.perf_counter()
            res = read_fn(image, tuple(int(round(x)) for x in o.box))
            ms = (time.perf_counter() - t0) * 1000.0
            text, conf = res if res else (None, 0.0)
            reads[(f.sequence_id, o.vehicle_id)].append(PlateRead(text, float(conf), ms))
    return reads


def _decide(policy: str, reads: list[PlateRead]) -> tuple[str, str, bool]:
    """``(raw_output, plate, abstained)`` for one vehicle under one policy."""
    from modules.plate_recognizer import PlateStabilizer

    usable = [r for r in reads if r.text]
    if policy == "last":
        r = usable[-1] if usable else None
        return (r.text or "", as_plate(r.text), False) if r else ("", "", False)
    if policy == "best_conf":
        r = max(usable, key=lambda x: x.confidence) if usable else None
        return (r.text or "", as_plate(r.text), False) if r else ("", "", False)
    if policy == "temporal":
        stab = PlateStabilizer()
        for r in reads:
            stab.add(r.text, conf=r.confidence)
        res = stab.result()
        return (res.stable or "", res.stable or "", not res.stable)
    raise ValueError(policy)


POLICIES = ("last", "best_conf", "temporal")


def _read_verbatim(truth: str, raw: str | None) -> bool:
    """The OCR produced the true plate itself — spaces/case aside, before any
    look-alike correction."""
    return bool(truth) and normalize_plate(raw or "") == truth


def _ocr_metrics(policy: str, truths: list[str], outs: list[tuple[str, str, bool]],
                 reads: list[list[PlateRead]]) -> dict:
    """One policy's metrics, defined the same way for every policy:

    * ``exact_match`` — right AND read verbatim: for a single-frame policy its
      chosen read; for the vote, at least one of the vehicle's reads.
    * ``normalized_match`` — the named plate (after the stabilizer's cleaning
      and look-alike correction) is right.
    * ``no_answer_rate`` — no plate named (1 - coverage), for any reason;
      ``abstention_rate`` — declined by rule (only the vote does that).
    * character accuracy, edit distance and invalid-format rate are over the
      vehicles a plate was named for."""
    n = len(truths)
    if n == 0:
        return {"vehicles": 0}
    rows = list(zip(truths, outs, reads))
    answered = [(t, raw, p, rs) for t, (raw, p, ab), rs in rows if not ab and p]
    right = [(t, raw, p, rs) for t, raw, p, rs in answered if p == t]
    if policy == "temporal":
        exact = sum(1 for t, _, _, rs in right if any(_read_verbatim(t, r.text) for r in rs))
    else:
        exact = sum(1 for t, raw, _, _ in right if _read_verbatim(t, raw))
    return {
        "vehicles": n,
        "coverage": round(len(answered) / n, 4),
        "no_answer_rate": round(1 - len(answered) / n, 4),
        "abstention_rate": round(sum(1 for _, (_, _, ab), _ in rows if ab) / n, 4),
        "exact_match": round(exact / n, 4),
        "normalized_match": round(len(right) / n, 4),
        "wrong_plate_rate": round((len(answered) - len(right)) / n, 4),
        "accuracy_when_answered": round(len(right) / len(answered), 4) if answered else None,
        "char_accuracy": (round(sum(char_accuracy(t, p) for t, _, p, _ in answered)
                                / len(answered), 4) if answered else None),
        "mean_edit_distance": (round(sum(levenshtein(t, p) for t, _, p, _ in answered)
                                     / len(answered), 3) if answered else None),
        "invalid_rate": (round(sum(1 for *_, p, _ in answered if not matches_structure(p))
                               / len(answered), 4) if answered else None),
    }


def ocr_report(ds: FieldDataset, vehicles: list[VehicleLabel], reads: dict) -> dict:
    """Policy metrics overall and per condition attribute, plus read latency.
    Only vehicles with a human-read plate and at least one read are scored."""
    scored = [v for v in vehicles if v.plate_text and reads.get(v.key)]
    outs = {p: {v.key: _decide(p, reads[v.key]) for v in scored} for p in POLICIES}
    lat = sorted(r.latency_ms for v in scored for r in reads[v.key])

    def block(subset):
        return {p: _ocr_metrics(p, [v.plate_text for v in subset],
                                [outs[p][v.key] for v in subset],
                                [reads[v.key] for v in subset]) for p in POLICIES}

    by_seq = frames_by_sequence(ds)
    conds = {v.key: vehicle_conditions(ds, v, by_seq.get(v.sequence_id, [])) for v in scored}
    attrs = ("lighting", "blur (worst frame)", "view angle", "plate size (distance)",
             "occlusion", "resolution")
    by: dict = {}
    for attr in attrs:
        by[attr] = {}
        for value in sorted({conds[v.key][attr] for v in scored}):
            by[attr][value] = block([v for v in scored if conds[v.key][attr] == value])
    return {
        "vehicles_scored": len(scored),
        "vehicles_skipped_no_plate_text_or_reads": len(vehicles) - len(scored),
        "reads": len(lat),
        "latency_ms": ({"mean": round(sum(lat) / len(lat), 2),
                        "p90": round(lat[int(0.9 * (len(lat) - 1))], 2)} if lat else None),
        "overall": block(scored), "by_condition": by,
        "scoring": "temporal = the shipped PlateStabilizer over all of a vehicle's reads; "
                   "single-frame policies get the same cleaning + look-alike correction",
    }


# -- end to end over labelled sequences --------------------------------------------------

@dataclass
class SequenceRun:
    sequence_id: str
    frames: dict = field(default_factory=dict)  # labelled frame_index -> record
    decisions: list = field(default_factory=list)
    confirmed: dict = field(default_factory=dict)  # track_id -> set of violations
    stable_plates: dict = field(default_factory=dict)  # track_id -> elected plate
    withheld: list = field(default_factory=list)
    frames_run: int = 0


def iter_frames(ds: FieldDataset, sequence_id: str):
    """``(frame_index, image)`` for every frame of a sequence, labelled or not —
    the temporal logic needs them all. ``frames_dir`` files are indexed by their
    numeric stem (else sorted position); a video by decode order."""
    import cv2

    seq = ds.sequences[sequence_id]
    if seq.frames_dir:
        d = os.path.join(ds.root, seq.frames_dir)
        names = sorted(n for n in os.listdir(d)
                       if os.path.splitext(n)[1].lower() in {".jpg", ".jpeg", ".png"})
        for pos, name in enumerate(names):
            stem = os.path.splitext(name)[0]
            idx = int(stem) if stem.isdigit() else pos
            img = cv2.imread(os.path.join(d, name))
            if img is not None:
                yield idx, img
        return
    cap = cv2.VideoCapture(os.path.join(ds.root, seq.video_path or ""))
    idx = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield idx, img
            idx += 1
    finally:
        cap.release()


def run_sequence(ds: FieldDataset, sequence_id: str, frames, detect, read_plate,
                 config=None) -> SequenceRun:
    """Drive the shipped ``ViolationPipeline`` over ``frames`` (``(index,
    image)``). ``detect(image) -> [DetBox]``; ``read_plate(image, box)``."""
    from modules.pipeline import DEFAULT_PIPELINE_CONFIG, ViolationPipeline

    seq = ds.sequences[sequence_id]
    labelled = {f.frame_index for f in ds.frames if f.sequence_id == sequence_id}
    pipe = ViolationPipeline(config or DEFAULT_PIPELINE_CONFIG)
    run = SequenceRun(sequence_id)
    for idx, image in frames:
        dets: list[DetBox] = detect(image)
        def reader(box, _im=image):
            return read_plate(_im, box)

        res = pipe.step(idx, dets, timestamp=idx / seq.fps, read_plate=reader)
        run.frames_run += 1
        for d in res.decisions:
            run.decisions.append({"track_id": d.track.track_id, "violation": d.violation,
                                  "plate": d.plate, "frame_index": d.frame_index})
        if idx in labelled:
            run.frames[idx] = {
                "dets": [(d.label, tuple(d.box)) for d in dets],
                "tracks": [(tf.track.track_id,
                            tuple(tf.track.body.box) if tf.track.body else None,
                            tuple(tf.track.plate_box) if tf.plate_visible and
                            tf.track.plate_box else None) for tf in res.tracks],
            }
    for tid in pipe.confirmed_track_ids:
        run.confirmed[tid] = pipe.confirmed_violations(tid)
        stab = pipe.stabilizers.get(tid)
        if stab is not None:
            run.stable_plates[tid] = stab.result().stable
    run.withheld = pipe.unfined_confirmations()
    return run


def _best(box, candidates):
    """Index of the candidate box with the highest IoU >= MATCH_IOU, else None."""
    best, best_i = MATCH_IOU, None
    for i, c in enumerate(candidates):
        if c is None:
            continue
        v = iou(box, c)
        if v >= best:
            best, best_i = v, i
    return best_i


STATE_CLASS = {"helmet": "WithHelmet", "no_helmet": "WithoutHelmet"}
ISSUED = ("correct fine", "wrong plate", "duplicate fine", "false fine",
          "fined without a visible plate")


def score_sequence(ds: FieldDataset, run: SequenceRun) -> dict:
    """Per labelled vehicle: stage-level measurements; one outcome per issued
    fine (correct / wrong plate / duplicate / false / fined without a visible
    plate) and one per labelled violation left unfined (missed / correctly
    withheld); for every error, the first failing stage. Plus phantom fines."""
    sid = run.sequence_id
    vehicles = [v for v in ds.vehicles.values() if v.sequence_id == sid]
    frames: dict[int, FrameLabel] = {f.frame_index: f for f in ds.frames
                                     if f.sequence_id == sid and f.frame_index in run.frames}
    track_votes: dict[int, dict[str, int]] = {}
    per_vehicle: dict[str, dict[str, Any]] = {}
    for v in vehicles:
        m: dict[str, Any] = {
            "rider_frames": 0, "rider_detected": 0, "class_frames": 0, "class_right": 0,
            "plate_frames": 0, "plate_detected": 0, "assoc_frames": 0, "assoc_right": 0,
            "tracks": [], "class_by_state": {}}
        per_vehicle[v.vehicle_id] = m
        for idx, f in frames.items():
            rec = run.frames[idx]
            riders = [o for o in f.objects if o.vehicle_id == v.vehicle_id and o.role == "rider"]
            plates = [o.box for o in f.objects if o.vehicle_id == v.vehicle_id
                      and o.role == "plate"]
            rider_dets = [(lab, b) for lab, b in rec["dets"] if lab in RIDER_LABELS]
            track_bodies = [t[1] for t in rec["tracks"]]
            main_track = None
            for o in riders:
                m["rider_frames"] += 1
                # Each rider is scored against its OWN state (a helmeted pillion
                # on a no-helmet vehicle is right to be called WithHelmet).
                state = "triple" if v.rider_count >= 3 else (o.helmet_state or "unknown")
                want = "TripleRiding" if state == "triple" else STATE_CLASS.get(state)
                j = _best(o.box, [b for _, b in rider_dets])
                if j is not None:
                    m["rider_detected"] += 1
                    if want:
                        right = rider_dets[j][0] == want
                        m["class_frames"] += 1
                        m["class_right"] += right
                        cell = m["class_by_state"].setdefault(state, [0, 0])
                        cell[0] += 1
                        cell[1] += right
                t = _best(o.box, track_bodies)
                if t is not None:
                    main_track = rec["tracks"][t][0]
                    m["tracks"].append(main_track)
                    track_votes.setdefault(main_track, {}).setdefault(v.vehicle_id, 0)
                    track_votes[main_track][v.vehicle_id] += 1
            for gp in plates:
                m["plate_frames"] += 1
                if _best(gp, [b for lab, b in rec["dets"] if lab == "Plate"]) is not None:
                    m["plate_detected"] += 1
                    if main_track is not None:
                        m["assoc_frames"] += 1
                        tplate = next((t[2] for t in rec["tracks"] if t[0] == main_track), None)
                        m["assoc_right"] += bool(tplate and iou(gp, tplate) >= MATCH_IOU)
    owner = {tid: max(votes.items(), key=lambda kv: kv[1])[0]
             for tid, votes in track_votes.items()}

    outcomes = []
    phantom = []
    fines_by_vehicle: dict[str, list] = {}
    for d in run.decisions:
        vid = owner.get(d["track_id"])
        if vid is None:
            phantom.append(d)
        else:
            fines_by_vehicle.setdefault(vid, []).append(d)

    for v in vehicles:
        m = per_vehicle[v.vehicle_id]
        tracks = m["tracks"]
        main = max(set(tracks), key=tracks.count) if tracks else None
        fines = fines_by_vehicle.get(v.vehicle_id, [])
        plate_ok = bool(v.plate_text) and v.plate_visibility != "none"

        def add(viol, kind, _v=v, _m=m, _main=main):
            stage = (_first_failure(_v, viol, _m, _main, run, kind)
                     if kind not in ("correct fine", "correctly withheld") else None)
            outcomes.append({"vehicle_id": _v.vehicle_id, "violation": viol, "outcome": kind,
                             "stage": stage})

        for viol in sorted(v.violations | {f["violation"] for f in fines}):
            issued = [f for f in fines if f["violation"] == viol]
            if viol not in v.violations:
                for _ in issued:
                    add(viol, "false fine")
            elif not plate_ok:
                for _ in issued:
                    add(viol, "fined without a visible plate")
                if not issued:
                    add(viol, "correctly withheld")
            else:
                right = [f for f in issued if f["plate"] == v.plate_text]
                add(viol, "correct fine" if right else "missed")
                for _ in right[1:]:
                    add(viol, "duplicate fine")
                for _ in (f for f in issued if f["plate"] != v.plate_text):
                    add(viol, "wrong plate")
    return {"sequence_id": sid, "vehicles": per_vehicle, "outcomes": outcomes,
            "phantom_fines": phantom, "withheld": run.withheld}


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def _low(num: int, den: int) -> bool:
    """A stage failed for this vehicle: it had frames to be judged on, and got
    fewer than half right. No applicable frames is not a failure."""
    return den > 0 and num / den < 0.5


def _first_failure(v, viol, m, main, run, kind) -> str:
    by_state = m["class_by_state"]
    if kind == "false fine":
        # A helmeted rider called bare-headed (or two riders called triple): the
        # detector is to blame if it mostly got the innocent riders' class wrong.
        # (For a no-helmet false fine, bare-headed riders are not innocent.)
        innocent = [c for st, c in by_state.items()
                    if not (viol == "no_helmet" and st == "no_helmet")]
        seen = sum(n for n, _r in innocent)
        wrong = sum(n - r for n, r in innocent)
        if seen and wrong / seen >= 0.5:
            return "detection: wrong class"
        return "decision rule: not confirmed"
    if kind == "duplicate fine":
        return "decision rule: duplicate"
    if kind == "fined without a visible plate":
        return "association: plate mislinked"  # its plate was never visible: it came from elsewhere
    if _low(m["rider_detected"], m["rider_frames"]):
        return "detection: rider missed"
    state = {"no_helmet": "no_helmet", "triple_riding": "triple"}.get(viol)
    if state and state in by_state and _low(by_state[state][1], by_state[state][0]):
        return "detection: wrong class"
    if main is None or _low(m["tracks"].count(main), m["rider_frames"]):
        return "tracking: track lost"
    if kind == "missed" and viol not in run.confirmed.get(main, set()):
        return "decision rule: not confirmed"
    if _low(m["plate_detected"], m["plate_frames"]):
        return "detection: plate missed"
    if _low(m["assoc_right"], m["assoc_frames"]):
        return "association: plate mislinked"
    return "ocr: plate misread"


def condition_matrix(ds: FieldDataset, scores: list[dict]) -> dict:
    """Aggregate per-sequence scores into one row per condition value."""
    by_seq = frames_by_sequence(ds)
    cond = {v.key: vehicle_conditions(ds, v, by_seq.get(v.sequence_id, []))
            for v in ds.vehicles.values()}

    def empty():
        return {"vehicles": 0, "expected_fines": 0, "correct": 0, "wrong_plate": 0,
                "missed": 0, "false_fines": 0, "duplicate_fines": 0, "correctly_withheld": 0,
                "fined_without_visible_plate": 0, "rider_frames": 0, "rider_detected": 0,
                "class_frames": 0, "class_right": 0, "plate_frames": 0, "plate_detected": 0,
                "assoc_frames": 0, "assoc_right": 0, "track_fragments": 0,
                "stages": {s: 0 for s in STAGES}}

    table: dict = {}
    overall = empty()
    phantom = 0
    kind_key = {"correct fine": "correct", "wrong plate": "wrong_plate", "missed": "missed",
                "false fine": "false_fines", "duplicate fine": "duplicate_fines",
                "correctly withheld": "correctly_withheld",
                "fined without a visible plate": "fined_without_visible_plate"}
    for sc in scores:
        phantom += len(sc["phantom_fines"])
        for vid, m in sc["vehicles"].items():
            key = (sc["sequence_id"], vid)
            rows = [overall] + [table.setdefault(a, {}).setdefault(val, empty())
                                for a, val in cond[key].items()]
            outs = [o for o in sc["outcomes"] if o["vehicle_id"] == vid]
            for row in rows:
                row["vehicles"] += 1
                for k in ("rider_frames", "rider_detected", "class_frames", "class_right",
                          "plate_frames", "plate_detected", "assoc_frames", "assoc_right"):
                    row[k] += m[k]
                row["track_fragments"] += max(0, len(set(m["tracks"])) - 1)
                for o in outs:
                    row[kind_key[o["outcome"]]] += 1
                    # One "correct fine" or "missed" per labelled violation with a
                    # visible plate; extra issued fines are counted, not expected.
                    if o["outcome"] in ("correct fine", "missed"):
                        row["expected_fines"] += 1
                    if o["stage"]:
                        row["stages"][o["stage"]] += 1

    def finish(row, phantoms=0):
        issued = (row["correct"] + row["wrong_plate"] + row["false_fines"]
                  + row["duplicate_fines"] + row["fined_without_visible_plate"] + phantoms)
        row["fines_issued"] = issued
        row["fine_precision"] = round(row["correct"] / issued, 4) if issued else None
        row["fine_recall"] = (round(row["correct"] / row["expected_fines"], 4)
                              if row["expected_fines"] else None)
        row["rider_recall"] = round(_rate(row["rider_detected"], row["rider_frames"]), 4)
        row["class_accuracy"] = round(_rate(row["class_right"], row["class_frames"]), 4)
        row["plate_recall"] = round(_rate(row["plate_detected"], row["plate_frames"]), 4)
        row["association_accuracy"] = round(_rate(row["assoc_right"], row["assoc_frames"]), 4)
        errs = row["stages"]
        worst = max(errs, key=errs.get) if any(errs.values()) else None
        row["bottleneck"] = worst
        row["bottleneck_errors"] = errs[worst] if worst else 0
        return row

    overall["phantom_fines"] = phantom
    overall = finish(overall, phantom)  # per-condition rows can't hold phantoms
    return {"overall": overall,
            "by_condition": {a: {val: finish(r) for val, r in sorted(vals.items())}
                             for a, vals in table.items()}}
