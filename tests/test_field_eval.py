"""Field evaluation: the OCR policy comparison on real-read style inputs, and the
end-to-end matrix that charges every wrong or missed fine to the first stage
that failed. The detector and reader are fakes driven by the labels, so each
test breaks exactly one stage and checks that stage takes the blame."""

import field_fixture
import pytest

from modules.association import DetBox
from modules.field_data import load_dataset
from modules.field_eval import (
    PlateRead,
    condition_matrix,
    ocr_report,
    run_sequence,
    score_sequence,
    vehicle_conditions,
)

W, H = 640, 480
BOXES = {"v1": {"rider": [100, 100, 180, 300], "plate": [115, 300, 165, 318]},
         "v2": {"rider": [400, 100, 480, 300], "plate": [415, 300, 465, 318]}}
PLATES = {"v1": "MH12AB1234", "v2": "KA05CD5678"}
N = 30


def _dataset(tmp_path, labelled_every=3):
    cams = [{"camera_id": "cam", "view": "rear", "resolution": [W, H], "fps": 25}]
    seqs = [{"sequence_id": "s1", "camera_id": "cam", "fps": 25, "frames_dir": "f",
             "lighting": "night", "weather": "clear"}]
    vehicles = [
        {"sequence_id": "s1", "vehicle_id": "v1", "first_frame": 0, "last_frame": N - 1,
         "plate_text": PLATES["v1"], "plate_visibility": "full", "rider_count": 1,
         "helmet_states": ["no_helmet"], "violations": ["no_helmet"], "occlusion": "none",
         "annotator": "a", "review_decision": "accepted"},
        {"sequence_id": "s1", "vehicle_id": "v2", "first_frame": 0, "last_frame": N - 1,
         "plate_text": PLATES["v2"], "plate_visibility": "full", "rider_count": 1,
         "helmet_states": ["helmet"], "violations": [], "occlusion": "partial",
         "annotator": "a", "review_decision": "accepted"},
    ]
    frames = [{"sequence_id": "s1", "frame_index": i, "frame_path": f"f/{i:06d}.png",
               "blur": "sharp", "annotator": "a",
               "objects": [
                   {"vehicle_id": vid, "role": role, "box": BOXES[vid][role],
                    **({"helmet_state": "no_helmet" if vid == "v1" else "helmet"}
                       if role == "rider" else {"plate_text": PLATES[vid]})}
                   for vid in ("v1", "v2") for role in ("rider", "plate")]}
              for i in range(0, N, labelled_every)]
    root = field_fixture.build(str(tmp_path / "e2e"), cameras=cams, sequences=seqs,
                               vehicles=vehicles, frames=frames, write_images=False)
    return load_dataset(root)


def _run(ds, *, labels=None, plates=True, ocr=None, phantom=False, rider_seen=True):
    labels = labels or {"v1": "WithoutHelmet", "v2": "WithHelmet"}
    ocr = ocr or PLATES

    def detect(_idx):
        dets = []
        for vid, b in BOXES.items():
            if rider_seen or vid != "v1":
                dets.append(DetBox(labels[vid], tuple(b["rider"]), 0.9))
            if plates:
                dets.append(DetBox("Plate", tuple(b["plate"]), 0.9))
        if phantom:
            dets += [DetBox("WithoutHelmet", (560, 100, 620, 300), 0.9),
                     DetBox("Plate", (565, 300, 615, 318), 0.9)]
        return dets

    def read(_idx, box):
        for vid, b in BOXES.items():
            if abs(box[0] - b["plate"][0]) < 5:
                return ocr[vid], 0.95
        return "DL01ZZ9999", 0.95

    run = run_sequence(ds, "s1", ((i, i) for i in range(N)), detect, read)
    return score_sequence(ds, run)


def _outcome(score, vid, violation, kind=None):
    return next(o for o in score["outcomes"]
                if o["vehicle_id"] == vid and o["violation"] == violation
                and (kind is None or o["outcome"] == kind))


def _kinds(score, vid, violation):
    return sorted(o["outcome"] for o in score["outcomes"]
                  if o["vehicle_id"] == vid and o["violation"] == violation)


def test_a_correct_run_fines_the_violator_with_its_own_plate(tmp_path):
    ds = _dataset(tmp_path)
    sc = _run(ds)
    assert _outcome(sc, "v1", "no_helmet")["outcome"] == "correct fine"
    assert not [o for o in sc["outcomes"] if o["vehicle_id"] == "v2"]  # nothing on v2
    m = sc["vehicles"]["v1"]
    assert m["rider_detected"] == m["rider_frames"] == 10
    assert m["assoc_right"] == m["assoc_frames"] > 0
    assert sc["phantom_fines"] == []


@pytest.mark.parametrize("kwargs, outcome, stage", [
    ({"ocr": {"v1": "MH12AB1284", "v2": PLATES["v2"]}}, "wrong plate", "ocr: plate misread"),
    ({"plates": False}, "missed", "detection: plate missed"),
    ({"labels": {"v1": "WithHelmet", "v2": "WithHelmet"}}, "missed",
     "detection: wrong class"),
    ({"rider_seen": False}, "missed", "detection: rider missed"),
])
def test_each_broken_stage_takes_the_blame(tmp_path, kwargs, outcome, stage):
    ds = _dataset(tmp_path)
    o = _outcome(_run(ds, **kwargs), "v1", "no_helmet", outcome)
    assert (o["outcome"], o["stage"]) == (outcome, stage)


def test_a_misread_plate_is_a_missed_violation_and_a_wrong_fine(tmp_path):
    ds = _dataset(tmp_path)
    sc = _run(ds, ocr={"v1": "MH12AB1284", "v2": PLATES["v2"]})
    assert _kinds(sc, "v1", "no_helmet") == ["missed", "wrong plate"]
    assert {o["stage"] for o in sc["outcomes"] if o["vehicle_id"] == "v1"} == \
        {"ocr: plate misread"}
    ov = condition_matrix(ds, [sc])["overall"]
    assert (ov["fine_precision"], ov["fine_recall"], ov["expected_fines"]) == (0.0, 0.0, 1)


def test_every_issued_fine_is_scored_not_just_one_per_vehicle(tmp_path):
    """Regression: a correct and a wrong-plate fine on one vehicle used to
    collapse into a single 'correct fine' (precision 1.0)."""
    from modules.field_eval import SequenceRun

    ds = _dataset(tmp_path)
    run = SequenceRun("s1")
    for i in range(0, N, 3):
        run.frames[i] = {"dets": [], "tracks": [(1, tuple(BOXES["v1"]["rider"]), None)]}
    run.decisions = [{"track_id": 1, "violation": "no_helmet", "plate": PLATES["v1"],
                      "frame_index": 12},
                     {"track_id": 1, "violation": "no_helmet", "plate": "MH12AB1284",
                      "frame_index": 20},
                     {"track_id": 9, "violation": "no_helmet", "plate": "DL01ZZ9999",
                      "frame_index": 21}]  # a track matching no labelled vehicle
    sc = score_sequence(ds, run)
    assert _kinds(sc, "v1", "no_helmet") == ["correct fine", "wrong plate"]
    ov = condition_matrix(ds, [sc])["overall"]
    assert ov["fines_issued"] == 3 and ov["phantom_fines"] == 1
    assert ov["fine_precision"] == round(1 / 3, 4) and ov["fine_recall"] == 1.0


def test_a_mixed_vehicle_scores_each_rider_against_its_own_state(tmp_path):
    """Regression: class accuracy used the vehicle-level class, so a perfect
    detector on a no-helmet rider with a helmeted pillion scored 50%."""
    from modules.field_eval import SequenceRun

    ds = _dataset(tmp_path)
    v1 = ds.vehicles[("s1", "v1")]
    v1.rider_count, v1.helmet_states = 2, ["no_helmet", "helmet"]
    pillion = (100, 40, 180, 100)
    run = SequenceRun("s1")
    for f in ds.frames:
        from modules.field_data import FrameObject
        f.objects.append(FrameObject("v1", "rider", pillion, helmet_state="helmet"))
        run.frames[f.frame_index] = {
            "dets": [("WithoutHelmet", tuple(BOXES["v1"]["rider"])), ("WithHelmet", pillion)],
            "tracks": []}
    m = score_sequence(ds, run)["vehicles"]["v1"]
    assert m["class_right"] == m["class_frames"] == 20



def test_a_helmeted_rider_called_bare_headed_is_a_false_fine_on_the_detector(tmp_path):
    ds = _dataset(tmp_path)
    sc = _run(ds, labels={"v1": "WithoutHelmet", "v2": "WithoutHelmet"})
    o = _outcome(sc, "v2", "no_helmet")
    assert (o["outcome"], o["stage"]) == ("false fine", "detection: wrong class")


def test_a_fine_on_an_unlabelled_vehicle_is_a_phantom(tmp_path):
    ds = _dataset(tmp_path)
    sc = _run(ds, phantom=True)
    assert len(sc["phantom_fines"]) == 1
    matrix = condition_matrix(ds, [sc])
    assert matrix["overall"]["phantom_fines"] == 1


def test_condition_matrix_names_the_bottleneck_per_condition(tmp_path):
    ds = _dataset(tmp_path)
    matrix = condition_matrix(ds, [_run(ds, plates=False)])
    night = matrix["by_condition"]["lighting"]["night"]
    assert night["expected_fines"] == 1 and night["correct"] == 0
    assert night["fine_recall"] == 0.0
    assert night["bottleneck"] == "detection: plate missed"
    assert matrix["by_condition"]["occlusion"]["partial"]["vehicles"] == 1  # v2


def test_vehicle_conditions_read_the_labels(tmp_path):
    ds = _dataset(tmp_path)
    c = vehicle_conditions(ds, ds.vehicles[("s1", "v1")])
    assert c["lighting"] == "night" and c["view angle"] == "rear"
    assert c["plate size (distance)"] == "medium (40-80 px)"
    assert c["vehicles in frame"] == "2-3" and c["crossing"] == "separate"
    assert c["resolution"] == "<720p"


def test_ocr_report_compares_policies_and_counts_abstention(tmp_path):
    ds = _dataset(tmp_path)
    v1, v2 = ds.vehicles[("s1", "v1")], ds.vehicles[("s1", "v2")]
    reads = {
        # three good reads, then a wrong LAST read that is itself a valid plate
        # (a look-alike like Z->2 would be corrected, as the pipeline does)
        v1.key: [PlateRead("MH12AB1234", 0.9, 10.0)] * 3 + [PlateRead("MH12AB1284", 0.3, 12.0)],
        # one read only: single-frame policies answer, the vote abstains
        v2.key: [PlateRead("KA05CD5678", 0.8, 11.0)],
    }
    rep = ocr_report(ds, [v1, v2], reads)
    o = rep["overall"]
    assert o["best_conf"]["normalized_match"] == 1.0
    assert o["last"]["normalized_match"] == 0.5 and o["last"]["wrong_plate_rate"] == 0.5
    assert o["temporal"]["coverage"] == 0.5 and o["temporal"]["abstention_rate"] == 0.5
    assert o["temporal"]["accuracy_when_answered"] == 1.0
    from modules.field_eval import as_plate

    assert as_plate("MH12AB1Z34") == "MH12AB1234"  # same correction as the stabilizer
    assert rep["reads"] == 5 and rep["latency_ms"]["mean"] == pytest.approx(10.6)
    assert set(rep["by_condition"]["occlusion"]) == {"none", "partial"}


def test_a_violation_with_an_invisible_plate_is_expected_to_be_withheld(tmp_path):
    ds = _dataset(tmp_path)
    v1 = ds.vehicles[("s1", "v1")]
    v1.plate_visibility, v1.plate_text = "none", ""
    o = _outcome(_run(ds, plates=False), "v1", "no_helmet")
    assert o["outcome"] == "correctly withheld" and o["stage"] is None


def test_evaluate_field_reports_not_measured_without_a_dataset(tmp_path):
    import json

    import evaluate_field

    rc = evaluate_field.main(["--dataset", str(tmp_path / "none"), "--out", str(tmp_path)])
    assert rc == 0
    assert "NOT MEASURED" in (tmp_path / "field_evaluation.md").read_text()
    assert json.loads((tmp_path / "field_evaluation.json").read_text())["measured"] is False


def test_evaluate_field_refuses_the_development_split(tmp_path):
    import evaluate_field

    ds = _dataset(tmp_path)
    assert evaluate_field.main(["--dataset", ds.root, "--split", "development",
                                "--out", str(tmp_path)]) == 2


def test_field_report_renders_every_section(tmp_path):
    import evaluate_field

    ds = _dataset(tmp_path)
    v1, v2 = ds.vehicles[("s1", "v1")], ds.vehicles[("s1", "v2")]
    reads = {v1.key: [PlateRead("MH12AB1234", 0.9, 10.0)] * 3,
             v2.key: [PlateRead("KA05CD5678", 0.9, 10.0)] * 3}
    md = evaluate_field._render({
        "dataset": "fixture", "dataset_version": "0", "split": "held_out", "sequences": 1,
        "vehicles": 2, "frames_run": N, "lock_hash": "sha256:x", "model": "m.pt",
        "pipeline_version": "1.1.0", "ocr": ocr_report(ds, [v1, v2], reads),
        "matrix": condition_matrix(ds, [_run(ds)])})
    assert "Fine precision **1.0**, recall **1.0**" in md
    assert "### OCR by lighting" in md and "| lighting: night |" in md


def test_frames_come_from_a_directory_by_index_or_from_a_video(tmp_path):
    import cv2
    import numpy as np

    from modules.field_eval import iter_frames

    root = field_fixture.build(str(tmp_path / "fr"))
    ds = load_dataset(root)
    idx = [i for i, _ in iter_frames(ds, "s1")]
    assert idx == [0, 1, 2]  # numeric stems -> frame indices
    video = tmp_path / "fr" / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for k in range(5):
        writer.write(np.full((48, 64, 3), k * 40, dtype=np.uint8))
    writer.release()
    ds.sequences["s1"].frames_dir = None
    ds.sequences["s1"].video_path = "clip.avi"
    assert [i for i, _ in iter_frames(ds, "s1")] == [0, 1, 2, 3, 4]


def test_plate_reads_are_taken_on_labelled_boxes_in_frame_order_and_timed(tmp_path):
    from modules.field_eval import collect_plate_reads

    ds = _dataset(tmp_path)
    seen = []

    def read(image, box):
        seen.append((image, box))
        return ("MH12AB1234", 0.9) if box[0] < 300 else None

    reads = collect_plate_reads(ds, [ds.vehicles[("s1", "v1")], ds.vehicles[("s1", "v2")]],
                                read, lambda f: f.frame_index)
    assert [s[0] for s in seen[:2]] == [0, 0]  # one image load serves both plates
    assert len(reads[("s1", "v1")]) == 10 and reads[("s1", "v1")][0].text == "MH12AB1234"
    assert reads[("s1", "v2")][0].text is None  # unreadable -> a missed read, not dropped
    assert all(r.latency_ms >= 0 for r in reads[("s1", "v1")])


def test_ocr_metrics_mean_the_same_thing_for_every_policy(tmp_path):
    """Regression: exact_match compared raw text for single-frame policies but
    the normalized vote for the temporal one, and only the vote could
    'abstain' — three identical reads scored 0 / 0 / 1."""
    ds = _dataset(tmp_path)
    v1, v2 = ds.vehicles[("s1", "v1")], ds.vehicles[("s1", "v2")]
    reads = {v1.key: [PlateRead("MH 12 AB 1234", 0.9, 5.0)] * 3,
             v2.key: [PlateRead(None, 0.0, 5.0)] * 3}  # nothing readable
    o = ocr_report(ds, [v1, v2], reads)["overall"]
    assert o["last"]["exact_match"] == o["best_conf"]["exact_match"] == \
        o["temporal"]["exact_match"] == 0.5
    for p in ("last", "best_conf", "temporal"):
        assert o[p]["no_answer_rate"] == 0.5
    assert o["temporal"]["abstention_rate"] == 0.5 and o["last"]["abstention_rate"] == 0.0


def test_a_fine_on_an_invisible_plate_is_blamed_on_association(tmp_path):
    """Regression: with no plate frames to judge, the plate-detection gate read
    0/0 as a failure and blamed 'plate missed'."""
    ds = _dataset(tmp_path)
    v1 = ds.vehicles[("s1", "v1")]
    v1.plate_visibility, v1.plate_text = "none", ""
    for f in ds.frames:
        f.objects = [o for o in f.objects if not (o.vehicle_id == "v1" and o.role == "plate")]
    o = _outcome(_run(ds), "v1", "no_helmet")
    assert (o["outcome"], o["stage"]) == ("fined without a visible plate",
                                          "association: plate mislinked")
