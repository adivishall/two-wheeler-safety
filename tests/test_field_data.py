"""Field dataset: the schema is enforced, splits are frozen per group, and
evaluation data cannot become training data by any route the guard knows."""

import json
import os

import cv2
import field_fixture
import pytest

from modules.field_data import (
    DEVELOPMENT,
    EVAL_SPLITS,
    EXTERNAL,
    FieldDataError,
    LeakageError,
    assign_splits,
    check_no_eval_leakage,
    coverage,
    export_training,
    frames_in,
    load_dataset,
    read_lock,
    split_of,
    training_frames,
)


@pytest.fixture
def ds(tmp_path):
    return load_dataset(field_fixture.build(str(tmp_path / "field")), check_files=True)


def test_a_valid_dataset_loads(ds):
    assert set(ds.cameras) == {"camA", "camB"}
    assert len(ds.vehicles) == 7 and len(ds.frames) == 21
    assert ds.vehicles[("s1", "v1")].violations == {"no_helmet"}
    assert ds.warnings == []


def _mutate_vehicle(tmp_path, **changes):
    vehicles = [{"sequence_id": "s1", "vehicle_id": "v1", "first_frame": 0, "last_frame": 2,
                 "plate_text": "MH12AB1234", "plate_visibility": "full", "rider_count": 1,
                 "helmet_states": ["no_helmet"], "violations": ["no_helmet"],
                 "occlusion": "none", "annotator": "ann1", **changes}]
    seqs = [{"sequence_id": "s1", "camera_id": "camA", "fps": 25, "frames_dir": "f"}]
    return field_fixture.build(str(tmp_path / "bad"), sequences=seqs, vehicles=vehicles,
                               frames=[], write_images=False)


@pytest.mark.parametrize("changes, fragment", [
    ({"rider_count": 3, "helmet_states": ["no_helmet", "helmet", "helmet"]},
     "triple_riding must be labelled"),
    ({"helmet_states": ["helmet"]}, "no rider is bare-headed"),
    ({"violations": [], "helmet_states": ["no_helmet"]}, "no_helmet is not labelled"),
    ({"plate_visibility": "none"}, "plate_text given for a plate labelled not visible"),
    ({"annotator": ""}, "annotator is required"),
    ({"violations": ["no_helmet", "jaywalking"]}, "unknown violation"),
    ({"last_frame": -5}, "last_frame < first_frame"),
])
def test_inconsistent_vehicle_labels_are_rejected(tmp_path, changes, fragment):
    with pytest.raises(FieldDataError) as err:
        load_dataset(_mutate_vehicle(tmp_path, **changes))
    assert any(fragment in issue for issue in err.value.issues), err.value.issues


def test_frame_paths_must_stay_inside_the_dataset(tmp_path):
    seqs = [{"sequence_id": "s1", "camera_id": "camA", "fps": 25, "frames_dir": "f"}]
    frames = [{"sequence_id": "s1", "frame_index": 0, "frame_path": "../../etc/x.png",
               "annotator": "a", "objects": []}]
    root = field_fixture.build(str(tmp_path / "p"), sequences=seqs, frames=frames,
                               write_images=False)
    with pytest.raises(FieldDataError, match="relative and inside"):
        load_dataset(root)


def test_boxes_are_checked_against_the_camera(tmp_path):
    seqs = [{"sequence_id": "s1", "camera_id": "camA", "fps": 25, "frames_dir": "f"}]
    frames = [{"sequence_id": "s1", "frame_index": 1, "frame_path": "f/1.png", "annotator": "a",
               "objects": [{"vehicle_id": "v1", "role": "rider", "box": [300, 10, 400, 50]}]}]
    root = field_fixture.build(str(tmp_path / "b"), sequences=seqs, frames=frames,
                               write_images=False)
    with pytest.raises(FieldDataError, match="outside the camera resolution"):
        load_dataset(root)


def test_a_plate_the_pipeline_would_reject_is_a_warning_not_an_error(tmp_path):
    root = _mutate_vehicle(tmp_path, plate_text="ABC 123", review_decision="accepted")
    ds = load_dataset(root)
    assert any("not shaped like an Indian plate" in w for w in ds.warnings)


def test_splits_are_by_group_deterministic_and_frozen(ds, tmp_path):
    lock = assign_splits(ds, external_cameras=["camB"], salt="t")
    assert split_of(ds, lock, "x1") == EXTERNAL
    before = {k: a["split"] for k, a in lock["assignments"].items()}
    # every frame of a sequence is in its sequence's split
    for f in ds.frames:
        assert split_of(ds, lock, f.sequence_id) in {DEVELOPMENT, *EVAL_SPLITS}
    # re-running with different ratios moves nothing that was already assigned
    again = assign_splits(ds, ratios={"development": 1.0, "validation": 0.0, "held_out": 0.0})
    assert {k: a["split"] for k, a in again["assignments"].items()} == before


def test_new_data_never_reshuffles_old_data(tmp_path):
    root = str(tmp_path / "grow")
    field_fixture.build(root)
    first = assign_splits(load_dataset(root), salt="t")
    meta = json.load(open(os.path.join(root, "dataset.json")))
    meta["sequences"].append({"sequence_id": "s9", "camera_id": "camA", "fps": 25,
                              "frames_dir": "frames/s9"})
    json.dump(meta, open(os.path.join(root, "dataset.json"), "w"))
    grown = assign_splits(load_dataset(root))
    for key, a in first["assignments"].items():
        assert grown["assignments"][key]["split"] == a["split"]
    assert "sequence:s9" in grown["assignments"]


def test_a_camera_cannot_become_external_after_it_was_used(ds):
    assign_splits(ds, salt="t")
    with pytest.raises(LeakageError, match="retroactively"):
        assign_splits(ds, external_cameras=["camA"])


def test_a_hand_edited_lock_is_refused(ds):
    lock = assign_splits(ds, external_cameras=["camB"], salt="t")
    key = next(k for k, a in lock["assignments"].items() if a["split"] != DEVELOPMENT)
    lock["assignments"][key]["split"] = DEVELOPMENT  # smuggle eval data into training
    with open(os.path.join(ds.root, "splits.lock.json"), "w") as fh:
        json.dump(lock, fh)
    with pytest.raises(LeakageError, match="edited by hand"):
        read_lock(ds.root)


def test_camera_day_grouping_keeps_a_cameras_day_together(tmp_path):
    root = field_fixture.build(str(tmp_path / "cd"))
    ds = load_dataset(root)
    lock = assign_splits(ds, group_by="camera_day", salt="t")
    by_day: dict = {}
    for sid, seq in ds.sequences.items():
        by_day.setdefault((seq.camera_id, seq.day), set()).add(split_of(ds, lock, sid))
    assert all(len(splits) == 1 for splits in by_day.values())


def test_training_export_contains_development_frames_only(ds, tmp_path):
    lock = assign_splits(ds, external_cameras=["camB"], salt="t")
    manifest = export_training(ds, lock, str(tmp_path / "train"))
    dev = frames_in(ds, lock, DEVELOPMENT)
    assert manifest["frames"] == len(dev) > 0
    exported = sorted(os.listdir(tmp_path / "train" / "images"))
    assert all(name.split("_")[0] in {f.sequence_id for f in dev} for name in exported)
    label = open(tmp_path / "train" / "labels" / exported[0].replace(".png", ".txt")).read()
    classes = sorted(int(line.split()[0]) for line in label.splitlines())
    assert classes == [0, 2]  # Plate, WithoutHelmet
    assert manifest["lock_hash"] == lock["lock_hash"]


def test_the_guard_catches_an_eval_frame_by_path_copy_and_reencode(ds, tmp_path):
    lock = assign_splits(ds, external_cameras=["camB"], salt="t")
    eval_frame = next(f for f in ds.frames if split_of(ds, lock, f.sequence_id) in EVAL_SPLITS)
    src = ds.frame_file(eval_frame)
    with pytest.raises(LeakageError, match="is an evaluation frame"):
        check_no_eval_leakage([src], ds, lock)
    copy = tmp_path / "renamed.png"
    copy.write_bytes(open(src, "rb").read())
    with pytest.raises(LeakageError, match="byte-identical"):
        check_no_eval_leakage([str(copy)], ds, lock)
    reenc = tmp_path / "reencoded.jpg"
    img = cv2.imread(src)
    cv2.imwrite(str(reenc), cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)),
                [cv2.IMWRITE_JPEG_QUALITY, 70])
    with pytest.raises(LeakageError, match="near-duplicate"):
        check_no_eval_leakage([str(reenc)], ds, lock)
    dev_frame = next(f for f in ds.frames if split_of(ds, lock, f.sequence_id) == DEVELOPMENT)
    check_no_eval_leakage([ds.frame_file(dev_frame)], ds, lock)  # a real dev frame passes


def test_rejected_and_uncertain_labels_are_not_trained_on(tmp_path):
    root = field_fixture.build(str(tmp_path / "q"))
    rows = [json.loads(line) for line in open(os.path.join(root, "frames.jsonl"))]
    rows[0]["review_decision"] = "rejected"
    rows[1]["label_confidence"] = "uncertain"
    with open(os.path.join(root, "frames.jsonl"), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    ds = load_dataset(root)
    lock = assign_splits(ds, salt="t",
                         ratios={"development": 1.0, "validation": 0.0, "held_out": 0.0})
    kept = {(f.sequence_id, f.frame_index) for f in training_frames(ds, lock)}
    assert (rows[0]["sequence_id"], 0) not in kept and (rows[1]["sequence_id"], 1) not in kept


def test_coverage_counts_labelled_vehicles_per_condition(ds):
    lock = assign_splits(ds, external_cameras=["camB"], salt="t")
    cov = coverage(ds, lock)
    assert cov[EXTERNAL]["camera"] == {"camB": 1}
    total_night = sum(c["lighting"].get("night", 0) for c in cov.values())
    assert total_night == 3


def test_the_trainer_refuses_a_config_that_contains_an_eval_frame(ds, tmp_path):
    """train_traffic.py --field-dataset runs the same guard before any training
    starts (exit 3), so a hand-built config can't bypass export_training."""
    import shutil

    import train_traffic

    lock = assign_splits(ds, external_cameras=["camB"], salt="t")
    eval_frame = next(f for f in ds.frames if split_of(ds, lock, f.sequence_id) in EVAL_SPLITS)
    train_dir = tmp_path / "yolo" / "train" / "images"
    train_dir.mkdir(parents=True)
    shutil.copyfile(ds.frame_file(eval_frame), train_dir / "innocent_name.png")
    (tmp_path / "yolo" / "train" / "labels").mkdir()
    (tmp_path / "yolo" / "train" / "labels" / "innocent_name.txt").write_text("")
    data = tmp_path / "yolo" / "data.yaml"
    data.write_text(f"path: {tmp_path / 'yolo'}\ntrain: train/images\nval: train/images\n"
                    "names: [Plate, WithHelmet, WithoutHelmet, TripleRiding]\n")
    assert train_traffic.main(["--data", str(data), "--field-dataset", ds.root]) == 3


def test_field_dataset_cli_round_trip(tmp_path, capsys):
    import field_dataset

    root = field_fixture.build(str(tmp_path / "cli"))
    assert field_dataset.main(["validate", root, "--check-files"]) == 0
    assert field_dataset.main(["export-train", root, str(tmp_path / "o")]) == 2  # no lock yet
    assert field_dataset.main(["assign-splits", root, "--external-camera", "camB",
                               "--salt", "t"]) == 0
    assert field_dataset.main(["export-train", root, str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "export_manifest.json").exists()
    bad = tmp_path / "broken"
    field_fixture.build(str(bad), schema_version="0.9", write_images=False)
    assert field_dataset.main(["validate", str(bad)]) == 2
    assert "schema_version" in capsys.readouterr().out
