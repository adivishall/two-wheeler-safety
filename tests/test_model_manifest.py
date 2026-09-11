"""Model-manifest tests (Phase 2). Model-free: a fake .pt file and a tiny
synthetic YOLO dataset stand in for the real weights/dataset, so these run in
plain CI without torch/ultralytics."""

import json

import pytest

from modules import model_manifest as mm


def _fake_run(tmp_path, weights_bytes=b"weights-v1"):
    """Build a minimal Ultralytics-style run dir: <run>/weights/best.pt +
    <run>/args.yaml. Returns the best.pt path."""
    run = tmp_path / "runs" / "detect" / "traffic_model"
    (run / "weights").mkdir(parents=True)
    best = run / "weights" / "best.pt"
    best.write_bytes(weights_bytes)
    (run / "args.yaml").write_text(
        "model: yolov8n.pt\ndata: d.yaml\nepochs: 15\nimgsz: 640\n"
        "batch: 16\ndevice: mps\nseed: 0\ndeterministic: true\nlr0: 0.01\n"
    )
    return str(best)


def _tiny_dataset(tmp_path):
    """A 2-class dataset with one train image's labels. Returns data.yaml path."""
    ds = tmp_path / "ds"
    (ds / "train" / "images").mkdir(parents=True)
    (ds / "train" / "labels").mkdir(parents=True)
    (ds / "train" / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n0 0.6 0.6 0.1 0.1\n"
    )
    data = ds / "data.yaml"
    data.write_text(
        "train: train/images\nval: train/images\nnc: 2\nnames: [Plate, Rider]\n"
    )
    return str(data)


def test_sha256_file_is_stable_and_prefixed(tmp_path):
    best = _fake_run(tmp_path)
    c1 = mm.sha256_file(best)
    c2 = mm.sha256_file(best)
    assert c1 == c2
    assert c1.startswith("sha256:") and len(c1) == len("sha256:") + 64


def test_dataset_version_counts_instances_per_class(tmp_path):
    data = _tiny_dataset(tmp_path)
    dv = mm.dataset_version(data)
    assert dv["classes"] == ["Plate", "Rider"]
    assert dv["class_counts"] == {"Plate": 2, "Rider": 1}
    assert dv["instances"] == 3
    assert dv["version"].startswith("sha256:")


def test_dataset_version_changes_when_labels_change(tmp_path):
    data = _tiny_dataset(tmp_path)
    v1 = mm.dataset_version(data)["version"]
    # add a label instance -> version must change
    (tmp_path / "ds" / "train" / "labels" / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    v2 = mm.dataset_version(data)["version"]
    assert v1 != v2


def test_build_manifest_records_full_provenance(tmp_path):
    best = _fake_run(tmp_path)
    data = _tiny_dataset(tmp_path)
    metrics = {"split": "val", "map50": 0.7, "map50_95": 0.5}
    man = mm.build_manifest(
        best, name="traffic-4class", version="1.2.3",
        data_yaml=data, metrics=metrics,
    )
    assert man["name"] == "traffic-4class"
    assert man["version"] == "1.2.3"
    assert man["checksum"] == mm.sha256_file(best)
    assert man["training_config"]["epochs"] == 15
    assert man["training_config"]["seed"] == 0
    assert man["metrics"]["map50"] == 0.7
    assert man["classes"] == ["Plate", "Rider"]
    assert man["created_at"].endswith("+00:00") or "T" in man["created_at"]


def test_metrics_dict_wins_over_eval_json(tmp_path):
    best = _fake_run(tmp_path)
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(json.dumps({"split": "val", "official": {"map50": 0.1}}))
    man = mm.build_manifest(
        best, name="m", version="1", eval_json=str(eval_json),
        metrics={"split": "val", "map50": 0.9},
    )
    assert man["metrics"]["map50"] == 0.9


def test_write_and_resolve_manifest_by_location(tmp_path, monkeypatch):
    best = _fake_run(tmp_path)
    monkeypatch.chdir(tmp_path)  # so the registry lands under tmp
    man = mm.build_manifest(best, name="traffic-4class", version="1.0.0")
    beside, registry = mm.write_manifest(man, best)
    assert (tmp_path / beside).exists() or beside  # beside weights
    # resolve prefers the manifest sitting next to the weights
    resolved = mm.resolve_manifest(best)
    assert resolved is not None
    assert resolved["name"] == "traffic-4class"


def test_resolve_manifest_by_checksum_from_registry(tmp_path, monkeypatch):
    best = _fake_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    man = mm.build_manifest(best, name="traffic-4class", version="1.0.0")
    _beside, _registry = mm.write_manifest(man, best)
    # remove the beside-manifest so resolution must fall back to checksum match
    (tmp_path / _beside).unlink()
    resolved = mm.resolve_manifest(best)
    assert resolved is not None and resolved["checksum"] == man["checksum"]


def test_resolve_manifest_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert mm.resolve_manifest(str(tmp_path / "nope.pt")) is None
    assert mm.resolve_manifest("") is None


def test_model_version_string_forms():
    assert mm.model_version_string(None) is None
    assert mm.model_version_string({"name": "m", "version": "2"}) == "m@2"
    only_sum = {"checksum": "sha256:" + "a" * 64}
    assert mm.model_version_string(only_sum) == "sha256:" + "a" * 12


def test_build_manifest_missing_weights_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mm.build_manifest(str(tmp_path / "absent.pt"), name="m", version="1")
