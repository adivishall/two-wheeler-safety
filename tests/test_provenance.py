"""Provenance: dataset fingerprints, run records, shared YOLO I/O helpers."""

import json
import shutil

import pytest

from modules import yolo_io
from modules.provenance import (
    dataset_fingerprint,
    git_state,
    run_provenance,
    split_fingerprint,
    weights_identity,
)


def _dataset(root, labels=None):
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    labels = labels or {"a": "0 0.5 0.5 0.2 0.2\n", "b": "1 0.4 0.4 0.1 0.1\n"}
    for stem, text in labels.items():
        (root / "train" / "images" / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff" + stem.encode())
        (root / "train" / "labels" / f"{stem}.txt").write_text(text)
    (root / "val" / "images" / "v.jpg").write_bytes(b"\xff\xd8\xffv")
    (root / "val" / "labels" / "v.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    data = root / "data.yaml"
    data.write_text("train: train/images\nval: val/images\nnames: [Plate, Rider]\n")
    return str(data)


def test_fingerprint_is_stable_and_independent_of_location(tmp_path):
    a = _dataset(tmp_path / "one")
    v1 = dataset_fingerprint(a)["version"]
    assert v1 == dataset_fingerprint(a)["version"]
    shutil.copytree(tmp_path / "one", tmp_path / "somewhere" / "else")
    moved = str(tmp_path / "somewhere" / "else" / "data.yaml")
    assert dataset_fingerprint(moved)["version"] == v1


def test_fingerprint_changes_when_a_box_moves_but_counts_do_not(tmp_path):
    """The schemes this replaced hashed per-class COUNTS, so this edit — same
    class, same count, different box — kept the same "version"."""
    a = _dataset(tmp_path / "a")
    b = _dataset(tmp_path / "b", {"a": "0 0.6 0.5 0.2 0.2\n", "b": "1 0.4 0.4 0.1 0.1\n"})
    assert dataset_fingerprint(a)["version"] != dataset_fingerprint(b)["version"]
    assert dataset_fingerprint(a)["splits"]["val"] == dataset_fingerprint(b)["splits"]["val"]


def test_fingerprint_changes_when_classes_are_reordered(tmp_path):
    a = _dataset(tmp_path / "a")
    v1 = dataset_fingerprint(a)["version"]
    (tmp_path / "a" / "data.yaml").write_text(
        "train: train/images\nval: val/images\nnames: [Rider, Plate]\n")
    assert dataset_fingerprint(a)["version"] != v1


def test_deep_mode_hashes_pixels(tmp_path):
    a = _dataset(tmp_path / "a")
    shallow = dataset_fingerprint(a)["version"]
    img = tmp_path / "a" / "train" / "images" / "a.jpg"
    img.write_bytes(b"\xff\xd8\xffX")  # same size, different bytes
    assert dataset_fingerprint(a)["version"] == shallow
    assert dataset_fingerprint(a, deep=True)["scheme"].endswith("image-bytes")


def test_split_fingerprint_and_run_provenance(tmp_path):
    data = _dataset(tmp_path / "d")
    assert split_fingerprint(data, "val").startswith("sha256:")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    prov = run_provenance(model_paths=[str(weights)], data_yaml=data, split="val",
                          config={"conf": 0.001})
    assert prov["models"][0]["sha256"].startswith("sha256:")
    assert prov["dataset"]["split"] == "val" and prov["dataset"]["split_images"] == 1
    assert prov["config"] == {"conf": 0.001}
    assert "python" in prov["environment"]
    json.dumps(prov)  # must be serialisable into a report


def test_missing_weights_have_no_identity(tmp_path):
    ident = weights_identity(str(tmp_path / "absent.pt"))
    assert ident["sha256"] is None and ident["version"] is None


def test_git_state_shape():
    st = git_state()
    assert set(st) == {"commit", "dirty"}


# -- shared YOLO I/O ----------------------------------------------------------

def test_split_pairs_are_sorted_and_labels_parse(tmp_path):
    data = _dataset(tmp_path / "d", {"z": "0 0.5 0.5 0.2 0.2\n", "a": "1 0.5 0.5 1 1\n"})
    cfg = yolo_io.load_data_yaml(data)
    pairs = list(yolo_io.iter_image_label_pairs(yolo_io.split_dir(cfg, "train")))
    assert [p[0].rsplit("/", 1)[1] for p in pairs] == ["a.jpg", "z.jpg"]
    gt = yolo_io.read_gt(pairs[0][1], w=100, h=50)
    assert gt == [(1, (0.0, 0.0, 100.0, 50.0))]
    assert yolo_io.read_gt(str(tmp_path / "none.txt"), 1, 1) == []
    assert list(yolo_io.iter_image_label_pairs(str(tmp_path / "missing"))) == []


def test_cache_key_is_stable_and_sensitive():
    k = yolo_io.cache_key(weights="w", split="s", conf=0.001)
    assert k == yolo_io.cache_key(conf=0.001, split="s", weights="w")
    assert k != yolo_io.cache_key(weights="w", split="s", conf=0.25)


def test_predict_split_with_a_fake_model(tmp_path):
    import cv2
    import numpy as np

    img_dir = tmp_path / "s" / "images"
    lbl_dir = tmp_path / "s" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "x.png"), np.zeros((20, 20, 3), np.uint8))
    (lbl_dir / "x.txt").write_text("0 0.5 0.5 0.5 0.5\n")

    class Box:
        cls, conf, xyxy = [0], [0.9], [[5.0, 5.0, 15.0, 15.0]]

    class Res:
        boxes = [Box()]

    class Model:
        def predict(self, source=None, **kw):
            assert kw["conf"] == yolo_io.AP_CONF
            return [Res()]

    seen = []
    recs = yolo_io.predict_split(Model(), yolo_io.iter_image_label_pairs(str(img_dir)), 1,
                                 corrupt=lambda im, key: seen.append(key) or im)
    assert seen == ["x.png"]
    assert recs[0].pred_tp.tolist() == [1.0]


def test_resolve_device_passthrough():
    assert yolo_io.resolve_device("cpu") == "cpu"
    assert yolo_io.resolve_device("auto") in {"cpu", "cuda", "mps"}


@pytest.mark.parametrize("names", ["[A, B]", "{0: A, 1: B}"])
def test_load_data_yaml_names(tmp_path, names):
    y = tmp_path / "d.yaml"
    y.write_text(f"names: {names}\n")
    assert yolo_io.load_data_yaml(str(y))["names"] == ["A", "B"]


def test_dirty_flag_ignores_generated_results(tmp_path, monkeypatch):
    """Regenerating tracked result files or editing docs must not mark the code
    dirty; editing code or a tracked data input must."""
    import subprocess

    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "eval" / "results").mkdir(parents=True)
    (tmp_path / "eval" / "results" / "r.json").write_text("{}")
    (tmp_path / "code.py").write_text("x = 1\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "GUIDE.md").write_text("guide")
    (tmp_path / "docs" / "input.json").write_text("{}")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    monkeypatch.chdir(tmp_path)
    assert git_state()["dirty"] is False
    (tmp_path / "eval" / "results" / "r.json").write_text('{"a": 1}')
    assert git_state()["dirty"] is False
    (tmp_path / "code.py").write_text("x = 2\n")
    assert git_state()["dirty"] is True
    git("checkout", "--", "code.py")
    assert git_state()["dirty"] is False
    (tmp_path / "docs" / "GUIDE.md").write_text("edited")  # tracked docs: not dirty
    assert git_state()["dirty"] is False
    (tmp_path / "docs" / "input.json").write_text('{"b": 2}')  # tracked data: dirty
    assert git_state()["dirty"] is True
    git("checkout", "--", "docs")
    (tmp_path / "notes.md").write_text("scratch")  # untracked non-code: not dirty
    assert git_state()["dirty"] is False
    (tmp_path / "new_rule.py").write_text("x = 3\n")  # untracked source: dirty
    assert git_state()["dirty"] is True
