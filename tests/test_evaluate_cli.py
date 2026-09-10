"""Tests for evaluate_model.py helpers that don't need the model or dataset.

The heavy Ultralytics import lives inside ``main()``, so importing the module
and exercising its parsing/IO helpers is safe in CI.
"""

import evaluate_model as ev


def test_parse_args_requires_model_and_data():
    args = ev.parse_args(["--model", "m.pt", "--data", "d.yaml", "--split", "test"])
    assert args.model == "m.pt" and args.data == "d.yaml" and args.split == "test"


def test_resolve_device_passthrough():
    assert ev.resolve_device("cpu") == "cpu"
    assert ev.resolve_device("cuda") == "cuda"
    # 'auto' resolves to something concrete (cpu on a CI box without a GPU)
    assert ev.resolve_device("auto") in {"cpu", "cuda", "mps"}


def test_read_gt_converts_yolo_normalized_to_pixels(tmp_path):
    label = tmp_path / "img.txt"
    # one box, class 2, centered, half the image wide/tall
    label.write_text("2 0.5 0.5 0.5 0.5\n")
    gt = ev._read_gt(str(label), w=100, h=200)
    assert len(gt) == 1
    cls, (x1, y1, x2, y2) = gt[0]
    assert cls == 2
    assert (x1, y1, x2, y2) == (25.0, 50.0, 75.0, 150.0)


def test_read_gt_missing_file_is_empty(tmp_path):
    assert ev._read_gt(str(tmp_path / "nope.txt"), 10, 10) == []


def test_load_data_yaml_names_list(tmp_path):
    y = tmp_path / "data.yaml"
    y.write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        "names: [Plate, WithHelmet, WithoutHelmet, TripleRiding]\n"
    )
    data = ev.load_data_yaml(str(y))
    assert data["names"] == ["Plate", "WithHelmet", "WithoutHelmet", "TripleRiding"]


def test_load_data_yaml_names_dict(tmp_path):
    y = tmp_path / "data.yaml"
    y.write_text("names:\n  0: Plate\n  1: WithHelmet\n")
    data = ev.load_data_yaml(str(y))
    assert data["names"] == ["Plate", "WithHelmet"]


def test_render_markdown_is_stable_without_optional_sections():
    md = ev._render_markdown({
        "name": "eval_x", "generated_at": "now", "model": "m.pt",
        "data": "d.yaml", "split": "val", "conf": 0.25, "iou": 0.5,
        "imgsz": 640, "device": "cpu",
    })
    assert "# Model evaluation — eval_x" in md
