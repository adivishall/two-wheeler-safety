"""The synthetic OCR sanity set: it must build, be honestly labelled, and be
loudly marked as *not* a benchmark.

These tests do not run OCR. They guard two things: that the generator produces a
dataset the evaluator can actually consume (right schema, readable images, one
row per plate x condition), and that nothing it writes could ever be mistaken
for a field measurement — every artifact is stamped ``synthetic``. The evaluator
itself is exercised end to end against real EasyOCR by hand, not in CI, because
CI has no model stack.
"""

from __future__ import annotations

import csv
import json
import os

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from make_ocr_sanity_set import (  # noqa: E402
    CONDITIONS,
    PLATES,
    build,
    degrade,
    parse_args,
    render_plate,
)


def test_render_plate_is_a_readable_image():
    img = render_plate("MH12AB1234")
    assert img.ndim == 3 and img.shape[2] == 3
    # Not a blank canvas: the drawn glyphs and border are darker than the plate.
    assert img.min() < 60 and img.max() > 200


def test_clean_condition_is_the_untouched_image():
    base = render_plate("KA05MN6789")
    assert np.array_equal(degrade(base, "clean"), base)


@pytest.mark.parametrize("condition", [c for c in CONDITIONS if c != "clean"])
def test_every_degradation_changes_the_image(condition):
    base = render_plate("KA05MN6789")
    out = degrade(base, condition, seed=3)
    assert out.shape == base.shape
    assert not np.array_equal(out, base), f"{condition} left the image untouched"


def test_degrade_is_deterministic_for_a_given_seed():
    base = render_plate("DL8CAF5031")
    # lowlight is the only stochastic condition (sensor noise); pin its seed.
    a = degrade(base, "lowlight", seed=7)
    b = degrade(base, "lowlight", seed=7)
    assert np.array_equal(a, b)


def test_unknown_condition_is_rejected():
    base = render_plate("TN22BC4567")
    with pytest.raises(ValueError):
        degrade(base, "sunny")


def test_build_writes_one_row_per_plate_and_condition(tmp_path):
    meta = build(str(tmp_path), plates=PLATES[:2], conditions=CONDITIONS)
    assert meta["images"] == 2 * len(CONDITIONS)
    imgs = os.listdir(tmp_path / "images")
    assert len(imgs) == meta["images"]


def test_labels_csv_has_exactly_the_documented_schema(tmp_path):
    build(str(tmp_path), plates=PLATES[:2])
    with open(tmp_path / "labels.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["image_path", "plate", "condition"]
        rows = list(reader)
    assert rows, "no rows written"
    for row in rows:
        assert row["condition"] in CONDITIONS
        # The evaluator resolves image_path relative to the CSV directory.
        assert os.path.exists(tmp_path / row["image_path"]), row["image_path"]
        assert row["plate"] in PLATES


def test_written_images_are_loadable_by_opencv(tmp_path):
    build(str(tmp_path), plates=PLATES[:1])
    with open(tmp_path / "labels.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        img = cv2.imread(str(tmp_path / row["image_path"]))
        assert img is not None, f"unreadable: {row['image_path']}"
        assert img.shape[2] == 3


def test_dataset_json_is_loudly_synthetic(tmp_path):
    build(str(tmp_path), plates=PLATES[:1])
    meta = json.loads((tmp_path / "dataset.json").read_text())
    assert meta["synthetic"] is True
    assert "not_a_benchmark" in meta
    assert "NOT" in meta["not_a_benchmark"]
    assert meta["schema"] == "image_path,plate,condition"


def test_returned_meta_is_marked_synthetic(tmp_path):
    meta = build(str(tmp_path), plates=PLATES[:1])
    assert meta["synthetic"] is True


def test_cli_defaults_to_all_conditions():
    args = parse_args([])
    assert args.conditions == list(CONDITIONS)
    assert args.out == "data/ocr_sanity"


def test_cli_rejects_an_unknown_condition():
    with pytest.raises(SystemExit):
        parse_args(["--conditions", "rainbow"])
