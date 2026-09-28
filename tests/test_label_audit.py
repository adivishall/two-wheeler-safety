"""Label-quality audit and synthetic corruptions (model-free)."""

import numpy as np
import pytest

from modules import corruptions
from modules.label_audit import (
    audit_split,
    disjoint_label_groups,
    dominant_source_share,
    parse_label_file,
    source_of,
)

NAMES = ["Plate", "WithHelmet", "WithoutHelmet", "TripleRiding"]


def _split(tmp_path, files):
    img = tmp_path / "images"
    lbl = tmp_path / "labels"
    img.mkdir(parents=True)
    lbl.mkdir(parents=True)
    pairs = []
    for name, text in files.items():
        (img / f"{name}.jpg").write_bytes(b"\xff\xd8\xff")
        lp = lbl / f"{name}.txt"
        if text is not None:
            lp.write_text(text)
        pairs.append((str(img / f"{name}.jpg"), str(lp)))
    return pairs


def test_parse_flags_every_kind_of_malformed_line():
    boxes, problems = parse_label_file(
        "0 0.5 0.5 0.1 0.1\n"
        "9 0.5 0.5 0.1 0.1\n"        # class out of range
        "1 0.5 0.5 0 0.1\n"          # zero size
        "2 1.5 0.5 0.1 0.1\n"        # centre outside the image (kept, flagged)
        "3 0.5\n"                    # field count
        "x 0.5 0.5 0.1 0.1\n", 4)    # not numeric
    kinds = sorted(k for k, _ in problems)
    assert kinds == ["class_out_of_range", "coords_out_of_range", "field_count",
                     "non_positive_size", "not_numeric"]
    assert [b[0] for b in boxes] == [0, 2]


def test_audit_counts_duplicates_contradictions_and_tiny_boxes(tmp_path):
    pairs = _split(tmp_path, {
        "ds1_aaaaaa_a": "0 0.5 0.9 0.01 0.01\n0 0.5 0.9 0.01 0.01\n",   # dup tiny plate
        "ds1_bbbbbb_b": "1 0.5 0.5 0.2 0.4\n2 0.5 0.5 0.2 0.4\n",       # contradiction
        "dst_cccccc_c": "3 0.5 0.5 0.6 0.8\n",
        "dst_dddddd_d": None,                                            # background
    })
    r = audit_split(pairs, NAMES)
    assert r["duplicate_boxes"] == 1
    assert r["helmet_contradictions"] == 1
    assert r["per_class"]["Plate"]["tiny_boxes"] == 2
    assert r["background_images"] == 1
    assert r["sources"] == {"ds1": 2, "dst": 2}
    assert dominant_source_share(r, "TripleRiding") == ("dst", 1.0)


def test_disjoint_label_groups_detects_a_glued_together_dataset(tmp_path):
    glued = audit_split(_split(tmp_path / "g", {
        "ds1_aaaaaa_a": "0 0.5 0.9 0.1 0.05\n2 0.5 0.5 0.2 0.4\n",
        "dst_bbbbbb_b": "3 0.5 0.5 0.6 0.8\n",
    }), NAMES)
    assert disjoint_label_groups(glued) == [["Plate", "WithoutHelmet"], ["TripleRiding"]]
    joined = audit_split(_split(tmp_path / "j", {
        "ds1_aaaaaa_a": "0 0.5 0.9 0.1 0.05\n3 0.5 0.5 0.6 0.8\n",
    }), NAMES)
    assert disjoint_label_groups(joined) == [["Plate", "TripleRiding"]]


def test_source_is_the_export_prefix():
    assert source_of("ds1_58e91e_BikesHelmets558_png.rf.x.jpg") == "ds1"
    assert source_of("/a/b/aug_8a4368db.jpg") == "aug"


@pytest.mark.parametrize("name", sorted(corruptions.CORRUPTIONS))
def test_corruptions_keep_shape_and_are_deterministic(name):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    f = corruptions.get(name)
    a, b = f(img, "k"), f(img, "k")
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, img)


def test_cache_tag_changes_with_parameters(monkeypatch):
    tag = corruptions.cache_tag("glare_mild")
    monkeypatch.setitem(corruptions.CORRUPTIONS, "glare_mild",
                        (corruptions.glare, {"strength": 0.9}))
    assert corruptions.cache_tag("glare_mild") != tag
