"""build_train_split.py: the de-duplicated training split must contain exactly
the images the script decided to keep, on every rerun."""

from __future__ import annotations

import os

import pytest

from build_train_split import choose_unique, drop_heldout_matches, materialise


def _touch_split(root, names):
    img_dir = os.path.join(root, "src", "images")
    lbl_dir = os.path.join(root, "src", "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    paths = []
    for n in names:
        p = os.path.join(img_dir, f"{n}.jpg")
        with open(p, "wb") as fh:
            fh.write(b"\xff\xd8\xff")
        with open(os.path.join(lbl_dir, f"{n}.txt"), "w") as fh:
            fh.write("0 0.5 0.5 0.1 0.1\n")
        paths.append(p)
    return paths


def test_materialise_rerun_removes_images_no_longer_kept(tmp_path):
    """Regression: a rerun that keeps fewer images (say a held-out set grew)
    left the previous run's links in train/images, so the 'leakage-free'
    training split still trained on the newly excluded image."""
    a, b, c = _touch_split(str(tmp_path), ["a", "b", "c"])
    out = os.path.join(str(tmp_path), "clean")
    assert materialise([a, b, c], out) == 3
    assert materialise([a, c], out) == 2
    assert sorted(os.listdir(os.path.join(out, "train", "images"))) == ["a.jpg", "c.jpg"]
    assert sorted(os.listdir(os.path.join(out, "train", "labels"))) == ["a.txt", "c.txt"]


def test_materialise_refuses_to_delete_a_real_file(tmp_path):
    (a,) = _touch_split(str(tmp_path), ["a"])
    out = os.path.join(str(tmp_path), "clean")
    real = os.path.join(out, "train", "images", "mine.jpg")
    os.makedirs(os.path.dirname(real))
    with open(real, "wb") as fh:
        fh.write(b"data")
    with pytest.raises(ValueError, match="mine.jpg"):
        materialise([a], out)
    assert os.path.isfile(real)


def test_choose_unique_collapses_only_identical_hashes():
    # Documents the actual behaviour: internal de-duplication is exact-dHash
    # only (distance 0); --max-distance applies to held-out matching.
    hashes = {"b.jpg": 0b1111, "a.jpg": 0b1111, "c.jpg": 0b1110}
    keep = choose_unique(hashes)
    assert keep == {0b1111: "a.jpg", 0b1110: "c.jpg"}


def test_drop_heldout_matches_uses_the_distance():
    train = {"t1": 0b0000, "t2": 0b1111_0000, "t3": 0b1111_1111_0000_0000}
    held = {"h": 0b0001}
    kept, dropped = drop_heldout_matches(sorted(train), train, held, max_distance=1)
    assert dropped == 1 and "t1" not in kept
    kept0, dropped0 = drop_heldout_matches(sorted(train), train, held, max_distance=0)
    assert dropped0 == 0 and kept0 == sorted(train)
