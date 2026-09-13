"""Split-hygiene checks: the leakage detector must find real duplicates and not
invent them, because the honesty of every held-out metric rests on it."""

from __future__ import annotations

import os

import numpy as np
import pytest

from modules.dataset_audit import (
    audit_splits,
    build_clean_split,
    dhash,
    find_duplicates_within,
    find_leaks,
    hamming,
    hash_directory,
    iter_images,
)

cv2 = pytest.importorskip("cv2")


def _gradient_image(seed: int, size: int = 128):
    """A deterministic textured image; different seeds look genuinely different."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(size // 8, size // 8, 3), dtype=np.uint8)
    return cv2.resize(base, (size, size), interpolation=cv2.INTER_LINEAR)


def test_dhash_is_stable_and_64_bit():
    img = _gradient_image(1)
    h = dhash(img)
    assert 0 <= h < 2**64
    assert dhash(img) == h  # deterministic


def test_dhash_survives_recompression_and_resize(tmp_path):
    """The transforms an augmentation/export pass applies must not change the hash
    much — that is the whole reason a perceptual hash is used instead of a
    checksum."""
    img = _gradient_image(2, size=256)
    path = tmp_path / "a.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    recompressed = cv2.imread(str(path))
    resized = cv2.resize(img, (200, 200))

    assert hamming(dhash(img), dhash(recompressed)) <= 5
    assert hamming(dhash(img), dhash(resized)) <= 5


def test_dhash_separates_different_images():
    a, b = _gradient_image(10), _gradient_image(99)
    assert hamming(dhash(a), dhash(b)) > 5


def test_dhash_rejects_none():
    with pytest.raises(ValueError):
        dhash(None)


def test_hamming_counts_differing_bits():
    assert hamming(0b1010, 0b1010) == 0
    assert hamming(0b1010, 0b1011) == 1
    assert hamming(0, 0xFF) == 8


def _write_split(root: str, name: str, seeds, *, with_labels: bool = True):
    img_dir = os.path.join(root, name, "images")
    lbl_dir = os.path.join(root, name, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for seed in seeds:
        cv2.imwrite(os.path.join(img_dir, f"img{seed}.jpg"), _gradient_image(seed))
        if with_labels:
            with open(os.path.join(lbl_dir, f"img{seed}.txt"), "w") as fh:
                fh.write("0 0.5 0.5 0.2 0.2\n")
    return img_dir


def test_find_leaks_flags_shared_images(tmp_path):
    train = _write_split(str(tmp_path), "train", [1, 2, 3])
    # seed 2 is shared with train; seeds 50/51 are not.
    held = _write_split(str(tmp_path), "test", [2, 50, 51])

    report = find_leaks(
        hash_directory(held), hash_directory(train), max_distance=5, split="test"
    )
    assert report.held_out_images == 3
    assert report.leaked_images == 1
    assert report.leak_rate == pytest.approx(1 / 3)
    assert len(report.leaked_paths) == 1
    assert "img2.jpg" in report.leaked_paths[0]


def test_find_leaks_clean_split_reports_zero(tmp_path):
    train = _write_split(str(tmp_path), "train", [1, 2, 3])
    held = _write_split(str(tmp_path), "test", [80, 81])
    report = find_leaks(hash_directory(held), hash_directory(train), max_distance=5)
    assert report.leaked_images == 0
    assert report.leaked_paths == []


def test_find_duplicates_within_counts_pairs(tmp_path):
    d = os.path.join(str(tmp_path), "s", "images")
    os.makedirs(d)
    img = _gradient_image(7)
    for name in ("a.jpg", "b.jpg"):  # byte-identical copies
        cv2.imwrite(os.path.join(d, name), img)
    cv2.imwrite(os.path.join(d, "c.jpg"), _gradient_image(77))
    assert find_duplicates_within(hash_directory(d)) == 1


def test_audit_splits_reports_clean_flag(tmp_path):
    _write_split(str(tmp_path), "train", [1, 2, 3])
    _write_split(str(tmp_path), "test", [90, 91])
    out = audit_splits(
        os.path.join(str(tmp_path), "train", "images"),
        {"test": os.path.join(str(tmp_path), "test", "images")},
        max_distance=5,
    )
    assert out["clean"] is True
    assert out["train_images"] == 3
    assert out["splits"]["test"]["held_out_images"] == 2
    assert "dHash" in out["method"]


def test_audit_splits_flags_leakage(tmp_path):
    _write_split(str(tmp_path), "train", [1, 2, 3])
    _write_split(str(tmp_path), "test", [3, 90])
    out = audit_splits(
        os.path.join(str(tmp_path), "train", "images"),
        {"test": os.path.join(str(tmp_path), "test", "images")},
        max_distance=5,
    )
    assert out["clean"] is False
    assert out["splits"]["test"]["leaked_images"] == 1


def test_build_clean_split_drops_leaked_and_keeps_labels(tmp_path):
    held = _write_split(str(tmp_path), "test", [5, 6, 7])
    leaked = [os.path.join(held, "img6.jpg")]
    out_root = os.path.join(str(tmp_path), "clean")

    result = build_clean_split(held, leaked, out_root, split_name="test")
    assert result["kept"] == 2
    assert result["dropped_leaked"] == 1
    assert result["images_without_labels"] == 0

    kept = sorted(os.listdir(os.path.join(out_root, "test", "images")))
    assert kept == ["img5.jpg", "img7.jpg"]
    labels = sorted(os.listdir(os.path.join(out_root, "test", "labels")))
    assert labels == ["img5.txt", "img7.txt"]


def test_build_clean_split_is_idempotent(tmp_path):
    """Re-running the audit must not fail on the symlinks it made last time."""
    held = _write_split(str(tmp_path), "test", [5, 6])
    out_root = os.path.join(str(tmp_path), "clean")
    build_clean_split(held, [], out_root, split_name="test")
    again = build_clean_split(held, [], out_root, split_name="test")
    assert again["kept"] == 2


def test_iter_images_is_sorted_and_filters_extensions(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for name in ("b.jpg", "a.png", "notes.txt", "c.JPEG"):
        (d / name).write_bytes(b"x")
    names = [os.path.basename(p) for p in iter_images(str(d))]
    assert names == ["a.png", "b.jpg", "c.JPEG"]


def test_hash_directory_skips_unreadable_files(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    (d / "broken.jpg").write_bytes(b"not an image")
    cv2.imwrite(str(d / "good.jpg"), _gradient_image(3))
    hashes = hash_directory(str(d))
    assert len(hashes) == 1
    assert "good.jpg" in next(iter(hashes))


def test_hash_directory_missing_dir_is_empty():
    assert hash_directory("/nonexistent/path/xyz") == {}
