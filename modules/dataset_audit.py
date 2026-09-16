"""Dataset split hygiene: is the held-out data actually held out?

[DATASET.md](../docs/DATASET.md) flagged leakage as **UNVERIFIED**: the training
images were re-hashed by an offline augmentation pass (``aug_<hash>.jpg``), so a
val/test image cannot be matched to its training copy by filename. That leaves
open the worst failure in an evaluation story — reporting metrics on images the
model was trained on.

This module closes that gap by comparing *pixel content* instead of names, using
a **difference hash (dHash)**: downscale to a small grey thumbnail and record
whether each pixel is brighter than its right-hand neighbour. The result is a
64-bit fingerprint that survives JPEG recompression, resizing, and mild
brightness/contrast shifts — exactly the transforms an augmentation pass applies
— while still separating genuinely different scenes. Two images are considered
near-duplicates when their hashes differ in at most ``max_distance`` bits.

Only numpy + cv2 are needed (both CI dependencies); no model, no torch. The
hashing itself is pure and unit-tested, so the honest-evaluation claim is backed
by code that runs in CI even though the images are gitignored.

Caveat worth stating plainly: dHash is a *perceptual* hash, not a proof. It
reliably catches recompression/resize/mild-photometric duplicates and it will
miss a heavy geometric augmentation (a large rotation or crop). So a clean report
is evidence of split hygiene, not a mathematical guarantee — see
:func:`audit_splits`' ``method`` field, which records exactly that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 8x9 grey thumbnail -> 8 rows x 8 comparisons = 64 bits.
HASH_SIZE = 8


def dhash(image, hash_size: int = HASH_SIZE) -> int:
    """Difference hash of a BGR/grey image as a ``hash_size**2``-bit integer.

    Resizing to ``(hash_size + 1) x hash_size`` and comparing horizontally
    adjacent pixels makes the hash depend on local *gradients* rather than
    absolute intensity, so it is stable under global brightness/contrast changes
    and JPEG artefacts.
    """
    import cv2
    import numpy as np

    if image is None:
        raise ValueError("dhash got no image")
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = np.asarray(small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in diff:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return int(a ^ b).bit_count()


def iter_images(directory: str):
    """Yield every image path under ``directory`` in sorted, deterministic order."""
    if not os.path.isdir(directory):
        return
    for dirpath, dirs, files in os.walk(directory):
        dirs.sort()
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(dirpath, name)


def hash_directory(directory: str, *, limit: int = 0) -> dict[str, int]:
    """``{image_path: dhash}`` for every readable image under ``directory``."""
    import cv2

    out: dict[str, int] = {}
    for i, path in enumerate(iter_images(directory)):
        if limit and i >= limit:
            break
        img = cv2.imread(path)
        if img is None:
            continue
        out[path] = dhash(img)
    return out


@dataclass
class LeakPair:
    held_out_image: str
    train_image: str
    distance: int


@dataclass
class LeakReport:
    """Result of comparing one held-out split against the training split."""

    split: str
    held_out_images: int = 0
    train_images: int = 0
    max_distance: int = 0
    leaked_images: int = 0
    examples: list = field(default_factory=list)
    # Every leaked held-out image path (the capped `examples` above is for the
    # human-readable report; this is what builds a clean split).
    leaked_paths: list = field(default_factory=list)

    @property
    def leak_rate(self) -> float:
        return self.leaked_images / self.held_out_images if self.held_out_images else 0.0

    def as_dict(self) -> dict:
        return {
            "split": self.split,
            "held_out_images": self.held_out_images,
            "train_images": self.train_images,
            "max_distance": self.max_distance,
            "leaked_images": self.leaked_images,
            "leak_rate": round(self.leak_rate, 4),
            "examples": [e.__dict__ for e in self.examples[:20]],
        }


def find_leaks(
    held_out: dict[str, int],
    train: dict[str, int],
    *,
    max_distance: int = 5,
    split: str = "held_out",
    max_examples: int = 20,
) -> LeakReport:
    """Report held-out images whose dHash is within ``max_distance`` of a train image.

    Exact matches (distance 0) are found through a dict lookup; only when a
    non-zero ``max_distance`` is requested does it fall back to the O(n*m) bit
    comparison, which is fine at this dataset's scale (11k x 400) and keeps the
    logic obvious.
    """
    report = LeakReport(
        split=split, held_out_images=len(held_out), train_images=len(train),
        max_distance=max_distance,
    )
    by_hash: dict[int, str] = {}
    for path, h in train.items():
        by_hash.setdefault(h, path)

    train_items = list(train.items())
    for path, h in sorted(held_out.items()):
        exact = by_hash.get(h)
        if exact is not None:
            report.leaked_images += 1
            report.leaked_paths.append(path)
            if len(report.examples) < max_examples:
                report.examples.append(LeakPair(path, exact, 0))
            continue
        if max_distance <= 0:
            continue
        best_path, best_dist = None, max_distance + 1
        for tpath, th in train_items:
            d = hamming(h, th)
            if d < best_dist:
                best_path, best_dist = tpath, d
                if d == 1:  # close enough to stop searching
                    break
        if best_path is not None and best_dist <= max_distance:
            report.leaked_images += 1
            report.leaked_paths.append(path)
            if len(report.examples) < max_examples:
                report.examples.append(LeakPair(path, best_path, best_dist))
    return report


def find_duplicates_within(hashes: dict[str, int], *, max_distance: int = 0) -> int:
    """Count near-duplicate *pairs* inside one split (a small val set padded with
    duplicates measures less than its image count suggests)."""
    items = sorted(hashes.items())
    dupes = 0
    for i, (_pa, ha) in enumerate(items):
        for _pb, hb in items[i + 1:]:
            if hamming(ha, hb) <= max_distance:
                dupes += 1
    return dupes


def audit_splits(
    train_dir: str,
    held_out_dirs: dict[str, str],
    *,
    max_distance: int = 5,
    limit: int = 0,
) -> dict:
    """Hash the training split once and compare every held-out split against it."""
    train = hash_directory(train_dir, limit=limit)
    reports = {}
    for split, directory in sorted(held_out_dirs.items()):
        held = hash_directory(directory, limit=limit)
        rep = find_leaks(held, train, max_distance=max_distance, split=split)
        d = rep.as_dict()
        d["internal_duplicate_pairs"] = find_duplicates_within(held)
        d["leaked_paths"] = list(rep.leaked_paths)
        reports[split] = d
    return {
        "method": (
            f"dHash({HASH_SIZE}x{HASH_SIZE}, 64-bit) with Hamming distance <= "
            f"{max_distance}; catches recompression/resize/photometric duplicates, "
            "not heavy geometric augmentation"
        ),
        "train_dir": train_dir,
        "train_images": len(train),
        "max_distance": max_distance,
        "splits": reports,
        "clean": all(r["leaked_images"] == 0 for r in reports.values()),
    }


def build_clean_split(
    held_out_dir: str,
    leaked_images: list,
    out_root: str,
    *,
    split_name: str = "test",
) -> dict:
    """Materialise a de-leaked copy of a split as a YOLO ``images/`` + ``labels/`` tree.

    Files are **symlinked**, not copied, so a clean split costs no disk and stays
    obviously derived from the original. The result is a normal YOLO split that
    ``evaluate_model.py`` and Ultralytics ``val()`` read with no special casing,
    which is the point: the honest number comes from the same tool as the
    inflated one, with only the data changed.
    """
    leaked = set(leaked_images)
    img_dir = os.path.join(out_root, split_name, "images")
    lbl_dir = os.path.join(out_root, split_name, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    kept = dropped = missing_labels = 0
    for src in iter_images(held_out_dir):
        if src in leaked:
            dropped += 1
            continue
        label_src = os.path.splitext(
            src.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        )[0] + ".txt"
        base = os.path.basename(src)
        _link(os.path.abspath(src), os.path.join(img_dir, base))
        if os.path.exists(label_src):
            _link(os.path.abspath(label_src),
                  os.path.join(lbl_dir, os.path.splitext(base)[0] + ".txt"))
        else:
            missing_labels += 1
        kept += 1
    return {
        "root": os.path.abspath(out_root),
        "split": split_name,
        "kept": kept,
        "dropped_leaked": dropped,
        "images_without_labels": missing_labels,
    }


def _link(src: str, dst: str) -> None:
    """Symlink ``src`` to ``dst``, replacing any stale link (idempotent reruns)."""
    if os.path.islink(dst) or os.path.exists(dst):
        os.unlink(dst)
    os.symlink(src, dst)
