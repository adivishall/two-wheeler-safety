"""Provenance for every generated result: which weights, which data, which code.

"Which exact model generated this result?" has to be answerable from the result
file alone. :func:`run_provenance` is attached to every evaluation/benchmark
report and records:

* the weights' SHA-256 (``best.pt`` is a path, not an identity) and the
  registered model version if a manifest matches that hash;
* the dataset's content fingerprint (:func:`dataset_fingerprint`);
* the git commit, and whether the working tree had uncommitted changes;
* library versions and the hardware the numbers were measured on.

:func:`dataset_fingerprint` is the ONE dataset-version scheme; the model
manifest and the dataset manifest both use it. The two schemes it replaced
hashed ``data.yaml`` plus per-class *counts* — so relabelling a box without
changing the counts kept the same "version" — and one of them folded the
dataset's absolute path into the hash, so moving the repo changed it.
"""

from __future__ import annotations

import glob
import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_KEYS = ("train", "val", "test")


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _split_dirs(data_yaml: str) -> dict[str, str]:
    cfg = _load_yaml(data_yaml)
    base = os.path.dirname(os.path.abspath(data_yaml))
    root = cfg.get("path") or base
    if not os.path.isabs(root):
        root = os.path.join(base, root)
    out = {}
    for key in SPLIT_KEYS:
        entry = cfg.get(key)
        if isinstance(entry, str) and entry:
            out[key] = entry if os.path.isabs(entry) else os.path.join(root, entry)
    return out


def dataset_fingerprint(data_yaml: str, *, deep: bool = False) -> dict:
    """Content fingerprint of a YOLO dataset, independent of where it lives.

    Per split, hashes the sorted list of ``(image name, image size, sha256 of
    its label file)``. So adding/removing/renaming an image, or changing any
    annotation, changes the version; moving the directory does not. With
    ``deep=True`` image *bytes* are hashed too (catches a same-size pixel
    edit; ~20 s on the 12 GB train split, so it is opt-in).

    The class list from ``data.yaml`` is part of the hash (reordering classes
    silently changes what every label means); absolute paths are not.
    """
    cfg = _load_yaml(data_yaml)
    names = cfg.get("names")
    classes = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names or [])
    top = hashlib.sha256(("classes:" + "|".join(classes)).encode())
    splits: dict[str, dict] = {}
    for split, img_dir in _split_dirs(data_yaml).items():
        if not os.path.isdir(img_dir):
            continue
        label_dir = img_dir.replace(os.sep + "images", os.sep + "labels")
        images = {
            os.path.splitext(n)[0]: n for n in os.listdir(img_dir)
            if os.path.splitext(n)[1].lower() in IMAGE_EXTS
        }
        labels = (
            {os.path.splitext(n)[0] for n in os.listdir(label_dir) if n.endswith(".txt")}
            if os.path.isdir(label_dir) else set()
        )
        h = hashlib.sha256()
        for stem in sorted(set(images) | labels):
            if stem in images:
                path = os.path.join(img_dir, images[stem])
                h.update(f"{images[stem]}:{os.path.getsize(path)}:".encode())
                if deep:
                    with open(path, "rb") as fh:
                        h.update(hashlib.sha256(fh.read()).digest())
            else:
                h.update(f"{stem}:noimage:".encode())
            if stem in labels:
                with open(os.path.join(label_dir, stem + ".txt"), "rb") as fh:
                    h.update(hashlib.sha256(fh.read()).digest())
            else:
                h.update(b"nolabel")
        n_images, n_labels = len(images), len(labels)
        digest = h.hexdigest()
        splits[split] = {"images": n_images, "label_files": n_labels,
                         "sha256": f"sha256:{digest[:16]}"}
        top.update(f"{split}:{digest}".encode())
    return {
        "version": f"sha256:{top.hexdigest()[:16]}",
        "scheme": "names+sizes+label-contents" + ("+image-bytes" if deep else ""),
        "classes": classes,
        "splits": splits,
    }


def split_fingerprint(data_yaml: str, split: str) -> str | None:
    """The fingerprint of one split (e.g. the evaluated one), or None."""
    return dataset_fingerprint(data_yaml)["splits"].get(split, {}).get("sha256")


# Generated outputs: regenerating results must not mark the CODE as dirty.
GENERATED_PATHS = ("eval/results", "data/dataset_manifest.json",
                   "data/DATASET_MANIFEST.md", "models/manifests")
# Documentation no evaluator reads: editing it while results regenerate must not
# stamp them dirty. Tracked data inputs (images, JSON) still count.
DOC_PATTERNS = ("*.md",)


def git_state() -> dict:
    """HEAD commit, and whether any tracked file OTHER than generated outputs
    and documentation has uncommitted changes (``dirty`` means the code that
    produced a result is not exactly the commit named)."""
    def run(*args):
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True,
                                 timeout=5, check=False)
            return out.stdout.strip()
        except Exception:  # noqa: BLE001 - git absent
            return ""

    commit = run("rev-parse", "--short", "HEAD") or None
    if not commit:
        return {"commit": None, "dirty": None}
    excludes = [f":(exclude){p}" for p in (*GENERATED_PATHS, *DOC_PATTERNS)]
    tracked = run("status", "--porcelain", "--untracked-files=no", "--", ".", *excludes)
    # Untracked SOURCE files count too: HEAD may import a module that exists
    # only in this working tree, and then the named commit can't reproduce the
    # result. (Untracked non-code — scratch notes, editor dirs — does not.)
    untracked = run("ls-files", "--others", "--exclude-standard", "--", "*.py",
                    "*.yaml", "*.yml", "*.toml", "*.txt")
    return {"commit": commit, "dirty": bool(tracked) or bool(untracked)}


def environment() -> dict:
    """Software + hardware the numbers were produced on."""
    env: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }
    for mod in ("torch", "ultralytics", "easyocr", "cv2", "numpy"):
        m = sys.modules.get(mod)
        if m is not None:
            env[mod] = getattr(m, "__version__", None)
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            env["cuda"] = bool(torch.cuda.is_available())
            env["mps"] = bool(torch.backends.mps.is_available())
        except Exception:  # noqa: BLE001
            pass
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
            env["chip"] = chip or None
        except Exception:  # noqa: BLE001
            pass
    return env


def weights_identity(model_path: str) -> dict:
    from modules.model_manifest import model_version_string, resolve_manifest, sha256_file

    out = {"path": model_path, "sha256": None, "version": None}
    if model_path and os.path.exists(model_path):
        out["sha256"] = sha256_file(model_path)
        try:
            out["version"] = model_version_string(resolve_manifest(model_path))
        except Exception:  # noqa: BLE001 - no manifest is not an error
            out["version"] = None
    return out


def run_provenance(*, model_paths=(), data_yaml: str | None = None,
                   split: str | None = None, config: dict | None = None) -> dict:
    """Everything needed to say exactly what produced a result file."""
    prov: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "environment": environment(),
        "models": [weights_identity(p) for p in model_paths],
    }
    if data_yaml and os.path.exists(data_yaml):
        fp = dataset_fingerprint(data_yaml)
        prov["dataset"] = {"data_yaml": data_yaml, "version": fp["version"],
                           "scheme": fp["scheme"], "split": split,
                           "split_version": fp["splits"].get(split, {}).get("sha256")
                           if split else None,
                           "split_images": fp["splits"].get(split, {}).get("images")
                           if split else None}
    if config:
        prov["config"] = config
    return prov


def glob_weights(root: str = "runs/detect") -> list[str]:
    return sorted(glob.glob(os.path.join(root, "*", "weights", "best.pt")))
