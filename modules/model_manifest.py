"""Lightweight, machine-readable model manifests (Phase 2).

A trained YOLO checkpoint is an opaque ``best.pt``: nothing in it says which
dataset produced it, on what code, with what hyper-parameters, or how well it
scored. That makes a detection un-auditable ("which model raised this fine?")
and a model swap silent. This module attaches a small JSON manifest to a
checkpoint recording exactly that provenance, and lets the runtime stamp the
model version onto every evidence package.

The manifest is deliberately **decoupled from torch/ultralytics**: it is built
from the files a training run leaves behind (``args.yaml`` for the training
config, ``results.csv`` for the training date) plus a SHA-256 of the weights and
the current git commit. So it can be generated, read, and tested in plain CI
without the ~2 GB inference stack.

Fields (see :func:`build_manifest`):

* ``name`` / ``version`` — human identity of the model line.
* ``training_date`` — from the run's ``results.csv`` mtime (falls back to the
  checkpoint mtime); the wall-clock the weights were produced.
* ``created_at`` — when this manifest file was written (UTC ISO-8601).
* ``dataset`` — path + a content ``version`` hash + per-class instance counts.
* ``git_commit`` — repo HEAD when the manifest was generated (best-effort).
* ``classes`` — the trained class names, in dataset order.
* ``training_config`` — the salient hyper-parameters from ``args.yaml``.
* ``metrics`` — validation metrics, if an ``evaluate_model.py`` JSON is supplied
  (never fabricated; absent when no eval was provided).
* ``checksum`` — ``sha256:<hex>`` of the weights file, the tamper-evident id.

Nothing here trains or infers; it only describes.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

MANIFEST_BASENAME = "model_manifest.json"
# Central registry of manifests for checkpoints that live outside the repo
# (weights are gitignored, but a small JSON manifest is committable here).
REGISTRY_DIR = os.path.join("models", "manifests")

_CONFIG_KEYS = (
    "model", "data", "epochs", "imgsz", "batch", "device", "workers",
    "optimizer", "lr0", "seed", "deterministic", "patience", "close_mosaic",
    "single_cls", "cos_lr", "amp", "fraction",
)


def sha256_file(path: str, *, chunk: int = 1 << 20) -> str:
    """Return ``sha256:<hex>`` for a file, read in chunks (weights are big)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit(short: bool = True) -> str | None:
    """Current repo HEAD, or None if git isn't available / not a repo."""
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else [
            "git", "rev-parse", "HEAD"
        ]
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=5, check=False
        )
        commit = out.stdout.strip()
        return commit or None
    except Exception:  # noqa: BLE001 - git absent / timeout
        return None


def _load_yaml(path: str) -> dict:
    import yaml  # pyyaml ships with ultralytics and is a CI dep

    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _class_names(cfg: dict) -> list[str]:
    names = cfg.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names or [])


def dataset_version(data_yaml: str) -> dict:
    """Describe a dataset for provenance: a content hash + per-class counts.

    The ``version`` hash covers the ``data.yaml`` contents *and* the per-class
    label instance counts, so any change to the classes or the label files
    changes the version. Returns ``{path, version, classes, class_counts,
    images, instances}``. Counting walks the label files (cheap, no image I/O).
    """
    cfg = _load_yaml(data_yaml)
    classes = _class_names(cfg)
    root = cfg.get("path") or os.path.dirname(os.path.abspath(data_yaml))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), root)

    counts = {c: 0 for c in classes}
    total_instances = 0
    total_images = 0
    seen_dirs: set[str] = set()  # dedup when splits alias the same directory
    for split_key in ("train", "val", "test"):
        entry = cfg.get(split_key)
        if not entry:
            continue
        split_dir = entry if os.path.isabs(entry) else os.path.join(root, entry)
        label_dir = split_dir.replace(os.sep + "images", os.sep + "labels")
        real = os.path.realpath(label_dir)
        if real in seen_dirs or not os.path.isdir(label_dir):
            continue
        seen_dirs.add(real)
        for lp in glob.glob(os.path.join(label_dir, "*.txt")):
            total_images += 1
            with open(lp) as fh:
                for line in fh:
                    parts = line.split()
                    if not parts:
                        continue
                    idx = int(float(parts[0]))
                    total_instances += 1
                    if 0 <= idx < len(classes):
                        counts[classes[idx]] += 1

    with open(data_yaml, "rb") as fh:
        yaml_bytes = fh.read()
    version_src = yaml_bytes + json.dumps(counts, sort_keys=True).encode()
    return {
        "path": data_yaml,
        "version": f"sha256:{_sha256_bytes(version_src)[:16]}",
        "classes": classes,
        "class_counts": counts,
        "label_files": total_images,
        "instances": total_instances,
    }


def _training_config(run_dir: str) -> dict:
    args_path = os.path.join(run_dir, "args.yaml")
    if not os.path.exists(args_path):
        return {}
    cfg = _load_yaml(args_path)
    return {k: cfg.get(k) for k in _CONFIG_KEYS if k in cfg}


def _training_date(run_dir: str, model_path: str) -> str:
    results = os.path.join(run_dir, "results.csv")
    src = results if os.path.exists(results) else model_path
    return datetime.fromtimestamp(
        os.path.getmtime(src), tz=timezone.utc
    ).isoformat()


def _metrics_from_eval(eval_json: str) -> dict:
    """Pull the headline + per-class metrics out of an evaluate_model.py JSON."""
    with open(eval_json) as fh:
        payload = json.load(fh)
    official = payload.get("official") or {}
    return {
        "source": os.path.relpath(eval_json),
        "split": payload.get("split"),
        "map50": official.get("map50"),
        "map50_95": official.get("map50_95"),
        "mean_precision": official.get("mean_precision"),
        "mean_recall": official.get("mean_recall"),
        "per_class": official.get("per_class"),
    }


def build_manifest(
    model_path: str,
    *,
    name: str,
    version: str,
    data_yaml: str | None = None,
    eval_json: str | None = None,
    metrics: dict | None = None,
    classes: list[str] | None = None,
) -> dict:
    """Assemble a manifest dict for a trained checkpoint (see module docstring).

    Only ``model_path``/``name``/``version`` are required. ``data_yaml`` adds
    dataset provenance + class names; ``eval_json`` (an ``evaluate_model.py``
    output) or an explicit ``metrics`` dict adds real validation metrics
    (``metrics`` wins when both are given, e.g. a training run passing its own
    ``val()`` numbers). Nothing is invented: a field is omitted (None) rather
    than guessed.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(model_path)))
    dataset = dataset_version(data_yaml) if data_yaml else None
    if classes is None and dataset is not None:
        classes = dataset["classes"]

    if metrics is not None:
        resolved_metrics = metrics
    elif eval_json is not None:
        resolved_metrics = _metrics_from_eval(eval_json)
    else:
        resolved_metrics = None

    return {
        "name": name,
        "version": version,
        "checksum": sha256_file(model_path),
        "weights": os.path.relpath(model_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_date": _training_date(run_dir, model_path),
        "git_commit": git_commit(),
        "classes": list(classes) if classes else None,
        "dataset": dataset,
        "training_config": _training_config(run_dir),
        "metrics": resolved_metrics,
    }


def write_manifest(manifest: dict, model_path: str) -> tuple[str, str]:
    """Write the manifest next to the weights *and* into the committable
    registry (``models/manifests/<name>.json``). Returns both paths."""
    beside = os.path.join(os.path.dirname(model_path), MANIFEST_BASENAME)
    with open(beside, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    os.makedirs(REGISTRY_DIR, exist_ok=True)
    registry = os.path.join(REGISTRY_DIR, f"{manifest['name']}.json")
    with open(registry, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return beside, registry


def resolve_manifest(model_path: str) -> dict | None:
    """Best-effort lookup of the manifest for a checkpoint the runtime loaded.

    Checks the manifest beside the weights first; falls back to any registry
    manifest whose ``checksum`` matches the on-disk weights. Returns None if no
    manifest is found (the runtime then simply records no model version).
    """
    if not model_path:
        return None
    beside = os.path.join(os.path.dirname(model_path), MANIFEST_BASENAME)
    if os.path.exists(beside):
        try:
            with open(beside) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass

    if not os.path.exists(model_path):
        return None
    try:
        checksum = sha256_file(model_path)
    except OSError:
        return None
    for path in glob.glob(os.path.join(REGISTRY_DIR, "*.json")):
        try:
            with open(path) as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        if m.get("checksum") == checksum:
            return m
    return None


def model_version_string(manifest: dict | None) -> str | None:
    """A compact ``name@version`` id for stamping onto evidence, or None."""
    if not manifest:
        return None
    name = manifest.get("name")
    version = manifest.get("version")
    if name and version:
        return f"{name}@{version}"
    checksum = manifest.get("checksum", "")
    return checksum[:19] if checksum else None


def _parse_args(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate a machine-readable manifest for a trained checkpoint.",
    )
    ap.add_argument("--model", required=True, help="path to YOLO weights (.pt)")
    ap.add_argument("--name", required=True, help="model line name, e.g. traffic-4class")
    ap.add_argument("--version", required=True, help="semantic version, e.g. 1.0.0")
    ap.add_argument("--data", default=None, help="dataset data.yaml for provenance")
    ap.add_argument("--metrics", default=None, help="evaluate_model.py JSON for metrics")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    manifest = build_manifest(
        args.model,
        name=args.name,
        version=args.version,
        data_yaml=args.data,
        eval_json=args.metrics,
    )
    beside, registry = write_manifest(manifest, args.model)
    print(f"Wrote manifest:\n  {beside}\n  {registry}")
    print(f"Model version: {model_version_string(manifest)}")
    print(f"Checksum:      {manifest['checksum']}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
