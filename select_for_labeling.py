"""Build the next labelling queue (see ``modules/active_learning.py``).

    python3 select_for_labeling.py --model runs/detect/traffic_model-2/weights/best.pt \\
        --train-data datasets/train_clean/data.yaml --eval-data eval/clean_splits/data.yaml \\
        --db traffic.db --budget 150 --device mps

Sources: detector predictions on the training pool (-> ``training_relabel``);
pending pipeline decisions with evidence sidecars and withheld violations from
the database (-> ``review``). Writes ``eval/results/labeling_queue.{json,md}``;
the report compares each signal's share in the queue with its base rate.

Evaluation images are NEVER chosen by model signals. Relabelling only the
held-out images where the model disagrees with the label inflates measured
accuracy: model-favoured "corrections" get made, while errors the model and the
label share are never found. Held-out labels are instead audited on a seeded,
uniform-random sample (``--eval-audit``), to be labelled without seeing
predictions (``evaluation_audit``, ``show_predictions: false``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from modules.logging_setup import configure_logging, get_logger

log = get_logger("select_for_labeling")


def _source_classes(rows_by_split: dict, names: list) -> dict:
    """Classes each upstream source ever labels, over every split we have."""
    from modules.label_audit import source_of

    seen: dict[str, set] = {}
    for rows in rows_by_split.values():
        for r in rows:
            s = source_of(os.path.basename(r["image"]))
            seen.setdefault(s, set()).update(names[c] for c, *_ in r["gt"])
    return seen


def detector_candidates(rows, split, purpose, names, source_classes, op, portable):
    import cv2

    from modules.active_learning import Candidate, image_signals, pattern_of
    from modules.dataset_audit import dhash
    from modules.label_audit import source_of

    out = []
    for r in rows:
        gt = [(names[c], tuple(b)) for c, *b in r["gt"]]
        preds = [(names[c], tuple(p[:4]), p[4]) for c, *p in r["preds"]]
        riders = sum(1 for n, _ in gt if n != "Plate")
        pred_riders = sum(1 for n, _, s in preds if n != "Plate" and s >= 0.25)
        src = source_of(os.path.basename(r["image"]))
        sig = image_signals(gt, preds, source_classes=source_classes.get(src),
                            op_thresholds=op, weak_riders=max(riders, pred_riders) >= 2)
        if not sig:
            continue
        img = cv2.imread(r["image"])
        out.append(Candidate(
            item_id=f"{split}:{os.path.basename(r['image'])}", source="detector_pool",
            purpose=purpose, signals=sig, pattern=f"{purpose}:{pattern_of(sig)}",
            image=portable(r["image"]), dhash=dhash(img) if img is not None else None,
            detail={"split": split, "upstream_source": src, "labelled": len(gt),
                    "predicted_confident": sum(1 for *_, s in preds if s >= 0.6)}))
    return out


def review_candidates(db_path, evidence_dir):
    from modules.active_learning import WITHHELD_SIGNAL, Candidate, pattern_of, review_signals
    from modules.db import Database

    db = Database(db_path)
    out = []
    items, offset = [], 0
    while True:  # list_violations pages at 500
        page = db.list_violations(review_status="pending", limit=500, offset=offset)["items"]
        if not page:
            break
        items += page
        offset += len(page)
    for v in items:
        detail = db.get_violation(v["id"]) or {}
        meta_url = (detail.get("evidence") or {}).get("metadata")
        if not meta_url:
            continue  # not pipeline output (no sidecar): nothing to learn from
        path = os.path.join(evidence_dir, os.path.basename(meta_url))
        try:
            with open(path) as fh:
                sidecar = json.load(fh)
        except (OSError, ValueError):
            continue
        sig = review_signals(v, sidecar)
        if sig:
            out.append(Candidate(item_id=f"violation:{v['id']}", source="review_queue",
                                 purpose="review", signals=sig,
                                 pattern=f"review:{pattern_of(sig)}",
                                 detail={"type": v.get("type"), "plate": v.get("plate"),
                                         "session": v.get("session_id")}))
    for ev in db.audit_events("violation_withheld"):
        meta = json.loads(ev.get("metadata") or "{}")
        for w in meta.get("withheld", [meta]):
            reason = w.get("reason")
            s = WITHHELD_SIGNAL.get(reason, 0.3)
            out.append(Candidate(item_id=f"withheld:{ev['record_id']}:{w.get('track_id')}",
                                 source="withheld", purpose="review",
                                 signals={"withheld": s}, pattern=f"review:withheld:{reason}",
                                 detail={"session": ev["record_id"], **w}))
    return out, len(items)


def blind_audit(images: list[tuple[str, str]], n: int, seed: int, portable) -> list[dict]:
    """A seeded uniform-random sample of evaluation images to relabel WITHOUT
    looking at predictions — the only unbiased way to measure (and fix)
    held-out label noise."""
    import random

    rng = random.Random(seed)
    chosen = rng.sample(sorted(images), min(n, len(images)))
    return [{"item_id": f"{split}:{os.path.basename(path)}", "image": portable(path),
             "purpose": "evaluation_audit", "show_predictions": False,
             "selection": f"uniform random, seed {seed}"} for split, path in chosen]


def render(queue, pool, pool_sizes, budget, review_pool, audit_n=0) -> str:
    from modules.active_learning import RESOLVES

    def share(items, key):
        return sum(1 for c in items if key in c.signals) / len(items) if items else 0.0

    keys = sorted({k for c in pool for k in c.signals})
    lines = ["# Labelling queue — active-learning selection", "",
             f"- Budget {budget}; selected **{len(queue)}** from "
             f"{sum(pool_sizes.values())} pool images "
             f"({', '.join(f'{k} {v}' for k, v in pool_sizes.items())}) and {review_pool} "
             "pending reviews; candidates with any signal: "
             f"{len(pool)}.",
             "- Priority ranks informativeness (`1 - Π(1 - signal)`), not a probability; "
             "selection decays repeats of a pattern and skips near-duplicate images.",
             "- Evaluation images are **not** model-selected: they get a separate, "
             "seeded uniform-random audit sample, labelled blind to predictions "
             "(`labeling_queue_eval_audit.json`). Model-guided relabelling of held-out "
             "data would inflate measured accuracy.",
             f"- Blind evaluation audit: **{audit_n}** val/test images.",
             "- Whether this queue beats random selection is **not yet measured** — that "
             "needs the labels back (docs/FIELD_EVALUATION.md §5).", "",
             "## Signal share: queue vs pool", "",
             "| signal | pool candidates | queue |", "|---|---:|---:|"]
    for k in keys:
        lines.append(f"| {k} | {share(pool, k):.0%} | {share(queue, k):.0%} |")
    by_pattern: dict[str, int] = {}
    for c in queue:
        by_pattern[c.pattern] = by_pattern.get(c.pattern, 0) + 1
    lines += ["", "## Queue by pattern — what a label would resolve", "",
              "| pattern | items | what it resolves | measured by |", "|---|---:|---|---|"]
    for pat, n in sorted(by_pattern.items(), key=lambda kv: -kv[1]):
        base = pat.split(":")[1]
        why, metric = RESOLVES.get(base, ("", ""))
        lines.append(f"| `{pat}` | {n} | {why} | {metric} |")
    src_q: dict[str, int] = {}
    src_p: dict[str, int] = {}
    for items, sink in ((queue, src_q), (pool, src_p)):
        for c in items:
            s = c.detail.get("upstream_source", c.source)
            sink[s] = sink.get(s, 0) + 1
    lines += ["", "## By upstream source", "",
              "`aug` images are offline-augmented copies whose original is not recoverable "
              "(`label_audit.source_of`): a label fixed on one fixes that copy only — as it "
              "would on an original, whose copies can't be found either. Dropping offline "
              "augmentation for online augmentation of cleaned originals is the structural "
              "fix (docs/ROADMAP.md).", "",
              "| source | pool candidates | queue |", "|---|---:|---:|"]
    for s in sorted(set(src_p) | set(src_q)):
        lines.append(f"| {s} | {src_p.get(s, 0)} | {src_q.get(s, 0)} |")
    lines += ["", "## Top 25", "", "| # | item | purpose | priority | signals |",
              "|---:|---|---|---:|---|"]
    for i, c in enumerate(queue[:25], 1):
        sig = ", ".join(f"{k} {v:.2f}" for k, v in sorted(c.signals.items(),
                                                        key=lambda kv: -kv[1]))
        lines.append(f"| {i} | `{c.item_id}` | {c.purpose} | {c.priority} | {sig} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    configure_logging()
    ap = argparse.ArgumentParser(description="Build the next labelling queue.")
    ap.add_argument("--model", default="runs/detect/traffic_model-2/weights/best.pt")
    ap.add_argument("--train-data", default="datasets/train_clean/data.yaml")
    ap.add_argument("--eval-data", default="eval/clean_splits/data.yaml")
    ap.add_argument("--db", default="traffic.db")
    ap.add_argument("--evidence-dir", default="evidence")
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--eval-audit", type=int, default=50,
                    help="uniform-random, model-blind audit sample of val/test images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decay", type=float, default=0.6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default="eval/cache")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--name", default="labeling_queue")
    args = ap.parse_args(argv)

    from evaluate_conditions import load_or_predict_boxes, operating_thresholds
    from modules import yolo_io
    from modules.active_learning import select
    from modules.provenance import portable_path, run_provenance

    device = yolo_io.resolve_device(args.device)
    op = operating_thresholds()
    rows_by_split, names = {}, []
    plan = [(args.train_data, "train"), (args.eval_data, "val"), (args.eval_data, "test")]
    have_model = os.path.exists(args.model)
    for data, split in plan:
        if have_model and os.path.exists(data):
            rows_by_split[split], names = load_or_predict_boxes(
                args.model, data, split, imgsz=640, device=device, cache_dir=args.cache_dir)
    source_classes = _source_classes(rows_by_split, names)
    pool = []
    if "train" in rows_by_split:
        pool += detector_candidates(rows_by_split["train"], "train", "training_relabel",
                                    names, source_classes, op, portable_path)
    audit = blind_audit([(split, r["image"]) for split in ("val", "test")
                         for r in rows_by_split.get(split, [])], args.eval_audit, args.seed,
                        portable_path)
    review_pool = 0
    if os.path.exists(args.db):
        rc, review_pool = review_candidates(args.db, args.evidence_dir)
        pool += rc
    queue = select(pool, args.budget, decay=args.decay)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.name}.json"), "w") as fh:
        json.dump([c.as_dict() for c in queue], fh, indent=1, sort_keys=True)
    with open(os.path.join(args.out, f"{args.name}_eval_audit.json"), "w") as fh:
        json.dump(audit, fh, indent=1, sort_keys=True)
    sizes = {k: len(v) for k, v in rows_by_split.items()}
    with open(os.path.join(args.out, f"{args.name}.md"), "w") as fh:
        fh.write(render(queue, pool, sizes, args.budget, review_pool, len(audit)))
    with open(os.path.join(args.out, f"{args.name}.provenance.json"), "w") as fh:
        json.dump(run_provenance(model_paths=[args.model] if have_model else [],
                                 config={"budget": args.budget, "decay": args.decay,
                                         "eval_audit": args.eval_audit, "seed": args.seed,
                                         "op_thresholds": op,
                                         "source_classes": {k: sorted(v) for k, v in
                                                            source_classes.items()}}),
                  fh, indent=2, sort_keys=True)
    print(f"selected {len(queue)} of {len(pool)} candidates -> {args.out}/{args.name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
