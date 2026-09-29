"""Compare training recipes by their seed means, not single checkpoints.

    R=runs/detect
    python3 compare_recipes.py --split val --device mps --baseline v1 \\
        --recipe v1=$R/traffic_model-2/weights/best.pt,$R/traffic_model_seed1/weights/best.pt \\
        --recipe v2_dedup=$R/traffic_model_v2_dedup/weights/best.pt

A recipe's AP is the mean over its seeds; differences are bootstrapped over
images and seeds together (``detection_stats.recipe_comparison``). This is the
promotion test in docs/RETRAINING_LOOP.md §7: fewer than three seeds on either
side is reported as ``insufficient seeds``, never as a verdict. Choose on val;
``--split test`` is for the one report after the choice. Predictions come from
the ``evaluate_uncertainty.py`` cache. Writes ``eval/results/<name>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from modules.logging_setup import configure_logging, get_logger

log = get_logger("compare_recipes")


def render(rep: dict, names: list[str], split: str) -> str:
    lines = [f"# Recipe comparison — seed means on `{split}` ({rep['images']} images)", "",
             f"- Baseline **{rep['baseline']}**. A recipe's AP = mean over its seeds; CIs "
             "resample images and seeds together. Verdicts need ≥ "
             f"{rep['min_seeds']} seeds per recipe.", "",
             "| recipe | seeds | " + " | ".join(names) + " | mAP@50 |",
             "|---|---:|" + "---|" * len(names) + "---:|"]
    for r, info in rep["recipes"].items():
        cells = []
        for n in names:
            pc = info["per_class"][n]
            sd = f" ± {pc['between_seed_sd']}" if pc["between_seed_sd"] is not None else ""
            cells.append(f"{pc['mean_ap50']}{sd}")
        lines.append(f"| {r} | {info['seeds']} | " + " | ".join(cells) + f" | {info['map50']} |")
    lines += ["", "± = between-seed SD (needs ≥ 2 seeds).", "",
              "| candidate − baseline | " + " | ".join(names) + " | mAP@50 | verdict |",
              "|---|" + "---|" * len(names) + "---|---|"]
    for r, c in rep["comparisons"].items():
        cells = [f"{c['per_class'][n]['diff']:+.3f} [{c['per_class'][n]['ci_low']:+.3f}, "
                 f"{c['per_class'][n]['ci_high']:+.3f}]" for n in names]
        m = c["map50"]
        lines.append(f"| {r} | " + " | ".join(cells) +
                     f" | {m['diff']:+.3f} [{m['ci_low']:+.3f}, {m['ci_high']:+.3f}] | "
                     f"**{c['verdict']}** |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    configure_logging()
    ap = argparse.ArgumentParser(description="Compare recipes by seed means.")
    ap.add_argument("--recipe", action="append", required=True, metavar="NAME=W1,W2,...")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--data", default="eval/clean_splits/data.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default="eval/cache")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    recipes: dict[str, list[str]] = {}
    for spec in args.recipe:
        name, _, paths = spec.partition("=")
        recipes[name] = [p for p in paths.split(",") if p]
    if args.baseline not in recipes:
        log.error("baseline %r is not one of the recipes", args.baseline)
        return 2
    missing = [p for ps in recipes.values() for p in ps if not os.path.exists(p)]
    if missing or not os.path.exists(args.data):
        log.error("not found: %s", missing or args.data)
        return 2
    if args.split == "test":
        log.warning("comparing on TEST: report only, never choose on this")

    from evaluate_uncertainty import load_or_predict
    from modules.detection_stats import recipe_comparison
    from modules.provenance import portable_path, run_provenance
    from modules.yolo_io import resolve_device

    device = resolve_device(args.device)
    records: dict[str, list] = {}
    names: list = []
    for r, paths in recipes.items():
        records[r] = []
        for p in paths:
            recs, names = load_or_predict(p, args.data, args.split, imgsz=640, device=device,
                                          cache_dir=args.cache_dir)
            records[r].append(recs)
    rep = recipe_comparison(records, args.baseline, list(names), n_boot=args.n_boot,
                            min_seeds=args.min_seeds)
    rep["split"] = args.split
    rep["checkpoints"] = {r: [portable_path(p) for p in ps] for r, ps in recipes.items()}
    rep["provenance"] = run_provenance(model_paths=[p for ps in recipes.values() for p in ps],
                                       data_yaml=args.data, split=args.split,
                                       config={"n_boot": args.n_boot})
    name = args.name or f"recipe_comparison_{args.split}"
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}.json"), "w") as fh:
        json.dump(rep, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out, f"{name}.md"), "w") as fh:
        fh.write(render(rep, list(names), args.split))
    for r, c in rep["comparisons"].items():
        print(f"{r} vs {args.baseline}: mAP@50 {c['map50']['diff']:+.3f} -> {c['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
