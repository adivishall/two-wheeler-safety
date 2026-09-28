"""How much does the detector degrade under blur, low light, glare, compression,
low resolution and occlusion? Measured, with CIs, on SYNTHETIC transforms.

    python3 evaluate_robustness.py                       # all corruptions, val split
    python3 evaluate_robustness.py --only glare_severe low_light_severe

Each corruption is applied to every held-out image (labels untouched, see
``modules/corruptions.py``), the model predicts on it, and AP@50 is compared to
the SAME images uncorrupted with a paired image bootstrap. A drop is reported as
significant only when its 95% CI excludes zero.

Robustness here means measured behaviour under changed inputs — not "it didn't
crash". The transforms approximate, they do not reproduce, real night / rain /
glare footage; every number in the output is stamped synthetic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from evaluate_uncertainty import DEFAULT_DATA, DEFAULT_MODEL, load_or_predict, model_label
from modules import corruptions, yolo_io
from modules.detection_stats import bootstrap_ap, paired_bootstrap
from modules.logging_setup import configure_logging, get_logger

log = get_logger("robustness")


def render_markdown(p: dict) -> str:
    names = p["class_names"]
    lines = [
        f"# Detector robustness under synthetic corruptions — `{p['model']}`",
        "",
        "> **Synthetic transforms of real held-out images.** Labels are untouched; "
        "only pixels change. This measures sensitivity to each transform, not "
        "performance on real night / rain / glare footage.",
        "",
        f"- Split **{p['split']}** of `{p['data']}` ({p['images']} images); predict-mode "
        f"AP@50 (the path the pipeline runs); {p['n_boot']} paired bootstrap resamples.",
        f"- Clean mAP@50: **{p['clean']['map50']['value']:.3f}** "
        f"[{p['clean']['map50']['ci_low']:.3f}, {p['clean']['map50']['ci_high']:.3f}]",
        "",
        "## Change in AP@50 vs the same images uncorrupted",
        "",
        "Cells: Δ AP@50 (★ = 95% CI excludes 0).",
        "",
        "| corruption | " + " | ".join(names) + " | mAP@50 |",
        "|---|" + "---:|" * (len(names) + 1),
    ]
    for name, r in p["corruptions"].items():
        d = r["paired_vs_clean"]

        def cell(x):
            if x["diff"] is None:
                return "—"
            return f"{x['diff']:+.3f}{' ★' if x['significant'] else ''}"

        lines.append(f"| `{name}` | " + " | ".join(cell(d["per_class"][n]) for n in names)
                     + f" | {cell(d['map50'])} |")
    lines += ["", "## Worst-hit class per corruption", ""]
    for name, r in p["corruptions"].items():
        pc = r["paired_vs_clean"]["per_class"]
        valid = {n: v for n, v in pc.items() if v["diff"] is not None}
        worst = min(valid, key=lambda n: valid[n]["diff"])
        lines.append(f"- `{name}`: {worst} ({valid[worst]['diff']:+.3f})")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--only", nargs="*", choices=sorted(corruptions.CORRUPTIONS), default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="eval/cache")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--name", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)
    if not os.path.exists(args.model) or not os.path.exists(args.data):
        log.error("model or data.yaml not found")
        return 2
    from modules.provenance import run_provenance

    device = yolo_io.resolve_device(args.device)
    clean, names = load_or_predict(args.model, args.data, args.split, imgsz=args.imgsz,
                                   device=device, cache_dir=args.cache_dir)
    results = {}
    for cname in args.only or list(corruptions.CORRUPTIONS):
        recs, _ = load_or_predict(args.model, args.data, args.split, imgsz=args.imgsz,
                                  device=device, cache_dir=args.cache_dir,
                                  # name AND parameters: retuning a transform
                                  # must never reuse the old predictions
                                  corruption=corruptions.cache_tag(cname),
                                  corrupt_fn=corruptions.get(cname))
        results[cname] = {
            "synthetic_transform": True,
            "params": corruptions.CORRUPTIONS[cname][1],
            "bootstrap": bootstrap_ap(recs, names, n_boot=args.bootstrap, seed=args.seed),
            "paired_vs_clean": paired_bootstrap(recs, clean, names, n_boot=args.bootstrap,
                                                seed=args.seed),
        }
        log.info("%s: mAP@50 %+.3f", cname, results[cname]["paired_vs_clean"]["map50"]["diff"])

    payload = {
        "name": args.name or f"robustness_{args.split}",
        "model": model_label(args.model),
        "data": args.data,
        "split": args.split,
        "images": len(clean),
        "n_boot": args.bootstrap,
        "class_names": names,
        "synthetic_transform": True,
        "clean": bootstrap_ap(clean, names, n_boot=args.bootstrap, seed=args.seed),
        "corruptions": results,
        "provenance": run_provenance(model_paths=[args.model], data_yaml=args.data,
                                     split=args.split,
                                     config={"conf": yolo_io.AP_CONF,
                                             "nms_iou": yolo_io.AP_NMS_IOU,
                                             "imgsz": args.imgsz}),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, payload["name"] + ".json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out, payload["name"] + ".md"), "w") as fh:
        fh.write(render_markdown(payload))
    print(f"wrote {os.path.join(args.out, payload['name'])}.md/.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
