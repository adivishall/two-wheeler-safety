"""Detector accuracy by measured image condition, on the real held-out images.

    python3 evaluate_conditions.py --model runs/detect/traffic_model-2/weights/best.pt \\
        --data eval/clean_splits/data.yaml --splits val test --device mps

Strata (lighting, sharpness, glare, crowding, source, object size) are measured
from pixels and boxes — see ``modules/image_conditions.py`` for each definition
and what it is a proxy for. Cut points are fitted on the first split (val) and
reused unchanged on the rest; a finding is reported when a stratum's AP differs
from the rest of the split with a bootstrap CI excluding zero, and is marked
confirmed only if test shows the same sign with its own CI excluding zero.

Writes ``eval/results/conditions_<split>.json`` and ``eval/results/conditions.md``.
Raw predictions are cached under ``eval/cache/`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from modules import yolo_io
from modules.logging_setup import configure_logging, get_logger

log = get_logger("evaluate_conditions")


def load_or_predict_boxes(model_path, data_yaml, split, *, imgsz, device, cache_dir):
    from modules.model_manifest import sha256_file
    from modules.provenance import split_fingerprint

    data = yolo_io.load_data_yaml(data_yaml)
    key = yolo_io.cache_key(kind="boxes", weights=sha256_file(model_path),
                            split=split_fingerprint(data_yaml, split), conf=yolo_io.AP_CONF,
                            nms=yolo_io.AP_NMS_IOU, imgsz=imgsz)
    path = os.path.join(cache_dir, f"boxes_{key}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)["rows"], data["names"]
    from ultralytics import YOLO

    log.info("predicting %s on %s", split, data_yaml)
    pairs = list(yolo_io.iter_image_label_pairs(yolo_io.split_dir(data, split)))
    rows = yolo_io.predict_boxes(YOLO(model_path), pairs, imgsz=imgsz, device=device)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"split": split, "rows": rows}, fh)
    return rows, data["names"]


def operating_thresholds() -> dict:
    from modules.config import load_config

    det = load_config().detection
    return {"Plate": det.conf_threshold, "WithHelmet": det.conf_threshold,
            "WithoutHelmet": det.helmet_min_conf, "TripleRiding": det.triple_min_conf}


def _cell(ap: dict) -> str:
    if ap["ap50"] is None:
        return "—"
    flag = "†" if ap["instances"] < 10 else ""
    return f"{ap['ap50']:.2f} [{ap['ci_low']:.2f}, {ap['ci_high']:.2f}] n={ap['instances']}{flag}"


def render_markdown(reports: dict, findings: list, tested: int, names: list,
                    model: str) -> str:
    first = next(iter(reports.values()))
    thr = first["thresholds"]
    lines = [
        "# Detector accuracy by measured image condition — real held-out images", "",
        f"- Model: `{model}`; AP@50 at conf ≥ {yolo_io.AP_CONF}, NMS IoU {yolo_io.AP_NMS_IOU}; "
        "95% image-bootstrap CIs; † = fewer than 10 instances (not interpretable).",
        "- Conditions are **measured from pixels and boxes, not labelled** — each is a "
        "proxy (`modules/image_conditions.py`). Cut points fitted on val, reused on test: "
        f"sharpness terciles {thr['sharpness_terciles']}, dark < {thr['luminance']['dark_below']} "
        f"mean luminance, glare ≥ {int(thr['glare']['share'] * 100)}% pixels ≥ "
        f"{thr['glare']['pixel']}.",
        f"- **Not measurable on this data**: resolution (all images pre-resized: "
        f"{', '.join(first['image_sizes'])}), camera angle, weather, true time of day, and "
        "anything past the detector (tracking, association, OCR, fines) — those need "
        "labelled field footage (`docs/FIELD_EVALUATION.md`).", "",
        "## Findings", "",
        f"Per class, strata whose AP differs from the rest of the split with a CI "
        f"excluding zero on val, among **{tested}** comparisons with ≥ 10 instances on "
        f"both sides — at 95%, about {tested * 0.05:.0f} would clear the bar by chance, "
        "so **only `confirmed` rows** (test agrees in sign, its own CI excluding zero) "
        "should be read as findings.", ""]
    if not findings:
        lines.append("None: no stratum differs significantly from the rest.")
    else:
        lines += ["| condition | class | val ΔAP [95% CI] | test ΔAP [95% CI] | status |",
                  "|---|---|---|---|---|"]
        for f in findings:
            t = f.get("test")
            tcell = (f"{t['diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]"
                     if t and t["diff"] is not None else "—")
            lines.append(f"| {f['attribute']} = **{f['value']}** ({f['images']} img) | "
                         f"{f['class']} | {f['diff']:+.3f} [{f['ci'][0]:+.3f}, "
                         f"{f['ci'][1]:+.3f}] | {tcell} | {f['status']} |")
    for split, rep in reports.items():
        lines += ["", f"## {split} — {rep['images']} images", "",
                  f"Overall mAP@50 {rep['overall']['ap']['map50']['value']}.", ""]
        for attr, values in rep["image_strata"].items():
            lines += [f"### {attr}", "",
                      "| value | images | " + " | ".join(names) + " |",
                      "|---|---:|" + "---|" * len(names)]
            for value, st in values.items():
                cells = " | ".join(_cell(st["ap"]["per_class"][n]) for n in names)
                lines.append(f"| {value} | {st['images']} | {cells} |")
            lines.append("")
        lines += ["### Recall at the operating thresholds", "",
                  "| condition | " + " | ".join(names) + " |",
                  "|---|" + "---:|" * len(names)]
        for attr, values in rep["image_strata"].items():
            for value, st in values.items():
                op = st["at_operating_threshold"]
                cells = " | ".join(
                    "—" if op[n]["gt"] == 0 else f"{op[n]['recall']:.2f} ({int(op[n]['gt'])})"
                    for n in names)
                lines.append(f"| {attr}: {value} | {cells} |")
        lines += ["", "### Object size (relative √area; COCO range rule)", "",
                  "| class | small | medium | large | small − large (paired) |",
                  "|---|---|---|---|---|"]
        for n, bands in rep["object_size"].items():
            d = bands.get("small_minus_large") or {}
            dcell = (f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
                     if d.get("diff") is not None else "—")
            lines.append(f"| {n} | " + " | ".join(_cell(bands[b]) for b in
                                                  ("small", "medium", "large"))
                         + f" | {dcell} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    configure_logging()
    ap = argparse.ArgumentParser(description="Detector AP by measured image condition.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="eval/clean_splits/data.yaml")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default="eval/cache")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args(argv)
    for p in (args.model, args.data):
        if not os.path.exists(p):
            log.error("not found: %s", p)
            return 2

    import cv2

    from modules.image_conditions import evaluate, fit_thresholds, image_properties, notable_gaps
    from modules.label_audit import source_of
    from modules.provenance import run_provenance

    device = yolo_io.resolve_device(args.device)
    op = operating_thresholds()
    reports: dict = {}
    thr = None
    names: list = []
    for split in args.splits:
        rows, names = load_or_predict_boxes(args.model, args.data, split, imgsz=args.imgsz,
                                            device=device, cache_dir=args.cache_dir)
        props = [image_properties(cv2.imread(r["image"])) for r in rows]
        sources = [source_of(os.path.basename(r["image"])) for r in rows]
        if thr is None:
            thr = fit_thresholds(rows, props, names)
        rep = evaluate(rows, props, sources, names, thr, op_thresholds=op,
                       n_boot=args.n_boot)
        rep["split"] = split
        rep["provenance"] = run_provenance(model_paths=[args.model], data_yaml=args.data,
                                           split=split, config={"n_boot": args.n_boot,
                                                                "op_thresholds": op})
        reports[split] = rep
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"conditions_{split}.json"), "w") as fh:
            json.dump(rep, fh, indent=2, sort_keys=True)

    first = reports[args.splits[0]]
    findings, tested = notable_gaps(first, names)
    second = reports.get(args.splits[1]) if len(args.splits) > 1 else None
    for f in findings:
        f["status"] = "val only"
        if second:
            if f["attribute"] == "object size":
                t = second["object_size"].get(f["class"], {}).get("small_minus_large")
            else:
                st = second["image_strata"].get(f["attribute"], {}).get(f["value"])
                g = st.get("gap_vs_rest") if st else None
                t = g["per_class"][f["class"]] if g else None
            if t:
                f["test"] = t
                same_sign = t["diff"] is not None and (t["diff"] > 0) == (f["diff"] > 0)
                f["status"] = ("confirmed" if same_sign and t["significant"] else
                               "same direction, not significant" if same_sign else
                               "not replicated")
    with open(os.path.join(args.out, "conditions.md"), "w") as fh:
        fh.write(render_markdown(reports, findings, tested, names, args.model))
    for f in findings:
        print(f"{f['attribute']}={f['value']} {f['class']}: {f['diff']:+.3f} "
              f"[{f['ci'][0]:+.3f}, {f['ci'][1]:+.3f}] -> {f['status']}")
    print(f"wrote {args.out}/conditions.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
