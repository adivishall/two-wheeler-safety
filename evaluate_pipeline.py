"""One command for every model-free evaluation in the project.

`evaluate_model.py` needs weights and a dataset. Everything *downstream* of the
detector — tracking, association, the violation state machines, OCR decision
policy, speed calibration, and the error budget — can be measured without either,
because those layers are driven by deterministic synthetic inputs through the
**real** runtime code. This is the entry point that runs all of it and writes one
set of results.

    python3 evaluate_pipeline.py                  # run everything, print a summary
    python3 evaluate_pipeline.py --only speed ocr # just those sections
    python3 evaluate_pipeline.py --out eval/results

Outputs (Phase 23 — generated, never hand-edited):

    eval/results/pipeline_evaluation.json   every number, machine-readable
    eval/results/pipeline_evaluation.md     the report the docs cite
    eval/results/pipeline_evaluation.csv    flat metric rows for spreadsheets
    eval/results/latest.json                the summary the dashboard reads

Because it needs no weights, no dataset and no network, CI runs it as a smoke
test — so the evaluation tooling itself cannot silently rot between the rare
occasions the full model evaluation is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

SECTIONS = ("system", "violations", "headroom", "budget", "ocr", "speed")


def run_system() -> dict:
    """Tracking / association / end-to-end fines over the synthetic scenarios."""
    from modules.system_eval import evaluate_all

    return evaluate_all()


def run_violations(trials: int) -> dict:
    """Per-violation pipeline decision quality + the confirm-window sweep."""
    from modules.pipeline_eval import evaluate_all

    full = evaluate_all(trials=trials)
    return {"helmet": full["helmet"], "triple_riding": full["triple_riding"]}


def run_headroom(trials: int) -> dict:
    """Head-to-head: naive single-frame fining vs the full pipeline."""
    from modules.pipeline_eval import pipeline_vs_single_frame

    return pipeline_vs_single_frame(trials=trials)


def run_budget(trials: int) -> dict:
    """Error budget: oracle ablation at the measured operating point, plus
    per-stage dose-response in one common unit."""
    from modules.pipeline_eval import (
        error_budget,
        helmet_scenarios,
        stage_tolerance,
        triple_riding_scenarios,
    )
    from modules.system_eval import builtin_scenarios

    scenarios = builtin_scenarios() + helmet_scenarios() + triple_riding_scenarios()
    out = error_budget(scenarios, trials=trials)
    out["tolerance"] = stage_tolerance(scenarios, trials=max(3, trials // 2))
    return out


def run_ocr() -> dict:
    """Single-frame vs temporal OCR decision policy under simulated noise."""
    from modules.ocr_temporal_eval import noise_sweep, run_simulation

    return {
        "default_noise": run_simulation().as_dict(),
        "noise_sweep": noise_sweep(),
    }


def run_speed() -> dict:
    """Speed MAE/RMSE/bias per calibration, for both motion geometries."""
    from modules.speed_eval import compare_calibrations

    return {
        "approach": compare_calibrations(direction="approach"),
        "lateral": compare_calibrations(direction="lateral"),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def flatten_rows(payload: dict) -> list[dict]:
    """Flatten the report into ``{section, subject, metric, value}`` CSV rows.

    A flat long format rather than a wide one: the sections have genuinely
    different metric sets, so any wide table would be mostly empty cells.
    """
    rows: list[dict] = []

    def add(section, subject, metrics: dict):
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                rows.append({"section": section, "subject": subject,
                             "metric": k, "value": v})

    sysr = payload.get("system")
    if sysr:
        add("system", "totals", sysr["totals"])
        for name, r in sysr["scenarios"].items():
            add("system", f"{name}.system", r["system"])
            add("tracking", name, r["tracking"])
            add("association", name, r["association"])

    vio = payload.get("violations")
    if vio:
        for key, block in vio.items():
            add("violations", key, block["decision"])
            for window, m in block["confirm_window_sweep"]["windows"].items():
                add("violations", f"{key}.confirm_window={window}", m)

    head = payload.get("headroom")
    if head:
        for rate, block in head["rates"].items():
            for policy in ("naive", "pipeline"):
                add("pipeline_vs_single_frame", f"noise={rate}.{policy}",
                    block[policy])

    budget = payload.get("budget")
    if budget:
        for rate, block in budget["by_ocr_rate"].items():
            add("error_budget", f"ocr_read_error={rate}.all_faults",
                {"f1": block["all_faults_f1"], "ceiling_gap": block["ceiling_gap"]})
            for s in block["stages"]:
                add("error_budget", f"ocr_read_error={rate}.{s['stage']}",
                    {k: v for k, v in s.items() if isinstance(v, (int, float))})

    ocr = payload.get("ocr")
    if ocr:
        for pol, m in ocr["default_noise"]["policies"].items():
            add("ocr", f"default.{pol}", m)
        for rate, rep in ocr["noise_sweep"]["sweep"].items():
            for pol, m in rep["policies"].items():
                add("ocr", f"noise={rate}.{pol}", m)

    speed = payload.get("speed")
    if speed:
        for direction, block in speed.items():
            for kind, m in block["results"].items():
                add("speed", f"{direction}.{kind}", m)
    return rows


def render_markdown(p: dict) -> str:
    lines = ["# Pipeline evaluation (model-free)", "",
             f"- Generated: {p['generated_at']}",
             "- Inputs: deterministic synthetic scenarios driven through the "
             "**shipped** `ViolationPipeline` (the object the video job runs per "
             "frame, with its defaults) and the real speed estimator.",
             "- No model weights, no dataset, no network. Every number is "
             "reproducible with `python3 evaluate_pipeline.py`.", "",
             "> These measure **pipeline logic**, not detector quality. Detector "
             "metrics live in `docs/MODEL_EVALUATION.md` and are a separate "
             "question — see the note on mixing them there.", ""]

    sysr = p.get("system")
    if sysr:
        t = sysr["totals"]
        lines += ["## End-to-end fines", "",
                  f"- precision **{t['precision']}**, recall **{t['recall']}**",
                  f"- TP {t['true_positives']} · FP {t['false_positives']} · "
                  f"FN {t['false_negatives']} · wrong-vehicle {t['wrong_vehicle']} "
                  f"· wrong-plate {t['wrong_plate']} · duplicates {t['duplicates']}",
                  "",
                  "| scenario | TP | FP | FN | ID switches | assoc acc |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, r in sysr["scenarios"].items():
            s, tr, a = r["system"], r["tracking"], r["association"]
            lines.append(
                f"| `{name}` | {s['true_positives']} | {s['false_positives']} | "
                f"{s['false_negatives']} | {tr['id_switches']} | {a['accuracy']} |"
            )
        lines.append("")

    vio = p.get("violations")
    if vio:
        lines += ["## Per-violation pipeline decision", "",
                  "How often the *complete* pipeline reaches the correct verdict "
                  "for a vehicle — not raw detector accuracy.", "",
                  "| violation | P | R | F1 | TP | FP | FN | TN |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for key, block in vio.items():
            d = block["decision"]
            lines.append(
                f"| `{key}` | {d['precision']} | {d['recall']} | {d['f1']} | "
                f"{d['true_positives']} | {d['false_positives']} | "
                f"{d['false_negatives']} | {d['true_negatives']} |"
            )
        lines += ["", "### What the streak length alone buys (sanity check)", "",
                  "`confirm_window=1` is single-frame fining. Helmet dwell gate off "
                  "here; these scenarios' flickers are one frame long by "
                  "construction, so this cannot choose a rule — "
                  "`eval/results/temporal_confirmation.md` does.", "",
                  "| violation | window | precision | recall |",
                  "|---|---:|---:|---:|"]
        for key, block in vio.items():
            for window, m in block["confirm_window_sweep"]["windows"].items():
                lines.append(f"| `{key}` | {window} | {m['precision']} | "
                             f"{m['recall']} |")
        lines.append("")

    head = p.get("headroom")
    if head:
        lines += ["## Is the pipeline better than a single-frame detector?", "",
                  "Both policies see **identical** detections, degraded by the "
                  "same helmet class-confusion noise. `naive` fines whenever any "
                  "single frame shows a violation box — no tracking, no temporal "
                  "confirmation, no contradiction check.", "",
                  head["note"], "",
                  "| detector noise | naive P | naive R | naive F1 | naive FPs | "
                  "pipeline P | pipeline R | pipeline F1 | pipeline FPs | F1 gain |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for rate, block in head["rates"].items():
            n, pl = block["naive"], block["pipeline"]
            lines.append(
                f"| {rate} | {n['precision']} | {n['recall']} | {n['f1']} | "
                f"{n['mean_false_positives']} | {pl['precision']} | "
                f"{pl['recall']} | {pl['f1']} | {pl['mean_false_positives']} | "
                f"**{block['pipeline_f1_advantage']:+.3f}** |"
            )
        lines += ["", "The naive policy always scores recall 1.000 because it "
                  "fines on anything — so the whole difference is precision. The "
                  "pipeline keeps precision near 1 at every noise level; its F1 "
                  "advantage holds in the range the detector actually operates in "
                  "(val-measured helmet flips 8–15% per frame) and reverses at "
                  "extreme symmetric noise, where it trades recall for precision. "
                  "That is the right trade for a system that fines people, and it "
                  "is a trade.", ""]

    budget = p.get("budget")
    if budget:
        op = budget["operating_point"]
        lines += ["## Error budget — oracle ablation at the measured operating point", "",
                  budget["interpretation"], "",
                  f"- Detector rates per vehicle per frame, from `{op['source']}`: "
                  "rider box missed " + ", ".join(
                      f"{k} {v:.1%}" for k, v in op["rider_miss"].items()) +
                  "; wrong class " + ", ".join(
                      f"{k}→{v[0]} {v[1]:.1%}" for k, v in op["class_flip"].items()) +
                  f"; plate missed {op['plate_miss']:.1%}.",
                  "- OCR read-error rate (≥1 wrong glyph per read) is **unmeasured** "
                  "— no labelled plate sequences exist — so it is swept.",
                  f"- No-fault F1 (the suite's by-design misses only): "
                  f"**{budget['no_fault_f1']}**; {budget['trials']} trials, seed "
                  f"{budget['seed']}.", ""]
        for rate, block in budget["by_ocr_rate"].items():
            lines += [f"**OCR read-error rate {float(rate):.0%}** — all faults on: F1 "
                      f"{block['all_faults_f1']} (gap to no-fault {block['ceiling_gap']})",
                      "", "| stage made perfect | F1 | recovered | share |",
                      "|---|---:|---:|---:|"]
            for s in block["stages"]:
                lines.append(f"| `{s['stage']}` — {s['description']} | "
                             f"{s['f1_if_perfect']} | **{s['recovered_f1']}** | "
                             f"{s['share_of_recovered']:.0%} |")
            lines.append("")
        robust = ("holds across the whole OCR sweep"
                  if budget["bottleneck_robust_to_ocr_assumption"]
                  else "does NOT hold across the whole OCR sweep — see the tables")
        lines += [f"Largest owner at the central OCR assumption: "
                  f"**`{budget['bottleneck']}`** ({robust}).", ""]
        tol = budget.get("tolerance")
        if tol:
            rates = list(next(iter(tol["f1_by_rate"].values())))
            lines += ["### Tolerance — one stage failing alone, same unit for all", "",
                      f"End-to-end fine F1 when only that stage fails ({tol['unit']}).",
                      "", "| stage | " + " | ".join(rates) + " |",
                      "|---|" + "---:|" * len(rates)]
            for stage, row in tol["f1_by_rate"].items():
                lines.append(f"| `{stage}` | " + " | ".join(str(row[r]) for r in rates) + " |")
            lines.append("")

    ocr = p.get("ocr")
    if ocr:
        lines += ["## OCR decision policy — single-frame vs temporal", "",
                  ocr["noise_sweep"]["note"], "",
                  "`coverage` is the share of vehicles the policy names at all; "
                  "`accuracy|answered` is how often it is right **when it "
                  "answers**. A wrong plate fines an innocent rider; an "
                  "abstention only misses a fine, so both are reported.", "",
                  "| substitution rate | policy | coverage | accuracy\\|answered |",
                  "|---|---|---:|---:|"]
        for rate, rep in ocr["noise_sweep"]["sweep"].items():
            for pol, m in rep["policies"].items():
                lines.append(f"| {rate} | `{pol}` | {m['coverage']} | "
                             f"{m['answered_accuracy']} |")
        lines.append("")

    speed = p.get("speed")
    if speed:
        lines += ["## Speed estimation", "",
                  "| motion | calibration | MAE (km/h) | RMSE | bias | "
                  "overspeed P | overspeed R |", "|---|---|---:|---:|---:|---:|---:|"]
        for direction, block in speed.items():
            for kind, m in block["results"].items():
                lines.append(
                    f"| {direction} | `{kind}` | {m['mae']} | {m['rmse']} | "
                    f"{m['bias']} | {m['overspeed_precision']} | "
                    f"{m['overspeed_recall']} |"
                )
            lines.append(f"| {direction} | **best: `{block['best']}`** | | | | | |")
        lines.append("")

    return "\n".join(lines) + "\n"


def build_summary(p: dict) -> dict:
    """The compact block the dashboard renders (Phase 24).

    Deliberately small: a handful of headline numbers plus provenance, so the
    dashboard shows evaluation status without turning into a metrics browser.
    """
    out: dict = {
        "generated_at": p["generated_at"],
        "source": "evaluate_pipeline.py",
    }
    if p.get("system"):
        t = p["system"]["totals"]
        out["end_to_end"] = {
            "precision": t["precision"], "recall": t["recall"],
            "true_positives": t["true_positives"],
            "false_positives": t["false_positives"],
            "false_negatives": t["false_negatives"],
            "scenarios": len(p["system"]["scenarios"]),
        }
    if p.get("violations"):
        out["violations"] = {
            k: {"precision": v["decision"]["precision"],
                "recall": v["decision"]["recall"],
                "f1": v["decision"]["f1"]}
            for k, v in p["violations"].items()
        }
    if p.get("headroom"):
        clean = p["headroom"]["rates"].get("0.00") or {}
        noisy = p["headroom"]["rates"].get("0.20") or {}
        out["vs_single_frame"] = {
            "clean_naive_f1": (clean.get("naive") or {}).get("f1"),
            "clean_pipeline_f1": (clean.get("pipeline") or {}).get("f1"),
            "noisy_naive_f1": (noisy.get("naive") or {}).get("f1"),
            "noisy_pipeline_f1": (noisy.get("pipeline") or {}).get("f1"),
        }
    if p.get("budget"):
        out["error_budget"] = {
            "method": p["budget"]["method"],
            "bottleneck": p["budget"]["bottleneck"],
            "robust_to_ocr_assumption": p["budget"]["bottleneck_robust_to_ocr_assumption"],
            "stages": [
                {"stage": s["stage"], "recovered_f1": s["recovered_f1"],
                 "share": s["share_of_recovered"]}
                for s in p["budget"]["stages"]
            ],
        }
    if p.get("ocr"):
        pol = p["ocr"]["default_noise"]["policies"]
        out["ocr_policy"] = {
            k: {"coverage": v["coverage"], "answered_accuracy": v["answered_accuracy"]}
            for k, v in pol.items()
        }
    if p.get("speed"):
        out["speed"] = {
            d: {"best": b["best"],
                "best_mae": b["results"][b["best"]]["mae"]}
            for d, b in p["speed"].items()
        }
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Run every model-free pipeline evaluation and write reports.",
    )
    ap.add_argument("--only", nargs="*", choices=SECTIONS, default=None,
                    help=f"run only these sections (default: all of {', '.join(SECTIONS)})")
    ap.add_argument("--out", default="eval/results", help="output directory")
    ap.add_argument("--name", default="pipeline_evaluation", help="report base name")
    ap.add_argument("--trials", type=int, default=30,
                    help="fault-injection trials per stage (default: 30)")
    ap.add_argument("--quiet", action="store_true", help="suppress the console summary")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sections = args.only or list(SECTIONS)

    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
    }
    if "system" in sections:
        payload["system"] = run_system()
    if "violations" in sections:
        payload["violations"] = run_violations(args.trials)
    if "headroom" in sections:
        payload["headroom"] = run_headroom(args.trials)
    if "budget" in sections:
        payload["budget"] = run_budget(args.trials)
    if "ocr" in sections:
        payload["ocr"] = run_ocr()
    if "speed" in sections:
        payload["speed"] = run_speed()

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, f"{args.name}.json")
    md_path = os.path.join(args.out, f"{args.name}.md")
    csv_path = os.path.join(args.out, f"{args.name}.csv")
    latest_path = os.path.join(args.out, "latest.json")

    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    with open(md_path, "w") as fh:
        fh.write(render_markdown(payload))
    rows = flatten_rows(payload)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["section", "subject", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    with open(latest_path, "w") as fh:
        json.dump(build_summary(payload), fh, indent=2, sort_keys=True)

    if not args.quiet:
        print("\n=== pipeline evaluation ===")
        if payload.get("system"):
            t = payload["system"]["totals"]
            print(f"end-to-end fines: precision={t['precision']:.3f} "
                  f"recall={t['recall']:.3f} "
                  f"(TP {t['true_positives']} / FP {t['false_positives']} / "
                  f"FN {t['false_negatives']})")
        for key, block in (payload.get("violations") or {}).items():
            d = block["decision"]
            print(f"{key:14s}: P={d['precision']:.3f} R={d['recall']:.3f} "
                  f"F1={d['f1']:.3f}")
        if payload.get("headroom"):
            print("pipeline vs naive single-frame fining:")
            for rate, block in payload["headroom"]["rates"].items():
                n, pl = block["naive"], block["pipeline"]
                print(f"   noise={rate}  naive F1={n['f1']:.3f} "
                      f"(FP {n['mean_false_positives']:.2f})  "
                      f"pipeline F1={pl['f1']:.3f} "
                      f"(FP {pl['mean_false_positives']:.2f})  "
                      f"gain {block['pipeline_f1_advantage']:+.3f}")
        if payload.get("budget"):
            b = payload["budget"]
            print(f"error budget ({b['method']}): largest owner {b['bottleneck']} "
                  f"(robust to OCR assumption: {b['bottleneck_robust_to_ocr_assumption']})")
            for s in b["stages"]:
                print(f"   {s['stage']:14s} recovers F1 {s['recovered_f1']:.4f} "
                      f"({s['share_of_recovered']:.0%})")
        if payload.get("ocr"):
            print("OCR policy (default noise):")
            for pol, m in payload["ocr"]["default_noise"]["policies"].items():
                print(f"   {pol:10s} coverage={m['coverage']:.2f} "
                      f"accuracy|answered={m['answered_accuracy']:.3f}")
        if payload.get("speed"):
            for d, b in payload["speed"].items():
                print(f"speed ({d}): best={b['best']} "
                      f"MAE={b['results'][b['best']]['mae']:.2f} km/h")
        print(f"\nWrote {md_path}\n      {json_path}\n      {csv_path}"
              f"\n      {latest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
