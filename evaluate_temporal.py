"""Choose the temporal confirmation rule by experiment (model-free).

    python3 evaluate_temporal.py                       # full run (~1-2 min)
    python3 evaluate_temporal.py --riders 20           # quick smoke run

Simulates riders whose per-frame helmet boxes are missed / mislabelled at the
rates the shipped detector shows on the VALIDATION split (read from
``--noise-from``), drives them through the shipped ``ViolationPipeline`` under
every candidate confirmation rule, selects a rule on a development seed under a
stated objective, and reports it on a separate seed. See
``modules/temporal_eval.py`` for the protocol and its limits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from modules.logging_setup import configure_logging
from modules.temporal_eval import (
    MAX_FALSE_FLAG,
    MEASURED_VAL_NOISE,
    noise_from_confusion,
    run_experiment,
)

DEFAULT_NOISE_FROM = "eval/results/eval_traffic_model-2_val_clean.json"


def render_markdown(p: dict) -> str:
    n = p["noise_model"]
    sel = p["selection"]
    lines = [
        "# Temporal confirmation — rule selection experiment",
        "",
        "> " + p["caveat"],
        "",
        f"- Noise model: `{n['name']}` — missed box {n['miss_violator']:.1%} (violator) / "
        f"{n['miss_compliant']:.1%} (compliant); helmeted rider labelled WithoutHelmet "
        f"**{n['false_violation']:.1%}**; bare-headed rider labelled WithHelmet "
        f"{n['false_compliant']:.1%} (per frame). Source: `{p.get('noise_source')}`.",
        f"- {p['n_riders_per_condition']} riders per condition (half violators), dwell "
        f"{', '.join(str(d) for d in p['dwells_frames'])} frames, stickiness "
        f"{', '.join(str(r) for r in p['stickiness'])} (P a frame repeats the last "
        "frame's outcome — real errors come in runs; unmeasured, so swept).",
        f"- Selected on dev seed {p['dev_seed']}, reported on test seed {p['test_seed']}.",
        f"- Objective: {sel['objective']}. Design conditions: "
        f"{', '.join(sel['design_conditions'])}.",
        "",
        f"**Selected rule: {sel['selected']}** ({sel['selected_by']}).",
        "",
        "## Selection (dev seed) — every candidate",
        "",
        "| rule | mean recall | worst false-flag rate | meets ≤"
        f"{MAX_FALSE_FLAG:.0%}? |",
        "|---|---:|---:|---|",
    ]
    for t in sorted(sel["table"], key=lambda t: (not t["feasible"], -t["mean_recall"])):
        lines.append(f"| {t['rule']} | {t['mean_recall']:.3f} | "
                     f"{t['worst_false_flag_rate']:.1%} | {'yes' if t['feasible'] else 'no'} |")
    lines += ["", "## Held-out seed — selected vs previous default vs single frame", "",
              "| condition | rule | precision | recall | false-flag rate |",
              "|---|---|---:|---:|---:|"]
    for cond, rules in p["test"].items():
        for name, m in rules.items():
            lines.append(f"| {cond} | {name} | {m['precision']:.3f} | {m['recall']:.3f} | "
                         f"{m['false_flag_rate']:.1%} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--noise-from", default=DEFAULT_NOISE_FROM,
                    help="evaluate_model.py JSON (val split) to take error rates from")
    ap.add_argument("--riders", type=int, default=600,
                    help="riders per class per condition (600; the selection was "
                         "re-checked against 150 and the same rule won at both sizes)")
    ap.add_argument("--dev-seed", type=int, default=101)
    ap.add_argument("--test-seed", type=int, default=202)
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--name", default="temporal_confirmation")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)
    noise = MEASURED_VAL_NOISE
    source = "modules/temporal_eval.py MEASURED_VAL_NOISE"
    if args.noise_from and os.path.exists(args.noise_from):
        with open(args.noise_from) as fh:
            ev = json.load(fh)
        if ev.get("split") != "val":
            print(f"refusing noise from split={ev.get('split')!r}: rules are chosen on val")
            return 2
        noise = noise_from_confusion(ev["error_analysis"]["confusion_matrix"],
                                     ev["class_names"], "measured (val, de-leaked)")
        source = args.noise_from
    out = run_experiment(dev_seed=args.dev_seed, test_seed=args.test_seed,
                         n_per_class=args.riders, noise=noise)
    out["noise_source"] = source
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, args.name + ".json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out, args.name + ".md"), "w") as fh:
        fh.write(render_markdown(out))
    print(f"selected: {out['selection']['selected']} ({out['selection']['selected_by']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
