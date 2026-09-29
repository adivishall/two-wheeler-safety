"""Reproducible performance benchmark for the detection pipeline.

Measures the things that decide whether the system keeps up with real footage:
model load time, single-image inference latency, EasyOCR latency on a plate
crop, end-to-end video throughput, and peak memory. Every number is *measured*
here — nothing is hard-coded — and the JSON it writes records the machine and
device so results are comparable across runs.

    python3 benchmark.py \\
        --model runs/detect/traffic_model-2/weights/best.pt \\
        --image samples/test.jpg --video demo_traffic.mp4 --max-frames 120 \\
        --ocr-lock-ab --devices cpu mps --micro

Every result states what a benchmark result must state to be comparable: the
device each number was ACTUALLY measured on (Ultralytics silently runs on the
CPU on Apple silicon unless told otherwise — an earlier version of this file
labelled CPU numbers "mps"), cold vs warm, batch size (always 1: the pipeline
processes frames one at a time), input resolution and a hash of each input
file, model identity by weights hash, library versions and hardware. Output:
``eval/results/benchmark.{json,md}``.

Sections are skipped cleanly when their input is absent (e.g. no ``--video``).
The repo ships no weights or video; the input hashes say which files were used.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

from modules.logging_setup import configure_logging, get_logger
from modules.yolo_io import resolve_device

log = get_logger("benchmark")


def summarize_times(times_ms: list[float]) -> dict:
    """Mean / p50 / p90 / FPS from a list of per-op millisecond timings."""
    if not times_ms:
        return {}
    ordered = sorted(times_ms)
    mean = sum(ordered) / len(ordered)
    return {
        "n": len(ordered),
        "mean_ms": round(mean, 2),
        "p50_ms": round(ordered[len(ordered) // 2], 2),
        "p90_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
        "fps": round(1000.0 / mean, 1) if mean else 0.0,
    }


def _peak_rss_mb() -> float | None:
    """Peak resident set size in MB, best-effort (Unix ru_maxrss)."""
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return round(rss / divisor, 1)
    except Exception:  # noqa: BLE001
        return None


def benchmark_image(model, reader, image_path, iterations, device=None) -> dict:
    import cv2

    img = cv2.imread(image_path)
    out = {"image": image_path, "device": device or "library default",
           "resolution": list(img.shape[1::-1]) if img is not None else None,
           "batch": 1}
    kw = {"verbose": False, **({"device": device} if device else {})}
    # Cold: the first call on this device pays lazy backend/graph init.
    t0 = time.perf_counter()
    model.predict(source=image_path, **kw)
    out["first_call_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = model.predict(source=image_path, **kw)[0]
        times.append((time.perf_counter() - t0) * 1000.0)
    out["inference"] = summarize_times(times)
    out["inference"]["state"] = "warm"

    # OCR latency on the first detected Plate crop (or the whole image if none).
    crop = img
    for b in result.boxes:
        if model.names[int(b.cls[0])] == "Plate":
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            crop = img[max(0, y1):y2, max(0, x1):x2]
            break
    if crop is not None and getattr(crop, "size", 0):
        reader.readtext(crop, detail=0)  # warm up
        ocr_times = []
        for _ in range(max(3, iterations // 3)):
            t0 = time.perf_counter()
            reader.readtext(crop, detail=0)
            ocr_times.append((time.perf_counter() - t0) * 1000.0)
        out["ocr"] = summarize_times(ocr_times)
    return out


def benchmark_video(model, reader, video_path, max_frames, *,
                    ocr_lock: bool = True, device: str | None = None) -> dict:
    import tempfile

    from modules.profiling import StageProfiler
    from modules.video_detector import process_video

    # process_video writes evidence next to its output video, so both live in a
    # temporary directory that is removed afterwards.
    tmp = tempfile.TemporaryDirectory()
    out_path = os.path.join(tmp.name, "out.mp4")
    # A real profiler makes process_video time each stage (YOLO/OCR/track/
    # evidence/DB/encode/read); it's a no-op unless one is passed, so the web
    # path pays nothing. The evidence/DB stages only fire on a confirmed
    # violation, so their share depends on the clip.
    profiler = StageProfiler()
    # The OCR lock is the one measured optimization in the pipeline, so the
    # benchmark has to be able to turn it off to measure it. It used to read
    # process_video's default no matter what, which made the documented
    # OCR_LOCK_CONFIDENCE=1.1 reproduction silently a no-op: both arms of the
    # A/B ran *with* the lock and the "+172%" could not be reproduced.
    # A confidence above 1.0 can never be reached, so the lock never engages.
    from modules.config import load_config

    cfg = load_config()
    lock_conf = cfg.detection.ocr_lock_confidence if ocr_lock else 1.1
    lock_obs = cfg.detection.ocr_lock_min_observations
    try:
        t0 = time.perf_counter()
        summary = process_video(
            video_path, model, reader, out_path,
            record_fn=lambda *a, **k: 0, max_frames=max_frames,
            profiler=profiler,
            ocr_lock_confidence=lock_conf,
            ocr_lock_min_observations=lock_obs,
            device=device,
        )
        elapsed = time.perf_counter() - t0
    finally:
        tmp.cleanup()
    frames = summary.get("frames", 0)
    return {
        "video": video_path,
        "device": device or "library default",
        "ocr_lock": ocr_lock,
        "ocr_lock_confidence": lock_conf,
        "frames": frames,
        "wall_seconds": round(elapsed, 2),
        "throughput_fps": round(frames / elapsed, 2) if elapsed else 0.0,
        # The pipeline's own count of OCR calls (one per plate read), not the
        # profiler's stage entries.
        "ocr_calls": summary.get("ocr_calls", (summary.get("profile", {}).get("stages", {})
                                               .get("ocr", {}).get("calls", 0))),
        "ms_per_frame": round(elapsed * 1000.0 / frames, 2) if frames else 0.0,
        "stage_profile": summary.get("profile", {}),
        # What the run decided, so an A/B can show a speed-up changed nothing.
        "decisions": decisions_of(summary),
    }


def decisions_of(summary: dict) -> dict:
    """The run's outcome: fines as (plate, violation) and withheld confirmations."""
    return {
        "fined": sorted([v.get("plate"), v.get("violation")]
                        for v in summary.get("violations") or []),
        "withheld": len(summary.get("unfined_confirmations") or []),
    }


def _media_info(path: str) -> dict:
    """Resolution / fps / frames and a content hash of a local input file."""
    import cv2

    from modules.model_manifest import sha256_file

    info: dict = {"path": path, "sha256": sha256_file(path)}
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        info.update(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=round(cap.get(cv2.CAP_PROP_FPS), 3),
                    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def benchmark_micro(model, reader, image_path, device) -> dict:
    """The stages the video profile can only sample once: DB writes, the
    model-free decision core per frame, and the HTTP/Flask overhead of
    /analyze over a direct analyze_image call on the same image and model."""
    import tempfile

    out: dict = {}
    tmp = tempfile.mkdtemp(prefix="bench_")

    from modules.db import Database

    db = Database(os.path.join(tmp, "b.db"))
    ev = {"original_path": "o.jpg", "annotated_path": "a.jpg",
          "plate_crop_path": "p.jpg", "metadata_path": "m.json"}
    times = []
    for i in range(200):
        t0 = time.perf_counter()
        db.record_fine(f"MH12AB{i:04d}", "no_helmet", "a.jpg", confidence=0.8,
                       track_id=i, evidence=ev)
        times.append((time.perf_counter() - t0) * 1000.0)
    out["db_record_fine"] = summarize_times(times)

    from modules.association import DetBox
    from modules.pipeline import ViolationPipeline

    pipe = ViolationPipeline()
    times = []
    for f in range(300):
        dets = []
        for k, x in enumerate((100, 300, 500)):
            x += 3 * f
            label = "WithoutHelmet" if k == 1 else "WithHelmet"
            dets += [DetBox(label, (x, 100, x + 60, 200), 0.9),
                     DetBox("Plate", (x + 10, 205, x + 50, 235), 0.9)]
        t0 = time.perf_counter()
        pipe.step(f, dets, timestamp=f / 25.0, read_plate=lambda b: ("MH12AB1234", 0.9))
        times.append((time.perf_counter() - t0) * 1000.0)
    out["decision_core_step_3_bikes"] = summarize_times(times)

    if image_path and os.path.exists(image_path):
        # Import the app against throwaway storage — never the real traffic.db.
        os.environ["TRAFFIC_DB_PATH"] = os.path.join(tmp, "app.db")
        os.environ["EVIDENCE_DIR"] = os.path.join(tmp, "evidence")
        if device:
            os.environ["DETECT_DEVICE"] = device
        import app as app_module
        from modules.detector import analyze_image

        app_module._models = (model, reader)
        client = app_module.app.test_client()
        with open(image_path, "rb") as fh:
            payload = fh.read()
        name = os.path.basename(image_path)

        def api_call():
            from io import BytesIO

            return client.post("/analyze", data={"image": (BytesIO(payload), name)},
                               content_type="multipart/form-data")

        api_call()
        analyze_image(image_path, model, reader, evidence_dir=os.path.join(tmp, "e"),
                      device=device)
        direct, api = [], []
        for _ in range(10):
            t0 = time.perf_counter()
            analyze_image(image_path, model, reader, evidence_dir=os.path.join(tmp, "e"),
                          device=device)
            direct.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            api_call()
            api.append((time.perf_counter() - t0) * 1000.0)
        d, a = summarize_times(direct), summarize_times(api)
        out["analyze_direct"] = d
        out["analyze_via_api"] = a
        out["api_overhead_ms"] = round(a["mean_ms"] - d["mean_ms"], 2)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark the detection pipeline.")
    ap.add_argument("--model", required=True, help="path to YOLO weights (.pt)")
    ap.add_argument("--image", default=None, help="sample image for inference/OCR timing")
    ap.add_argument("--video", default=None, help="sample video for throughput timing")
    ap.add_argument("--iterations", type=int, default=30, help="image inference repeats")
    ap.add_argument("--max-frames", type=int, default=200, help="cap for video timing")
    ap.add_argument("--no-ocr-lock", action="store_true",
                    help="disable the plate OCR lock so OCR runs on every "
                         "plated frame — the baseline arm of the OCR-lock A/B")
    ap.add_argument("--ocr-lock-ab", action="store_true",
                    help="run the video benchmark twice (lock on and off) and "
                         "report the measured speedup")
    ap.add_argument("--device", default="auto",
                    help="device for the video benchmark: cuda | mps | cpu | auto")
    ap.add_argument("--devices", nargs="*", default=None,
                    help="also time single-image inference on each of these devices")
    ap.add_argument("--micro", action="store_true",
                    help="also time DB writes, the decision core, and API overhead")
    ap.add_argument("--out", default="eval/results", help="output directory")
    ap.add_argument("--name", default="benchmark", help="report name")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)

    if not os.path.exists(args.model):
        log.error("model weights not found: %s", args.model)
        return 2
    if not args.image and not args.video:
        log.error("give at least one of --image / --video to benchmark")
        return 2

    device = resolve_device(args.device)
    from modules.detector import load_models
    from modules.provenance import run_provenance

    t0 = time.perf_counter()
    model, reader = load_models(args.model)
    load_seconds = round(time.perf_counter() - t0, 2)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": device,
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or platform.machine(),
        },
        "model_load_seconds": load_seconds,
        "model_load_state": "cold: YOLO + EasyOCR constructed from disk in this process",
        "inputs": {},
    }

    if args.image and os.path.exists(args.image):
        payload["inputs"]["image"] = _media_info(args.image)
        payload["image_benchmark"] = benchmark_image(
            model, reader, args.image, args.iterations, device=device)
        for dev in args.devices or []:
            payload.setdefault("image_by_device", {})[dev] = benchmark_image(
                model, reader, args.image, args.iterations, device=dev)
    elif args.image:
        log.warning("image not found: %s", args.image)

    if args.video and os.path.exists(args.video):
        payload["inputs"]["video"] = _media_info(args.video)
        payload["video_benchmark"] = benchmark_video(
            model, reader, args.video, args.max_frames,
            ocr_lock=not args.no_ocr_lock, device=device,
        )
        if args.ocr_lock_ab:
            # The baseline arm: same clip, same model, lock disabled. Both arms
            # in one process so the comparison is not across machine states.
            payload["video_benchmark_no_ocr_lock"] = benchmark_video(
                model, reader, args.video, args.max_frames, ocr_lock=False,
                device=device,
            )
            locked = payload["video_benchmark"]["throughput_fps"]
            unlocked = payload["video_benchmark_no_ocr_lock"]["throughput_fps"]
            payload["ocr_lock_speedup"] = {
                "locked_fps": locked,
                "unlocked_fps": unlocked,
                "speedup_pct": round(100 * (locked - unlocked) / unlocked, 1)
                if unlocked else 0.0,
                "locked_ocr_calls": payload["video_benchmark"]["ocr_calls"],
                "unlocked_ocr_calls":
                    payload["video_benchmark_no_ocr_lock"]["ocr_calls"],
                "same_decisions": (payload["video_benchmark"]["decisions"]
                                   == payload["video_benchmark_no_ocr_lock"]["decisions"]),
            }
    elif args.video:
        log.warning("video not found: %s", args.video)

    if args.micro:
        payload["micro"] = benchmark_micro(model, reader, args.image, device)

    payload["peak_rss_mb"] = _peak_rss_mb()
    payload["provenance"] = run_provenance(model_paths=[args.model],
                                           config={"device": device, "batch": 1,
                                                   "iterations": args.iterations,
                                                   "max_frames": args.max_frames})

    ab = payload.get("ocr_lock_speedup")
    if ab:
        print(f"\nOCR lock A/B: {ab['unlocked_fps']} FPS "
              f"({ab['unlocked_ocr_calls']} OCR calls) -> {ab['locked_fps']} FPS "
              f"({ab['locked_ocr_calls']} OCR calls) = {ab['speedup_pct']:+.1f}%")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out, f"{args.name}.md"), "w") as fh:
        fh.write(render_markdown(payload))

    _print_summary(payload)
    print(f"\nBenchmark: {path} (+ .md)")
    return 0


def render_markdown(p: dict) -> str:
    env = p.get("provenance", {}).get("environment", {})
    mdl = (p.get("provenance", {}).get("models") or [{}])[0]
    lines = ["# Performance benchmark", "",
             f"- Generated: {p['generated_at']}",
             f"- Hardware: {env.get('chip') or env.get('processor')} · {env.get('platform')}",
             f"- Software: python {env.get('python')} · torch {env.get('torch')} · "
             f"ultralytics {env.get('ultralytics')} · opencv {env.get('cv2')}",
             f"- Model: `{mdl.get('version')}` (`{str(mdl.get('sha256'))[:23]}…`)",
             "- Batch size 1 throughout (the pipeline processes frames one at a time).",
             f"- Model load (cold, YOLO + EasyOCR): **{p['model_load_seconds']} s**",
             ""]
    for key, info in p.get("inputs", {}).items():
        lines.append(f"- Input {key}: `{info['path']}` {info.get('width')}×{info.get('height')}"
                     + (f", {info.get('frames')} frames @ {info.get('fps')} fps"
                        if key == "video" else "")
                     + f", `{info['sha256'][:23]}…`")
    lines.append("")
    rows = []
    ib = p.get("image_benchmark")
    if ib:
        rows.append((ib["device"], ib))
    for dev, b in (p.get("image_by_device") or {}).items():
        rows.append((dev, b))
    if rows:
        lines += ["## Single image — YOLO inference", "",
                  "| device | first call (cold) | warm mean | p50 | p90 | FPS |",
                  "|---|---:|---:|---:|---:|---:|"]
        seen = set()
        for dev, b in rows:
            if dev in seen:
                continue
            seen.add(dev)
            i = b["inference"]
            lines.append(f"| {dev} | {b['first_call_ms']} ms | {i['mean_ms']} ms | "
                         f"{i['p50_ms']} | {i['p90_ms']} | {i['fps']} |")
        if ib and ib.get("ocr"):
            lines += ["", f"EasyOCR on the plate crop: {ib['ocr']['mean_ms']} ms mean "
                      f"(p90 {ib['ocr']['p90_ms']})."]
        lines.append("")
    for key, title in (("video_benchmark", "OCR lock on (shipped)"),
                       ("video_benchmark_no_ocr_lock", "OCR lock off")):
        vb = p.get(key)
        if not vb:
            continue
        prof = vb.get("stage_profile") or {}
        lines += [f"## Video — {title} — device {vb['device']}", "",
                  f"{vb['frames']} frames in {vb['wall_seconds']} s → "
                  f"**{vb['throughput_fps']} FPS** ({vb['ms_per_frame']} ms/frame), "
                  f"{vb['ocr_calls']} OCR calls.", "",
                  "| stage | % of wall | ms/call | calls |", "|---|---:|---:|---:|"]
        for name, st in (prof.get("stages") or {}).items():
            lines.append(f"| {name} | {st['pct_of_wall']}% | {st['ms_per_call']} | {st['calls']} |")
        other = prof.get("unaccounted_pct_of_wall", 0)
        lines += [f"| unaccounted (draw, glue) | {other}% | | |", ""]
    ab = p.get("ocr_lock_speedup")
    if ab:
        lines += [f"OCR lock A/B (same clip, same process): {ab['unlocked_fps']} → "
                  f"**{ab['locked_fps']} FPS ({ab['speedup_pct']:+.1f}%)**, "
                  f"{ab['unlocked_ocr_calls']} → {ab['locked_ocr_calls']} OCR calls."]
        if "same_decisions" in ab:
            dec = p["video_benchmark"]["decisions"]
            fined = ", ".join(f"{pl} {vi}" for pl, vi in dec["fined"]) or "none"
            lines.append(
                f"Same decisions in both arms: **{'yes' if ab['same_decisions'] else 'NO'}** "
                f"(fined: {fined}; withheld: {dec['withheld']}).")
        lines.append("")
    mi = p.get("micro")
    if mi:
        lines += ["## Micro-benchmarks", "", "| operation | mean | p90 |", "|---|---:|---:|"]
        for name in ("db_record_fine", "decision_core_step_3_bikes", "analyze_direct",
                     "analyze_via_api"):
            if name in mi:
                lines.append(f"| {name} | {mi[name]['mean_ms']} ms | {mi[name]['p90_ms']} ms |")
        if "api_overhead_ms" in mi:
            lines += ["", f"HTTP + Flask + validation + DB overhead of `/analyze` over a direct "
                      f"call: **{mi['api_overhead_ms']} ms**."]
        lines.append("")
    if p.get("peak_rss_mb") is not None:
        lines.append(f"Peak RSS: {p['peak_rss_mb']} MB.")
    return "\n".join(lines) + "\n"


def _print_summary(p: dict) -> None:
    print("\n=== Benchmark ===")
    print(f"Device: {p['device']}  |  model load (cold): {p['model_load_seconds']}s")
    ib = p.get("image_benchmark")
    if ib and ib.get("inference"):
        i = ib["inference"]
        print(f"Image inference: {i['mean_ms']} ms mean ({i['fps']} FPS), "
              f"p50 {i['p50_ms']} / p90 {i['p90_ms']}")
    if ib and ib.get("ocr"):
        o = ib["ocr"]
        print(f"OCR (plate crop): {o['mean_ms']} ms mean")
    vb = p.get("video_benchmark")
    if vb:
        print(f"Video: {vb['frames']} frames in {vb['wall_seconds']}s "
              f"-> {vb['throughput_fps']} FPS ({vb['ms_per_frame']} ms/frame)")
        _print_stage_profile(vb.get("stage_profile") or {})
    if p.get("peak_rss_mb") is not None:
        print(f"Peak memory: {p['peak_rss_mb']} MB")


def _print_stage_profile(profile: dict) -> None:
    """Print where per-frame time went, most expensive first, so a bottleneck
    is obvious. ``unaccounted`` is honest slack (drawing, state machines, glue),
    never folded into a stage."""
    stages = profile.get("stages") or {}
    if not stages:
        return
    print("  Stage breakdown (% of video wall time):")
    for name, s in stages.items():
        print(f"    {name:<9} {s['pct_of_wall']:>5.1f}%  "
              f"({s['seconds']}s, {s['ms_per_call']} ms/call x{s['calls']})")
    print(f"    {'other':<9} {profile.get('unaccounted_pct_of_wall', 0.0):>5.1f}%  "
          f"({profile.get('unaccounted_seconds', 0.0)}s)")


if __name__ == "__main__":
    sys.exit(main())
