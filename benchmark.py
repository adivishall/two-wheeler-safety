"""Reproducible performance benchmark for the detection pipeline.

Measures the things that decide whether the system keeps up with real footage:
model load time, single-image inference latency, EasyOCR latency on a plate
crop, end-to-end video throughput, and peak memory. Every number is *measured*
here — nothing is hard-coded — and the JSON it writes records the machine and
device so results are comparable across runs.

    python3 benchmark.py \\
        --model runs/detect/traffic_model-2/weights/best.pt \\
        --image test.jpg --video sample.mp4 --iterations 30

Sections are skipped cleanly when their input is absent (e.g. no ``--video``),
so the tool is useful even with just a sample image. The repo ships no weights
or video, so this is run locally against the owner's model/footage.
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


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def benchmark_image(model, reader, image_path, iterations) -> dict:
    import cv2

    out = {"image": image_path}
    # Inference latency.
    model.predict(source=image_path, verbose=False)  # warm up
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = model.predict(source=image_path, verbose=False)[0]
        times.append((time.perf_counter() - t0) * 1000.0)
    out["inference"] = summarize_times(times)

    # OCR latency on the first detected Plate crop (or the whole image if none).
    img = cv2.imread(image_path)
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


def benchmark_video(model, reader, video_path, max_frames) -> dict:
    import tempfile

    from modules.video_detector import process_video

    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        t0 = time.perf_counter()
        summary = process_video(
            video_path, model, reader, out_path,
            record_fn=lambda *a, **k: 0, max_frames=max_frames,
        )
        elapsed = time.perf_counter() - t0
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass
    frames = summary.get("frames", 0)
    return {
        "video": video_path,
        "frames": frames,
        "wall_seconds": round(elapsed, 2),
        "throughput_fps": round(frames / elapsed, 2) if elapsed else 0.0,
        "ms_per_frame": round(elapsed * 1000.0 / frames, 2) if frames else 0.0,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark the detection pipeline.")
    ap.add_argument("--model", required=True, help="path to YOLO weights (.pt)")
    ap.add_argument("--image", default=None, help="sample image for inference/OCR timing")
    ap.add_argument("--video", default=None, help="sample video for throughput timing")
    ap.add_argument("--iterations", type=int, default=30, help="image inference repeats")
    ap.add_argument("--max-frames", type=int, default=200, help="cap for video timing")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--out", default="reports", help="output directory")
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
    }

    if args.image and os.path.exists(args.image):
        payload["image_benchmark"] = benchmark_image(
            model, reader, args.image, args.iterations
        )
    elif args.image:
        log.warning("image not found: %s", args.image)

    if args.video and os.path.exists(args.video):
        payload["video_benchmark"] = benchmark_video(
            model, reader, args.video, args.max_frames
        )
    elif args.video:
        log.warning("video not found: %s", args.video)

    payload["peak_rss_mb"] = _peak_rss_mb()

    os.makedirs(args.out, exist_ok=True)
    name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = os.path.join(args.out, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    _print_summary(payload)
    print(f"\nBenchmark JSON: {path}")
    return 0


def _print_summary(p: dict) -> None:
    print("\n=== Benchmark ===")
    print(f"Device: {p['device']}  |  model load: {p['model_load_seconds']}s")
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
    if p.get("peak_rss_mb") is not None:
        print(f"Peak memory: {p['peak_rss_mb']} MB")


if __name__ == "__main__":
    sys.exit(main())
