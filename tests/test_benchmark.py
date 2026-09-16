"""Tests for benchmark.py helpers that don't need the model."""

import benchmark as bench


def test_summarize_times_empty():
    assert bench.summarize_times([]) == {}


def test_summarize_times_stats():
    s = bench.summarize_times([10.0, 20.0, 30.0, 40.0])
    assert s["n"] == 4
    assert s["min_ms"] == 10.0 and s["max_ms"] == 40.0
    assert s["mean_ms"] == 25.0
    assert s["fps"] == round(1000.0 / 25.0, 1)


def test_resolve_device_passthrough():
    assert bench.resolve_device("cpu") == "cpu"
    assert bench.resolve_device("auto") in {"cpu", "cuda", "mps"}


def test_parse_args():
    args = bench.parse_args(["--model", "m.pt", "--image", "a.jpg", "--iterations", "5"])
    assert args.model == "m.pt" and args.image == "a.jpg" and args.iterations == 5


def test_benchmark_forwards_the_ocr_lock_setting(monkeypatch):
    """Regression: benchmark.py used to call process_video without passing the
    OCR-lock settings, so --no-ocr-lock (and the documented
    OCR_LOCK_CONFIDENCE=1.1) silently did nothing and *both* arms of the
    OCR-lock A/B ran with the lock on. The published +172% speedup was therefore
    not measuring what it claimed."""
    import benchmark

    captured = {}

    def fake_process_video(*args, **kwargs):
        captured.update(kwargs)
        return {"frames": 10, "profile": {"stages": {"ocr": {"calls": 3}}}}

    monkeypatch.setattr("modules.video_detector.process_video", fake_process_video)

    benchmark.benchmark_video(None, None, "v.mp4", 10, ocr_lock=False)
    assert captured["ocr_lock_confidence"] > 1.0, \
        "a confidence above 1.0 is what makes the lock unreachable"

    captured.clear()
    benchmark.benchmark_video(None, None, "v.mp4", 10, ocr_lock=True)
    assert captured["ocr_lock_confidence"] <= 1.0


def test_benchmark_reports_ocr_call_count(monkeypatch):
    """Without the call count the two A/B arms are indistinguishable in the
    output — which is exactly how the broken A/B went unnoticed."""
    import benchmark

    monkeypatch.setattr(
        "modules.video_detector.process_video",
        lambda *a, **k: {"frames": 10, "profile": {"stages": {"ocr": {"calls": 7}}}},
    )
    out = benchmark.benchmark_video(None, None, "v.mp4", 10)
    assert out["ocr_calls"] == 7
    assert out["ocr_lock"] is True
