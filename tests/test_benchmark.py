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


def test_benchmark_records_what_each_arm_decided(monkeypatch):
    """A speed-up is only a speed-up if the decisions are unchanged; the A/B
    report compares them, so each arm must carry its fines and withheld count."""
    import benchmark

    summary = {"frames": 10, "ocr_calls": 2,
               "violations": [{"plate": "MH12AB1234", "violation": "no_helmet"}],
               "unfined_confirmations": [{"track": 3}]}
    monkeypatch.setattr("modules.video_detector.process_video", lambda *a, **k: summary)
    out = benchmark.benchmark_video(None, None, "v.mp4", 10)
    assert out["decisions"] == {"fined": [["MH12AB1234", "no_helmet"]], "withheld": 1}


def test_benchmark_video_leaves_no_evidence_behind(monkeypatch):
    """process_video writes evidence beside its output video; the benchmark's
    output lives in a temporary directory that must be gone afterwards."""
    import os

    import benchmark

    seen = {}

    def fake_process_video(video, model, reader, output_path, **kw):
        seen["dir"] = os.path.dirname(output_path)
        open(os.path.join(seen["dir"], "evidence.jpg"), "w").close()
        return {"frames": 1}

    monkeypatch.setattr("modules.video_detector.process_video", fake_process_video)
    benchmark.benchmark_video(None, None, "v.mp4", 1)
    assert not os.path.exists(seen["dir"])


def test_arms_are_interleaved_and_the_median_run_is_reported():
    """Two back-to-back runs of one config differed by 20% FPS, so arms run
    interleaved (A B, B A, ...) and the median run is what gets reported."""
    order = []
    fps = iter([30.0, 17.0, 16.0, 26.0, 33.0, 18.0])

    def run(lock):
        order.append(lock)
        return {"throughput_fps": next(fps), "ocr_calls": 30 if lock else 120,
                "decisions": {"fined": [["MH12AB1234", "no_helmet"]], "withheld": 0}}

    locked, unlocked = bench.run_interleaved(run, [True, False], 3)
    assert order == [True, False, False, True, True, False]
    med = bench.median_run(locked)
    assert med["throughput_fps"] == 30.0 and med["fps_min"] == 26.0 and med["fps_max"] == 33.0
    ab = bench.lock_speedup(locked, unlocked)
    assert ab["unlocked_fps"] == 17.0 and ab["speedup_pct"] == round(100 * 13 / 17, 1)
    assert ab["same_decisions"] is True and ab["repeats"] == 3


def test_a_changed_decision_is_reported_even_if_faster():
    def arm(fps, fined):
        return {"throughput_fps": fps, "ocr_calls": 1,
                "decisions": {"fined": fined, "withheld": 0}}

    ab = bench.lock_speedup([arm(30.0, [["A", "no_helmet"]])], [arm(15.0, [["B", "no_helmet"]])])
    assert ab["same_decisions"] is False


def test_paired_overhead_says_when_it_is_below_the_noise():
    noisy = bench.paired_difference([45.0, 50.0, 44.0, 52.0, 47.0], [47.0, 46.0, 49.0, 48.0, 45.0])
    assert noisy["distinguishable"] is False
    clear = bench.paired_difference([55.0, 56.0, 54.0, 57.0, 55.0], [45.0, 46.0, 44.0, 47.0, 45.0])
    assert clear["distinguishable"] is True and clear["median_ms"] == 10.0
