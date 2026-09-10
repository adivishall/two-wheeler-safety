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
