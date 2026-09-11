"""Tests for the per-stage profiler (Phase 21).

These use real (tiny) sleeps so the accounting math is exercised without the
model — the profiler itself is model-agnostic.
"""

import time

from modules.profiling import NULL, StageProfiler


def test_stage_accumulates_time_and_counts():
    p = StageProfiler()
    for _ in range(3):
        with p.stage("yolo"):
            time.sleep(0.005)
    assert p.counts["yolo"] == 3
    assert p.totals["yolo"] >= 0.012  # 3 x ~5ms, allowing scheduler slack


def test_summary_percentages_and_unaccounted():
    p = StageProfiler()
    with p.stage("yolo"):
        time.sleep(0.02)
    with p.stage("ocr"):
        time.sleep(0.01)
    # Reference wall is larger than the measured stages, so the remainder is
    # reported honestly as unaccounted rather than inflating a stage.
    p.set_wall(0.05)
    s = p.summary()
    assert s["wall_seconds"] == 0.05
    assert set(s["stages"]) == {"yolo", "ocr"}
    # yolo took ~2x ocr, so it ranks first and has the larger share.
    names = list(s["stages"])
    assert names[0] == "yolo"
    assert s["stages"]["yolo"]["pct_of_wall"] > s["stages"]["ocr"]["pct_of_wall"]
    assert s["unaccounted_seconds"] > 0
    # measured + unaccounted reconstructs the wall (within rounding).
    assert abs(s["measured_seconds"] + s["unaccounted_seconds"] - s["wall_seconds"]) < 0.01


def test_add_records_externally_timed_stage():
    p = StageProfiler()
    p.add("db", 0.123)
    p.set_wall(1.0)
    s = p.summary()
    assert s["stages"]["db"]["seconds"] == 0.123
    assert s["stages"]["db"]["calls"] == 1
    assert s["stages"]["db"]["pct_of_wall"] == 12.3


def test_null_profiler_is_a_noop():
    with NULL.stage("anything"):
        pass
    NULL.add("x", 1.0)
    NULL.set_wall(1.0)
    assert NULL.summary() == {}


def test_summary_without_wall_uses_measured():
    p = StageProfiler()
    p.add("encode", 0.2)
    s = p.summary()
    # No set_wall: wall defaults to measured, so nothing is unaccounted.
    assert s["wall_seconds"] == 0.2
    assert s["unaccounted_seconds"] == 0.0
