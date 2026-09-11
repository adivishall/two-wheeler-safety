"""Lightweight per-stage timing for the video pipeline.

Phase 21 asks a specific, honest question: *where does the per-frame time
actually go* — YOLO inference, OCR, tracking/association, evidence writing, the
DB call, or video encoding? The answer decides what (if anything) is worth
optimizing; guessing does not.

:class:`StageProfiler` accumulates wall time per named stage via a cheap
context manager. It only ever wraps **non-overlapping** calls, so the summed
stage time never exceeds the frame-loop wall time; whatever is left (Python
glue, drawing, the state machines) is reported honestly as ``unaccounted``
rather than being silently folded into a stage.

Production stays untouched: :data:`NULL` is a zero-overhead profiler, and
``process_video`` uses it unless a real profiler is passed (only the benchmark
does). Nothing here fabricates a number — every value comes from
``time.perf_counter`` around a real call.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class StageProfiler:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._wall: float | None = None

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def add(self, name: str, seconds: float) -> None:
        """Record a stage duration measured elsewhere (e.g. a call already
        timed for another reason)."""
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def set_wall(self, seconds: float) -> None:
        """Set the reference wall time the percentages are taken against
        (typically the frame-loop elapsed time)."""
        self._wall = seconds

    def summary(self) -> dict:
        measured = sum(self.totals.values())
        wall = self._wall if self._wall is not None else measured
        stages = {}
        for name, total in sorted(self.totals.items(), key=lambda kv: kv[1], reverse=True):
            calls = self.counts[name]
            stages[name] = {
                "seconds": round(total, 4),
                "calls": calls,
                "pct_of_wall": round(100 * total / wall, 1) if wall else 0.0,
                "ms_per_call": round(1000 * total / calls, 3) if calls else 0.0,
            }
        unaccounted = max(0.0, wall - measured)
        return {
            "wall_seconds": round(wall, 3),
            "measured_seconds": round(measured, 3),
            "unaccounted_seconds": round(unaccounted, 3),
            "unaccounted_pct_of_wall": round(100 * unaccounted / wall, 1) if wall else 0.0,
            "stages": stages,
        }


class _NullProfiler:
    """Zero-overhead stand-in used in production (no timing, no dict churn)."""

    @contextmanager
    def stage(self, name: str):
        yield

    def add(self, name: str, seconds: float) -> None:
        pass

    def set_wall(self, seconds: float) -> None:
        pass

    def summary(self) -> dict:
        return {}


NULL = _NullProfiler()
