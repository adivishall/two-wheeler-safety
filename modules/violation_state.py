"""Temporal violation state machines.

A single frame's detection is noisy, so a violation should be *confirmed over
time* before it becomes a fine, and should return gracefully to a clean state
when the evidence goes away. Two small, deterministic state machines do this
per vehicle:

* :class:`HelmetStateMachine` (Phase 4) — states unknown / helmet /
  no_helmet_candidate / confirmed_no_helmet / ambiguous. It **preserves the
  contradiction safeguard**: when the model asserts helmet *and* no-helmet on
  one rider (an ``ambiguous`` frame) it advances neither and reports
  ``ambiguous`` rather than guessing. A no-helmet call must clear a confidence
  threshold and persist for a configurable window before it confirms, and a
  helmet frame returns a not-yet-confirmed rider to ``helmet``.

* :class:`TripleRidingStateMachine` (Phase 5) — states none / candidate /
  confirmed / cleared, with configurable persistence to confirm and to clear.

Both remember the best (highest-confidence) supporting frame and never
"un-confirm" once confirmed, so a fine is issued at most once. The pipeline
records on ``confirmed`` (not only the transition frame) so that if the plate
isn't yet readable at the confirming frame the fine still lands once it is.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class HelmetState(enum.Enum):
    UNKNOWN = "unknown"
    HELMET = "helmet"
    NO_HELMET_CANDIDATE = "no_helmet_candidate"
    CONFIRMED_NO_HELMET = "confirmed_no_helmet"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class HelmetConfig:
    confirm_window: int = 5  # consecutive no-helmet frames needed to confirm
    min_conf: float = 0.3  # model confidence a no-helmet frame must clear
    miss_tolerance: int = 1  # blank frames tolerated before a candidate decays


DEFAULT_HELMET_CONFIG = HelmetConfig()


class HelmetStateMachine:
    def __init__(self, config: HelmetConfig = DEFAULT_HELMET_CONFIG):
        self.config = config
        self.state = HelmetState.UNKNOWN
        self.candidate_frames = 0
        self.supporting_frames = 0
        self.best_conf = 0.0
        self.best_frame: int | None = None
        self._misses = 0
        self._ever_confirmed = False
        self._just_confirmed = False

    def update(
        self,
        *,
        has_helmet: bool,
        has_no_helmet: bool,
        no_helmet_conf: float,
        ambiguous: bool,
        frame_idx: int,
    ) -> HelmetState:
        self._just_confirmed = False

        if ambiguous:
            # Model contradicts itself on one rider: trust neither, advance
            # nothing. A prior confirmation is not undone.
            self.candidate_frames = 0
            self._misses = 0
            if not self._ever_confirmed:
                self.state = HelmetState.AMBIGUOUS
            return self.state

        if has_no_helmet and no_helmet_conf >= self.config.min_conf:
            self.candidate_frames += 1
            self.supporting_frames += 1
            self._misses = 0
            if no_helmet_conf > self.best_conf:
                self.best_conf = no_helmet_conf
                self.best_frame = frame_idx
            if self._ever_confirmed:
                self.state = HelmetState.CONFIRMED_NO_HELMET
            elif self.candidate_frames >= self.config.confirm_window:
                self.state = HelmetState.CONFIRMED_NO_HELMET
                self._ever_confirmed = True
                self._just_confirmed = True
            else:
                self.state = HelmetState.NO_HELMET_CANDIDATE
            return self.state

        if has_helmet:
            # Graceful return: a not-yet-confirmed rider goes back to helmet.
            self.candidate_frames = 0
            self._misses = 0
            if not self._ever_confirmed:
                self.state = HelmetState.HELMET
            return self.state

        # Nothing observed this frame.
        self._misses += 1
        if self._misses > self.config.miss_tolerance:
            self.candidate_frames = 0
            if not self._ever_confirmed:
                self.state = HelmetState.UNKNOWN
        return self.state

    @property
    def confirmed(self) -> bool:
        return self._ever_confirmed

    @property
    def just_confirmed(self) -> bool:
        return self._just_confirmed


class TripleState(enum.Enum):
    NONE = "none"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    CLEARED = "cleared"


@dataclass(frozen=True)
class TripleConfig:
    confirm_window: int = 5  # consecutive triple-riding frames needed to confirm
    min_conf: float = 0.3
    clear_after: int = 5  # blank frames before an active state is CLEARED


DEFAULT_TRIPLE_CONFIG = TripleConfig()


class TripleRidingStateMachine:
    def __init__(self, config: TripleConfig = DEFAULT_TRIPLE_CONFIG):
        self.config = config
        self.state = TripleState.NONE
        self.candidate_frames = 0
        self.frames_observed = 0
        self.best_conf = 0.0
        self.best_frame: int | None = None
        self._misses = 0
        self._ever_confirmed = False
        self._just_confirmed = False

    def update(
        self, *, has_triple: bool, conf: float, frame_idx: int
    ) -> TripleState:
        self._just_confirmed = False

        if has_triple and conf >= self.config.min_conf:
            self.candidate_frames += 1
            self.frames_observed += 1
            self._misses = 0
            if conf > self.best_conf:
                self.best_conf = conf
                self.best_frame = frame_idx
            if self._ever_confirmed:
                self.state = TripleState.CONFIRMED
            elif self.candidate_frames >= self.config.confirm_window:
                self.state = TripleState.CONFIRMED
                self._ever_confirmed = True
                self._just_confirmed = True
            else:
                self.state = TripleState.CANDIDATE
            return self.state

        # Not observed this frame.
        self.candidate_frames = 0
        self._misses += 1
        if self._misses >= self.config.clear_after and self.state in (
            TripleState.CANDIDATE,
            TripleState.CONFIRMED,
        ):
            self.state = TripleState.CLEARED
        return self.state

    @property
    def confirmed(self) -> bool:
        return self._ever_confirmed

    @property
    def just_confirmed(self) -> bool:
        return self._just_confirmed
