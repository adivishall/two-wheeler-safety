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
"un-confirm" once confirmed, so a fine is issued at most once.

Both support two confirmation rules, chosen by config:

* **consecutive** (``vote_window=None``) — the violation must hold for
  ``confirm_window`` frames in a row; one contrary frame starts over.
* **k-of-n vote** (``vote_window=n``) — confirm once ``confirm_window`` of the
  last ``n`` *observed* frames support the violation. Frames with no rider box
  are not evidence either way and do not enter the window. One flipped frame
  costs one vote instead of the whole streak, so recall survives per-frame
  detector noise that a strict streak cannot. Which rule ships, and why, is
  measured in ``modules/temporal_eval.py``. The pipeline
records on ``confirmed`` (not only the transition frame) so that if the plate
isn't yet readable at the confirming frame the fine still lands once it is.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass


class HelmetState(enum.Enum):
    UNKNOWN = "unknown"
    HELMET = "helmet"
    NO_HELMET_CANDIDATE = "no_helmet_candidate"
    CONFIRMED_NO_HELMET = "confirmed_no_helmet"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class HelmetConfig:
    confirm_window: int = 5  # supporting no-helmet frames needed to confirm
    min_conf: float = 0.3  # model confidence a no-helmet frame must clear
    miss_tolerance: int = 1  # (consecutive rule) blank frames before a candidate decays
    vote_window: int | None = None  # k-of-n: window of observed frames; None = consecutive
    # Track-level gate: additionally require at least `min_observed` frames with
    # a rider box, of which at least `min_fraction` support the violation. A
    # streak alone gives a long-lived compliant rider many chances to produce
    # one lucky run of flips; the fraction converges instead. 0 / 0.0 = off.
    min_observed: int = 0
    min_fraction: float = 0.0


DEFAULT_HELMET_CONFIG = HelmetConfig()


class HelmetStateMachine:
    def __init__(self, config: HelmetConfig = DEFAULT_HELMET_CONFIG):
        self.config = config
        self.state = HelmetState.UNKNOWN
        self.candidate_frames = 0
        self.supporting_frames = 0
        self.observed_frames = 0
        self.best_conf = 0.0
        self.best_frame: int | None = None
        self._misses = 0
        self._ever_confirmed = False
        self._just_confirmed = False
        n = config.vote_window
        self._votes: deque | None = (
            deque(maxlen=n) if n is not None and n > config.confirm_window else None
        )

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
        if ambiguous or has_helmet or has_no_helmet:
            self.observed_frames += 1
        if self._votes is not None:
            return self._update_vote(
                has_helmet=has_helmet, has_no_helmet=has_no_helmet,
                no_helmet_conf=no_helmet_conf, ambiguous=ambiguous, frame_idx=frame_idx,
            )

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
            elif self.candidate_frames >= self.config.confirm_window and self._gate_ok():
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

    def _update_vote(
        self, *, has_helmet, has_no_helmet, no_helmet_conf, ambiguous, frame_idx,
    ) -> HelmetState:
        assert self._votes is not None
        observed = ambiguous or has_helmet or has_no_helmet
        if not observed:
            return self.state  # a blank frame is not evidence either way
        # An ambiguous frame (helmet AND no-helmet on one rider) is observed but
        # never supports: the contradiction safeguard, as a zero vote.
        support = has_no_helmet and not ambiguous and no_helmet_conf >= self.config.min_conf
        self._votes.append(1 if support else 0)
        self.candidate_frames = sum(self._votes)
        if support:
            self.supporting_frames += 1
            if no_helmet_conf > self.best_conf:
                self.best_conf = no_helmet_conf
                self.best_frame = frame_idx
        if self._ever_confirmed:
            return self.state
        if self.candidate_frames >= self.config.confirm_window and self._gate_ok():
            self.state = HelmetState.CONFIRMED_NO_HELMET
            self._ever_confirmed = True
            self._just_confirmed = True
        elif ambiguous:
            self.state = HelmetState.AMBIGUOUS
        elif self.candidate_frames > 0:
            self.state = HelmetState.NO_HELMET_CANDIDATE
        else:
            self.state = HelmetState.HELMET
        return self.state

    def _gate_ok(self) -> bool:
        cfg = self.config
        if self.observed_frames < cfg.min_observed:
            return False
        if cfg.min_fraction <= 0.0:
            return True
        return self.supporting_frames >= cfg.min_fraction * self.observed_frames

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
    confirm_window: int = 5  # supporting triple-riding frames needed to confirm
    min_conf: float = 0.3
    clear_after: int = 5  # blank frames before an active state is CLEARED
    vote_window: int | None = None  # k-of-n: window of observed frames; None = consecutive


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
        n = config.vote_window
        self._votes: deque | None = (
            deque(maxlen=n) if n is not None and n > config.confirm_window else None
        )

    def update(
        self, *, has_triple: bool, conf: float, frame_idx: int,
        observed: bool | None = None,
    ) -> TripleState:
        """``observed`` says whether a rider box was present this frame (so a
        non-triple frame is evidence *against*); None infers it from
        ``has_triple`` for callers that only know about triple boxes."""
        self._just_confirmed = False
        if self._votes is not None:
            return self._update_vote(
                has_triple=has_triple, conf=conf, frame_idx=frame_idx,
                observed=has_triple if observed is None else observed,
            )

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

    def _update_vote(self, *, has_triple, conf, frame_idx, observed) -> TripleState:
        assert self._votes is not None
        if not observed:
            self._misses += 1
            if self._misses >= self.config.clear_after and self.state is TripleState.CANDIDATE:
                self.state = TripleState.CLEARED
            return self.state
        self._misses = 0
        support = has_triple and conf >= self.config.min_conf
        self._votes.append(1 if support else 0)
        self.candidate_frames = sum(self._votes)
        if support:
            self.frames_observed += 1
            if conf > self.best_conf:
                self.best_conf = conf
                self.best_frame = frame_idx
        if self._ever_confirmed:
            self.state = TripleState.CONFIRMED
        elif self.candidate_frames >= self.config.confirm_window:
            self.state = TripleState.CONFIRMED
            self._ever_confirmed = True
            self._just_confirmed = True
        elif self.candidate_frames > 0:
            self.state = TripleState.CANDIDATE
        return self.state

    @property
    def confirmed(self) -> bool:
        return self._ever_confirmed

    @property
    def just_confirmed(self) -> bool:
        return self._just_confirmed
