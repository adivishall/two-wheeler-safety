"""Helmet + triple-riding temporal state machine tests (pure, deterministic)."""

from modules.violation_state import (
    HelmetConfig,
    HelmetState,
    HelmetStateMachine,
    TripleConfig,
    TripleRidingStateMachine,
    TripleState,
)


def _no_helmet(sm, conf=0.9, frame=0):
    return sm.update(has_helmet=False, has_no_helmet=True, no_helmet_conf=conf,
                     ambiguous=False, frame_idx=frame)


def _helmet(sm, frame=0):
    return sm.update(has_helmet=True, has_no_helmet=False, no_helmet_conf=0.0,
                     ambiguous=False, frame_idx=frame)


def _ambiguous(sm, frame=0):
    return sm.update(has_helmet=True, has_no_helmet=True, no_helmet_conf=0.9,
                     ambiguous=True, frame_idx=frame)


def _blank(sm, frame=0):
    return sm.update(has_helmet=False, has_no_helmet=False, no_helmet_conf=0.0,
                     ambiguous=False, frame_idx=frame)


# --- helmet -----------------------------------------------------------------

def test_helmet_confirms_only_after_window():
    sm = HelmetStateMachine(HelmetConfig(confirm_window=5))
    for f in range(4):
        assert _no_helmet(sm, frame=f) is HelmetState.NO_HELMET_CANDIDATE
    assert _no_helmet(sm, frame=4) is HelmetState.CONFIRMED_NO_HELMET
    assert sm.confirmed and sm.just_confirmed


def test_helmet_below_confidence_threshold_does_not_count():
    sm = HelmetStateMachine(HelmetConfig(confirm_window=3, min_conf=0.5))
    for f in range(6):
        _no_helmet(sm, conf=0.2, frame=f)  # under threshold
    assert not sm.confirmed


def test_ambiguous_frame_blocks_confirmation():
    # The contradiction safeguard: helmet+no_helmet on one rider trusts neither.
    sm = HelmetStateMachine(HelmetConfig(confirm_window=3))
    _no_helmet(sm, frame=0)
    _no_helmet(sm, frame=1)
    assert _ambiguous(sm, frame=2) is HelmetState.AMBIGUOUS
    assert sm.candidate_frames == 0  # progress reset
    assert not sm.confirmed


def test_helmet_frame_returns_candidate_to_helmet():
    sm = HelmetStateMachine(HelmetConfig(confirm_window=5))
    _no_helmet(sm, frame=0)
    _no_helmet(sm, frame=1)
    assert _helmet(sm, frame=2) is HelmetState.HELMET
    assert not sm.confirmed


def test_helmet_confirmation_is_sticky_and_records_best_frame():
    sm = HelmetStateMachine(HelmetConfig(confirm_window=2))
    _no_helmet(sm, conf=0.7, frame=0)
    _no_helmet(sm, conf=0.95, frame=1)  # confirms here, best conf
    assert sm.state is HelmetState.CONFIRMED_NO_HELMET
    # A later helmet frame does not un-confirm a rider already fined.
    _helmet(sm, frame=2)
    assert sm.confirmed
    assert sm.best_conf == 0.95 and sm.best_frame == 1


def test_blank_frames_decay_candidate_within_tolerance():
    sm = HelmetStateMachine(HelmetConfig(confirm_window=5, miss_tolerance=1))
    _no_helmet(sm, frame=0)
    _no_helmet(sm, frame=1)
    _blank(sm, frame=2)   # within tolerance, candidate held
    assert sm.candidate_frames == 2
    _blank(sm, frame=3)   # exceeds tolerance, decays
    assert sm.candidate_frames == 0
    assert sm.state is HelmetState.UNKNOWN


# --- triple riding ----------------------------------------------------------

def test_triple_confirms_after_window():
    sm = TripleRidingStateMachine(TripleConfig(confirm_window=4))
    for f in range(3):
        assert sm.update(has_triple=True, conf=0.8, frame_idx=f) is TripleState.CANDIDATE
    assert sm.update(has_triple=True, conf=0.8, frame_idx=3) is TripleState.CONFIRMED
    assert sm.confirmed


def test_triple_clears_after_absence():
    sm = TripleRidingStateMachine(TripleConfig(confirm_window=2, clear_after=3))
    sm.update(has_triple=True, conf=0.8, frame_idx=0)
    sm.update(has_triple=True, conf=0.8, frame_idx=1)  # confirmed
    for f in range(2, 5):
        sm.update(has_triple=False, conf=0.0, frame_idx=f)
    assert sm.state is TripleState.CLEARED
    assert sm.confirmed  # still counts as having happened (fined once)


def test_triple_single_frame_does_not_confirm():
    sm = TripleRidingStateMachine(TripleConfig(confirm_window=5))
    sm.update(has_triple=True, conf=0.9, frame_idx=0)
    assert not sm.confirmed
