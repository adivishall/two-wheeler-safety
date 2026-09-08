"""Plate correction + temporal stabilization tests (pure, deterministic)."""

from modules.plate_info import is_valid_plate, matches_structure
from modules.plate_recognizer import (
    PlateConfig,
    PlateStabilizer,
    correct_plate,
)

# --- correction -------------------------------------------------------------

def test_already_valid_plate_is_unchanged():
    assert correct_plate("MH02DL4596") == ("MH02DL4596", 0)


def test_bh_series_recognized_without_edits():
    assert correct_plate("22BH1234AA") == ("22BH1234AA", 0)


def test_digit_lookalike_corrected_in_number_position():
    # 'O'->'0' in the RTO digits; 'B'->'8', 'S'->'5', 'I'->'1' in the number.
    assert correct_plate("MHO2DL4596")[0] == "MH02DL4596"
    assert correct_plate("MH02DL459B")[0] == "MH02DL4598"
    assert correct_plate("MH02DL459S")[0] == "MH02DL4595"
    assert correct_plate("MH02DL45I6")[0] == "MH02DL4516"


def test_letter_lookalike_corrected_in_series_position():
    corrected, edits = correct_plate("MH02D14596")  # '1'->'I' in series
    assert corrected == "MH02DI4596"
    assert edits == 1


def test_correction_is_bounded_and_refuses_to_hallucinate():
    # All-digit junk needs >=3 substitutions (2 just for the state letters) to
    # look like a plate, over the default budget of 2.
    assert correct_plate("12345678") == (None, 0)
    assert correct_plate("XYZ") == (None, 0)  # too short to be any plate
    assert correct_plate("") == (None, 0)
    # A caller can widen the budget explicitly, and then it does coerce.
    assert correct_plate("12345678", PlateConfig(max_edits=3))[0] is not None


def test_correction_can_be_disabled():
    assert correct_plate("MHO2DL4596", PlateConfig(enable_correction=False)) == (None, 0)


def test_matches_structure_vs_known_state():
    assert matches_structure("MH99XY1234") is True   # shaped like a plate
    assert is_valid_plate("MH99XY1234") is True       # MH is a known state
    assert is_valid_plate("QZ99XY1234") is False      # QZ is not a state
    assert matches_structure("QZ99XY1234") is True     # ...but still plate-shaped


# --- temporal stabilization -------------------------------------------------

def test_single_observation_never_elects_a_plate():
    stab = PlateStabilizer()  # min_observations defaults to 2
    stab.add("MH02DL4596", conf=0.99)
    assert stab.stable_plate is None
    assert stab.num_observations == 1


def test_majority_valid_reading_wins():
    stab = PlateStabilizer()
    stab.add("MH02DL4596", conf=0.9)
    stab.add("MH02DL4596", conf=0.8)
    stab.add("MH02DL4590", conf=0.4)  # one noisy frame disagrees
    assert stab.stable_plate == "MH02DL4596"
    assert stab.disagreement_count == 1
    assert 0.0 < stab.confidence <= 1.0


def test_noisy_frames_corrected_then_agree():
    # Two frames read the same plate but with look-alike noise; after
    # correction they agree and elect the true plate.
    stab = PlateStabilizer()
    stab.add("MHO2DL4596", conf=0.7)   # O instead of 0
    stab.add("MH02DL459B", conf=0.7)   # B instead of 8 -> different plate!
    stab.add("MH02DL4596", conf=0.9)
    stab.add("MH02DL4596", conf=0.9)
    assert stab.stable_plate == "MH02DL4596"


def test_only_invalid_readings_elect_nothing():
    stab = PlateStabilizer()
    stab.add("!!!", conf=0.9)          # empty after normalization -> skipped
    stab.add("12345678", conf=0.9)     # can't be coerced within edit budget
    assert stab.stable_plate is None


def test_result_exposes_full_state():
    stab = PlateStabilizer()
    stab.add("MH02DL4596", conf=0.9)
    stab.add("MH02DL4596", conf=0.9)
    r = stab.result()
    assert r.stable == "MH02DL4596"
    assert r.num_observations == 2
    assert r.disagreement_count == 0
    assert r.valid is True
    assert r.confidence > 0
