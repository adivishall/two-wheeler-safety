"""OCR evaluation + preprocessing tests (model-free: no EasyOCR/torch)."""

import numpy as np
import pytest

from modules import ocr_preprocess as pp
from modules.ocr_eval import (
    OcrObservation,
    char_accuracy,
    evaluate,
    evaluate_with_conditions,
    levenshtein,
)

# -- metrics ----------------------------------------------------------------

def test_levenshtein_basics():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("abc", "") == 3
    assert levenshtein("MH12AB1234", "MH12AB1284") == 1


def test_char_accuracy_partial_credit():
    assert char_accuracy("MH12AB1234", "MH12AB1234") == 1.0
    assert char_accuracy("MH12AB1234", "MH12AB1284") == 0.9
    assert char_accuracy("", "") == 1.0
    assert char_accuracy("", "junk") == 0.0


def test_evaluate_all_correct():
    obs = [
        OcrObservation("MH12AB1234", "MH12AB1234", 0.9),
        OcrObservation("KA05MN6789", "KA05MN6789", 0.8),
    ]
    m = evaluate(obs)
    assert m.exact_match == 1.0
    assert m.normalized_match == 1.0
    assert m.char_accuracy == 1.0
    assert m.mean_edit_distance == 0.0
    assert m.invalid_rate == 0.0


def test_normalized_match_ignores_spacing_and_case():
    obs = [OcrObservation("MH 12 AB 1234", "mh12ab1234", 0.7)]
    m = evaluate(obs)
    assert m.normalized_match == 1.0
    assert m.exact_match == 0.0  # raw strings differ


def test_invalid_rate_counts_structure_failures():
    obs = [
        OcrObservation("MH12AB1234", "MH12AB1234", 0.9),  # valid
        OcrObservation("MH12AB1234", "XYZ!!", 0.4),       # junk -> invalid
    ]
    m = evaluate(obs)
    assert m.invalid_rate == 0.5


def test_confidence_separation_is_correct_minus_wrong():
    obs = [
        OcrObservation("MH12AB1234", "MH12AB1234", 0.9),  # correct
        OcrObservation("KA05MN6789", "KA05MN0000", 0.3),  # wrong
    ]
    m = evaluate(obs)
    assert m.mean_conf_correct == pytest.approx(0.9)
    assert m.mean_conf_wrong == pytest.approx(0.3)
    assert m.confidence_separation == pytest.approx(0.6)


def test_empty_evaluation():
    assert evaluate([]).n == 0


def test_by_condition_breakdown():
    obs = [
        OcrObservation("MH12AB1234", "MH12AB1234", 0.9, condition="clean"),
        OcrObservation("KA05MN6789", "KA05MN0000", 0.3, condition="blur"),
        OcrObservation("KA05MN6789", "KA05MN6789", 0.8, condition="blur"),
    ]
    rep = evaluate_with_conditions(obs)
    assert rep.overall.n == 3
    assert set(rep.by_condition) == {"clean", "blur"}
    assert rep.by_condition["clean"].normalized_match == 1.0
    assert rep.by_condition["blur"].normalized_match == 0.5


# -- preprocessing ----------------------------------------------------------

def _img(h=20, w=80):
    return (np.random.default_rng(0).integers(0, 255, (h, w, 3))).astype(np.uint8)


@pytest.mark.parametrize("preset", sorted(pp.PIPELINES))
def test_every_pipeline_runs_and_returns_bgr(preset):
    out = pp.apply_pipeline(_img(), preset)
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.dtype == np.uint8


def test_upscale_grows_small_crops_only():
    small = _img(h=20, w=80)
    grown = pp.upscale(small, min_height=64)
    assert grown.shape[0] >= 64  # 20 * min(4, 64/20)=3.2 -> 64
    big = _img(h=100, w=200)
    assert pp.upscale(big, min_height=64).shape == big.shape  # no-op when tall enough


def test_upscale_respects_max_scale_cap():
    tiny = _img(h=10, w=40)  # needs 6.4x for 64px, but cap is 4x
    grown = pp.upscale(tiny, min_height=64, max_scale=4.0)
    assert grown.shape[0] == 40  # 10 * 4.0


def test_pad_adds_border():
    out = pp.pad(_img(20, 20), border=10)
    assert out.shape[0] == 40 and out.shape[1] == 40


def test_perspective_correct_requires_four_corners():
    with pytest.raises(ValueError):
        pp.perspective_correct(_img(), [[0, 0], [1, 1]])


def test_perspective_correct_warps_to_rectangle():
    img = _img(50, 100)
    corners = [[5, 5], [95, 8], [90, 45], [8, 42]]
    out = pp.perspective_correct(img, corners)
    assert out.ndim == 3 and out.shape[0] > 0 and out.shape[1] > 0


# -- CLI wiring (with a fake reader, no EasyOCR) ----------------------------

def test_read_labels_resolves_paths(tmp_path):
    from evaluate_ocr import read_labels

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("image_path,plate,condition\na.jpg,MH12AB1234,clean\n")
    rows = read_labels(str(csv_path))
    assert rows[0]["plate"] == "MH12AB1234"
    assert rows[0]["condition"] == "clean"
    assert rows[0]["image_path"].endswith("a.jpg")
    assert rows[0]["image_path"].startswith(str(tmp_path))


def test_ocr_one_with_fake_reader():
    from evaluate_ocr import ocr_one

    class FakeReader:
        def readtext(self, img, detail=1):
            return [([[0, 0]], "MH12 AB", 0.8), ([[0, 0]], "1234", 0.6)]

    text, conf = ocr_one(FakeReader(), _img(), "none")
    assert text == "MH12AB1234"
    assert conf == pytest.approx(0.7)
