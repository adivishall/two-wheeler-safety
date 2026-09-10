import main


def test_centroid():
    assert main.centroid((0, 0, 10, 10)) == (5.0, 5.0)


def test_clean_plate_strips_punctuation_and_uppercases():
    assert main.clean_plate("mh-12 ab*1234") == "MH12AB1234"


def test_clean_plate_returns_none_for_empty_input():
    assert main.clean_plate(None) is None
    assert main.clean_plate("") is None


def test_iou_of_identical_boxes_is_one():
    box = (0, 0, 100, 100)
    assert main.iou(box, box) == 1.0


def test_iou_of_non_overlapping_boxes_is_zero():
    assert main.iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_iou_real_contradiction_case():
    # plate6.jpeg: a real photo of a helmeted rider where the model output
    # both WithHelmet and WithoutHelmet on nearly the same box (IoU 0.95),
    # and the higher-confidence one (WithoutHelmet) was the wrong answer.
    without_helmet = (240, 274, 880, 1242)
    with_helmet = (229, 263, 890, 1235)
    assert main.iou(without_helmet, with_helmet) > 0.9


def test_iou_real_two_different_riders_case():
    # test.jpg: two different riders side by side, one helmeted one not --
    # these boxes must NOT be treated as a contradiction.
    without_helmet = (884, 102, 1401, 1041)
    with_helmet = (63, 75, 571, 1046)
    assert main.iou(without_helmet, with_helmet) == 0.0


def test_is_contradicted_true_for_overlapping_boxes():
    without_helmet = (240, 274, 880, 1242)
    with_helmet = (229, 263, 890, 1235)
    assert main.is_contradicted(without_helmet, [with_helmet]) is True


def test_is_contradicted_false_when_no_helmet_boxes_present():
    without_helmet = (240, 274, 880, 1242)
    assert main.is_contradicted(without_helmet, []) is False


def test_is_contradicted_false_for_distant_boxes():
    without_helmet = (884, 102, 1401, 1041)
    with_helmet = (63, 75, 571, 1046)
    assert main.is_contradicted(without_helmet, [with_helmet]) is False


def test_nearest_plate_id_returns_none_when_no_plates_tracked():
    assert main.nearest_plate_id((0, 0, 10, 10), {}) is None


def test_nearest_plate_id_picks_the_closer_plate():
    tracked_plates = {
        0: (0, 0, 10, 10),
        1: (1000, 1000, 1010, 1010),
    }
    assert main.nearest_plate_id((5, 5, 15, 15), tracked_plates) == 0
    assert main.nearest_plate_id((995, 995, 1005, 1005), tracked_plates) == 1


def test_import_main_is_dependency_light():
    # Importing the CLI module must not pull in torch/ultralytics/easyocr (they
    # load lazily inside main()), so tests/CI stay fast and model-free. Checked
    # in a clean subprocess so other tests in the suite can't pollute the result.
    import subprocess
    import sys
    code = (
        "import main, sys; "
        "bad=[m for m in ('torch','ultralytics','easyocr') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"import main pulled in: {out.stdout.strip()}"


def test_parse_args_defaults_and_flags():
    args = main.parse_args(["--source", "clip.mp4", "--no-report", "--max-frames", "50"])
    assert args.source == "clip.mp4" and args.no_report is True and args.max_frames == 50


def test_local_recorder_returns_fine_amount_without_network():
    # report=False must never touch the network; it returns the fine amount.
    rec = main._make_recorder("http://unused", None, report=False)
    assert rec("MH12AB1234", "no_helmet", "evidence/x.jpg") == 500
    assert rec("MH12AB1234", "triple_riding", None) == 1000
