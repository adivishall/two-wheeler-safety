"""The plate-identity safeguards of the decision core (model-free).

Both cases were found by an independent review of the shared pipeline:
the OCR lock froze a track's plate, so a track that switched to another vehicle
fined the first vehicle's plate; and a confirmed violation whose plate only
stabilised on frames without a rider box was never fined.
"""

from modules.association import DetBox
from modules.pipeline import PipelineConfig, ViolationPipeline


def _frame(x, label, *, rider=True, plate=True):
    dets = []
    if rider:
        dets.append(DetBox(label, (x, 100, x + 60, 200), 0.9))
    if plate:
        dets.append(DetBox("Plate", (x + 10, 205, x + 50, 235), 0.9))
    return dets


def _run(frames, texts, cfg=None):
    """texts[i] is what OCR returns on frame i (None = unreadable)."""
    pipe = ViolationPipeline(cfg or PipelineConfig())
    fines = []
    for i, dets in enumerate(frames):
        res = pipe.step(i, dets, timestamp=i / 25.0,
                        read_plate=lambda box, t=texts[i]: (t, 0.9) if t else None)
        fines += [(d.plate, d.violation) for d in res.decisions]
    return pipe, fines


def test_identity_switch_under_the_ocr_lock_does_not_fine_the_first_plate():
    # Frames 0-9: helmeted rider A (plate locks on MH12AB1234). Frames 10-59:
    # the SAME track continues on bare-headed rider B, plate KA05MN6789.
    frames = [_frame(100 + 2 * i, "WithHelmet" if i < 10 else "WithoutHelmet")
              for i in range(60)]
    texts = ["MH12AB1234"] * 10 + ["KA05MN6789"] * 50
    pipe, fines = _run(frames, texts)
    assert ("MH12AB1234", "no_helmet") not in fines
    assert fines == [("KA05MN6789", "no_helmet")]
    assert len(pipe.stabilizers) == 1  # it really was one track throughout


def test_locked_plate_is_rechecked_not_frozen():
    frames = [_frame(100 + 2 * i, "WithHelmet") for i in range(40)]
    texts = ["MH12AB1234"] * 40
    pipe, _ = _run(frames, texts)
    # 5 reads to lock, then a re-check every 10 frames instead of never.
    assert pipe.ocr_calls == 5 + 3


def test_violation_fined_when_plate_stabilises_only_without_a_rider_box():
    # Rider (no helmet) + unreadable plate on frames 0-19, then only the plate
    # is detected (rider box missed) and it reads cleanly on frames 20-39.
    frames = [_frame(100 + 2 * i, "WithoutHelmet", rider=i < 20) for i in range(40)]
    texts = [None] * 20 + ["MH12AB1234"] * 20
    pipe, fines = _run(frames, texts)
    assert fines == [("MH12AB1234", "no_helmet")]
    assert pipe.unfined_confirmations() == []


def test_stale_plate_is_not_used_for_a_late_confirmation():
    # The plate reads cleanly on frames 0-4 only (afterwards it is detected but
    # unreadable). The rider is helmeted until frame 29 and bare-headed from 30,
    # so the violation confirms at frame 34 — 30 frames after the last reading
    # that agreed with the elected plate. That plate is too old to vouch for
    # the vehicle now on the track, so the fine is withheld, with that reason.
    cfg = PipelineConfig(helmet_min_observed=0, helmet_min_fraction=0.0)
    frames = [_frame(100 + 2 * i, "WithHelmet" if i < 30 else "WithoutHelmet")
              for i in range(60)]
    texts = ["MH12AB1234"] * 5 + [None] * 55
    pipe, fines = _run(frames, texts, cfg)
    assert fines == []
    assert [h["reason"] for h in pipe.unfined_confirmations()] == \
        ["plate_not_recently_confirmed"]
