from utils.tracker import CentroidTracker


def test_new_boxes_get_registered():
    tracker = CentroidTracker()
    result = tracker.update([(0, 0, 50, 50), (200, 200, 250, 250)])
    assert len(result) == 2


def test_ids_persist_across_frames_when_boxes_move_slightly():
    tracker = CentroidTracker()
    frame1 = tracker.update([(0, 0, 50, 50), (200, 200, 250, 250)])
    frame2 = tracker.update([(5, 5, 55, 55), (205, 205, 255, 255)])
    assert set(frame2.keys()) == set(frame1.keys())


def test_disappeared_objects_are_not_returned():
    tracker = CentroidTracker()
    tracker.update([(0, 0, 50, 50), (200, 200, 250, 250)])
    # a third, unrelated box appears far away; the first two should not
    # be returned just because they're still remembered internally
    result = tracker.update([(600, 600, 650, 650)])
    assert len(result) == 1


def test_empty_frame_returns_no_objects():
    tracker = CentroidTracker()
    tracker.update([(0, 0, 50, 50)])
    result = tracker.update([])
    assert result == {}


def test_reappearing_object_keeps_its_original_id():
    tracker = CentroidTracker()
    frame1 = tracker.update([(0, 0, 50, 50)])
    original_id = next(iter(frame1))

    tracker.update([])  # object briefly not detected

    frame3 = tracker.update([(3, 3, 53, 53)])
    assert original_id in frame3


def test_object_is_dropped_after_max_disappeared_frames():
    tracker = CentroidTracker(max_disappeared=2)
    frame1 = tracker.update([(0, 0, 50, 50)])
    original_id = next(iter(frame1))

    tracker.update([])
    tracker.update([])
    tracker.update([])  # 3 misses > max_disappeared of 2

    frame_after = tracker.update([(3, 3, 53, 53)])
    assert original_id not in frame_after
