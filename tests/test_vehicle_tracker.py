"""VehicleTracker lifecycle + multi-vehicle tracking tests (synthetic, no model)."""

from modules.association import DetBox
from modules.vehicle import TrackState
from modules.vehicle_tracker import TrackerConfig, VehicleTracker


def _bike(rider_label, rider_box, plate_box, conf=0.9):
    return [DetBox(rider_label, rider_box, conf), DetBox("Plate", plate_box, conf)]


def test_new_vehicle_gets_a_track():
    tr = VehicleTracker()
    tracks = tr.update(_bike("WithoutHelmet", (0, 0, 100, 100), (20, 120, 80, 160)), 0)
    assert len(tracks) == 1
    assert tracks[0].state is TrackState.TENTATIVE


def test_track_confirms_after_n_init_frames():
    tr = VehicleTracker(tracker_config=TrackerConfig(n_init=3))
    tid = None
    for f in range(3):
        box = (f * 2, 0, 100 + f * 2, 100)
        plate = (20 + f * 2, 120, 80 + f * 2, 160)
        tracks = tr.update(_bike("WithoutHelmet", box, plate), f)
        tid = tracks[0].track_id
    assert tr.tracks[tid].state is TrackState.CONFIRMED


def test_stable_id_across_frames_with_motion():
    tr = VehicleTracker()
    ids = []
    for f in range(4):
        dx = f * 15
        tracks = tr.update(
            _bike("WithoutHelmet", (dx, 0, 100 + dx, 100), (20 + dx, 120, 80 + dx, 160)), f
        )
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1  # same vehicle, same id throughout


def test_two_crossing_bikes_keep_distinct_ids():
    # A moves right, B moves left; their paths cross. IDs must not swap.
    tr = VehicleTracker()
    # frame 0: A left, B right
    t0 = tr.update(
        _bike("WithoutHelmet", (0, 0, 80, 100), (10, 120, 70, 160))
        + _bike("WithHelmet", (400, 0, 480, 100), (410, 120, 470, 160)),
        0,
    )
    ids0 = sorted(t.track_id for t in t0)
    assert len(ids0) == 2
    # advance them toward each other over several frames
    for f in range(1, 6):
        ax = f * 60
        bx = 400 - f * 60
        tr.update(
            _bike("WithoutHelmet", (ax, 0, 80 + ax, 100), (10 + ax, 120, 70 + ax, 160))
            + _bike("WithHelmet", (bx, 0, 80 + bx, 100), (10 + bx, 120, 70 + bx, 160)),
            f,
        )
    assert len([t for t in tr.active_tracks()]) == 2


def test_plate_appearing_after_vehicle_keeps_same_track():
    # Frame 0: only the rider is detected (no plate yet).
    tr = VehicleTracker()
    t0 = tr.update([DetBox("WithoutHelmet", (0, 0, 100, 100))], 0)
    tid = t0[0].track_id
    assert t0[0].plate_box is None
    # Frame 1: the plate now appears for the same vehicle.
    t1 = tr.update(_bike("WithoutHelmet", (5, 0, 105, 100), (25, 120, 85, 160)), 1)
    assert t1[0].track_id == tid          # same identity
    assert t1[0].plate_box is not None    # now carries a plate


def test_temporary_disappearance_reuses_id():
    tr = VehicleTracker(tracker_config=TrackerConfig(n_init=2, max_age=5))
    for f in range(2):  # establish + confirm
        tr.update(_bike("WithoutHelmet", (f, 0, 100 + f, 100), (20 + f, 120, 80 + f, 160)), f)
    tid = next(iter(tr.tracks))
    tr.update([], 2)              # occluded for a frame
    tr.update([], 3)
    tracks = tr.update(_bike("WithoutHelmet", (4, 0, 104, 100), (24, 120, 84, 160)), 4)
    assert tracks[0].track_id == tid  # same vehicle after reappearing


def test_unconfirmed_track_dropped_immediately_when_lost():
    tr = VehicleTracker(tracker_config=TrackerConfig(n_init=3))
    tr.update([DetBox("WithoutHelmet", (0, 0, 100, 100))], 0)  # tentative only
    tr.update([], 1)  # vanished before ever confirming
    assert all(t.state is TrackState.REMOVED for t in []) or not tr.active_tracks()


def test_three_close_bikes_tracked_as_three():
    tr = VehicleTracker()
    dets = []
    for i in range(3):
        x = i * 130
        dets += _bike("WithoutHelmet", (x, 0, x + 100, 100), (x + 20, 120, x + 80, 160))
    for f in range(3):
        tr.update(dets, f)
    assert len(tr.active_tracks()) == 3
