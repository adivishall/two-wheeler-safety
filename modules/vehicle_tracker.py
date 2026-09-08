"""Multi-object vehicle tracker.

Each frame:

1. :func:`~modules.association.associate` groups the raw detections into
   per-vehicle instances (body + plate, one-to-one).
2. Each instance gets an *anchor* box (the union of its body and plate boxes).
3. Existing tracks are motion-predicted (last box shifted by velocity) and
   matched to this frame's anchors with a gated Hungarian assignment on a
   ``(1 - IoU) + normalised-centroid-distance`` cost -- a global one-to-one
   match, not per-object nearest-neighbour.
4. Matched tracks are updated (box, velocity via EMA, hit count, lifecycle);
   unmatched tracks age and are eventually removed; unmatched anchors spawn new
   tentative tracks.

Stable IDs come from a monotonic counter; all iteration is over sorted keys so
the output is deterministic. This is what the video pipeline uses instead of
``CentroidTracker`` + ``nearest_plate_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.association import (
    DEFAULT_CONFIG,
    AssociationConfig,
    DetBox,
    associate,
)
from modules.geometry import Box, centroid, centroid_distance, diagonal, iou, union_box
from modules.vehicle import TrackState, VehicleTrack

_BIG = 1e9


@dataclass(frozen=True)
class TrackerConfig:
    max_age: int = 15  # frames a track survives unmatched before removal
    n_init: int = 3  # matched frames before a track is CONFIRMED
    w_iou: float = 1.0  # cost weight on (1 - IoU)
    w_dist: float = 0.5  # cost weight on normalised centroid distance
    gate_iou: float = 0.0  # pairs with IoU <= this AND far apart are forbidden
    gate_distance_diag: float = 1.5  # "far apart" = > this * anchor diagonal
    velocity_alpha: float = 0.5  # EMA smoothing for velocity


DEFAULT_TRACKER_CONFIG = TrackerConfig()


def _anchor_box(body_box: Box | None, plate_box: Box | None) -> Box:
    if body_box is not None and plate_box is not None:
        return union_box(body_box, plate_box)
    return body_box if body_box is not None else plate_box  # one is non-None


class VehicleTracker:
    def __init__(
        self,
        config: AssociationConfig = DEFAULT_CONFIG,
        tracker_config: TrackerConfig = DEFAULT_TRACKER_CONFIG,
    ):
        self.config = config
        self.tc = tracker_config
        self._next_id = 0
        self.tracks: dict[int, VehicleTrack] = {}

    # -- public API ---------------------------------------------------------

    def update(self, dets: list[DetBox], frame_idx: int) -> list[VehicleTrack]:
        """Advance the tracker by one frame; return tracks matched this frame."""
        result = associate(dets, self.config)

        # Build this frame's vehicle instances as (anchor, body, plate_box).
        instances = []
        for body_i, plate_i in result.pairs:
            body = result.bodies[body_i] if body_i is not None else None
            body_box = body.box if body is not None else None
            plate_box = result.plates[plate_i] if plate_i is not None else None
            instances.append((_anchor_box(body_box, plate_box), body, plate_box))

        track_ids = sorted(self.tracks)
        matches, unmatched_tracks, unmatched_instances = self._match(
            track_ids, [inst[0] for inst in instances]
        )

        for tid, inst_idx in matches:
            self._update_track(self.tracks[tid], instances[inst_idx], frame_idx)

        for tid in unmatched_tracks:
            self._mark_missed(self.tracks[tid])

        for inst_idx in unmatched_instances:
            self._create_track(instances[inst_idx], frame_idx)

        for tid in [t for t, tr in self.tracks.items() if tr.state is TrackState.REMOVED]:
            del self.tracks[tid]

        # Everything seen this frame: matched tracks (reset to 0) and tracks
        # newly created this frame (also 0). Sorted for deterministic output.
        return [
            self.tracks[tid]
            for tid in sorted(self.tracks)
            if self.tracks[tid].time_since_update == 0
            and self.tracks[tid].last_seen_frame == frame_idx
        ]

    def active_tracks(self) -> list[VehicleTrack]:
        return [t for t in self.tracks.values() if t.state is not TrackState.REMOVED]

    # -- matching -----------------------------------------------------------

    def _match(self, track_ids: list[int], anchors: list[Box]):
        if not track_ids or not anchors:
            return [], list(track_ids), list(range(len(anchors)))

        cost = [[0.0] * len(anchors) for _ in track_ids]
        for r, tid in enumerate(track_ids):
            pred = self._predicted_box(self.tracks[tid])
            for c, anchor in enumerate(anchors):
                overlap = iou(pred, anchor)
                diag = diagonal(anchor) or 1.0
                dist = centroid_distance(pred, anchor) / diag
                gated = overlap <= self.tc.gate_iou and dist > self.tc.gate_distance_diag
                cost[r][c] = (
                    _BIG
                    if gated
                    else self.tc.w_iou * (1.0 - overlap) + self.tc.w_dist * dist
                )

        # Local import keeps the assignment routine in one place.
        from modules.association import gated_min_cost_matching

        pairs = gated_min_cost_matching(cost, gate=_BIG)
        matched_rows = {r for r, _ in pairs}
        matched_cols = {c for _, c in pairs}
        matches = [(track_ids[r], c) for r, c in pairs]
        unmatched_tracks = [track_ids[r] for r in range(len(track_ids)) if r not in matched_rows]
        unmatched_instances = [c for c in range(len(anchors)) if c not in matched_cols]
        return matches, unmatched_tracks, unmatched_instances

    def _predicted_box(self, track: VehicleTrack) -> Box:
        vx, vy = track.velocity
        x1, y1, x2, y2 = track.box
        return (
            round(x1 + vx),
            round(y1 + vy),
            round(x2 + vx),
            round(y2 + vy),
        )

    # -- lifecycle ----------------------------------------------------------

    def _create_track(self, instance, frame_idx: int) -> None:
        anchor, body, plate_box = instance
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = VehicleTrack(
            track_id=tid,
            box=anchor,
            first_seen_frame=frame_idx,
            last_seen_frame=frame_idx,
            plate_box=plate_box,
            body=body,
            state=TrackState.TENTATIVE,
        )

    def _update_track(self, track: VehicleTrack, instance, frame_idx: int) -> None:
        anchor, body, plate_box = instance
        old_cx, old_cy = centroid(track.box)
        new_cx, new_cy = centroid(anchor)
        a = self.tc.velocity_alpha
        track.velocity = (
            a * (new_cx - old_cx) + (1 - a) * track.velocity[0],
            a * (new_cy - old_cy) + (1 - a) * track.velocity[1],
        )
        track.box = anchor
        track.body = body
        if plate_box is not None:
            track.plate_box = plate_box
        track.hits += 1
        track.age += 1
        track.time_since_update = 0
        track.last_seen_frame = frame_idx
        newly_confirmed = (
            track.state is TrackState.TENTATIVE and track.hits >= self.tc.n_init
        )
        if newly_confirmed or track.state is TrackState.LOST:
            track.state = TrackState.CONFIRMED

    def _mark_missed(self, track: VehicleTrack) -> None:
        track.age += 1
        track.time_since_update += 1
        if track.state is TrackState.TENTATIVE:
            # A track that never confirmed and vanished is noise -> drop fast.
            track.state = TrackState.REMOVED
        elif track.time_since_update > self.tc.max_age:
            track.state = TrackState.REMOVED
        else:
            track.state = TrackState.LOST
