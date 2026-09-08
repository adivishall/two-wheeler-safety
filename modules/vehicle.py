"""First-class tracked-vehicle state.

A :class:`VehicleTrack` is one physical two-wheeler followed across frames. It
replaces the old design where a *plate* box was the only tracked entity and
violations were attached to it after the fact. The track owns its identity
(stable ``track_id``), its motion, its associated plate box, and its current
violation signals.

Several fields are placeholders that later phases populate (temporal plate
voting, helmet/triple/overspeed state machines). They default to neutral values
so the object is meaningful from Phase 1 onward without those phases.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from modules.association import VehicleBody
from modules.geometry import Box, centroid


class TrackState(enum.Enum):
    """Lifecycle of a track.

    TENTATIVE: newly created, not yet seen on enough frames to trust.
    CONFIRMED: seen on ``n_init`` frames -- a real, stable vehicle.
    LOST:      currently unmatched (occluded / missed detection) but retained.
    REMOVED:   unmatched past ``max_age``; scheduled for deletion.
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class VehicleTrack:
    track_id: int
    box: Box  # anchor box: union of the body and plate boxes seen this frame
    first_seen_frame: int
    last_seen_frame: int

    velocity: tuple[float, float] = (0.0, 0.0)
    hits: int = 1  # frames matched
    age: int = 1  # frames since creation
    time_since_update: int = 0  # frames since last match
    state: TrackState = TrackState.TENTATIVE

    # Association outputs for the current frame.
    plate_box: Box | None = None
    body: VehicleBody | None = None

    # --- filled by later phases (kept here so the shape is stable) ---
    plate_observations: list = field(default_factory=list)  # Phase 2
    stable_plate: str | None = None  # Phase 2
    plate_confidence: float = 0.0  # Phase 2/3
    helmet_state: str = "unknown"  # Phase 4
    triple_state: str = "none"  # Phase 5
    overspeed_state: str = "none"  # Phase 6
    violation_history: list = field(default_factory=list)
    evidence_frames: dict = field(default_factory=dict)

    @property
    def centroid(self) -> tuple[float, float]:
        return centroid(self.box)

    @property
    def confirmed(self) -> bool:
        return self.state is TrackState.CONFIRMED
