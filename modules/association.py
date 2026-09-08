"""Vehicle-level association.

The trained model emits four independent box classes -- ``Plate``,
``WithHelmet``, ``WithoutHelmet``, ``TripleRiding`` -- with no notion of which
boxes belong to the same physical two-wheeler. The old pipeline attributed a
violation box to the *nearest tracked plate* (``nearest_plate_id``), which
mis-assigns whenever two bikes are close: "nearest" is a purely local, greedy
signal and one plate can end up owning two riders.

This module replaces that with a two-step, globally consistent association:

1. **Merge body boxes into vehicle bodies** (`merge_bodies`): rider/violation
   boxes (helmet, no-helmet, triple-riding) that strongly overlap or contain
   one another are the same physical vehicle, so they're grouped by
   connected-components on IoU/containment. This also folds the
   helmet-contradiction case (WithHelmet + WithoutHelmet on one rider) into a
   single body flagged ``ambiguous_helmet``.

2. **Assign plates to bodies one-to-one** (`associate`): a cost matrix combines
   centroid distance, horizontal-span overlap (a plate sits under *its* rider),
   and vertical ordering (the plate is lower than the rider), then the
   **Hungarian algorithm** finds the globally minimum-cost one-to-one matching.
   Pairs above a gate are rejected, so a plate with no plausible body stays
   unmatched rather than being force-assigned. One-to-one is guaranteed, so a
   single violation can never be attributed to two vehicles.

All weights/gates live in :class:`AssociationConfig` and every function is pure
and deterministic (stable tie-breaking by input order), so the behaviour is
unit-testable with synthetic boxes and no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modules.geometry import (
    Box,
    centroid,
    centroid_distance,
    containment,
    diagonal,
    horizontal_overlap_ratio,
    iou,
    union_box,
)

_BIG = 1e9  # stand-in for "forbidden pair" inside the cost matrix


# ---------------------------------------------------------------------------
# Hungarian (Kuhn-Munkres) minimum-cost assignment
# ---------------------------------------------------------------------------

def hungarian(cost: list[list[float]]) -> list[int]:
    """Minimum-cost one-to-one assignment.

    Args:
        cost: an ``n x m`` matrix of non-negative costs.

    Returns:
        ``assign`` of length ``n`` where ``assign[i]`` is the column matched to
        row ``i``, or ``-1`` if that row is unmatched (only when ``n > m``).

    This is the classic O(n^2 m) potentials/augmenting-path formulation. It
    requires ``rows <= cols``; when there are more rows than columns we solve
    the transpose and invert the mapping, so callers needn't care.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if m == 0:
        return [-1] * n

    if n > m:  # solve the transpose so rows <= cols, then invert
        transposed = [[cost[i][j] for i in range(n)] for j in range(m)]
        col_for_row_t = hungarian(transposed)  # length m, values in [0, n)
        assign = [-1] * n
        for tcol, trow in enumerate(col_for_row_t):
            if trow != -1:
                assign[trow] = tcol
        return assign

    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)  # p[j] = row (1-indexed) assigned to column j
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assign = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assign[p[j] - 1] = j - 1
    return assign


def gated_min_cost_matching(
    cost: list[list[float]], gate: float
) -> list[tuple[int, int]]:
    """One-to-one matching via :func:`hungarian`, dropping any pair whose cost
    is ``>= gate``. Returns ``(row, col)`` pairs sorted by row."""
    if not cost or not cost[0]:
        return []
    assign = hungarian(cost)
    pairs = []
    for r, c in enumerate(assign):
        if c != -1 and cost[r][c] < gate:
            pairs.append((r, c))
    return pairs


# ---------------------------------------------------------------------------
# Vehicle bodies
# ---------------------------------------------------------------------------

@dataclass
class DetBox:
    """One raw detection: a class label, box, and model confidence."""

    label: str
    box: Box
    conf: float = 1.0


@dataclass
class VehicleBody:
    """A physical two-wheeler's rider region, merged from overlapping body
    boxes, with the violation signals present on it this frame."""

    box: Box
    has_helmet: bool = False
    has_no_helmet: bool = False
    has_triple: bool = False
    no_helmet_conf: float = 0.0
    triple_conf: float = 0.0
    member_labels: list[str] = field(default_factory=list)

    @property
    def ambiguous_helmet(self) -> bool:
        """Model contradicted itself (helmet *and* no-helmet on one rider) --
        the same safeguard the old ``is_contradicted`` gave, at vehicle scope."""
        return self.has_helmet and self.has_no_helmet

    @property
    def no_helmet_violation(self) -> bool:
        """No-helmet only when it isn't contradicted by a helmet box here."""
        return self.has_no_helmet and not self.has_helmet


@dataclass(frozen=True)
class AssociationConfig:
    """Tunable weights/gates for association (all documented, all overridable)."""

    # Body merging: two body boxes are the same vehicle if their IoU exceeds
    # this, or one is largely contained in the other (helmet inside a triple).
    body_merge_iou: float = 0.3
    body_merge_containment: float = 0.6

    # Plate<->body cost = w_dist * (dist / body_diag)
    #                   + w_hoverlap * (1 - horizontal_overlap_ratio)
    #                   + w_vertical * vertical_penalty
    w_dist: float = 1.0
    w_hoverlap: float = 1.5
    w_vertical: float = 0.5

    # Reject a plate<->body pair when it has no horizontal overlap AND its
    # centroids are farther apart than this multiple of the body's diagonal.
    gate_distance_diag: float = 1.2

    _BODY_LABELS = ("WithHelmet", "WithoutHelmet", "TripleRiding")


DEFAULT_CONFIG = AssociationConfig()


def _union_find(n: int) -> list[int]:
    return list(range(n))


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)  # deterministic: keep lower root


def merge_bodies(
    dets: list[DetBox], config: AssociationConfig = DEFAULT_CONFIG
) -> list[VehicleBody]:
    """Group rider/violation boxes into one :class:`VehicleBody` per vehicle.

    Boxes are connected when IoU > ``body_merge_iou`` or one is contained in
    the other by > ``body_merge_containment`` (folds a helmet inside a
    triple-riding body, or a contradicting helmet/no-helmet pair, into one).
    """
    body_dets = [d for d in dets if d.label in AssociationConfig._BODY_LABELS]
    n = len(body_dets)
    parent = _union_find(n)

    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = body_dets[i].box, body_dets[j].box
            if (
                iou(bi, bj) > config.body_merge_iou
                or containment(bi, bj) > config.body_merge_containment
                or containment(bj, bi) > config.body_merge_containment
            ):
                _union(parent, i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(_find(parent, i), []).append(i)

    bodies = []
    for root in sorted(groups):  # deterministic order
        members = groups[root]
        box = body_dets[members[0]].box
        body = VehicleBody(box=box)
        for idx in members:
            d = body_dets[idx]
            box = union_box(box, d.box)
            body.member_labels.append(d.label)
            if d.label == "WithHelmet":
                body.has_helmet = True
            elif d.label == "WithoutHelmet":
                body.has_no_helmet = True
                body.no_helmet_conf = max(body.no_helmet_conf, d.conf)
            elif d.label == "TripleRiding":
                body.has_triple = True
                body.triple_conf = max(body.triple_conf, d.conf)
        body.box = box
        bodies.append(body)
    return bodies


def _plate_body_cost(
    plate: Box, body: Box, config: AssociationConfig
) -> tuple[float, bool]:
    """Return (cost, gated). ``gated`` is True when the pair is forbidden."""
    body_diag = diagonal(body) or 1.0
    dist = centroid_distance(plate, body) / body_diag
    hoverlap = horizontal_overlap_ratio(plate, body)

    # Plate should sit lower than (or level with) the rider. Penalise a plate
    # whose centre is *above* the body's centre, scaled by body height.
    _, py = centroid(plate)
    _, by = centroid(body)
    body_h = max(1.0, body[3] - body[1])
    vertical_penalty = max(0.0, (by - py)) / body_h

    cost = (
        config.w_dist * dist
        + config.w_hoverlap * (1.0 - hoverlap)
        + config.w_vertical * vertical_penalty
    )

    gated = hoverlap == 0.0 and dist > config.gate_distance_diag
    return cost, gated


@dataclass
class AssociationResult:
    """Vehicle instances for one frame after association."""

    # (body_index_or_None, plate_index_or_None) per vehicle instance.
    pairs: list[tuple[int | None, int | None]]
    bodies: list[VehicleBody]
    plates: list[Box]


def associate(
    dets: list[DetBox], config: AssociationConfig = DEFAULT_CONFIG
) -> AssociationResult:
    """Group detections into per-vehicle instances for one frame.

    Every body gets at most one plate and every plate at most one body (a
    globally minimum-cost one-to-one matching). Bodies with no plausible plate,
    and plates with no plausible body, still become vehicle instances (a plate
    alone carries identity but no violation; a body alone carries a violation
    that can't yet be fined without a plate).
    """
    bodies = merge_bodies(dets, config)
    plates = [d.box for d in dets if d.label == "Plate"]

    pairs: list[tuple[int | None, int | None]] = []

    if bodies and plates:
        cost = [[0.0] * len(plates) for _ in bodies]
        for bi, body in enumerate(bodies):
            for pi, plate in enumerate(plates):
                c, gated = _plate_body_cost(plate, body.box, config)
                cost[bi][pi] = _BIG if gated else c
        matched = gated_min_cost_matching(cost, gate=_BIG)
        matched_bodies = {b for b, _ in matched}
        matched_plates = {p for _, p in matched}
        for bi, pi in matched:
            pairs.append((bi, pi))
        for bi in range(len(bodies)):
            if bi not in matched_bodies:
                pairs.append((bi, None))
        for pi in range(len(plates)):
            if pi not in matched_plates:
                pairs.append((None, pi))
    else:
        pairs.extend((bi, None) for bi in range(len(bodies)))
        pairs.extend((None, pi) for pi in range(len(plates)))

    return AssociationResult(pairs=pairs, bodies=bodies, plates=plates)
