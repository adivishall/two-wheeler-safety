"""Temporally-stable plate recognition.

A single OCR frame is unreliable: characters flicker (``0``/``O``, ``1``/``I``,
``8``/``B``, ``5``/``S``), whole readings drop out, and a bad frame can invent a
plate that never existed. The old pipeline stored the *last* frame's raw
reading and could fine from it.

Two reusable pieces fix that:

* :func:`correct_plate` — context-aware look-alike correction. It does **not**
  blindly swap characters: it only proposes a fix that makes the string match a
  real Indian plate *structure* (standard or BH), coercing each character toward
  the class its position requires (state = letters, RTO = digits, series =
  letters, number = digits), and only if the edit distance is within a bound.

* :class:`PlateStabilizer` — collects per-vehicle OCR observations across
  frames, scores each by OCR confidence and structural validity, and elects a
  stable plate by weighted temporal voting. It refuses to report a plate until
  it has seen enough observations, so a fine is never issued from one frame.

All thresholds live in :class:`PlateConfig`; every function is pure/deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.plate_info import (
    decode_plate,
    matches_structure,
    normalize_plate,
)

# Look-alikes, split by the direction of the fix. Conservative on purpose:
# only genuinely confusable pairs, and only applied to push a character toward
# the class its plate position requires.
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
             "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}


@dataclass(frozen=True)
class PlateConfig:
    min_observations: int = 2  # never elect a plate from a single frame
    min_confidence: float = 0.35  # agreement score the winner must clear
    max_edits: int = 2  # cap on look-alike substitutions per correction (conservative:
    #                     an all-digit junk string needs >=3 to look like a plate)
    enable_correction: bool = True
    invalid_weight: float = 0.25  # weight multiplier for a structurally-invalid reading
    require_known_state: bool = False  # if True, the stable plate must decode()


DEFAULT_PLATE_CONFIG = PlateConfig()


def _coerce_letter(ch: str) -> str | None:
    if ch.isalpha():
        return ch
    return _TO_LETTER.get(ch)


def _coerce_digit(ch: str) -> str | None:
    if ch.isdigit():
        return ch
    return _TO_DIGIT.get(ch)


def _coerce_segment(seg: str, kind: str) -> tuple[str, int] | None:
    """Coerce every char in ``seg`` to ``kind`` ('letter'/'digit'); return the
    coerced segment and the number of characters changed, or None if a char
    can't be coerced."""
    out = []
    edits = 0
    coerce = _coerce_letter if kind == "letter" else _coerce_digit
    for ch in seg:
        fixed = coerce(ch)
        if fixed is None:
            return None
        if fixed != ch:
            edits += 1
        out.append(fixed)
    return "".join(out), edits


def _coerce_standard(s: str) -> tuple[str, int] | None:
    """Try every plausible standard segmentation
    (state=2, rto=1-2, series=1-3, number=1-4) and return the coerced plate
    with the fewest edits, or None."""
    n = len(s)
    # Each candidate scored by (edits, prefer rto_len=2, prefer longer series,
    # then the string itself) so ties resolve deterministically.
    best_score: tuple[int, int, int, str] | None = None
    best_candidate: tuple[str, int] | None = None
    for rto_len in (1, 2):
        for series_len in (1, 2, 3):
            number_len = n - 2 - rto_len - series_len
            if not 1 <= number_len <= 4:
                continue
            i = 2
            state = _coerce_segment(s[0:2], "letter")
            rto = _coerce_segment(s[i:i + rto_len], "digit")
            i += rto_len
            series = _coerce_segment(s[i:i + series_len], "letter")
            i += series_len
            number = _coerce_segment(s[i:i + number_len], "digit")
            if not (state and rto and series and number):
                continue
            candidate = state[0] + rto[0] + series[0] + number[0]
            edits = state[1] + rto[1] + series[1] + number[1]
            score = (edits, -rto_len, -series_len, candidate)
            if best_score is None or score < best_score:
                best_score = score
                best_candidate = (candidate, edits)
    return best_candidate


def _coerce_bh(s: str) -> tuple[str, int] | None:
    """BH series: 2 digits, 'BH', 4 digits, 1-2 letters."""
    n = len(s)
    for tail_len in (1, 2):
        if 4 + tail_len + 4 != n:  # 2 + 2(BH) + 4 + tail
            continue
        year = _coerce_segment(s[0:2], "digit")
        bh = _coerce_segment(s[2:4], "letter")
        number = _coerce_segment(s[4:8], "digit")
        tail = _coerce_segment(s[8:8 + tail_len], "letter")
        if not (year and bh and number and tail) or bh[0] != "BH":
            continue
        candidate = year[0] + "BH" + number[0] + tail[0]
        edits = year[1] + bh[1] + number[1] + tail[1]
        return candidate, edits
    return None


def correct_plate(
    text: str, config: PlateConfig = DEFAULT_PLATE_CONFIG
) -> tuple[str | None, int]:
    """Return ``(corrected_plate, edits)`` if a small, structure-valid
    correction exists, else ``(None, 0)``.

    A string that is already well-formed returns itself with 0 edits.
    """
    norm = normalize_plate(text)
    if not norm:
        return None, 0
    if matches_structure(norm):
        return norm, 0
    if not config.enable_correction:
        return None, 0

    best: tuple[str, int] | None = None
    for coerced in (_coerce_standard(norm), _coerce_bh(norm)):
        if coerced and coerced[1] <= config.max_edits and (
            best is None or coerced[1] < best[1]
        ):
            best = coerced
    if best is None:
        return None, 0
    return best


@dataclass
class Observation:
    raw: str
    normalized: str
    used: str  # corrected plate if a valid correction was found, else normalized
    conf: float
    valid: bool
    edits: int


@dataclass
class PlateResult:
    raw: str | None
    normalized: str | None
    corrected: str | None
    stable: str | None
    confidence: float
    num_observations: int
    disagreement_count: int
    valid: bool


class PlateStabilizer:
    """Accumulate OCR observations for one vehicle and elect a stable plate."""

    def __init__(self, config: PlateConfig = DEFAULT_PLATE_CONFIG):
        self.config = config
        self.observations: list[Observation] = []

    def add(self, raw_text: str | None, conf: float = 1.0) -> None:
        norm = normalize_plate(raw_text or "")
        if not norm:
            return
        corrected, edits = correct_plate(norm, self.config)
        if corrected is not None:
            used, valid = corrected, True
        else:
            used, valid = norm, False
            edits = 0
        self.observations.append(
            Observation(raw=raw_text or "", normalized=norm, used=used,
                        conf=float(conf), valid=valid, edits=edits)
        )

    # -- aggregation --------------------------------------------------------

    def _tally(self) -> tuple[dict[str, float], float]:
        weights: dict[str, float] = {}
        total = 0.0
        for o in self.observations:
            w = o.conf * (1.0 if o.valid else self.config.invalid_weight)
            weights[o.used] = weights.get(o.used, 0.0) + w
            total += w
        return weights, total

    def result(self) -> PlateResult:
        last = self.observations[-1] if self.observations else None
        num = len(self.observations)

        stable: str | None = None
        confidence = 0.0

        valid_obs = [o for o in self.observations if o.valid]
        if valid_obs and num >= self.config.min_observations:
            weights, total = self._tally()
            # Only structurally-valid readings are eligible to win.
            candidates = {o.used for o in valid_obs}
            winner = max(
                candidates, key=lambda p: (weights.get(p, 0.0), p)
            )
            confidence = weights.get(winner, 0.0) / total if total else 0.0
            known_ok = (
                not self.config.require_known_state
                or decode_plate(winner).get("recognized", False)
            )
            if confidence >= self.config.min_confidence and known_ok:
                stable = winner

        disagreement = (
            sum(1 for o in self.observations if o.used != stable) if stable else 0
        )
        return PlateResult(
            raw=last.raw if last else None,
            normalized=last.normalized if last else None,
            corrected=stable,
            stable=stable,
            confidence=round(confidence, 4),
            num_observations=num,
            disagreement_count=disagreement,
            valid=stable is not None,
        )

    # convenience passthroughs
    @property
    def stable_plate(self) -> str | None:
        return self.result().stable

    @property
    def confidence(self) -> float:
        return self.result().confidence

    @property
    def num_observations(self) -> int:
        return len(self.observations)

    @property
    def disagreement_count(self) -> int:
        return self.result().disagreement_count
