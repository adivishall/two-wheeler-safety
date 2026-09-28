"""Input validation and lightweight security helpers for the web app.

Kept dependency-free and separate from Flask so every rule is unit-testable
without a request context. Covers:

* allowed upload extensions (image / video) and image / video-container
  magic-byte sniffing,
* turning a caller-supplied ``image_path`` into a safe evidence basename
  (no directory traversal, must be an allowed image type),
* plate / violation / amount validation,
* a small in-memory per-key sliding-window rate limiter.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

from modules.plate_info import normalize_plate

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# First bytes that identify common image formats (content, not just name).
_IMAGE_MAGIC = (
    b"\xff\xd8\xff",              # JPEG
    b"\x89PNG\r\n\x1a\n",        # PNG
    b"BM",                        # BMP
)

MAX_PLATE_LEN = 15
MAX_VIOLATION_LEN = 64
MAX_AMOUNT = 1_000_000


def file_extension(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def is_allowed_extension(filename: str, allowed: set[str]) -> bool:
    return file_extension(filename) in allowed


def sniff_image(head: bytes) -> bool:
    """True if ``head`` (first bytes of a file) looks like a supported image."""
    if not head:
        return False
    if head[:4] == b"RIFF":  # RIFF is also AVI/WAV: only RIFF....WEBP is an image
        return head[8:12] == b"WEBP"
    return any(head.startswith(sig) for sig in _IMAGE_MAGIC)


def sniff_video(head: bytes) -> bool:
    """True if ``head`` (first 12+ bytes) looks like a supported video container:
    ISO-BMFF (.mp4/.mov: ``ftyp`` box at offset 4), AVI (``RIFF....AVI ``), or
    Matroska/WebM (EBML magic). Rejects a renamed non-video before a worker
    thread ever hands it to the decoder."""
    if not head or len(head) < 12:
        return False
    if head[4:8] == b"ftyp":
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return True
    return head[:4] == b"\x1a\x45\xdf\xa3"


def safe_extension(filename: str, allowed: set[str], default: str) -> str:
    """Return the file's extension if allowed, else ``default`` — used to build
    a server-controlled temp-file suffix from an untrusted upload name."""
    ext = file_extension(filename)
    return ext if ext in allowed else default


def safe_evidence_name(path_or_name: str | None) -> str | None:
    """Reduce a caller-supplied path to a safe evidence basename.

    Strips any directory component (defeats ``../`` traversal) and requires an
    allowed image extension. Returns None if there's nothing usable.
    """
    if not path_or_name:
        return None
    base = os.path.basename(str(path_or_name).replace("\\", "/"))
    if not base or base in (".", ".."):
        return None
    if file_extension(base) not in IMAGE_EXTENSIONS:
        return None
    # keep only sane filename characters
    if any(c in base for c in ("/", "\0")):
        return None
    return base


def safe_json_name(name: str | None) -> str | None:
    """A bare ``*.json`` basename (an evidence sidecar), or None. Same rules as
    :func:`safe_evidence_name`: no directory part, no traversal, no NUL."""
    if not name:
        return None
    base = os.path.basename(str(name).replace("\\", "/"))
    if base != name or base in (".", "..") or "\0" in base:
        return None
    return base if file_extension(base) == ".json" else None


def validate_plate(plate: str | None) -> str | None:
    """Return the normalized plate if it's plausible (alphanumeric, bounded
    length), else None. Structural/format validation happens upstream in the
    OCR stabilizer; this is the last-line guard on the write API."""
    norm = normalize_plate(plate or "")
    if not (2 <= len(norm) <= MAX_PLATE_LEN):
        return None
    return norm


def validate_violation(violation: str | None) -> bool:
    """Accept any short, safe violation slug (letters/digits/underscore).

    Deliberately not restricted to a fixed set: unknown types are allowed and
    fall back to the default fine amount, matching existing behaviour."""
    if not violation or len(violation) > MAX_VIOLATION_LEN:
        return False
    return all(c.isalnum() or c == "_" for c in violation)


def validate_amount(amount) -> bool:
    return isinstance(amount, int) and 0 < amount <= MAX_AMOUNT


class RateLimiter:
    """Per-key sliding-window limiter (in-memory, thread-safe). Fine for a
    single-process deployment; not a distributed limiter."""

    # Sweep idle keys at most this often, so one-off clients (every distinct IP
    # that ever hit a write route) can't grow the table without bound.
    _SWEEP_EVERY_S = 60.0

    def __init__(self, max_requests: int, window_s: float):
        self.max_requests = max_requests
        self.window_s = window_s
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            cutoff = now - self.window_s
            if now - self._last_sweep >= self._SWEEP_EVERY_S:
                self._sweep(cutoff)
                self._last_sweep = now
            dq = self._hits.setdefault(key, deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.max_requests:
                return False
            dq.append(now)
            return True

    def _sweep(self, cutoff: float) -> None:
        for k in [k for k, dq in self._hits.items() if not dq or dq[-1] < cutoff]:
            del self._hits[k]

    def tracked_keys(self) -> int:
        with self._lock:
            return len(self._hits)
