"""Central, typed configuration for the whole project.

Before this module, runtime knobs were scattered across ``app.py`` (upload/job
limits, API key, model/DB paths), ``main.py`` (speed limit, streak threshold),
and the pipeline modules (detector confidence, temporal windows). That made it
hard to see what was tunable, and easy to let the CLI and the web app drift to
different defaults.

Everything tunable now lives here as a small tree of frozen dataclasses, each
built from environment variables via :meth:`from_env`. The design goals are:

* **One place to look.** Every environment variable the project reads is listed
  in this file (and documented in the README).
* **No import-time surprises.** ``from_env`` reads the environment *when called*,
  not at import. ``app.py`` calls it at module load, so re-importing the app in
  a test (with a patched ``TRAFFIC_DB_PATH`` / ``DETECT_API_KEY``) picks up the
  new values, exactly as it did when the reads were inline.
* **Typed and validated.** Numeric limits are parsed with sensible fallbacks so
  a malformed env var fails loudly rather than silently disabling a limit.

This is intentionally not a configuration *framework* — no file loaders, no
schema library. A dataclass tree is enough for a single-process app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Default location of the trained YOLO weights. Nothing in the repo ships the
# weights (they are gitignored); this is where ``train_traffic.py`` writes them
# by default and where every entry point looks unless ``MODEL_PATH`` overrides.
DEFAULT_MODEL_PATH = "runs/detect/traffic_model-2/weights/best.pt"

# Fine amounts (₹) by violation type. Unknown types fall back to DEFAULT_FINE.
DEFAULT_FINE_AMOUNTS = {
    "no_helmet": 500,
    "triple_riding": 1000,
    "overspeed": 700,
}
DEFAULT_FINE = 300

# The model's trained classes, in the order the dataset defines them. Used by
# the evaluation tooling and documented so downstream code never hard-codes a
# stray class name.
CLASS_NAMES = ("Plate", "WithHelmet", "WithoutHelmet", "TripleRiding")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: str | None) -> str | None:
    raw = os.environ.get(name)
    return raw if raw not in (None, "") else default


@dataclass(frozen=True)
class ServerConfig:
    """Flask server + upload/rate-limit/job knobs (formerly inline in app.py)."""

    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    detect_api_key: str | None = None
    max_upload_mb: int = 200
    rate_limit_per_min: int = 120
    max_concurrent_video_jobs: int = 2
    job_max_age_s: int = 3600
    max_video_mb: int = 100
    max_video_seconds: int = 300

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            host=_env_str("HOST", "127.0.0.1"),
            port=_env_int("PORT", 5000),
            debug=_env_bool("FLASK_DEBUG", False),
            detect_api_key=_env_str("DETECT_API_KEY", None),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 200),
            rate_limit_per_min=_env_int("RATE_LIMIT_PER_MIN", 120),
            max_concurrent_video_jobs=_env_int("MAX_CONCURRENT_VIDEO_JOBS", 2),
            job_max_age_s=_env_int("JOB_MAX_AGE_S", 3600),
            max_video_mb=_env_int("MAX_VIDEO_MB", 100),
            max_video_seconds=_env_int("MAX_VIDEO_SECONDS", 300),
        )


@dataclass(frozen=True)
class DetectionConfig:
    """Detector + temporal-confirmation + speed thresholds.

    These match the defaults previously baked into ``detector.py``,
    ``violation_state.py``, ``video_detector.py``, and ``speed.py`` so behaviour
    is unchanged; centralising them just makes them discoverable and overridable.
    """

    conf_threshold: float = 0.25  # YOLO min box confidence
    streak_threshold: int = 5  # consecutive frames to confirm a violation
    helmet_min_conf: float = 0.3  # no-helmet frame must clear this to count
    triple_min_conf: float = 0.3  # triple-riding frame must clear this
    speed_limit_kmh: float = 40.0
    max_video_width: int = 1280  # frames wider than this are downscaled
    contradiction_iou: float = 0.1  # helmet/no-helmet overlap = same rider

    @classmethod
    def from_env(cls) -> "DetectionConfig":
        return cls(
            conf_threshold=_env_float("DETECT_CONF_THRESHOLD", 0.25),
            streak_threshold=_env_int("STREAK_THRESHOLD", 5),
            helmet_min_conf=_env_float("HELMET_MIN_CONF", 0.3),
            triple_min_conf=_env_float("TRIPLE_MIN_CONF", 0.3),
            speed_limit_kmh=_env_float("SPEED_LIMIT_KMH", 40.0),
            max_video_width=_env_int("MAX_VIDEO_WIDTH", 1280),
            contradiction_iou=_env_float("CONTRADICTION_IOU", 0.1),
        )


@dataclass(frozen=True)
class AppConfig:
    """Top-level config: storage paths + the two sub-configs above."""

    model_path: str = DEFAULT_MODEL_PATH
    db_path: str = "traffic.db"
    evidence_dir: str = "evidence"
    log_level: str = "INFO"
    server: ServerConfig = field(default_factory=ServerConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            model_path=_env_str("MODEL_PATH", DEFAULT_MODEL_PATH),
            db_path=_env_str("TRAFFIC_DB_PATH", "traffic.db"),
            evidence_dir=_env_str("EVIDENCE_DIR", "evidence"),
            log_level=_env_str("LOG_LEVEL", "INFO"),
            server=ServerConfig.from_env(),
            detection=DetectionConfig.from_env(),
        )


def fine_amount(violation: str) -> int:
    """The fine (₹) for a violation type, with a default for unknown types."""
    return DEFAULT_FINE_AMOUNTS.get(violation, DEFAULT_FINE)


def load_config() -> AppConfig:
    """Build an :class:`AppConfig` from the current environment.

    Call this at module load in an entry point (``app.py``, the CLIs) so that a
    test which patches the environment and re-imports still sees fresh values.
    """
    return AppConfig.from_env()
