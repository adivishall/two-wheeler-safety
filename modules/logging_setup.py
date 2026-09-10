"""Structured application logging.

The pipeline previously diagnosed itself with ``print()``, which mixes
user-facing CLI output with internal diagnostics and can't be filtered or
routed. This module gives the *application* layer (Flask app, video pipeline,
job manager, database) a real logger while leaving the CLIs' friendly
``print()`` summaries alone.

Usage:

    from modules.logging_setup import configure_logging, get_logger
    configure_logging()             # once, at process start
    log = get_logger(__name__)
    log.info("processing video job %s", job_id)

* ``configure_logging`` is idempotent (safe to call from every entry point and
  from tests) and honours ``LOG_LEVEL`` / the level passed in.
* Log lines are single-line ``time level logger: message`` so they grep cleanly
  and don't leak multi-line payloads.
* Nothing here logs plate strings or image bytes at INFO — callers decide what
  is safe to include, and the privacy note in the README applies.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Everything in the project logs under this root so a single logger name
# controls verbosity for the whole app.
ROOT = "tws"


def configure_logging(level: str | int | None = None) -> None:
    """Attach a single stderr handler to the project root logger.

    Idempotent: repeated calls only adjust the level, never stack handlers
    (important because tests re-import the app repeatedly).
    """
    global _CONFIGURED

    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger(ROOT)
    root.setLevel(level)
    # Don't propagate to the Python root (avoids duplicate lines if the host
    # process also configured logging, e.g. gunicorn).
    root.propagate = False

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setLevel(level)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the project root (``tws.<name>``).

    Auto-configures with defaults if ``configure_logging`` hasn't run yet, so a
    library import never emits the "No handlers could be found" warning.
    """
    if not _CONFIGURED:
        configure_logging()
    short = name.split(".")[-1] if name else "app"
    return logging.getLogger(f"{ROOT}.{short}")
