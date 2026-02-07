"""Structured logging configuration for FriendlyFace.

Environment variables:
    FF_LOG_FORMAT  -- ``json`` for structured JSON output, ``text`` for human-readable (default).
    FF_LOG_LEVEL   -- Python log level name (default: ``INFO``).
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Any


def _is_json_mode() -> bool:
    """Return True when structured JSON logging is requested."""
    return os.environ.get("FF_LOG_FORMAT", "text").lower() == "json"


def _get_log_level() -> int:
    """Return the numeric log level from FF_LOG_LEVEL (default INFO)."""
    name = os.environ.get("FF_LOG_LEVEL", "INFO").upper()
    numeric = getattr(logging, name, None)
    if not isinstance(numeric, int):
        return logging.INFO
    return numeric


class StructuredJsonFormatter(logging.Formatter):
    """JSON formatter that emits one JSON object per log line.

    Uses ``pythonjsonlogger`` under the hood but injects the FriendlyFace
    specific fields (request_id, path, method, status_code, duration_ms)
    when they are present on the LogRecord.
    """

    def __init__(self) -> None:
        super().__init__()
        from pythonjsonlogger.json import JsonFormatter

        self._inner = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Attach structured extras so they appear in the JSON output.
        extras: dict[str, Any] = {}
        for key in (
            "request_id",
            "path",
            "method",
            "status_code",
            "duration_ms",
        ):
            value = getattr(record, key, None)
            if value is not None:
                extras[key] = value

        # Attach traceback as a structured field when present.
        if record.exc_info and record.exc_info[1] is not None:
            extras["traceback"] = traceback.format_exception(*record.exc_info)
            # Prevent the inner formatter from also appending the traceback
            # as free-form text.
            record.exc_info = None
            record.exc_text = None

        # Inject extras onto the record so JsonFormatter serialises them.
        for k, v in extras.items():
            setattr(record, k, v)

        return self._inner.format(record)


def setup_logging() -> None:
    """Configure the root logger according to FF_LOG_FORMAT and FF_LOG_LEVEL."""
    level = _get_log_level()
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers so we don't double-log during tests.
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)

    if _is_json_mode():
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    root.addHandler(handler)


def log_startup_info() -> None:
    """Emit a structured startup log line with platform configuration."""
    import friendlyface

    logger = logging.getLogger("friendlyface")
    storage_backend = os.environ.get("FF_STORAGE", "sqlite")
    did_seed_set = bool(os.environ.get("FF_DID_SEED"))

    logger.info(
        "FriendlyFace started",
        extra={
            "version": friendlyface.__version__,
            "storage_backend": storage_backend,
            "rate_limit_config": os.environ.get("FF_RATE_LIMIT", "none"),
            "did_seed_status": "configured" if did_seed_set else "random",
        },
    )
