"""Structured JSON logging utility with correlation ID support.

Provides a JSON log formatter and per-request correlation ID tracking
using Python's built-in logging module and contextvars (no external deps).
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone

# Context variable for per-request correlation IDs.
correlation_id_var: ContextVar[str | None] = ContextVar(
    "correlation_id", default=None
)


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current async/thread context."""
    correlation_id_var.set(cid)


def get_correlation_id() -> str | None:
    """Return the current correlation ID, or ``None`` if unset."""
    return correlation_id_var.get()


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Each line contains:
    - ``timestamp`` – ISO-8601 UTC timestamp
    - ``level`` – log level name
    - ``component`` – logger name (set via *get_logger*)
    - ``correlation_id`` – per-request ID from *correlation_id_var*
    - ``message`` – the formatted log message
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "correlation_id": get_correlation_id(),
            "message": record.getMessage(),
        }
        return json.dumps(entry, default=str)


def get_logger(component: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a logger configured with the JSON formatter.

    Parameters
    ----------
    component:
        Identifier for the subsystem (e.g. ``"orchestrator"``,
        ``"pdf_parser"``).  Used as the logger *name* and appears in
        every log record under the ``component`` key.
    level:
        Minimum severity.  Defaults to ``INFO``.
    """
    logger = logging.getLogger(component)

    # Avoid adding duplicate handlers when called more than once for the
    # same component (e.g. during tests or module reloads).
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        # Prevent propagation to the root logger so messages aren't
        # duplicated when the root logger also has handlers.
        logger.propagate = False

    return logger
