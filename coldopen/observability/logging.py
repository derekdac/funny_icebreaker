"""Structured JSON logging with explicit intent-vs-outcome tracking."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from coldopen.observability.pii import scrub_mapping, scrub_pii


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON with PII scrubbing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": scrub_pii(record.getMessage()),
        }
        extra_fields = getattr(record, "structured_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(scrub_mapping(extra_fields))
        return json.dumps(payload, default=str)


def get_logger(name: str = "coldopen") -> logging.Logger:
    """Return a configured JSON logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_tool_intent(
    logger: logging.Logger,
    *,
    trace_id: str,
    agent_name: str,
    tool_name: str,
    intended_action: str,
    arguments: dict[str, Any],
) -> None:
    """Record the agent's intended action before tool execution."""
    logger.info(
        "tool_intent",
        extra={
            "structured_fields": {
                "event_type": "intent",
                "trace_id": trace_id,
                "agent": agent_name,
                "tool": tool_name,
                "intended_action": intended_action,
                "arguments": arguments,
            }
        },
    )


def log_tool_outcome(
    logger: logging.Logger,
    *,
    trace_id: str,
    agent_name: str,
    tool_name: str,
    status: str,
    outcome_summary: str,
    duration_ms: float,
) -> None:
    """Record the actual outcome after tool execution."""
    logger.info(
        "tool_outcome",
        extra={
            "structured_fields": {
                "event_type": "outcome",
                "trace_id": trace_id,
                "agent": agent_name,
                "tool": tool_name,
                "status": status,
                "outcome_summary": outcome_summary,
                "duration_ms": round(duration_ms, 2),
            }
        },
    )
