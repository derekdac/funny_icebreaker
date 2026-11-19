"""PII redaction pipeline for logs, prompts, and persistent memory."""

from __future__ import annotations

import os
import re
from typing import Any

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_API_KEY_RE = re.compile(r"\b(?:AIza[0-9A-Za-z\-_]{35,}|sk-[A-Za-z0-9]{20,})\b")


def scrub_pii(text: str, use_cloud_dlp: bool | None = None) -> str:
    """Redact emails, phone numbers, SSNs, and API keys before logging or storage.

    Args:
        text: Raw input or output text to sanitize.
        use_cloud_dlp: When True, delegates to Google Cloud DLP de-identification
            before applying local regex safeguards. Defaults to COLDOPEN_ENABLE_CLOUD_DLP.

    Returns:
        Sanitized string with sensitive tokens replaced by typed placeholders.
    """
    if not text:
        return text

    dlp_enabled = (
        use_cloud_dlp
        if use_cloud_dlp is not None
        else os.getenv("COLDOPEN_ENABLE_CLOUD_DLP", "false").lower() == "true"
    )
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")

    sanitized = text
    if dlp_enabled and project_id:
        sanitized = _scrub_with_cloud_dlp(sanitized, project_id)

    sanitized = _API_KEY_RE.sub("[REDACTED_SECRET]", sanitized)
    sanitized = _EMAIL_RE.sub("[REDACTED_EMAIL]", sanitized)
    sanitized = _SSN_RE.sub("[REDACTED_SSN]", sanitized)
    sanitized = _PHONE_RE.sub("[REDACTED_PHONE]", sanitized)
    return sanitized


def scrub_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact PII from dictionary values prior to JSON logging."""
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            cleaned[key] = scrub_pii(value)
        elif isinstance(value, dict):
            cleaned[key] = scrub_mapping(value)
        elif isinstance(value, list):
            cleaned[key] = [
                scrub_pii(item) if isinstance(item, str) else item for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _scrub_with_cloud_dlp(text: str, project_id: str) -> str:
    """Invoke Cloud DLP deidentify_content for enterprise-grade PII scrubbing."""
    from google.cloud import dlp_v2  # lazy import for offline unit tests

    dlp = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    info_types = [
        {"name": "EMAIL_ADDRESS"},
        {"name": "PHONE_NUMBER"},
        {"name": "US_SOCIAL_SECURITY_NUMBER"},
        {"name": "PERSON_NAME"},
    ]
    inspect_config = {"info_types": info_types}
    deidentify_config = {
        "info_type_transformations": {
            "transformations": [
                {"primitive_transformation": {"replace_with_info_type_config": {}}}
            ]
        }
    }
    response = dlp.deidentify_content(
        request={
            "parent": parent,
            "deidentify_config": deidentify_config,
            "inspect_config": inspect_config,
            "item": {"value": text},
        }
    )
    return response.item.value
