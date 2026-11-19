"""Secure secret and configuration management via Google Cloud Secret Manager."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated runtime configuration without hardcoded credentials."""

    project_id: str
    location: str
    db_path: str
    use_secret_manager: bool
    enable_cloud_dlp: bool


def load_secret(secret_id: str, project_id: str | None = None, env_fallback: str | None = None) -> str:
    """Retrieve a secret from Google Cloud Secret Manager, with optional local env fallback.

    Args:
        secret_id: Name of the secret in Google Cloud Secret Manager.
        project_id: GCP project ID hosting the secret. Defaults to GOOGLE_CLOUD_PROJECT.
        env_fallback: Environment variable name to check when Secret Manager is disabled.

    Returns:
        The secret string value, or an empty string if unavailable in local dev mode.
    """
    use_sm = os.getenv("COLDOPEN_USE_SECRET_MANAGER", "false").lower() == "true"
    resolved_project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")

    if use_sm and resolved_project:
        from google.cloud import secretmanager  # lazy import for offline unit tests

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{resolved_project}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("utf-8").strip()

    if env_fallback:
        return os.getenv(env_fallback, "").strip()
    return ""


def get_runtime_config() -> RuntimeConfig:
    """Build runtime configuration from environment variables."""
    return RuntimeConfig(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        db_path=os.getenv("COLDOPEN_DB_PATH", "coldopen_state.db"),
        use_secret_manager=os.getenv("COLDOPEN_USE_SECRET_MANAGER", "false").lower() == "true",
        enable_cloud_dlp=os.getenv("COLDOPEN_ENABLE_CLOUD_DLP", "false").lower() == "true",
    )
