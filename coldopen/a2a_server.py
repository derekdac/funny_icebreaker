"""A2A server entrypoint for serving Funny Icebreaker Game Agent with A2UI enabled."""

from __future__ import annotations

import os

from google.adk.a2a.utils.agent_to_a2a import to_a2a

# Enable A2UI payloads by default when launched via the A2A server.
os.environ.setdefault("COLDOPEN_ENABLE_A2UI", "true")

from coldopen.agents.pipeline import build_coldopen_pipeline  # noqa: E402

a2a_agent = build_coldopen_pipeline()
app = to_a2a(a2a_agent, port=int(os.getenv("PORT", "8000")))
