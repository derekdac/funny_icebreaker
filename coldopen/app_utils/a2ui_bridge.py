"""A2UI-over-A2A bridge compatible with a2a-sdk 1.x and ADK 2.x.

Negotiates A2UI v0.9 / v0.8 per A2A request via ``X-A2A-Extensions``, exposes
``SendA2uiToClientToolset`` only to A2UI-capable clients, and converts
``send_a2ui_json_to_client`` tool responses into A2A ``DataPart`` payloads with
``mimeType="application/json+a2ui"``.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any

from a2a.server.agent_execution import RequestContext
from a2a.types import AgentExtension, Part
from a2ui.constants import A2UI_TOOL_NAME, A2UI_VALIDATED_JSON_KEY
from a2ui.core.basic_catalog import BasicCatalog
from a2ui.processor.processor import A2uiRequestProcessor
from google.adk.a2a.converters import part_converter
from google.adk.a2a.executor.config import ExecuteInterceptor
from google.adk.agents.readonly_context import ReadonlyContext
from google.genai import types as genai_types

A2UI_EXTENSION_V0_8 = "https://a2ui.org/a2a-extension/a2ui/v0.8"
A2UI_EXTENSION_V0_9 = "https://a2ui.org/a2a-extension/a2ui/v0.9"
A2UI_MIME_TYPE = "application/json+a2ui"

_ACTIVE_A2UI_VERSION: ContextVar[str | None] = ContextVar(
    "coldopen_active_a2ui_version", default=None
)

_PROCESSORS: dict[str, A2uiRequestProcessor] = {
    "0.8": A2uiRequestProcessor(catalogs=[BasicCatalog("0.8")]),
    "0.9": A2uiRequestProcessor(catalogs=[BasicCatalog("0.9")]),
}


def build_a2ui_extensions() -> list[AgentExtension]:
    """Advertise both A2UI v0.9 and v0.8 extensions on the A2A Agent Card."""
    return [
        AgentExtension(
            uri=A2UI_EXTENSION_V0_9,
            description="Ability to render A2UI v0.9 interactive UI surfaces.",
            params={"supportedCatalogIds": [BasicCatalog("0.9").catalog_id]},
        ),
        AgentExtension(
            uri=A2UI_EXTENSION_V0_8,
            description="Ability to render A2UI v0.8 interactive UI surfaces.",
            params={"supportedCatalogIds": [BasicCatalog("0.8").catalog_id]},
        ),
    ]


def a2ui_enabled_provider(_ctx: ReadonlyContext) -> bool:
    """Enable A2UI when negotiated on the active A2A request or forced via env var."""
    if _ACTIVE_A2UI_VERSION.get() is not None:
        return True
    return os.getenv("COLDOPEN_ENABLE_A2UI", "false").lower() == "true"


def a2ui_processor_provider(_ctx: ReadonlyContext) -> A2uiRequestProcessor:
    """Return the A2uiRequestProcessor matching the negotiated protocol version."""
    version = _ACTIVE_A2UI_VERSION.get() or "0.9"
    return _PROCESSORS.get(version, _PROCESSORS["0.9"])


def _make_a2ui_data_part(message: dict[str, Any]) -> Part:
    """Build an A2A DataPart tagged with application/json+a2ui."""
    part = part_converter._compat.make_data_part(
        data=message,
        metadata={"mimeType": A2UI_MIME_TYPE},
    )
    if getattr(part_converter._compat, "IS_A2A_V1", False):
        part.media_type = A2UI_MIME_TYPE
    return part


def a2ui_genai_part_to_a2a_part(
    part: genai_types.Part,
) -> Part | list[Part] | None:
    """Convert ADK GenAI parts to A2A parts, translating A2UI tool outputs to DataParts."""
    if part.function_call and part.function_call.name == A2UI_TOOL_NAME:
        return None

    if part.function_response and part.function_response.name == A2UI_TOOL_NAME:
        response = part.function_response.response or {}
        validated = response.get(A2UI_VALIDATED_JSON_KEY)
        if isinstance(validated, list):
            return [
                _make_a2ui_data_part(msg)
                for msg in validated
                if isinstance(msg, dict)
            ]
        return None

    return part_converter.convert_genai_part_to_a2a_part(part)


def a2ui_execute_interceptor() -> ExecuteInterceptor:
    """Negotiate A2UI extension per A2A request and reset context on completion."""

    async def _before_agent(context: RequestContext) -> RequestContext:
        requested = set(context.requested_extensions or ())
        if A2UI_EXTENSION_V0_9 in requested:
            _ACTIVE_A2UI_VERSION.set("0.9")
            context.add_activated_extension(A2UI_EXTENSION_V0_9)
        elif A2UI_EXTENSION_V0_8 in requested:
            _ACTIVE_A2UI_VERSION.set("0.8")
            context.add_activated_extension(A2UI_EXTENSION_V0_8)
        else:
            _ACTIVE_A2UI_VERSION.set(None)
        return context

    async def _after_agent(_exec_ctx: Any, event: Any) -> Any:
        _ACTIVE_A2UI_VERSION.set(None)
        return event

    return ExecuteInterceptor(
        before_agent=_before_agent,
        after_agent=_after_agent,
    )
