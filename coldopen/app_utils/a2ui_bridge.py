"""A2UI-over-A2A bridge compatible with a2a-sdk 1.x and ADK 2.x.

Negotiates A2UI v0.9 / v0.8 per A2A request via ``X-A2A-Extensions``, exposes
``SendA2uiToClientToolset`` only to A2UI-capable clients, and converts
``send_a2ui_json_to_client`` tool responses into A2A ``DataPart`` payloads with
``mimeType="application/json+a2ui"``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextvars import ContextVar
from typing import Any

from a2a.server.agent_execution import RequestContext
from a2a.types import AgentExtension, Part
from a2ui.constants import A2UI_TOOL_NAME, A2UI_VALIDATED_JSON_KEY
from a2ui.core.basic_catalog import BasicCatalog
from a2ui.inference_formats import to_message_dicts
from a2ui.parser.response_part import A2uiPart, ErrorPart, TextPart
from a2ui.processor.processor import A2uiRequestProcessor
from google.adk.a2a.converters import part_converter
from google.adk.a2a.executor.config import ExecuteInterceptor
from google.adk.agents.readonly_context import ReadonlyContext
from google.genai import types as genai_types

_LOGGER = logging.getLogger(__name__)

A2UI_EXTENSION_V0_8 = "https://a2ui.org/a2a-extension/a2ui/v0.8"
A2UI_EXTENSION_V0_9 = "https://a2ui.org/a2a-extension/a2ui/v0.9"
A2UI_MIME_TYPE = "application/json+a2ui"

_A2UI_CLIENT_ACTION_KEYS = ("userAction", "action")
_A2UI_TAG_DETECT_RE = re.compile(r"<a2ui(?:-json)?\b", re.IGNORECASE)
_A2UI_EXPRESS_TAG_RE = re.compile(r"<a2ui\b", re.IGNORECASE)
_A2UI_JSON_BLOCK_RE = re.compile(
    r"<a2ui-json\b[^>]*>(.*?)</a2ui-json>",
    re.IGNORECASE | re.DOTALL,
)
_A2UI_STRIP_TAGS_RE = re.compile(
    r"<a2ui(?:-json)?\b[^>]*>.*?(?:</a2ui(?:-json)?>|\Z)",
    re.IGNORECASE | re.DOTALL,
)

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


def a2ui_processor_provider(
    _ctx: ReadonlyContext | None = None,
) -> A2uiRequestProcessor:
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


def _is_a2ui_data_part(
    part: Part,
    meta: dict[str, Any],
    data: Any,
) -> bool:
    """Return True when an inbound A2A DataPart carries an A2UI payload or action."""
    if getattr(part, "media_type", None) == A2UI_MIME_TYPE:
        return True
    if meta.get("mimeType") == A2UI_MIME_TYPE:
        return True
    if isinstance(data, dict) and any(k in data for k in _A2UI_CLIENT_ACTION_KEYS):
        return True
    return False


def a2ui_a2a_part_to_genai_part(part: Part) -> genai_types.Part | None:
    """Convert inbound A2A parts to GenAI parts, surfacing A2UI DataParts as readable text.

    Converting ``application/json+a2ui`` and A2UI ``userAction``/``action`` DataParts
    into ``Part.from_text(...)`` ensures SFW policy guardrails and history compaction
    inspect the payload and the LLM receives the structured client action cleanly.
    """
    if part_converter._compat.is_data_part(part):
        meta = part_converter._compat.part_metadata(part) or {}
        data = part_converter._compat.data_part_dict(part)
        if _is_a2ui_data_part(part, meta, data):
            if data is not None:
                payload_json = json.dumps(data, ensure_ascii=False)
            else:
                payload_json = part_converter._compat.data_part_blob_bytes(part).decode(
                    "utf-8", errors="replace"
                )
            return genai_types.Part.from_text(
                text=f"[A2UI Client Action]\n{payload_json}"
            )

    return part_converter.convert_a2a_part_to_genai_part(part)


def _strip_raw_a2ui_markup(text: str) -> str:
    """Remove any raw <a2ui> or <a2ui-json> blocks (including unclosed tags)."""
    return _A2UI_STRIP_TAGS_RE.sub("", text)


def _emit_text_part(text: str, out: list[Part]) -> None:
    """Append a cleaned non-empty A2A text Part to ``out``."""
    cleaned = _strip_raw_a2ui_markup(text).strip()
    if cleaned:
        converted = part_converter.convert_genai_part_to_a2a_part(
            genai_types.Part.from_text(text=cleaned)
        )
        if converted is not None:
            out.append(converted)


def _parse_a2ui_express_segment(
    text: str,
    processor: A2uiRequestProcessor,
) -> list[Part]:
    """Parse a text segment that may contain <a2ui>...</a2ui> Express DSL blocks."""
    if not _A2UI_EXPRESS_TAG_RE.search(text):
        out: list[Part] = []
        _emit_text_part(text, out)
        return out

    out = []
    try:
        stream = processor.open_stream(after_rejected_block="continue")
        parsed_parts = [*stream.feed(text), *stream.close()]
    except Exception as exc:
        _LOGGER.warning("Failed to parse inline <a2ui> block: %s", exc)
        _emit_text_part(text, out)
        return out

    for seg in parsed_parts:
        if isinstance(seg, TextPart):
            _emit_text_part(seg.text, out)
        elif isinstance(seg, A2uiPart):
            for msg in to_message_dicts(seg.a2ui):
                if isinstance(msg, dict):
                    out.append(_make_a2ui_data_part(msg))
        elif isinstance(seg, ErrorPart):
            _LOGGER.warning("Dropped malformed inline <a2ui> block: %s", seg.error)
    return out


def _convert_text_with_a2ui_tags(
    text: str,
    processor: A2uiRequestProcessor,
) -> list[Part]:
    """Extract <a2ui-json> and <a2ui> blocks into DataParts without leaking raw DSL."""
    out: list[Part] = []
    cursor = 0
    for match in _A2UI_JSON_BLOCK_RE.finditer(text):
        prefix = text[cursor : match.start()]
        if prefix:
            out.extend(_parse_a2ui_express_segment(prefix, processor))
        raw_json = match.group(1).strip()
        try:
            payload = json.loads(raw_json)
            items = (
                [payload]
                if isinstance(payload, dict)
                else (payload if isinstance(payload, list) else [])
            )
            for item in items:
                if isinstance(item, dict):
                    out.append(_make_a2ui_data_part(item))
        except Exception as exc:
            _LOGGER.warning("Dropped malformed inline <a2ui-json> block: %s", exc)
        cursor = match.end()

    tail = text[cursor:]
    if tail:
        out.extend(_parse_a2ui_express_segment(tail, processor))
    return out


def a2ui_genai_part_to_a2a_part(
    part: genai_types.Part,
) -> Part | list[Part] | None:
    """Convert ADK GenAI parts to A2A parts, translating A2UI tool outputs and inline blocks to DataParts."""
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

    if part.text and _A2UI_TAG_DETECT_RE.search(part.text):
        processor = a2ui_processor_provider(None)
        converted_parts = _convert_text_with_a2ui_tags(part.text, processor)
        if not converted_parts:
            return None
        return converted_parts

    return part_converter.convert_genai_part_to_a2a_part(part)


def a2ui_execute_interceptor() -> ExecuteInterceptor:
    """Negotiate A2UI extension per A2A request and reset context on completion."""

    async def _before_agent(context: RequestContext) -> RequestContext:
        requested = set(context.requested_extensions or ())
        if A2UI_EXTENSION_V0_9 in requested:
            _ACTIVE_A2UI_VERSION.set("0.9")
            if hasattr(context, "add_activated_extension"):
                context.add_activated_extension(A2UI_EXTENSION_V0_9)
        elif A2UI_EXTENSION_V0_8 in requested:
            _ACTIVE_A2UI_VERSION.set("0.8")
            if hasattr(context, "add_activated_extension"):
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
