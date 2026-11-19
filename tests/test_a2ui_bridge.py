"""Unit tests for the A2UI-over-A2A bridge, inbound/outbound converters, and prompt guidance."""

from __future__ import annotations

import json
from types import SimpleNamespace

from a2ui.constants import A2UI_TOOL_NAME, A2UI_VALIDATED_JSON_KEY
from google.adk.a2a.converters import part_converter
from google.genai import types as genai_types

from coldopen.app_utils import a2a as a2a_module
from coldopen.app_utils.a2ui_bridge import (
    A2UI_MIME_TYPE,
    _ACTIVE_A2UI_VERSION,
    a2ui_a2a_part_to_genai_part,
    a2ui_genai_part_to_a2a_part,
)
from coldopen.guardrails.policy import adk_policy_and_compaction_callback
from coldopen.prompts.constitution import HOST_INSTRUCTION


def _make_data_part(
    data: dict,
    *,
    metadata: dict | None = None,
    media_type: str | None = None,
):
    part = part_converter._compat.make_data_part(data=data, metadata=metadata)
    if media_type is not None:
        part.media_type = media_type
    return part


class TestInboundA2uiConverter:
    def test_converts_media_type_a2ui_data_part_to_text(self) -> None:
        payload = {
            "userAction": {
                "name": "select_theme",
                "context": {"theme": "IKEA Furniture or Death Metal Band"},
            }
        }
        part = _make_data_part(payload, media_type=A2UI_MIME_TYPE)

        converted = a2ui_a2a_part_to_genai_part(part)

        assert converted is not None
        assert converted.text is not None
        assert converted.inline_data is None
        assert "select_theme" in converted.text
        assert "IKEA Furniture or Death Metal Band" in converted.text
        # Ensure the embedded JSON is valid and round-trips cleanly.
        json_body = converted.text.split("\n", 1)[1]
        assert json.loads(json_body) == payload

    def test_converts_metadata_mime_type_a2ui_data_part_to_text(self) -> None:
        payload = {"customA2ui": {"surfaceId": "main", "value": "confirm"}}
        part = _make_data_part(payload, metadata={"mimeType": A2UI_MIME_TYPE})

        converted = a2ui_a2a_part_to_genai_part(part)

        assert converted is not None
        assert converted.text is not None
        assert json.loads(converted.text.split("\n", 1)[1]) == payload

    def test_converts_v0_9_action_without_mime_type_to_text(self) -> None:
        payload = {
            "action": {
                "name": "confirm_deck",
                "context": {"draft_id": "draft-123"},
            }
        }
        part = _make_data_part(payload)

        converted = a2ui_a2a_part_to_genai_part(part)

        assert converted is not None
        assert converted.text is not None
        assert "confirm_deck" in converted.text
        assert "draft-123" in converted.text

    def test_inbound_a2ui_payload_is_screened_by_sfw_guardrail(self) -> None:
        payload = {
            "userAction": {
                "name": "submit_custom_theme",
                "context": {"theme": "NSFW workplace jokes"},
            }
        }
        part = _make_data_part(
            payload,
            metadata={"mimeType": A2UI_MIME_TYPE},
            media_type=A2UI_MIME_TYPE,
        )
        genai_part = a2ui_a2a_part_to_genai_part(part)
        assert genai_part is not None

        llm_request = SimpleNamespace(
            contents=[genai_types.Content(role="user", parts=[genai_part])]
        )
        response = adk_policy_and_compaction_callback(SimpleNamespace(), llm_request)

        assert response is not None
        assert response.content is not None
        blocked_text = response.content.parts[0].text
        assert blocked_text is not None
        assert "Blocked by SFW/policy guardrail" in blocked_text

    def test_delegates_non_a2ui_parts_to_default_converter(self) -> None:
        text_part = part_converter._compat.make_text_part("Hello ColdOpen")
        converted_text = a2ui_a2a_part_to_genai_part(text_part)
        assert converted_text is not None
        assert converted_text.text == "Hello ColdOpen"

        generic_data = _make_data_part({"foo": "bar"})
        converted_generic = a2ui_a2a_part_to_genai_part(generic_data)
        assert converted_generic is not None
        assert converted_generic.text is None
        assert converted_generic.inline_data is not None


class TestOutboundA2uiConverter:
    def test_converts_valid_express_dsl_and_preserves_surrounding_prose(self) -> None:
        token = _ACTIVE_A2UI_VERSION.set("0.9")
        try:
            genai_part = genai_types.Part.from_text(
                text='Pick a theme:\n<a2ui>\nroot = Text("Hello world")\n</a2ui>\nOr type your own!'
            )
            result = a2ui_genai_part_to_a2a_part(genai_part)
        finally:
            _ACTIVE_A2UI_VERSION.reset(token)

        assert isinstance(result, list)
        texts = [
            part_converter._compat.part_text(p)
            for p in result
            if part_converter._compat.is_text_part(p)
        ]
        data_parts = [p for p in result if part_converter._compat.is_data_part(p)]

        assert texts == ["Pick a theme:", "Or type your own!"]
        assert len(data_parts) >= 2
        for dp in data_parts:
            assert getattr(dp, "media_type", None) == A2UI_MIME_TYPE
            assert part_converter._compat.part_metadata(dp).get("mimeType") == A2UI_MIME_TYPE
        for t in texts:
            assert "<a2ui" not in t.lower()

    def test_converts_a2ui_json_block_and_preserves_prose(self) -> None:
        msg = {
            "version": "v0.9",
            "createSurface": {"surfaceId": "s1", "catalogId": "basic"},
        }
        genai_part = genai_types.Part.from_text(
            text=f"Intro\n<a2ui-json>[{json.dumps(msg)}]</a2ui-json>\nOutro"
        )

        result = a2ui_genai_part_to_a2a_part(genai_part)

        assert isinstance(result, list)
        texts = [
            part_converter._compat.part_text(p)
            for p in result
            if part_converter._compat.is_text_part(p)
        ]
        data_parts = [p for p in result if part_converter._compat.is_data_part(p)]

        assert texts == ["Intro", "Outro"]
        assert len(data_parts) == 1
        assert part_converter._compat.data_part_dict(data_parts[0]) == msg
        assert getattr(data_parts[0], "media_type", None) == A2UI_MIME_TYPE

    def test_strips_malformed_express_dsl_without_leaking_raw_tags(self) -> None:
        token = _ACTIVE_A2UI_VERSION.set("0.9")
        try:
            genai_part = genai_types.Part.from_text(
                text="Before\n<a2ui>\nnot valid syntax !!!\n</a2ui>\nAfter"
            )
            result = a2ui_genai_part_to_a2a_part(genai_part)
        finally:
            _ACTIVE_A2UI_VERSION.reset(token)

        assert isinstance(result, list)
        texts = [
            part_converter._compat.part_text(p)
            for p in result
            if part_converter._compat.is_text_part(p)
        ]
        assert texts == ["Before", "After"]
        assert all("<a2ui" not in t.lower() for t in texts)

    def test_returns_none_when_only_malformed_a2ui_markup_present(self) -> None:
        token = _ACTIVE_A2UI_VERSION.set("0.9")
        try:
            bad_express = genai_types.Part.from_text(text="<a2ui>bad !!!</a2ui>")
            bad_json = genai_types.Part.from_text(text="<a2ui-json>{not json</a2ui-json>")
            unclosed_json = genai_types.Part.from_text(text="<a2ui-json>{unclosed")

            assert a2ui_genai_part_to_a2a_part(bad_express) is None
            assert a2ui_genai_part_to_a2a_part(bad_json) is None
            assert a2ui_genai_part_to_a2a_part(unclosed_json) is None
        finally:
            _ACTIVE_A2UI_VERSION.reset(token)

    def test_mixed_valid_and_invalid_express_blocks(self) -> None:
        token = _ACTIVE_A2UI_VERSION.set("0.9")
        try:
            genai_part = genai_types.Part.from_text(
                text='Intro\n<a2ui>\nroot = Text("Good")\n</a2ui>\nMid\n<a2ui>\nbad !!!\n</a2ui>\nEnd'
            )
            result = a2ui_genai_part_to_a2a_part(genai_part)
        finally:
            _ACTIVE_A2UI_VERSION.reset(token)

        assert isinstance(result, list)
        texts = [
            part_converter._compat.part_text(p)
            for p in result
            if part_converter._compat.is_text_part(p)
        ]
        data_parts = [p for p in result if part_converter._compat.is_data_part(p)]
        assert texts == ["Intro", "Mid", "End"]
        assert len(data_parts) >= 2

    def test_plain_text_and_a2ui_tool_parts_still_work(self) -> None:
        plain = a2ui_genai_part_to_a2a_part(genai_types.Part.from_text(text="Just prose."))
        assert plain is not None
        assert not isinstance(plain, list)
        assert part_converter._compat.part_text(plain) == "Just prose."

        fc_part = genai_types.Part(
            function_call=genai_types.FunctionCall(name=A2UI_TOOL_NAME, args={})
        )
        assert a2ui_genai_part_to_a2a_part(fc_part) is None

        fr_part = genai_types.Part(
            function_response=genai_types.FunctionResponse(
                name=A2UI_TOOL_NAME,
                response={A2UI_VALIDATED_JSON_KEY: [{"createSurface": {"surfaceId": "s1"}}]},
            )
        )
        fr_result = a2ui_genai_part_to_a2a_part(fr_part)
        assert isinstance(fr_result, list)
        assert len(fr_result) == 1
        assert getattr(fr_result[0], "media_type", None) == A2UI_MIME_TYPE


class TestWiringAndConstitution:
    def test_a2a_module_wires_inbound_converter(self) -> None:
        import inspect

        src = inspect.getsource(a2a_module.attach_a2a_routes)
        assert "a2a_part_converter=a2ui_a2a_part_to_genai_part" in src
        assert "gen_ai_part_converter=a2ui_genai_part_to_a2a_part" in src

    def test_host_instruction_documents_a2ui_actions_and_draft_summary(self) -> None:
        assert "userAction" in HOST_INSTRUCTION
        assert "action" in HOST_INSTRUCTION
        assert "skip_summarization=True" in HOST_INSTRUCTION
        assert "{draft_summary?}" in HOST_INSTRUCTION
