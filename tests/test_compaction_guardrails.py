"""Unit tests for safe ADK compaction boundaries and OpenTelemetry tracing idempotency."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.genai import types
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from coldopen.guardrails.policy import adk_policy_and_compaction_callback
from coldopen.memory.compaction import compact_history
from coldopen.observability import tracing


def _user_text(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def _model_text(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part.from_text(text=text)])


def _model_fc(name: str, args: dict[str, object] | None = None) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args or {}))],
    )


def _user_fr(name: str, response: dict[str, object] | None = None) -> types.Content:
    return types.Content(
        role="user",
        parts=[types.Part(function_response=types.FunctionResponse(name=name, response=response or {"ok": True}))],
    )


def _assert_valid_turn_structure(contents: list[types.Content]) -> None:
    """Assert Gemini turn invariants: no orphaned function_response, no adjacent user turns, no empty text parts."""
    assert contents, "Compacted contents must not be empty"
    assert contents[0].role == "user"
    assert not any(getattr(p, "function_response", None) is not None for p in (contents[0].parts or []))

    for idx, content in enumerate(contents):
        parts = content.parts or []
        for part in parts:
            if getattr(part, "text", None) is not None:
                assert part.text != "", f"Empty text part found at index {idx}"

        has_fr = any(getattr(p, "function_response", None) is not None for p in parts)
        if has_fr:
            assert idx > 0, "function_response cannot be at index 0"
            prev_parts = contents[idx - 1].parts or []
            assert contents[idx - 1].role == "model"
            assert any(
                getattr(p, "function_call", None) is not None for p in prev_parts
            ), f"Orphaned function_response at index {idx} without preceding function_call"

        if idx > 0:
            prev_role = contents[idx - 1].role
            curr_role = content.role
            prev_has_fr = any(
                getattr(p, "function_response", None) is not None for p in (contents[idx - 1].parts or [])
            )
            if prev_role == "user" and curr_role == "user" and not prev_has_fr and not has_fr:
                raise AssertionError(f"Adjacent plain user turns at indices {idx - 1} and {idx}")


def test_compaction_preserves_function_call_response_pair_at_boundary() -> None:
    # 10 turns: contents[-6] is index 4 (_user_fr) whose matching _model_fc is at index 3.
    contents = [
        _user_text("Hi ColdOpen"),            # 0
        _model_text("Hello! Pick a theme."),  # 1
        _user_text("IKEA or Death Metal"),    # 2 (clean user turn before boundary)
        _model_fc("ideator_agent"),           # 3 (contents[-7])
        _user_fr("ideator_agent"),            # 4 (contents[-6] — MUST NOT split here)
        _model_fc("save_and_audit_draft"),    # 5
        _user_fr("save_and_audit_draft"),     # 6
        _model_text("Draft ready!"),          # 7
        _user_text("Looks great, build it"),  # 8
        _model_text("Building now..."),       # 9
    ]
    req = SimpleNamespace(contents=list(contents))

    resp = adk_policy_and_compaction_callback(SimpleNamespace(), req)  # type: ignore[arg-type]
    assert resp is None

    _assert_valid_turn_structure(req.contents)
    # Should have split at index 2 (dropping 0..1) and merged summary into index 2's user turn.
    assert len(req.contents) == 8
    first_parts = req.contents[0].parts or []
    assert len(first_parts) == 2
    assert "Earlier session summary:" in (first_parts[0].text or "")
    assert "Hi ColdOpen" in (first_parts[0].text or "")
    assert first_parts[1].text == "IKEA or Death Metal"
    # Tool call / response pairs remain intact at indices 1-2 and 3-4.
    assert req.contents[1].parts[0].function_call.name == "ideator_agent"
    assert req.contents[2].parts[0].function_response.name == "ideator_agent"


def test_compaction_scans_forward_when_no_clean_user_turn_before_target() -> None:
    # Index 0 is user, indices 1..5 are tool calls/responses + model reply, index 6 is next clean user.
    contents = [
        _user_text("Start game"),             # 0
        _model_fc("tool_1"),                  # 1
        _user_fr("tool_1"),                   # 2
        _model_fc("tool_2"),                  # 3
        _user_fr("tool_2"),                   # 4 (target_idx = 10 - 6 = 4)
        _model_text("Tool results ready"),    # 5
        _user_text("Proceed to compile"),     # 6 (first clean user turn > 0)
        _model_fc("compile_pptx"),            # 7
        _user_fr("compile_pptx"),             # 8
        _model_text("Deck compiled"),         # 9
    ]
    req = SimpleNamespace(contents=list(contents))

    resp = adk_policy_and_compaction_callback(SimpleNamespace(), req)  # type: ignore[arg-type]
    assert resp is None

    _assert_valid_turn_structure(req.contents)
    assert len(req.contents) == 4
    first_parts = req.contents[0].parts or []
    assert "Earlier session summary:" in (first_parts[0].text or "")
    assert "Start game" in (first_parts[0].text or "")
    assert first_parts[1].text == "Proceed to compile"


def test_compaction_skips_when_only_single_active_tool_chain() -> None:
    # No clean user turn at any index >= 1 -> must NOT split anywhere.
    contents = [
        _user_text("Run long workflow"),
        _model_fc("step_1"),
        _user_fr("step_1"),
        _model_fc("step_2"),
        _user_fr("step_2"),
        _model_fc("step_3"),
        _user_fr("step_3"),
        _model_fc("step_4"),
        _user_fr("step_4"),
    ]
    req = SimpleNamespace(contents=list(contents))

    resp = adk_policy_and_compaction_callback(SimpleNamespace(), req)  # type: ignore[arg-type]
    assert resp is None
    assert len(req.contents) == len(contents)
    _assert_valid_turn_structure(req.contents)


def test_compaction_empty_summary_prefix_does_not_prepend_empty_text_part() -> None:
    # Dropped turns (0..2) contain only tool calls/responses with no text -> summary_prefix == "".
    contents = [
        _model_fc("init_tool"),               # 0 (no text)
        _user_fr("init_tool"),                # 1 (no text)
        _model_fc("preload_tool"),            # 2 (no text)
        _user_fr("preload_tool"),             # 3 (no text)
        _user_text("Create Font or Cheese"),  # 4 (clean user turn, target_idx = 10 - 6 = 4)
        _model_fc("ideator_agent"),           # 5
        _user_fr("ideator_agent"),            # 6
        _model_text("Draft created"),         # 7
        _user_text("Confirm"),                # 8
        _model_text("Done"),                  # 9
    ]
    req = SimpleNamespace(contents=list(contents))

    resp = adk_policy_and_compaction_callback(SimpleNamespace(), req)  # type: ignore[arg-type]
    assert resp is None

    _assert_valid_turn_structure(req.contents)
    assert len(req.contents) == 6
    # First kept turn must NOT have an empty summary part prepended.
    first_parts = req.contents[0].parts or []
    assert len(first_parts) == 1
    assert first_parts[0].text == "Create Font or Cheese"


def test_compact_history_backward_compatible() -> None:
    messages = [{"role": "user", "content": f"Turn {i}: brainstorm"} for i in range(10)]
    result = compact_history(messages, max_recent_turns=4, max_token_budget=500)
    assert result.dropped_count == 6
    assert len(result.recent_messages) == 4
    assert "Earlier session summary:" in result.summary_prefix


def test_init_tracing_and_traced_span_respect_existing_provider(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exporter = InMemorySpanExporter()
    existing_provider = TracerProvider()
    existing_provider.add_span_processor(SimpleSpanProcessor(exporter))

    monkeypatch.setattr(tracing, "_INITIALIZED", False)
    with (
        patch.object(trace, "get_tracer_provider", return_value=existing_provider),
        patch.object(trace, "set_tracer_provider") as mock_set_provider,
        patch.object(trace, "get_tracer", side_effect=existing_provider.get_tracer),
        caplog.at_level(logging.WARNING),
    ):
        tracer_1 = tracing.init_tracing(service_name="coldopen")
        tracer_2 = tracing.init_tracing(service_name="coldopen")

        assert tracer_1 is not None
        assert tracer_2 is not None
        mock_set_provider.assert_not_called()
        assert "Overriding of current TracerProvider is not allowed" not in caplog.text

        with tracing.traced_span("compile_deck", attributes={"team": "eng", "items": 10, "ok": True}) as span:
            assert span is not None

        mock_set_provider.assert_not_called()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "compile_deck"
    assert spans[0].attributes == {"team": "eng", "items": 10, "ok": True}


def test_init_tracing_registers_once_when_proxy_provider_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy_provider = trace.ProxyTracerProvider()
    monkeypatch.setattr(tracing, "_INITIALIZED", False)

    with (
        patch.object(trace, "get_tracer_provider", return_value=proxy_provider),
        patch.object(trace, "set_tracer_provider") as mock_set_provider,
    ):
        tracing.init_tracing(service_name="coldopen", export_to_console=False)
        tracing.init_tracing(service_name="coldopen", export_to_console=False)
        assert mock_set_provider.call_count == 1
