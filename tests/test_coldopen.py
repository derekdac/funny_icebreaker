"""Unit tests for ColdOpen tools, memory, guardrails, and PII redaction."""

from __future__ import annotations

import asyncio
from pathlib import Path

from coldopen.guardrails.policy import evaluate_policy_guardrail, request_host_approval
from coldopen.memory.compaction import compact_history
from coldopen.memory.consolidator import schedule_memory_consolidation
from coldopen.memory.store import ColdOpenStore
from coldopen.observability.pii import scrub_pii
from coldopen.tools.auditor import audit_headline_leaks, check_sequence_entropy
from coldopen.tools.compiler import (
    compile_pptx_deck,
    compile_pptx_from_draft,
    get_team_played_history,
    record_deck_feedback,
    save_and_audit_draft,
)
from coldopen.tools.schemas import DeckManifest, SlideItem


def _sample_manifest(team_id: str = "eng-team") -> DeckManifest:
    categories = ["A", "A", "B", "A", "B", "B"]
    items = [
        SlideItem(
            term=f"Specimen {idx + 1}",
            category=cat,  # type: ignore[arg-type]
            category_label="IKEA Product" if cat == "A" else "Death Metal Band",
            subtitle="Verified binary round item",
            speaker_notes=f"Item {idx + 1}: {'IKEA' if cat == 'A' else 'Death Metal'}",
        )
        for idx, cat in enumerate(categories)
    ]
    return DeckManifest(
        team_id=team_id,
        deck_title="IKEA or Death Metal?",
        category_a_name="IKEA Product",
        category_b_name="Death Metal Band",
        items=items,
    )


def test_pii_redaction() -> None:
    raw = "Contact alice@example.com or 555-123-4567 with key AIzaSyDummyKeyValue123456789012345678901"
    cleaned = scrub_pii(raw, use_cloud_dlp=False)
    assert "alice@example.com" not in cleaned
    assert "555-123-4567" not in cleaned
    assert "AIzaSy" not in cleaned
    assert "[REDACTED_EMAIL]" in cleaned
    assert "[REDACTED_PHONE]" in cleaned
    assert "[REDACTED_SECRET]" in cleaned


def test_guided_error_handling_on_empty_inputs() -> None:
    leak_err = audit_headline_leaks([], ["ikea"])
    assert leak_err["ok"] is False
    assert "recovery_instruction" in leak_err

    entropy_err = check_sequence_entropy(["A", "B"])
    assert entropy_err["ok"] is False
    assert "recovery_instruction" in entropy_err


def test_history_compaction() -> None:
    messages = [{"role": "user", "content": f"Turn {i}: brainstorm items"} for i in range(10)]
    compacted = compact_history(messages, max_recent_turns=4, max_token_budget=500)
    assert compacted.dropped_count == 6
    assert len(compacted.recent_messages) == 4
    assert "Earlier session summary:" in compacted.summary_prefix


def test_compile_pptx_and_sqlite_dedup(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    pptx_path = tmp_path / "deck.pptx"
    manifest = _sample_manifest("alpha-team")

    result = compile_pptx_deck(manifest, output_path=str(pptx_path), db_path=str(db_path))
    assert result["ok"] is True
    assert pptx_path.exists()
    assert result["slide_count"] == 14  # 1 title + 6*2 pairs + 1 outro
    assert result["recorded_items"] == 6

    history = get_team_played_history("alpha-team", db_path=str(db_path))
    assert history["count"] == 6
    assert "Specimen 1" in history["played_terms"]  # type: ignore[operator]


def test_async_memory_consolidation(tmp_path: Path) -> None:
    async def _run() -> dict[str, object]:
        store = ColdOpenStore(db_path=tmp_path / "async.db")
        task = schedule_memory_consolidation(store, team_id="beta-team", theme="Font or Cheese", item_count=18)
        return await task

    summary = asyncio.run(_run())
    assert summary["total_decks_generated"] == 1
    assert summary["total_items_played"] == 18


def test_guardrail_and_hitl_gate() -> None:
    allowed, _ = evaluate_policy_guardrail("IKEA or Death Metal for global eng team")
    assert allowed is True

    blocked, reason = evaluate_policy_guardrail("Include someone's ssn and password")
    assert blocked is False
    assert "Blocked" in reason

    manifest = _sample_manifest()
    approved, _ = request_host_approval(manifest, prompt_fn=lambda _: "y")
    assert approved is True
    rejected, _ = request_host_approval(manifest, prompt_fn=lambda _: "n")
    assert rejected is False


def test_save_and_audit_draft_zero_copy_compile(tmp_path: Path) -> None:
    db_path = str(tmp_path / "draft.db")
    pptx_path = str(tmp_path / "from_draft.pptx")
    manifest = _sample_manifest("draft-team")

    saved = save_and_audit_draft(manifest, banned_category_terms=["ikea", "death"], db_path=db_path)
    assert saved["ok"] is True
    assert saved["passed"] is True

    compiled = compile_pptx_from_draft(str(saved["draft_id"]), output_path=pptx_path, db_path=db_path)
    assert compiled["ok"] is True
    assert Path(pptx_path).exists()


def test_title_filename_slug_and_reveal_image(tmp_path: Path, monkeypatch: object) -> None:
    from PIL import Image
    from pptx import Presentation

    img_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), color=(30, 120, 210)).save(img_path)

    manifest = _sample_manifest("img-team")
    manifest.items[0].image_path = str(img_path)
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

    compiled = compile_pptx_deck(manifest, db_path=str(tmp_path / "db.sqlite"))
    assert compiled["ok"] is True
    expected = tmp_path / "ikea_or_death_metal.pptx"
    assert expected.exists()
    prs = Presentation(str(expected))
    assert any(shape.shape_type == 13 for shape in prs.slides[2].shapes)  # 13 = PICTURE


def test_a2ui_toolset_wired_on_root_agent() -> None:
    from a2ui.adk import SendA2uiToClientToolset
    from coldopen.agent import root_agent

    assert any(isinstance(t, SendA2uiToClientToolset) for t in root_agent.tools)


def test_sfw_guardrail_blocks_nsfw_draft(tmp_path: Path) -> None:
    manifest = _sample_manifest("sfw-team")
    manifest.items[0].term = "Explicit adult content item"
    res = save_and_audit_draft(manifest, banned_category_terms=["ikea"], db_path=str(tmp_path / "sfw.db"))
    assert res["ok"] is False
    assert res["stage"] == "sfw_guardrail"


def test_layout_modes_and_feedback_loop(tmp_path: Path) -> None:
    from PIL import Image
    from pptx import Presentation

    db_path = str(tmp_path / "fb.db")
    img_path = tmp_path / "ignored.png"
    Image.new("RGB", (100, 100), color=(10, 20, 30)).save(img_path)

    manifest = _sample_manifest("fb-team")
    manifest.layout_mode = "pill_only"
    manifest.items[0].image_path = str(img_path)
    out_pptx = tmp_path / "pill.pptx"

    compiled = compile_pptx_deck(manifest, output_path=str(out_pptx), db_path=db_path)
    assert compiled["ok"] is True
    prs = Presentation(str(out_pptx))
    assert not any(shape.shape_type == 13 for shape in prs.slides[2].shapes)

    fb = record_deck_feedback("IKEA or Death Metal?", 5, "Great round!", team_id="fb-team", db_path=db_path)
    assert fb["ok"] is True
    history = get_team_played_history("fb-team", db_path=db_path)
    assert len(history["recent_feedback"]) == 1  # type: ignore[arg-type]


def test_a2a_server_and_callback_trajectory() -> None:
    from types import SimpleNamespace
    from google.genai import types
    from coldopen.a2a_server import app
    from coldopen.guardrails.policy import adk_policy_and_compaction_callback

    assert app is not None

    blocked_req = SimpleNamespace(
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="Make an nsfw deck")])]
    )
    resp = adk_policy_and_compaction_callback(SimpleNamespace(), blocked_req)  # type: ignore[arg-type]
    assert resp is not None
    assert "SFW" in resp.content.parts[0].text  # type: ignore[union-attr]


async def test_compile_and_save_pptx_artifact(tmp_path: Path) -> None:
    from coldopen.tools.compiler import PPTX_MIME_TYPE, compile_and_save_pptx_artifact

    db_path = str(tmp_path / "artifact.db")
    pptx_path = str(tmp_path / "artifact_deck.pptx")
    saved = save_and_audit_draft(_sample_manifest("artifact-team"), banned_category_terms=["ikea"], db_path=db_path)
    assert saved["passed"] is True

    captured: list[tuple[str, object]] = []

    class _FakeToolContext:
        async def save_artifact(self, *, filename: str, artifact: object) -> int:
            captured.append((filename, artifact))
            return 0

    result = await compile_and_save_pptx_artifact(
        draft_id=str(saved["draft_id"]),
        tool_context=_FakeToolContext(),  # type: ignore[arg-type]
        output_path=pptx_path,
        db_path=db_path,
    )
    assert result["artifact_filename"] == "artifact_deck.pptx"
    assert result["artifact_version"] == 0
    assert result["mime_type"] == PPTX_MIME_TYPE
    assert len(captured) == 1
    assert captured[0][0] == "artifact_deck.pptx"
    assert captured[0][1].inline_data.mime_type == PPTX_MIME_TYPE  # type: ignore[union-attr]
    assert len(captured[0][1].inline_data.data) > 0  # type: ignore[union-attr]
