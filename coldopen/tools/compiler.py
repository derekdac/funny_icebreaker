"""Minimalist 16:9 PowerPoint deck compiler and SQLite history recorder."""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from google.adk.tools import ToolContext
from google.genai import types
from PIL import Image

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from pydantic import ValidationError

from coldopen.guardrails.policy import check_manifest_sfw
from coldopen.memory.consolidator import schedule_memory_consolidation
from coldopen.memory.store import ColdOpenStore
from coldopen.observability.logging import get_logger, log_tool_intent, log_tool_outcome
from coldopen.observability.tracing import traced_span
from coldopen.tools.auditor import audit_headline_leaks, check_sequence_entropy
from coldopen.tools.schemas import CompileDeckResult, DeckManifest, ToolError

_LOGGER = get_logger("coldopen.tools.compiler")

_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
_INK = RGBColor(0x20, 0x21, 0x24)
_MUTED = RGBColor(0x5F, 0x63, 0x68)
_CAT_A_COLOR = RGBColor(0x1A, 0x73, 0xE8)  # Crisp blue pill
_CAT_B_COLOR = RGBColor(0xD9, 0x30, 0x25)  # Crisp coral pill


def _set_white_background(slide: object) -> None:
    fill = slide.background.fill  # type: ignore[attr-defined]
    fill.solid()
    fill.fore_color.rgb = _WHITE


def _add_centered_text(
    slide: object,
    text: str,
    *,
    top: float,
    height: float,
    font_size: int,
    bold: bool = True,
    color: RGBColor = _INK,
) -> None:
    box = slide.shapes.add_textbox(Inches(0.8), Inches(top), Inches(11.733), Inches(height))  # type: ignore[attr-defined]
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.name = "Arial"
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color


def _title_to_filename(deck_title: str) -> str:
    """Convert a deck title like 'IKEA or Death Metal?' to 'ikea_or_death_metal.pptx'."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", deck_title).strip("_").lower()
    return f"{slug or 'coldopen_deck'}.pptx"


def _add_centered_picture(slide: object, image_path: str, top_in: float = 1.65, max_h_in: float = 3.7, max_w_in: float = 8.5) -> None:
    """Insert an image scaled proportionally and centered horizontally on a 13.333in slide."""
    with Image.open(image_path) as img:
        w_px, h_px = img.size
    aspect = (w_px / h_px) if h_px else 1.5
    height_in = max_h_in
    width_in = height_in * aspect
    if width_in > max_w_in:
        width_in = max_w_in
        height_in = width_in / aspect
    left_in = (13.333 - width_in) / 2.0
    slide.shapes.add_picture(  # type: ignore[attr-defined]
        image_path,
        Inches(left_in),
        Inches(top_in),
        width=Inches(width_in),
        height=Inches(height_in),
    )


def compile_pptx_deck(
    manifest: DeckManifest,
    output_path: str | None = None,
    db_path: str = "coldopen_state.db",
) -> dict[str, object]:
    """Compile a validated DeckManifest into a minimalist 16:9 PowerPoint deck.

    Args:
        manifest: Structured DeckManifest Pydantic model containing title and items.
        output_path: Destination filesystem path for the compiled .pptx file.
        db_path: SQLite database path for recording played items.

    Returns:
        Dictionary matching CompileDeckResult on success, or ToolError with recovery
        instructions if validation or file writing fails.
    """
    start = time.perf_counter()
    trace_id = uuid.uuid4().hex[:12]
    log_tool_intent(
        _LOGGER,
        trace_id=trace_id,
        agent_name="deck_compiler",
        tool_name="compile_pptx_deck",
        intended_action="Validate manifest, build 16:9 PPTX slides, and record team history",
        arguments={"output_path": output_path},
    )

    with traced_span("tool.compile_pptx_deck", {"output_path": output_path}):
        try:
            validated = (
                manifest
                if isinstance(manifest, DeckManifest)
                else DeckManifest.model_validate(manifest)
            )
        except ValidationError as exc:
            err = ToolError(
                error_code="INVALID_MANIFEST",
                message=f"Manifest validation failed: {exc.error_count()} errors.",
                recovery_instruction="Fix the invalid fields in DeckManifest (ensure 5-25 items with valid 'A'/'B' categories) and retry compile_pptx_deck.",
            )
            log_tool_outcome(
                _LOGGER,
                trace_id=trace_id,
                agent_name="deck_compiler",
                tool_name="compile_pptx_deck",
                status="error",
                outcome_summary=err.message,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return err.model_dump()

        resolved_output = (
            output_path
            if output_path and output_path != "coldopen_deck.pptx"
            else _title_to_filename(validated.deck_title)
        )

        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        blank_layout = prs.slide_layouts[6]

        # 1. Title Slide
        title_slide = prs.slides.add_slide(blank_layout)
        _set_white_background(title_slide)
        _add_centered_text(title_slide, validated.deck_title, top=2.6, height=1.8, font_size=52)

        # 2. Question & Reveal Pairs
        for item in validated.items:
            q_slide = prs.slides.add_slide(blank_layout)
            _set_white_background(q_slide)
            if validated.layout_mode == "emoji_prompt":
                q_font = 84
            else:
                q_font = 56 if len(item.term) <= 28 else 32
            _add_centered_text(q_slide, item.term, top=2.4, height=2.5, font_size=q_font)

            r_slide = prs.slides.add_slide(blank_layout)
            _set_white_background(r_slide)
            _add_centered_text(r_slide, item.term, top=0.55, height=0.95, font_size=34)

            has_image = (
                validated.layout_mode != "pill_only"
                and bool(item.image_path and Path(item.image_path).exists())
            )
            if has_image:
                _add_centered_picture(r_slide, str(item.image_path), top_in=1.65, max_h_in=3.7)
                pill_top = 5.55
            else:
                pill_top = 2.9

            pill = r_slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(4.166),
                Inches(pill_top),
                Inches(5.0),
                Inches(0.68 if has_image else 0.95),
            )
            pill.fill.solid()
            pill.fill.fore_color.rgb = _CAT_A_COLOR if item.category == "A" else _CAT_B_COLOR
            pill.line.fill.background()
            tf = pill.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = item.category_label.upper()
            run.font.name = "Arial"
            run.font.size = Pt(18 if has_image else 24)
            run.font.bold = True
            run.font.color.rgb = _WHITE

            _add_centered_text(
                r_slide,
                item.subtitle,
                top=6.35 if has_image else 6.0,
                height=0.75,
                font_size=18,
                bold=False,
                color=_MUTED,
            )
            r_slide.notes_slide.notes_text_frame.text = item.speaker_notes

        # 3. Outro Slide
        outro = prs.slides.add_slide(blank_layout)
        _set_white_background(outro)
        _add_centered_text(outro, "Thanks for playing!", top=2.6, height=1.8, font_size=52)

        out = Path(resolved_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        prs.save(str(out))

        store = ColdOpenStore(db_path=db_path)
        recorded = store.record_played_items(
            team_id=validated.team_id,
            theme=validated.deck_title,
            items=[{"term": i.term, "category": i.category} for i in validated.items],
        )

        try:
            asyncio.get_running_loop()
            schedule_memory_consolidation(
                store,
                team_id=validated.team_id,
                theme=validated.deck_title,
                item_count=len(validated.items),
            )
        except RuntimeError:
            pass  # No running event loop in synchronous CLI/unit-test calls

        result = CompileDeckResult(
            output_path=str(out),
            slide_count=len(prs.slides),
            recorded_items=recorded,
        )
        log_tool_outcome(
            _LOGGER,
            trace_id=trace_id,
            agent_name="deck_compiler",
            tool_name="compile_pptx_deck",
            status="ok",
            outcome_summary=f"slides={result.slide_count}, recorded={recorded}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        return result.model_dump()


def get_team_played_history(team_id: str = "default", db_path: str = "coldopen_state.db") -> dict[str, object]:
    """Retrieve previously played terms and recent host feedback to prevent repeats and improve quality.

    Args:
        team_id: Unique identifier for the team or recurring meeting.
        db_path: SQLite database path.

    Returns:
        Dictionary containing played_terms, count, and recent_feedback.
    """
    store = ColdOpenStore(db_path=db_path)
    terms = store.get_played_terms(team_id)
    feedback = store.get_recent_feedback(team_id)
    return {
        "ok": True,
        "team_id": team_id,
        "played_terms": terms,
        "count": len(terms),
        "recent_feedback": feedback,
    }


def record_deck_feedback(
    deck_title: str,
    rating: int,
    notes: str,
    team_id: str = "default",
    db_path: str = "coldopen_state.db",
) -> dict[str, object]:
    """Store post-game host feedback (1-5 rating and notes) in SQLite memory.

    Args:
        deck_title: Title of the deck being rated.
        rating: Integer score from 1 (poor) to 5 (great).
        notes: Specific feedback on which items landed or missed.
        team_id: Optional history bucket identifier.
        db_path: SQLite database path.

    Returns:
        Confirmation dictionary or ToolError if rating is out of range.
    """
    if rating < 1 or rating > 5:
        return ToolError(
            error_code="INVALID_RATING",
            message=f"Rating must be between 1 and 5, got {rating}.",
            recovery_instruction="Ask the user for a rating from 1 to 5 and retry record_deck_feedback.",
        ).model_dump()
    ColdOpenStore(db_path=db_path).record_feedback(team_id, deck_title, rating, notes)
    return {"ok": True, "deck_title": deck_title, "rating": rating}


def save_and_audit_draft(
    manifest: DeckManifest,
    banned_category_terms: list[str],
    db_path: str = "coldopen_state.db",
) -> dict[str, object]:
    """Validate SFW safety, leak-audit, entropy-check, and persist a draft manifest under a draft_id.

    Args:
        manifest: Complete DeckManifest proposed by the Ideator.
        banned_category_terms: Category keywords that must not appear in question headlines.
        db_path: SQLite database path for zero-copy draft handoff.

    Returns:
        Dictionary with draft_id and audit status, or ToolError/violations if checks fail.
    """
    validated = manifest if isinstance(manifest, DeckManifest) else DeckManifest.model_validate(manifest)
    is_sfw, sfw_reason = check_manifest_sfw(validated)
    if not is_sfw:
        return {
            "ok": False,
            "passed": False,
            "stage": "sfw_guardrail",
            "details": sfw_reason,
            "recovery_instruction": "Replace any NSFW, profane, or workplace-unsafe items and call save_and_audit_draft again.",
        }

    leak_res = audit_headline_leaks(validated.items, banned_category_terms)
    if not leak_res.get("ok") or not leak_res.get("passed"):
        return {
            "ok": False,
            "passed": False,
            "stage": "leak_audit",
            "details": leak_res,
            "recovery_instruction": "Fix the flagged headlines so they contain no category keywords, then call save_and_audit_draft again.",
        }

    entropy_res = check_sequence_entropy([item.category for item in validated.items])
    if not entropy_res.get("ok") or not entropy_res.get("passed"):
        return {
            "ok": False,
            "passed": False,
            "stage": "entropy_check",
            "details": entropy_res,
            "recovery_instruction": "Reorder items into irregular A/B clusters (runs of 1, 2, and 3) and call save_and_audit_draft again.",
        }

    draft_id = f"draft-{uuid.uuid4().hex[:8]}"
    ColdOpenStore(db_path=db_path).save_draft_manifest(draft_id, validated.model_dump())
    return {
        "ok": True,
        "passed": True,
        "draft_id": draft_id,
        "deck_title": validated.deck_title,
        "item_count": len(validated.items),
    }


def compile_pptx_from_draft(
    draft_id: str,
    output_path: str | None = None,
    db_path: str = "coldopen_state.db",
) -> dict[str, object]:
    """Compile a saved draft_id into a 16:9 PowerPoint deck without re-serializing JSON.

    Args:
        draft_id: Identifier returned by save_and_audit_draft.
        output_path: Destination filesystem path for the compiled .pptx file.
        db_path: SQLite database path storing the draft and played history.

    Returns:
        Dictionary matching CompileDeckResult on success, or ToolError if draft_id is missing.
    """
    manifest_dict = ColdOpenStore(db_path=db_path).get_draft_manifest(draft_id)
    if not manifest_dict:
        return ToolError(
            error_code="DRAFT_NOT_FOUND",
            message=f"No draft manifest found for draft_id '{draft_id}'.",
            recovery_instruction="Run save_and_audit_draft first to obtain a valid draft_id.",
        ).model_dump()
    return compile_pptx_deck(
        DeckManifest.model_validate(manifest_dict),
        output_path=output_path,
        db_path=db_path,
    )


PPTX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def _upload_pptx_to_gcs(pptx_path: Path, bucket_name: str, draft_id: str) -> str | None:
    """Optionally upload compiled PPTX to GCS and return an authenticated download URL."""
    try:
        from google.cloud import storage

        client = storage.Client()
        blob_name = f"coldopen-decks/{draft_id}/{pptx_path.name}"
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_filename(str(pptx_path), content_type=PPTX_MIME_TYPE)
        return f"https://storage.cloud.google.com/{bucket_name}/{blob_name}"
    except Exception as exc:  # pragma: no cover - optional cloud export
        _LOGGER.warning("GCS export skipped for %s: %s", pptx_path.name, exc)
        return None


async def compile_and_save_pptx_artifact(
    draft_id: str,
    tool_context: ToolContext,
    output_path: str | None = None,
    db_path: str = "coldopen_state.db",
) -> dict[str, Any]:
    """Compile a saved draft_id to PPTX and publish it as a downloadable ADK artifact.

    Args:
        draft_id: Identifier returned by save_and_audit_draft.
        tool_context: ADK ToolContext used to save the compiled deck as a session artifact.
        output_path: Optional destination filesystem path for the compiled .pptx file.
        db_path: SQLite database path storing the draft and played history.

    Returns:
        CompileDeckResult fields enriched with artifact_filename, artifact_version, and
        optional download_url (when COLDOPEN_EXPORT_BUCKET is configured).
    """
    result = compile_pptx_from_draft(
        draft_id=draft_id,
        output_path=output_path,
        db_path=db_path,
    )
    if "error_code" in result:
        return result

    pptx_path = Path(str(result["output_path"]))
    pptx_bytes = pptx_path.read_bytes()
    artifact_part = types.Part.from_bytes(data=pptx_bytes, mime_type=PPTX_MIME_TYPE)
    version = await tool_context.save_artifact(
        filename=pptx_path.name,
        artifact=artifact_part,
    )
    result["artifact_filename"] = pptx_path.name
    result["artifact_version"] = version
    result["mime_type"] = PPTX_MIME_TYPE

    export_bucket = os.getenv("COLDOPEN_EXPORT_BUCKET", "").strip()
    if export_bucket:
        download_url = _upload_pptx_to_gcs(pptx_path, export_bucket, draft_id)
        if download_url:
            result["download_url"] = download_url

    return result
