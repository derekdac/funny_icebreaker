"""Explicit Pydantic JSON schemas for ColdOpen tools and structured agent outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SlideItem(BaseModel):
    """Single binary guessing round item with reveal metadata and presenter notes."""

    term: str = Field(..., min_length=2, max_length=140, description="Question headline or term shown on the question slide.")
    category: Literal["A", "B"] = Field(..., description="Ground-truth binary bucket ('A' or 'B').")
    category_label: str = Field(..., min_length=2, max_length=60, description="Human-readable category name, e.g. 'IKEA Product' or 'Death Metal Band'.")
    subtitle: str = Field(..., min_length=3, max_length=160, description="Short 1-line descriptor shown on the reveal slide.")
    speaker_notes: str = Field(..., min_length=5, max_length=400, description="Pronunciation, bold category, and 1-line trivia context for the presenter.")
    image_path: str | None = Field(default=None, description="Optional local path to a verified specimen image for the reveal slide.")


class DeckManifest(BaseModel):
    """Complete 15-20 item deck specification ready for audit or compilation."""

    team_id: str = Field(default="default", min_length=1, max_length=80, description="Optional history bucket for deduplicating past decks.")
    deck_title: str = Field(..., min_length=3, max_length=100, description="Deck title, e.g. 'IKEA or Death Metal?'.")
    category_a_name: str = Field(..., min_length=2, max_length=60, description="Display name for category A.")
    category_b_name: str = Field(..., min_length=2, max_length=60, description="Display name for category B.")
    layout_mode: Literal["photo_reveal", "pill_only", "emoji_prompt"] = Field(
        default="photo_reveal",
        description="Visual layout: 'photo_reveal' (image + pill), 'pill_only' (text-only themes), or 'emoji_prompt' (large 84pt emoji questions).",
    )
    cover_image_path: str | None = Field(
        default=None,
        description="Optional local or gs:// path to the AI-generated 16:9 cover slide photo.",
    )
    items: list[SlideItem] = Field(..., min_length=5, max_length=25, description="Ordered list of guessing round items.")


class ToolError(BaseModel):
    """Guided error payload returned to the LLM with actionable recovery steps."""

    ok: Literal[False] = False
    error_code: str = Field(..., description="Machine-readable error identifier.")
    message: str = Field(..., description="Human-readable explanation of what failed.")
    recovery_instruction: str = Field(..., description="Specific step the agent should take to fix the input and retry.")


class LeakAuditResult(BaseModel):
    """Result of scanning question headlines for category giveaways."""

    ok: Literal[True] = True
    passed: bool = Field(..., description="True when zero question terms leak their category.")
    flagged_indices: list[int] = Field(default_factory=list, description="Zero-based indices of items with category leaks.")
    violations: list[str] = Field(default_factory=list, description="Specific leak explanations for each flagged item.")


class EntropyCheckResult(BaseModel):
    """Result of auditing the A/B answer sequence for predictable patterns."""

    ok: Literal[True] = True
    passed: bool = Field(..., description="True when the sequence avoids alternating or repeating patterns.")
    sequence: str = Field(..., description="Compact string representation such as 'AABABBBAA'.")
    reason: str = Field(..., description="Explanation of pass or failure.")


class SpecimenResult(BaseModel):
    """Result of fetching an authentic Wikimedia specimen image."""

    ok: Literal[True] = True
    term: str
    local_path: str | None = Field(default=None, description="Local path to downloaded image if found.")
    source_url: str | None = Field(default=None, description="Canonical Wikimedia image URL.")
    found: bool = Field(..., description="Whether an authentic specimen image was retrieved.")


class CompileDeckResult(BaseModel):
    """Result of compiling the minimalist 16:9 PowerPoint deck."""

    ok: Literal[True] = True
    output_path: str = Field(..., description="Filesystem path to the generated .pptx file.")
    slide_count: int = Field(..., description="Total number of slides generated (title + 2*N + outro).")
    recorded_items: int = Field(..., description="Number of newly recorded items persisted to SQLite.")
