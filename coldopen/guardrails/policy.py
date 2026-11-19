"""Guardrails, content policy checks, and Human-in-the-Loop (HITL) approval hooks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

from coldopen.memory.compaction import compact_history, summarize_turns
from coldopen.observability.logging import get_logger
from coldopen.tools.schemas import DeckManifest

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models import LlmRequest, LlmResponse

_LOGGER = get_logger("coldopen.guardrails")

_BLOCKED_PATTERNS = [
    re.compile(
        r"\b(?:nsfw|explicit|porn|erotic|nude|nudity|profanity|gore|beheading|"
        r"cocaine|heroin|methamphetamine|ssn|password|secret_key)\b",
        re.IGNORECASE,
    ),
]


def evaluate_policy_guardrail(text: str) -> tuple[bool, str]:
    """Screen prompts and generated items against workplace safety and data-leakage rules.

    Args:
        text: Input brief or generated item text to validate.

    Returns:
        Tuple of (is_allowed, reason).
    """
    for pattern in _BLOCKED_PATTERNS:
        match = pattern.search(text)
        if match:
            reason = f"Blocked by SFW/policy guardrail: disallowed term '{match.group(0)}'."
            _LOGGER.warning("policy_guardrail_triggered", extra={"structured_fields": {"reason": reason}})
            return False, reason
    return True, "ok"


def check_manifest_sfw(manifest: DeckManifest) -> tuple[bool, str]:
    """Validate every item in a DeckManifest for workplace Safe-For-Work compliance.

    Args:
        manifest: Draft DeckManifest to inspect before saving or compiling.

    Returns:
        Tuple of (is_safe, violation_reason).
    """
    ok, reason = evaluate_policy_guardrail(manifest.deck_title)
    if not ok:
        return False, reason
    for idx, item in enumerate(manifest.items):
        combined = f"{item.term} {item.category_label} {item.subtitle} {item.speaker_notes}"
        ok, reason = evaluate_policy_guardrail(combined)
        if not ok:
            return False, f"Item {idx} failed SFW check: {reason}"
    return True, "ok"


def request_host_approval(
    manifest: DeckManifest,
    prompt_fn: Callable[[str], str] | None = None,
    auto_approve: bool = False,
) -> tuple[bool, str]:
    """Human-in-the-Loop gate requiring explicit host sign-off before asset sourcing and compilation.

    Args:
        manifest: Audited DeckManifest ready for host review.
        prompt_fn: Optional input function (defaults to built-in input) for interactive CLI testing.
        auto_approve: Explicit flag for non-interactive automated demos.

    Returns:
        Tuple of (approved, host_feedback).
    """
    if auto_approve:
        _LOGGER.info("hitl_auto_approved", extra={"structured_fields": {"title": manifest.deck_title}})
        return True, "approved"

    preview = "\n".join(
        f"  {idx + 1:02d}. [{item.category}] {item.term}"
        for idx, item in enumerate(manifest.items)
    )
    message = (
        f"\n--- Host Approval Required: {manifest.deck_title} ({len(manifest.items)} items) ---\n"
        f"{preview}\n"
        "Approve this manifest for image sourcing and PPTX compilation? [y/N]: "
    )
    reader = prompt_fn or input
    reply = reader(message).strip().lower()
    approved = reply in {"y", "yes"}
    _LOGGER.info(
        "hitl_decision",
        extra={"structured_fields": {"approved": approved, "item_count": len(manifest.items)}},
    )
    return approved, ("approved" if approved else reply or "rejected")


def _is_clean_user_turn(content: object) -> bool:
    """Return True if content is a user turn containing no function_response/function_call parts."""
    role = (getattr(content, "role", None) or "").lower()
    if role != "user":
        return False
    parts = getattr(content, "parts", None) or []
    return not any(
        getattr(p, "function_response", None) is not None
        or getattr(p, "function_call", None) is not None
        for p in parts
    )


def _find_safe_split_index(contents: list[object], *, max_recent_turns: int = 6) -> int | None:
    """Find a split index near the last `max_recent_turns` turns starting on a clean user turn.

    Searches backward from the target index down to index 1 first (preferring to preserve
    the full active tool-call chain), then forward to the end of `contents` if needed.
    Returns None if no safe boundary > 0 exists.
    """
    target_idx = max(1, len(contents) - max_recent_turns)
    for idx in range(target_idx, 0, -1):
        if _is_clean_user_turn(contents[idx]):
            return idx
    for idx in range(target_idx + 1, len(contents)):
        if _is_clean_user_turn(contents[idx]):
            return idx
    return None


def adk_policy_and_compaction_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """ADK before_model_callback enforcing safety guardrails and sliding-window compaction.

    Args:
        callback_context: Active ADK callback context for the current agent turn.
        llm_request: Outgoing LLM request containing conversation contents.

    Returns:
        An immediate LlmResponse if blocked by policy guardrails, or None to proceed
        with compacted contents.
    """
    from google.adk.models import LlmResponse
    from google.genai import types

    contents = list(llm_request.contents or [])
    if contents:
        last_parts = getattr(contents[-1], "parts", None) or []
        latest_text = " ".join(p.text for p in last_parts if getattr(p, "text", None))
        allowed, reason = evaluate_policy_guardrail(latest_text)
        if not allowed:
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=reason)],
                )
            )

    if len(contents) > 8:
        split_idx = _find_safe_split_index(contents, max_recent_turns=6)
        if split_idx is not None and split_idx > 0:
            serialized = [
                {
                    "role": getattr(c, "role", "user") or "user",
                    "content": " ".join(
                        p.text for p in (getattr(c, "parts", None) or []) if getattr(p, "text", None)
                    ),
                }
                for c in contents
            ]
            compacted = compact_history(
                serialized,
                max_recent_turns=len(contents) - split_idx,
                max_token_budget=2500,
            )
            final_split_idx = split_idx
            if compacted.dropped_count > split_idx:
                for idx in range(compacted.dropped_count, len(contents)):
                    if _is_clean_user_turn(contents[idx]):
                        final_split_idx = idx
                        break

            summary_prefix = summarize_turns(serialized[:final_split_idx]).strip()
            kept_contents = list(contents[final_split_idx:])
            if summary_prefix:
                first_kept = kept_contents[0]
                if (getattr(first_kept, "role", None) or "").lower() == "user":
                    first_parts = list(getattr(first_kept, "parts", None) or [])
                    merged_first = types.Content(
                        role=getattr(first_kept, "role", "user") or "user",
                        parts=[types.Part.from_text(text=summary_prefix), *first_parts],
                    )
                    llm_request.contents = [merged_first, *kept_contents[1:]]
                else:
                    prefix = types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=summary_prefix)],
                    )
                    llm_request.contents = [prefix, *kept_contents]
            else:
                llm_request.contents = kept_contents

            _LOGGER.info(
                "adk_history_compacted",
                extra={"structured_fields": {"dropped_turns": final_split_idx}},
            )
    return None
