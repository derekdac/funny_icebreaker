"""Context bloat management via sliding-window and summary compaction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompactionResult:
    """Result of compacting conversation history to fit token/turn budgets."""

    summary_prefix: str
    recent_messages: list[dict[str, str]]
    dropped_count: int


def estimate_tokens(text: str) -> int:
    """Fast heuristic token estimator (~4 chars per token)."""
    return max(1, len(text) // 4)


def compact_history(
    messages: list[dict[str, str]],
    *,
    max_recent_turns: int = 6,
    max_token_budget: int = 2500,
) -> CompactionResult:
    """Compact older conversation turns into a concise digest while keeping recent turns intact.

    Args:
        messages: Ordered list of {'role': ..., 'content': ...} dicts.
        max_recent_turns: Maximum number of trailing turns to preserve verbatim.
        max_token_budget: Approximate token ceiling for the combined context.

    Returns:
        CompactionResult containing a digest of older turns and the preserved tail.
    """
    if len(messages) <= max_recent_turns:
        total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        if total_tokens <= max_token_budget:
            return CompactionResult(summary_prefix="", recent_messages=list(messages), dropped_count=0)

    split_idx = max(0, len(messages) - max_recent_turns)
    older = messages[:split_idx]
    recent = list(messages[split_idx:])

    # Trim recent turns further if a single turn is unusually large.
    while len(recent) > 1 and sum(estimate_tokens(m.get("content", "")) for m in recent) > max_token_budget:
        older.append(recent.pop(0))

    digest_lines = [
        f"- {msg.get('role', 'user')}: {msg.get('content', '').strip()[:120]}"
        for msg in older
        if msg.get("content", "").strip()
    ]
    summary_prefix = (
        "Earlier session summary:\n" + "\n".join(digest_lines[-8:])
        if digest_lines
        else ""
    )
    return CompactionResult(
        summary_prefix=summary_prefix,
        recent_messages=recent,
        dropped_count=len(older),
    )
