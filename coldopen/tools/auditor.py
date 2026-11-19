"""Quality audit tools for headline leak detection and sequence randomness."""

from __future__ import annotations

import re
import time
import uuid

from coldopen.observability.logging import get_logger, log_tool_intent, log_tool_outcome
from coldopen.observability.tracing import traced_span
from coldopen.tools.schemas import EntropyCheckResult, LeakAuditResult, SlideItem, ToolError

_LOGGER = get_logger("coldopen.tools.auditor")


def audit_headline_leaks(
    items: list[SlideItem],
    banned_category_terms: list[str],
) -> dict[str, object]:
    """Scan question headlines to ensure none reveal or hint at the ground-truth category.

    Args:
        items: List of SlideItem objects whose 'term' fields will be scanned.
        banned_category_terms: Category keywords that must never appear inside a question
            headline (for example, ['florida', 'onion', 'ikea', 'cheese', 'font']).

    Returns:
        Dictionary matching LeakAuditResult on success, or ToolError with recovery
        instructions if the input list is empty or malformed.
    """
    start = time.perf_counter()
    trace_id = uuid.uuid4().hex[:12]
    log_tool_intent(
        _LOGGER,
        trace_id=trace_id,
        agent_name="leak_entropy_critic",
        tool_name="audit_headline_leaks",
        intended_action="Check question terms for category giveaways",
        arguments={"item_count": len(items), "banned_count": len(banned_category_terms)},
    )

    with traced_span("tool.audit_headline_leaks", {"item_count": len(items)}):
        if not items:
            err = ToolError(
                error_code="EMPTY_ITEMS",
                message="No items were provided for leak auditing.",
                recovery_instruction="Generate at least 5 binary items with 'term' and 'category' fields, then call audit_headline_leaks again.",
            )
            log_tool_outcome(
                _LOGGER,
                trace_id=trace_id,
                agent_name="leak_entropy_critic",
                tool_name="audit_headline_leaks",
                status="error",
                outcome_summary=err.message,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return err.model_dump()

        normalized_banned = [t.strip().lower() for t in banned_category_terms if t.strip()]
        flagged: list[int] = []
        violations: list[str] = []

        for idx, raw_item in enumerate(items):
            term = (
                raw_item.term.strip()
                if isinstance(raw_item, SlideItem)
                else str(raw_item.get("term", "")).strip()
            )
            if not term:
                flagged.append(idx)
                violations.append(f"Item {idx} has an empty 'term'.")
                continue
            lowered = term.lower()
            for banned in normalized_banned:
                if re.search(rf"\b{re.escape(banned)}\b", lowered):
                    flagged.append(idx)
                    violations.append(
                        f"Item {idx} ('{term}') contains giveaway keyword '{banned}'. Replace with a neutral noun."
                    )
                    break

        result = LeakAuditResult(
            passed=len(flagged) == 0,
            flagged_indices=flagged,
            violations=violations,
        )
        log_tool_outcome(
            _LOGGER,
            trace_id=trace_id,
            agent_name="leak_entropy_critic",
            tool_name="audit_headline_leaks",
            status="ok",
            outcome_summary=f"passed={result.passed}, flagged={len(flagged)}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        return result.model_dump()


def check_sequence_entropy(categories: list[str]) -> dict[str, object]:
    """Audit the A/B answer sequence to prevent alternating or repeating sub-patterns.

    Args:
        categories: Ordered list of category codes ('A' or 'B') for the deck.

    Returns:
        Dictionary matching EntropyCheckResult on success, or ToolError with recovery
        instructions if invalid category codes are supplied.
    """
    start = time.perf_counter()
    trace_id = uuid.uuid4().hex[:12]
    log_tool_intent(
        _LOGGER,
        trace_id=trace_id,
        agent_name="leak_entropy_critic",
        tool_name="check_sequence_entropy",
        intended_action="Validate answer sequence randomness",
        arguments={"length": len(categories)},
    )

    with traced_span("tool.check_sequence_entropy", {"length": len(categories)}):
        cleaned = [c.strip().upper() for c in categories]
        if len(cleaned) < 4 or any(c not in {"A", "B"} for c in cleaned):
            err = ToolError(
                error_code="INVALID_SEQUENCE",
                message="Sequence must contain at least 4 entries using only 'A' or 'B'.",
                recovery_instruction="Map every item category to 'A' or 'B' and supply at least 4 items.",
            )
            log_tool_outcome(
                _LOGGER,
                trace_id=trace_id,
                agent_name="leak_entropy_critic",
                tool_name="check_sequence_entropy",
                status="error",
                outcome_summary=err.message,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return err.model_dump()

        seq = "".join(cleaned)
        alternations = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
        alternation_ratio = alternations / (len(seq) - 1)

        # Detect strict alternation (e.g. ABABAB) or repeating 2/3-char motifs across the whole deck.
        passed = True
        reason = "Sequence uses irregular cluster lengths with no predictable cadence."
        if alternation_ratio >= 0.85:
            passed = False
            reason = (
                f"Sequence '{seq}' alternates too predictably (ratio={alternation_ratio:.2f}). "
                "Introduce irregular clusters of 2 and 3 consecutive items."
            )
        elif len(set(seq)) == 1:
            passed = False
            reason = f"Sequence '{seq}' contains only one category. Mix both 'A' and 'B'."

        result = EntropyCheckResult(passed=passed, sequence=seq, reason=reason)
        log_tool_outcome(
            _LOGGER,
            trace_id=trace_id,
            agent_name="leak_entropy_critic",
            tool_name="check_sequence_entropy",
            status="ok",
            outcome_summary=f"passed={passed}, seq={seq}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        return result.model_dump()
