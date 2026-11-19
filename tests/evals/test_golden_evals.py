"""Golden-dataset regression evaluation suite for ColdOpen quality gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coldopen.tools.auditor import audit_headline_leaks, check_sequence_entropy

_GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
_GOLDEN = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _GOLDEN["leak_eval_cases"], ids=lambda c: c["name"])
def test_golden_headline_leak_eval(case: dict[str, object]) -> None:
    result = audit_headline_leaks(
        items=case["items"],  # type: ignore[arg-type]
        banned_category_terms=case["banned_terms"],  # type: ignore[arg-type]
    )
    assert result["ok"] is True
    assert result["passed"] is case["expected_passed"]


@pytest.mark.parametrize("case", _GOLDEN["entropy_eval_cases"], ids=lambda c: c["name"])
def test_golden_sequence_entropy_eval(case: dict[str, object]) -> None:
    result = check_sequence_entropy(categories=case["sequence"])  # type: ignore[arg-type]
    assert result["ok"] is True
    assert result["passed"] is case["expected_passed"]
