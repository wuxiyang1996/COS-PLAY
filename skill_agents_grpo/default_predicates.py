"""
Shared default predicate extractor used by stage3_mvp and contract_verification.

Single source of truth for the no-op fallback so callers can plug in
domain-specific extractors (vision, rule-based, LLM, etc.).
"""

from __future__ import annotations

from typing import Any, Dict


def default_extract_predicates(obs: Any) -> Dict[str, float]:
    """Fallback extractor: returns empty predicates.

    Replace with a domain-specific extractor for real use.
    """
    return {}
