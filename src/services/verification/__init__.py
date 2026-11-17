"""Automated verification lane for assembled designs."""

from __future__ import annotations

from src.services.verification.verifier import (
    AxeScore,
    Finding,
    VerificationReport,
    axe_accessibility_score,
    axe_score_html,
    parse_axe_summary,
    report_from_dict,
    verify_html,
)

__all__ = [
    "AxeScore",
    "Finding",
    "VerificationReport",
    "axe_accessibility_score",
    "axe_score_html",
    "parse_axe_summary",
    "report_from_dict",
    "verify_html",
]
