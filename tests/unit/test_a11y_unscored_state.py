"""An accessibility assessment that did not happen must not report a number.

``_parse_report`` used to return ``AccessibilityReport(overall_score=50.0)`` for an
unparseable LLM reply, and ``float(...get("score", 50))`` for a reply that merely
omitted the key. Both manufacture a measurement: 50.0 is persisted to
``redesign_jobs.accessibility_score``, sent to the webhook, and — being below the
70.0 auto-fix threshold — drives the fixer to rewrite the WHOLE document from a
prompt whose issue list literally reads "No critical issues found."

UNSCORED is now its own state: ``overall_score is None`` + ``unscored_reason``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

import pytest

from src.services.accessibility_scorer import AccessibilityScorer

if TYPE_CHECKING:
    from w2v_common.clients import WMClient


class _StubWMResponse:
    def __init__(self, output: object) -> None:
        self.output = output
        self.usage_metrics = {"prompt_tokens": 1, "completion_tokens": 2}


class _ScriptedWM:
    """Replies with ``reply`` to scoring calls and records every prompt."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[str] = []

    async def message(self, **kwargs: object) -> _StubWMResponse:
        self.prompts.append(str(kwargs.get("message", "")))
        return _StubWMResponse(self._reply)


def _scorer(reply: str) -> tuple[AccessibilityScorer, _ScriptedWM]:
    wm = _ScriptedWM(reply)
    return AccessibilityScorer(wm_client=cast("WMClient", wm)), wm


# --------------------------------------------------------------------------- #
# 1. Unparseable reply
# --------------------------------------------------------------------------- #


def test_unparseable_llm_reply_is_unscored_not_fifty(caplog):
    scorer, _ = _scorer("")
    with caplog.at_level(logging.ERROR):
        report = scorer._parse_report("I am sorry, I cannot evaluate this page.")

    assert report.overall_score is None, (
        "an unparseable reply must not manufacture a score in the real score domain"
    )
    assert report.unscored_reason is not None
    assert report.to_dict()["overall_score"] is None
    assert any("UNSCORED" in r.message for r in caplog.records if r.levelno >= logging.ERROR)


def test_mis_sliced_fenced_block_is_unscored():
    """The ``IndexError`` branch the original code explicitly anticipated."""
    scorer, _ = _scorer("")
    report = scorer._parse_report("here you go: ```json")
    assert report.overall_score is None


# --------------------------------------------------------------------------- #
# 2. Valid JSON that OMITS the keys
# --------------------------------------------------------------------------- #


def test_valid_json_without_overall_score_is_unscored():
    scorer, _ = _scorer("")
    report = scorer._parse_report(json.dumps({"categories": []}))
    assert report.overall_score is None
    assert report.unscored_reason is not None


def test_category_without_score_is_unscored_not_fifty():
    scorer, _ = _scorer("")
    report = scorer._parse_report(
        json.dumps({"overall_score": 88, "categories": [{"category": "alt_text"}]})
    )
    assert report.overall_score == 88.0
    assert report.categories[0].score is None, (
        "a category whose score the LLM omitted must not be reported as 50"
    )


# --------------------------------------------------------------------------- #
# 3. An unscored report must NOT enter the auto-fix loop
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unscored_report_does_not_trigger_the_auto_fixer(caplog):
    """50.0 < 70.0 used to send the whole document to be rewritten blind."""
    html = "<html><body><h1>Hi</h1></body></html>"
    scorer, wm = _scorer("not json at all")

    with caplog.at_level(logging.ERROR):
        out_html, report = await scorer.score_and_fix(html, max_attempts=2, threshold=70.0)

    fix_prompts = [p for p in wm.prompts if p.startswith("Fix the following accessibility issues")]
    assert fix_prompts == [], "the auto-fixer must not run on a page that was never scored"
    assert out_html == html, "the original document must survive an unscored assessment"
    assert report.overall_score is None
    assert report.fix_skipped_reason is not None
    assert any("auto-fix SKIPPED" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_genuinely_low_score_still_enters_the_auto_fixer():
    """The measured path is unchanged: a real 10.0 does trigger the fix."""
    html = "<html><body><h1>Hi</h1></body></html>"
    scorer, wm = _scorer(
        json.dumps(
            {
                "overall_score": 10,
                "categories": [
                    {
                        "category": "alt_text",
                        "score": 10,
                        "findings": [{"severity": "critical", "description": "no alt"}],
                    }
                ],
            }
        )
    )
    await scorer.score_and_fix(html, max_attempts=2, threshold=70.0)
    fix_prompts = [p for p in wm.prompts if p.startswith("Fix the following accessibility issues")]
    assert len(fix_prompts) == 1
