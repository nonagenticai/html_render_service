"""The accessibility auto-fix must never store a truncated document.

``_build_fix_prompt`` embedded ``html[:50000]``, so any page over 50k chars was
handed to the LLM cut mid-tag. Whatever came back replaced the *whole* page:
``score_and_fix`` returns it, ``finalize_phases._phase_accessibility`` stores it as
``redesign.html`` and the API serves it. The deterministic validation ran EARLIER
(``service.py`` :215) and is never re-run over the fixed output — so a document
ending in ``<div class=`` shipped as a completed job.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from src.services.accessibility_scorer import MAX_FIX_INPUT_CHARS, AccessibilityScorer

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

_LOW_SCORE = json.dumps(
    {
        "overall_score": 10,
        "categories": [
            {
                "category": "alt_text",
                "score": 10,
                "findings": [
                    {
                        "severity": "critical",
                        "description": "images lack alt",
                        "recommendation": "add alt",
                    }
                ],
            }
        ],
    }
)


class _StubWMResponse:
    def __init__(self, output: object) -> None:
        self.output = output
        self.usage_metrics = {"prompt_tokens": 1, "completion_tokens": 2}


class _ScriptedWM:
    """Returns the low score for every scoring call; ``fix_reply`` for fix calls."""

    def __init__(self, fix_reply: str) -> None:
        self._fix_reply = fix_reply
        self.prompts: list[str] = []

    async def message(self, **kwargs: object) -> _StubWMResponse:
        prompt = str(kwargs.get("message", ""))
        self.prompts.append(prompt)
        if prompt.startswith("Fix the following accessibility issues"):
            return _StubWMResponse(self._fix_reply)
        return _StubWMResponse(f"```json\n{_LOW_SCORE}\n```")

    @property
    def fix_prompts(self) -> list[str]:
        return [p for p in self.prompts if p.startswith("Fix the following accessibility issues")]


def _oversize_html() -> str:
    filler = "<p>lorem ipsum dolor sit amet consectetur</p>\n" * 2000
    html = f"<!DOCTYPE html><html><body><h1>Big</h1>\n{filler}</body></html>"
    assert len(html) > MAX_FIX_INPUT_CHARS
    return html


@pytest.mark.asyncio
async def test_oversize_html_is_never_sent_truncated_to_the_fixer() -> None:
    html = _oversize_html()
    wm = _ScriptedWM("```html\n<html><body>whatever</body></html>\n```")
    scorer = AccessibilityScorer(wm_client=cast("WMClient", wm))

    out, report = await scorer.score_and_fix(html, threshold=70.0)

    assert wm.fix_prompts == [], "oversize HTML must not be handed to the auto-fixer at all"
    assert out == html, "the original (valid) document must be preserved"
    assert report.html_was_fixed is False


@pytest.mark.asyncio
async def test_a_fix_truncated_mid_tag_is_rejected() -> None:
    html = "<!DOCTYPE html><html><body><h1>Small</h1><img src='a.png'></body></html>"
    # A reply cut mid-tag — exactly what the truncated-input fix produced.
    wm = _ScriptedWM("```html\n<!DOCTYPE html><html><body><h1>Small</h1><div class=\n```")
    scorer = AccessibilityScorer(wm_client=cast("WMClient", wm))

    out, report = await scorer.score_and_fix(html, threshold=70.0)

    assert out == html, "a truncated fix must not replace the page"
    assert report.html_was_fixed is False


@pytest.mark.asyncio
async def test_a_well_formed_fix_is_still_accepted() -> None:
    html = "<!DOCTYPE html><html><body><h1>Small</h1><img src='a.png'></body></html>"
    fixed = "<!DOCTYPE html><html><body><h1>Small</h1><img src='a.png' alt='a'></body></html>"
    wm = _ScriptedWM(f"```html\n{fixed}\n```")
    scorer = AccessibilityScorer(wm_client=cast("WMClient", wm))

    out, report = await scorer.score_and_fix(html, threshold=70.0)

    assert out == fixed
    assert report.html_was_fixed is True


def test_fix_prompt_embeds_the_whole_document() -> None:
    """No silent slicing left in the prompt builder."""
    from src.services.accessibility_scorer import AccessibilityReport

    scorer = AccessibilityScorer(wm_client=cast("WMClient", _ScriptedWM("")))
    html = "<html><body>" + ("x" * (MAX_FIX_INPUT_CHARS + 500)) + "</body></html>"
    prompt = scorer._build_fix_prompt(html, AccessibilityReport(overall_score=10.0))
    assert html in prompt
