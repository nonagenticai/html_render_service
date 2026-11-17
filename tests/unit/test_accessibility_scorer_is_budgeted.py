"""The accessibility scorer's LLM call runs under the job's generation budget.

THE DEFECT this locks (#130). ``AccessibilityScorer._call_llm`` issued its WM call
directly, with no ``_budget_window``/``GenerationBudget`` in scope at all — the seam
``HtmlGenerator`` uses for every other LLM dispatch. A job could exceed its own
deadline in the accessibility phase (which runs AFTER generation, in the same job),
and the LLM spend for that tail was invisible to the budget accounting.

THE FIX'S CONTRACT, locked here: an exhausted budget SKIPS the scorer call — it never
raises out of ``score``/``score_and_fix`` — because the design is already generated
and paid for (see ``accessibility_scorer.py``'s own precedent for
``MAX_FIX_INPUT_CHARS``: "failing the job is worse ... SKIP the fix ... say so
loudly"). The skip must be visible: ``unscored_reason`` / ``fix_skipped_reason``.

Hermetic: a scripted worker_management, no DB, no net.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from src.services.accessibility_scorer import AccessibilityScorer
from src.services.html_gen.section_run import GenerationBudget

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

_SCORE_JSON = """```json
{"overall_score": 40, "categories": [
  {"category": "alt_text", "score": 40, "findings": [
    {"severity": "major", "description": "missing alt", "recommendation": "add alt"}
  ]}
]}
```"""

_FIXED_HTML = "```html\n<html><body><h1>ok</h1></body></html>\n```"


class _Resp:
    def __init__(self, output: object) -> None:
        self.output = output
        self.usage_metrics = {"prompt_tokens": 1, "completion_tokens": 1}


class _WM:
    def __init__(self, outputs: list[object]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict] = []

    async def message(self, **kwargs: object) -> _Resp:
        self.calls.append(kwargs)
        return _Resp(self._outputs.pop(0))


def _spent_budget() -> GenerationBudget:
    """A budget whose deadline is already reached (less than MIN_CALL_SECONDS left)."""
    return GenerationBudget(deadline=100.0, clock=lambda: 100.0)


def _scorer(wm: _WM) -> AccessibilityScorer:
    return AccessibilityScorer(wm_client=cast("WMClient", wm))


# ── the seam is consulted at all ──────────────────────────────────────────────


async def test_score_consumes_the_job_budget() -> None:
    wm = _WM([_SCORE_JSON])
    budget = GenerationBudget()

    report = await _scorer(wm).score("<html></html>", budget=budget)

    assert report.overall_score == 40.0
    assert budget.calls == 1, "the scorer's LLM call was not accounted against the job budget"


async def test_the_auto_fix_call_also_consumes_the_job_budget() -> None:
    wm = _WM([_SCORE_JSON, _FIXED_HTML, _SCORE_JSON.replace("40", "95")])
    budget = GenerationBudget()

    _, report = await _scorer(wm).score_and_fix("<html></html>", threshold=70.0, budget=budget)

    assert budget.calls == 3, "score + fix + rescore must all spend the job budget"
    assert report.html_was_fixed is True


# ── exhaustion SKIPS, it never raises out of score()/score_and_fix() ──────────


async def test_score_is_skipped_not_raised_when_the_budget_is_exhausted() -> None:
    wm = _WM([_SCORE_JSON])  # never reached

    report = await _scorer(wm).score("<html></html>", budget=_spent_budget())

    assert wm.calls == [], "a scoring call started after the job deadline"
    assert report.overall_score is None
    assert report.unscored_reason is not None
    assert "budget" in report.unscored_reason.lower()


async def test_score_and_fix_skips_scoring_when_the_budget_is_exhausted() -> None:
    wm = _WM([_SCORE_JSON])  # never reached

    html_out, report = await _scorer(wm).score_and_fix(
        "<html>original</html>", threshold=70.0, budget=_spent_budget()
    )

    assert wm.calls == [], "a scoring call started after the job deadline"
    assert html_out == "<html>original</html>", "the original document must survive untouched"
    assert report.fix_skipped_reason is not None
    assert "budget" in report.fix_skipped_reason.lower()


async def test_the_auto_fix_call_is_skipped_when_the_budget_runs_out_mid_loop() -> None:
    """The initial score succeeds (below threshold); the budget dies before the fix call."""
    calls = {"n": 0}

    def _clock() -> float:
        # First spend (n=0->1) sees plenty of time; every later spend sees none.
        calls["n"] += 1
        return 0.0 if calls["n"] <= 1 else 100.0

    budget = GenerationBudget(deadline=100.0, clock=_clock)
    wm = _WM([_SCORE_JSON, _FIXED_HTML])  # fix call never reached

    html_out, report = await _scorer(wm).score_and_fix(
        "<html>original</html>", threshold=70.0, budget=budget
    )

    assert len(wm.calls) == 1, "the auto-fix call ran after the job budget was exhausted"
    assert html_out == "<html>original</html>", "a paid, already-generated design must survive"
    assert report.fix_skipped_reason is not None
    assert "budget" in report.fix_skipped_reason.lower()


async def test_no_budget_means_unbounded_back_compat() -> None:
    """``budget=None`` (the pre-#130 default) must keep working, unbounded."""
    wm = _WM([_SCORE_JSON])

    report = await _scorer(wm).score("<html></html>")

    assert report.overall_score == 40.0
    assert len(wm.calls) == 1
