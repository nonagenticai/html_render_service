"""LLM-based accessibility scoring and auto-fix routed through worker_management."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.llm_client import get_wm_client
from src.llm_credentials import LLMConfigLike, model_override_from
from src.services.html_gen.html_utils import _reject_reason
from src.services.html_gen.section_run import BudgetExhausted, GenerationBudget, budget_window

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

logger = logging.getLogger(__name__)

# Largest document we are willing to hand to the auto-fixer. The fix prompt used to
# embed ``html[:50000]``, so anything larger was "fixed" from a document cut mid-tag
# and the truncated result replaced the WHOLE page. Chunking is not an option here —
# an a11y fix is document-scoped (heading order, landmarks, contrast) and splitting
# the tree would produce fragments the model re-wraps into new documents. Failing the
# job is worse: the design is already generated and paid for, and the auto-fix is an
# advisory improvement, not a correctness requirement. So: SKIP the fix for oversize
# input, keep the original valid document, and say so loudly in the report + logs.
MAX_FIX_INPUT_CHARS = 50_000

ACCESSIBILITY_CATEGORIES = [
    "color_contrast",
    "alt_text",
    "heading_hierarchy",
    "keyboard_navigation",
    "aria_labels",
    "form_labels",
]


@dataclass
class AccessibilityFinding:
    """A single accessibility issue."""

    category: str
    severity: str  # critical, major, minor
    description: str
    element: str | None = None
    recommendation: str | None = None
    wcag_criterion: str | None = None


@dataclass
class CategoryScore:
    """Score for a single accessibility category."""

    category: str
    # 0-100, or None when the LLM reply omitted this category's score. A default
    # number here (it used to be 50) is indistinguishable from a measured 50.
    score: float | None
    findings: list[AccessibilityFinding] = field(default_factory=list)


@dataclass
class AccessibilityReport:
    """Complete accessibility assessment report."""

    # ``None`` means UNSCORED — the assessment did not happen (unparseable LLM
    # reply, or a reply that omitted the score). It is deliberately NOT a number:
    # the previous 50.0 fallback landed in the same domain as a real score, was
    # persisted to `redesign_jobs.accessibility_score`, sent to the webhook, and
    # (being < the 70.0 threshold) drove the auto-fixer to rewrite the whole
    # document from an issue list reading "No critical issues found."
    overall_score: float | None
    categories: list[CategoryScore] = field(default_factory=list)
    findings: list[AccessibilityFinding] = field(default_factory=list)
    html_was_fixed: bool = False
    # Why the auto-fix did not run / was rejected. Surfaced so a completed job
    # visibly says "the fix was skipped" instead of silently looking unfixable.
    fix_skipped_reason: str | None = None
    # Why there is no score. Set iff ``overall_score is None``.
    unscored_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "unscored_reason": self.unscored_reason,
            "categories": [
                {
                    "category": c.category,
                    "score": c.score,
                    "findings": [
                        {
                            "category": f.category,
                            "severity": f.severity,
                            "description": f.description,
                            "element": f.element,
                            "recommendation": f.recommendation,
                            "wcag_criterion": f.wcag_criterion,
                        }
                        for f in c.findings
                    ],
                }
                for c in self.categories
            ],
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "description": f.description,
                    "element": f.element,
                    "recommendation": f.recommendation,
                    "wcag_criterion": f.wcag_criterion,
                }
                for f in self.findings
            ],
            "html_was_fixed": self.html_was_fixed,
            "fix_skipped_reason": self.fix_skipped_reason,
        }


_SCORING_PROMPT = """Evaluate the following HTML for WCAG 2.1 AA accessibility compliance.

Analyze these 6 categories and return a JSON object:

1. **color_contrast**: Text/background contrast ratios
2. **alt_text**: Image alt attributes
3. **heading_hierarchy**: Proper h1-h6 nesting
4. **keyboard_navigation**: Focus management, tab order, skip links
5. **aria_labels**: ARIA attributes on interactive elements
6. **form_labels**: Form input labels and associations

Return ONLY a JSON object in this exact format:
```json
{{
  "overall_score": <0-100>,
  "categories": [
    {{
      "category": "<category_name>",
      "score": <0-100>,
      "findings": [
        {{
          "severity": "<critical|major|minor>",
          "description": "<issue description>",
          "element": "<the problematic HTML element or selector>",
          "recommendation": "<how to fix>",
          "wcag_criterion": "<WCAG criterion, e.g. 1.1.1>"
        }}
      ]
    }}
  ]
}}
```

HTML to evaluate:
```html
{html}
```"""


# The structural page check now lives in ``html_gen.html_utils`` so the generation
# path (``_require_html``) and this auto-fixer gate share ONE implementation.
# Re-exported under the original name; imported for its side-effect-free use below.
__all__ = ["AccessibilityScorer", "_reject_reason"]


class AccessibilityScorer:
    """WCAG 2.1 AA accessibility evaluator routed through worker_management."""

    WORKER_NAME = "redesign-accessibility-scorer"

    def __init__(self, wm_client: WMClient | None = None) -> None:
        self._wm = wm_client if wm_client is not None else get_wm_client()

    async def score(
        self,
        html: str,
        llm_config: LLMConfigLike | None = None,
        budget: GenerationBudget | None = None,
    ) -> AccessibilityReport:
        """Score HTML for accessibility compliance.

        Args:
            html: Raw HTML string to evaluate (truncated to 50k chars).
            llm_config: Optional LLM configuration override.
            budget: The job's :class:`GenerationBudget`, when this scorer is
                running inside a job (see :func:`~.html_gen.section_run.budget_window`).
                ``None`` leaves the call unbounded — direct/ad-hoc callers may
                not have a job budget to hand it.

        Returns:
            AccessibilityReport with overall score and category breakdowns.
            When ``budget`` is exhausted, an UNSCORED report is returned
            (never raised) — see the module docstring's rationale at
            :data:`MAX_FIX_INPUT_CHARS`: the design is already generated and
            paid for, so a budget refusal here is an advisory skip, not a job
            failure.
        """
        logger.info("Starting accessibility scoring (html_length=%d)", len(html))
        import time

        t0 = time.monotonic()
        prompt = _SCORING_PROMPT.format(html=html[:50000])
        try:
            response_text = await self._call_llm(prompt, llm_config=llm_config, budget=budget)
        except BudgetExhausted as exc:
            logger.warning(
                "Accessibility scoring SKIPPED: job budget exhausted (%s) — the design is "
                "already generated and paid for, skipping rather than failing the job",
                exc,
            )
            return AccessibilityReport(
                overall_score=None,
                unscored_reason=f"job budget exhausted before accessibility scoring ({exc})",
            )
        report = self._parse_report(response_text)
        elapsed = time.monotonic() - t0
        logger.info(
            "Accessibility scoring completed in %.2fs (score=%s)",
            elapsed,
            "UNSCORED" if report.overall_score is None else f"{report.overall_score:.1f}",
        )
        return report

    async def score_and_fix(
        self,
        html: str,
        max_attempts: int = 2,
        threshold: float = 70.0,
        llm_config: LLMConfigLike | None = None,
        budget: GenerationBudget | None = None,
    ) -> tuple[str, AccessibilityReport]:
        """Score and optionally fix accessibility issues.

        Returns the (potentially fixed) HTML and the final report. ``budget``
        is the job's :class:`GenerationBudget` (see :meth:`score`); an
        exhaustion during the auto-fix loop stops it and records
        ``fix_skipped_reason`` — same "skip, don't fail" precedent as every
        other skip in this method.
        """
        report = await self.score(html, llm_config=llm_config, budget=budget)

        initial_score = report.overall_score
        if initial_score is None:
            # UNSCORED is not "below threshold". The auto-fixer would rewrite the
            # WHOLE document from `report.findings`, which on an unscored report is
            # empty — i.e. a prompt whose issue list literally reads "No critical
            # issues found." Keep the original valid document and say why.
            logger.error(
                "Accessibility auto-fix SKIPPED: the page was never scored (%s) — "
                "an unscored report must not be treated as below-threshold",
                report.unscored_reason,
            )
            report.fix_skipped_reason = (
                f"accessibility not scored ({report.unscored_reason}); auto-fix not attempted"
            )
            return html, report

        if initial_score >= threshold or max_attempts <= 1:
            return html, report

        if len(html) > MAX_FIX_INPUT_CHARS:
            # Never hand the fixer a mid-tag slice: whatever it returns replaces the
            # whole page. Keep the original valid document and record why.
            logger.warning(
                "Accessibility auto-fix SKIPPED: html is %d chars (cap %d) — fixing a "
                "truncated document would corrupt the page",
                len(html),
                MAX_FIX_INPUT_CHARS,
            )
            report.fix_skipped_reason = (
                f"html too large for auto-fix ({len(html)} > {MAX_FIX_INPUT_CHARS} chars)"
            )
            return html, report

        # Attempt auto-fix
        for attempt in range(1, max_attempts):
            logger.warning(
                "Accessibility score %.1f below threshold %.1f, attempting auto-fix (attempt %d/%d)",
                report.overall_score,
                threshold,
                attempt,
                max_attempts - 1,
            )
            fix_prompt = self._build_fix_prompt(html, report)
            try:
                fixed_response = await self._call_llm(
                    fix_prompt, llm_config=llm_config, budget=budget
                )
            except BudgetExhausted as exc:
                logger.warning(
                    "Accessibility auto-fix SKIPPED on attempt %d: job budget exhausted (%s) — "
                    "the design is already generated and paid for, keeping the original",
                    attempt,
                    exc,
                )
                report.fix_skipped_reason = f"job budget exhausted during auto-fix ({exc})"
                break
            fixed_html = self._extract_html(fixed_response)
            if not fixed_html:
                logger.warning("Auto-fix attempt %d returned no valid HTML, stopping", attempt)
                report.fix_skipped_reason = "auto-fix returned no HTML block"
                break

            # Re-validate the FIXED output before it can replace the page. The
            # deterministic html validation ran earlier in the pipeline over the
            # pre-fix document and is never re-run over this one.
            rejection = _reject_reason(fixed_html)
            if rejection is not None:
                logger.warning(
                    "Auto-fix attempt %d produced unusable HTML (%s), keeping the original",
                    attempt,
                    rejection,
                )
                report.fix_skipped_reason = f"auto-fix output rejected: {rejection}"
                break

            html = fixed_html
            report = await self.score(html, llm_config=llm_config, budget=budget)
            report.html_was_fixed = True

            rescore = report.overall_score
            if rescore is None:
                # The re-score of the FIXED document failed. We cannot claim the
                # fix helped, and we must not loop on a measurement we do not have.
                logger.error(
                    "Accessibility re-score after auto-fix attempt %d is UNSCORED (%s), stopping",
                    attempt,
                    report.unscored_reason,
                )
                report.fix_skipped_reason = (
                    f"re-score after auto-fix not scored ({report.unscored_reason})"
                )
                break

            if rescore >= threshold:
                logger.info(
                    "Accessibility auto-fix succeeded on attempt %d (score: %.1f)",
                    attempt,
                    rescore,
                )
                break

        return html, report

    async def _call_llm(
        self,
        prompt: str,
        llm_config: LLMConfigLike | None = None,
        budget: GenerationBudget | None = None,
    ) -> str:
        """Send a prompt to WM and return the response text.

        Runs under ``budget`` — the same enforcement seam ``HtmlGenerator``
        uses (:func:`~.html_gen.section_run.budget_window`) — when the caller
        has a job budget to hand it. Raises :class:`BudgetExhausted` when the
        call may not start, or when the job deadline expires while it is in
        flight; callers decide whether that means skip or fail (see
        :meth:`score`/:meth:`score_and_fix`). ``budget=None`` leaves the call
        unbounded, same as before #130.
        """
        import time

        model_override = model_override_from(llm_config)
        logger.debug("LLM call for accessibility evaluation (worker=%s)", self.WORKER_NAME)
        t0 = time.monotonic()
        if budget is not None:
            async with budget_window(budget, extra=False):
                response = await self._wm.message(
                    worker_name=self.WORKER_NAME,
                    message=prompt,
                    model_override=model_override,
                )
        else:
            response = await self._wm.message(
                worker_name=self.WORKER_NAME,
                message=prompt,
                model_override=model_override,
            )
        elapsed = time.monotonic() - t0
        logger.debug("LLM accessibility call completed in %.2fs", elapsed)
        if isinstance(response.output, str):
            return response.output
        return json.dumps(response.output)

    def _parse_report(self, response_text: str) -> AccessibilityReport:
        """Parse LLM JSON response into an AccessibilityReport."""
        try:
            # Extract JSON from response (may be wrapped in code blocks)
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError) as exc:
            # A reply we could not parse means the page was NOT assessed. Returning
            # a number here (it used to be 50.0) manufactures a measurement: it is
            # persisted, webhooked, and fed to the auto-fixer as "mediocre page".
            # UNSCORED is its own state.
            logger.error(
                "Accessibility report UNSCORED: could not parse the LLM reply "
                "(%s: %s); first 200 chars: %r",
                type(exc).__name__,
                exc,
                response_text[:200],
            )
            return AccessibilityReport(
                overall_score=None,
                unscored_reason=f"unparseable LLM reply ({type(exc).__name__})",
            )

        categories: list[CategoryScore] = []
        all_findings: list[AccessibilityFinding] = []

        for cat_data in data.get("categories", []):
            findings = []
            for f_data in cat_data.get("findings", []):
                finding = AccessibilityFinding(
                    category=cat_data.get("category", "unknown"),
                    severity=f_data.get("severity", "minor"),
                    description=f_data.get("description", ""),
                    element=f_data.get("element"),
                    recommendation=f_data.get("recommendation"),
                    wcag_criterion=f_data.get("wcag_criterion"),
                )
                findings.append(finding)
                all_findings.append(finding)

            # Valid JSON that OMITS "score" hits the same gap as an unparseable
            # reply: a default number is a measurement nobody made. Absent -> None.
            raw_cat_score = cat_data.get("score")
            if raw_cat_score is None:
                logger.warning(
                    "Accessibility category %r UNSCORED: the LLM reply omitted its score",
                    cat_data.get("category", "unknown"),
                )
            categories.append(
                CategoryScore(
                    category=cat_data.get("category", "unknown"),
                    score=None if raw_cat_score is None else float(raw_cat_score),
                    findings=findings,
                )
            )

        raw_overall = data.get("overall_score")
        if raw_overall is None:
            logger.error(
                "Accessibility report UNSCORED: the LLM reply parsed but omitted "
                "'overall_score' (keys present: %s)",
                sorted(data) if isinstance(data, dict) else type(data).__name__,
            )
            return AccessibilityReport(
                overall_score=None,
                categories=categories,
                findings=all_findings,
                unscored_reason="LLM reply omitted 'overall_score'",
            )

        return AccessibilityReport(
            overall_score=float(raw_overall),
            categories=categories,
            findings=all_findings,
        )

    def _build_fix_prompt(self, html: str, report: AccessibilityReport) -> str:
        """Build a prompt to fix accessibility issues."""
        issues = []
        for f in report.findings:
            if f.severity in ("critical", "major"):
                issues.append(f"- [{f.severity}] {f.description}")
                if f.recommendation:
                    issues.append(f"  Fix: {f.recommendation}")

        issues_text = "\n".join(issues) if issues else "No critical issues found."

        return f"""Fix the following accessibility issues in this HTML while preserving its visual design and content.

## Accessibility Issues to Fix
{issues_text}

## HTML to Fix
```html
{html}
```

Output the FIXED HTML in a ```html code block. Only fix accessibility issues, do not change the design."""

    def _extract_html(self, response: str) -> str | None:
        """Extract HTML code block from LLM response."""
        import re

        pattern = r"```html\s*\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
