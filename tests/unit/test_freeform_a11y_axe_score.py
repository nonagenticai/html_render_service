"""A freeform design must report an accessibility score, or say why it has none.

Live 2026-10-19, run ``01a15193-…``, job ``01a15194-361b-…``: a web_design
(freeform, ``scope=full_page``) job completed and its ``accessibility_score`` was
null. Not a failed scan — NO scan: the axe score was only derived inside the
assembly verify lane, and the LLM scorer is opt-in (``run_accessibility_check``
defaults False), so a freeform job was never measured and the null carried no
reason. These tests pin: successful scan -> numeric score; failed scan ->
explicit ``unscored_reason`` (null is never read as clean); existing scores and
LLM unscored reports are never overridden.
"""

from __future__ import annotations

import pytest

from src.services.design_builder.service import RedesignService
from src.services.design_builder.types import _A11yResult
from src.services.verification import verifier
from src.services.verification.verifier import _BrowserProbe, axe_score_html

_EMPTY = _A11yResult(score=None, report=None, redesigned_html_path=None)


def _probe_returning(probe: _BrowserProbe):
    async def _fake(html: str, *, timeout_ms: int, run_axe: bool) -> _BrowserProbe:
        assert run_axe is True
        return probe

    return _fake


@pytest.mark.asyncio
async def test_successful_scan_returns_numeric_score(monkeypatch: pytest.MonkeyPatch) -> None:
    axe = {"violations": [{"id": "color-contrast", "impact": "serious"}]}
    monkeypatch.setattr(
        verifier, "_probe_in_browser", _probe_returning(_BrowserProbe(axe_result=axe))
    )
    result = await axe_score_html("<html></html>")
    assert result.score == 90.0
    assert result.unscored_reason is None
    assert result.impact_counts["serious"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe",
    [_BrowserProbe(axe_result=None), _BrowserProbe(render_error="page failed to load: boom")],
)
async def test_failed_scan_is_explicit_not_null(
    monkeypatch: pytest.MonkeyPatch, probe: _BrowserProbe
) -> None:
    monkeypatch.setattr(verifier, "_probe_in_browser", _probe_returning(probe))
    result = await axe_score_html("<html></html>")
    assert result.score is None
    assert result.unscored_reason


@pytest.mark.asyncio
async def test_browser_crash_is_explicit_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(html: str, *, timeout_ms: int, run_axe: bool) -> _BrowserProbe:
        raise RuntimeError("no chromium")

    monkeypatch.setattr(verifier, "_probe_in_browser", _boom)
    result = await axe_score_html("<html></html>")
    assert result.score is None
    assert result.unscored_reason == "axe scan failed (RuntimeError)"


@pytest.mark.asyncio
async def test_freeform_job_with_nothing_measured_gets_axe_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axe = {"violations": []}
    monkeypatch.setattr(
        verifier, "_probe_in_browser", _probe_returning(_BrowserProbe(axe_result=axe))
    )
    filled = await RedesignService._backfill_a11y(_EMPTY, None, "<html></html>")
    assert filled.score == 100.0
    assert filled.report is not None
    assert filled.report["source"] == "axe"
    assert filled.report["overall_score"] == 100.0
    assert filled.report["unscored_reason"] is None


@pytest.mark.asyncio
async def test_freeform_failed_scan_persists_unscored_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        verifier, "_probe_in_browser", _probe_returning(_BrowserProbe(axe_result=None))
    )
    filled = await RedesignService._backfill_a11y(_EMPTY, None, "<html></html>")
    assert filled.score is None
    assert filled.report is not None
    assert filled.report["overall_score"] is None
    assert filled.report["unscored_reason"] == "axe scan produced no result"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("a11y", "verification"),
    [
        # LLM score present -> untouched.
        (_A11yResult(score=88.0, report={"overall_score": 88.0}, redesigned_html_path=None), None),
        # LLM ran but could not score -> its own unscored_reason stands.
        (
            _A11yResult(
                score=None,
                report={"overall_score": None, "unscored_reason": "unparseable LLM reply"},
                redesigned_html_path=None,
            ),
            None,
        ),
        # Assembly verify lane already scored it.
        (_EMPTY, {"checks": {"accessibility_score": 75.0}}),
    ],
)
async def test_no_rescan_when_something_already_measured(
    monkeypatch: pytest.MonkeyPatch, a11y: _A11yResult, verification: dict | None
) -> None:
    async def _must_not_run(html: str, *, timeout_ms: int, run_axe: bool) -> _BrowserProbe:
        raise AssertionError("axe re-scan must not run")

    monkeypatch.setattr(verifier, "_probe_in_browser", _must_not_run)
    filled = await RedesignService._backfill_a11y(a11y, verification, "<html></html>")
    if a11y.score is not None:
        assert filled is a11y
    elif a11y.report is not None:
        assert filled.report == a11y.report
    else:
        assert filled.score == 75.0
