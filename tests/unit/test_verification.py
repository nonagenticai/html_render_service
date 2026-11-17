"""Tests for the automated verification lane.

Pure report/parsing tests always run; the browser-backed checks skip when the
Playwright chromium binary is unavailable (matching test_screenshot_renderer).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.catalog import assemble_page
from src.services.verification import (
    VerificationReport,
    parse_axe_summary,
    report_from_dict,
    verify_html,
)
from src.services.verification.verifier import (
    Finding,
    _apply_a11y_checks,
    _axe_blocking,
    _axe_impact_counts,
    _placeholder_links,
    _split_console_errors,
    axe_accessibility_score,
)


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        return bool(path) and Path(path).exists()
    except Exception:  # noqa: BLE001 — probe: any failure means "no chromium"
        return False


_requires_chromium = pytest.mark.skipif(
    not _chromium_available(), reason="Playwright chromium not installed"
)


# --------------------------------------------------------------------------- #
# Pure report / parsing
# --------------------------------------------------------------------------- #


def test_report_roundtrip():
    rep = VerificationReport(
        status="failed",
        findings=[Finding("console", "hard", "1 error", {"errors": ["boom"]})],
        checks={"console_errors": ["boom"]},
    )
    d = rep.to_dict()
    back = report_from_dict(d)
    assert back.status == "failed"
    assert back.hard_findings[0].check == "console"
    assert back.checks["console_errors"] == ["boom"]


def test_report_status_and_hard_findings():
    rep = VerificationReport(
        status="passed",
        findings=[Finding("links", "advisory", "3 placeholder links")],
    )
    assert rep.hard_findings == []
    assert "passed" in rep.summary()


def test_parse_axe_summary_only_hard():
    rep = VerificationReport(
        status="failed",
        findings=[
            Finding("a11y", "hard", "contrast", {"violations": ["color-contrast"]}),
            Finding("links", "advisory", "placeholder"),
        ],
    )
    summary = parse_axe_summary(rep)
    assert "color-contrast" in summary
    assert "placeholder" not in summary  # advisory excluded


def test_axe_blocking_filters_impact():
    axe_result = {
        "violations": [
            {"id": "color-contrast", "impact": "serious"},
            {"id": "landmark", "impact": "moderate"},
            {"id": "aria", "impact": "critical"},
        ]
    }
    ids = {v["id"] for v in _axe_blocking(axe_result)}
    assert ids == {"color-contrast", "aria"}
    assert _axe_blocking(None) == []


def test_placeholder_links_counts_empty_and_hash():
    html = '<a href="/real">ok</a><a href="#">x</a><a href="">y</a><a>z</a>'
    assert _placeholder_links(html) == 3  # "#", "", missing


def test_axe_accessibility_score_clean_page_is_100():
    assert axe_accessibility_score({"violations": []}) == 100.0


def test_axe_accessibility_score_none_when_axe_absent():
    # Absence of an axe result is not a passing score — must be None.
    assert axe_accessibility_score(None) is None


def test_axe_accessibility_score_weights_by_impact_and_clamps():
    axe_result = {
        "violations": [
            {"id": "aria", "impact": "critical"},  # -25
            {"id": "color-contrast", "impact": "serious"},  # -10
            {"id": "landmark", "impact": "moderate"},  # -4
            {"id": "region", "impact": "minor"},  # -1
            {"id": "mystery", "impact": None},  # unknown -> minor -> -1
        ]
    }
    # 100 - (25+10+4+1+1) = 59
    assert axe_accessibility_score(axe_result) == 59.0
    # A pile of critical violations clamps at 0, never negative.
    heavy = {"violations": [{"id": f"c{i}", "impact": "critical"} for i in range(10)]}
    assert axe_accessibility_score(heavy) == 0.0


def test_axe_impact_counts_maps_unknown_to_minor():
    axe_result = {
        "violations": [
            {"id": "a", "impact": "critical"},
            {"id": "b", "impact": "minor"},
            {"id": "c", "impact": "bogus"},
        ]
    }
    counts = _axe_impact_counts(axe_result)
    assert counts == {"critical": 1, "serious": 0, "moderate": 0, "minor": 2}


def test_a11y_from_verification_backfills_axe_score():
    from src.services.design_builder.service import RedesignService
    from src.services.design_builder.types import _A11yResult

    empty = _A11yResult(score=None, report=None, redesigned_html_path=None)
    report = {"checks": {"accessibility_score": 82.5, "a11y_impact_counts": {"serious": 1}}}
    filled = RedesignService._a11y_from_verification(empty, report)
    assert filled.score == 82.5
    assert filled.report is not None
    assert filled.report["source"] == "axe"

    # No verification / no axe score -> unchanged (still null).
    assert RedesignService._a11y_from_verification(empty, None).score is None
    # Never overrides an existing (LLM) score.
    scored = _A11yResult(score=90.0, report={"overall_score": 90.0}, redesigned_html_path=None)
    assert RedesignService._a11y_from_verification(scored, report) is scored


# --------------------------------------------------------------------------- #
# Browser-backed checks
# --------------------------------------------------------------------------- #

_GOOD_PLAN = [
    {"type": "navbar", "content": {"brand": "Acme", "cta_label": "Join", "cta_href": "/j"}},
    {
        "type": "hero",
        "content": {
            "headline": "Ship faster",
            "subheading": "Deploy in minutes",
            "primary_label": "Start",
            "primary_href": "/s",
        },
    },
    {
        "type": "features",
        "content": {"headline": "Why", "items": [{"title": "Fast", "body": "Quick"}]},
    },
    {
        "type": "footer",
        "content": {"brand": "Acme", "copyright": "2026", "links": [{"label": "P", "href": "/p"}]},
    },
]


@_requires_chromium
@pytest.mark.asyncio
async def test_verify_good_design_passes():
    from src.services.browser_pool import close_browser_pool

    html = assemble_page(_GOOD_PLAN, brand={"name": "Acme", "primary_color": "#2563eb"}).html
    try:
        rep = await verify_html(html)
        assert rep.status == "passed", rep.summary()
        assert rep.checks["a11y_serious_violations"] == 0
    finally:
        await close_browser_pool()


@_requires_chromium
@pytest.mark.asyncio
async def test_verify_detects_js_error():
    from src.services.browser_pool import close_browser_pool

    html = assemble_page(_GOOD_PLAN, brand={"name": "Acme"}).html
    broken = html.replace("</body>", "<script>throw new Error('boom')</script></body>")
    try:
        rep = await verify_html(broken)
        assert rep.status == "failed"
        assert any(f.check == "console" for f in rep.hard_findings)
    finally:
        await close_browser_pool()


def test_split_console_errors_separates_resource_load_echoes():
    hard, resource = _split_console_errors(
        [
            "Failed to load resource: net::ERR_NAME_NOT_RESOLVED",
            "Uncaught TypeError: x is not a function",
            "Failed to load resource: the server responded with a status of 404 ()",
        ]
    )
    assert hard == ["Uncaught TypeError: x is not a function"]
    assert len(resource) == 2


@_requires_chromium
@pytest.mark.asyncio
async def test_verify_unreachable_subresources_are_advisory_not_hard():
    # Fonts/images blocked by pod egress make Chromium log "Failed to load
    # resource" console errors. They must not fail the design: the failed
    # requests are already reported as an advisory ``assets`` finding.
    from src.services.browser_pool import close_browser_pool

    html = assemble_page(_GOOD_PLAN, brand={"name": "Acme"}).html
    html = html.replace(
        "</head>",
        '<link rel="stylesheet" href="https://fonts.invalid/x.css"></head>',
    ).replace("</body>", '<img alt="" src="https://images.invalid/a.png"></body>')
    try:
        rep = await verify_html(html)
        assert not any(f.check == "console" for f in rep.hard_findings), rep.summary()
        assert rep.checks["console_resource_errors"], rep.checks
        assert any(f.check == "assets" and f.severity == "advisory" for f in rep.findings)
    finally:
        await close_browser_pool()


@_requires_chromium
@pytest.mark.asyncio
async def test_verify_http_404_subresource_is_an_advisory_finding():
    # A 404 is a completed request: no ``requestfailed`` fires, so the console
    # echo is its only trace. It must surface as an advisory finding, not hard
    # and not silently in ``checks`` only.
    import http.server
    import threading

    from src.services.browser_pool import close_browser_pool

    class _NotFound(http.server.BaseHTTPRequestHandler):
        # Typed image/png so Chromium's ORB does not turn the 404 into a
        # network failure (which ``requestfailed`` would already cover).
        def do_GET(self):
            self.send_response(404)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format, *args):  # noqa: A002 — base-class signature
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NotFound)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    html = assemble_page(_GOOD_PLAN, brand={"name": "Acme"}).html.replace(
        "</body>", f'<img alt="" src="http://{host}:{port}/missing.png"></body>'
    )
    try:
        with pytest.MonkeyPatch.context() as mp:
            # Loopback is a private address the SSRF guard blocks; allow it here.
            mp.setattr(
                "src.services.screenshot_renderer._minio_allow_hosts",
                lambda: [host],
            )
            rep = await verify_html(html)
        assert not any(f.check == "console" for f in rep.hard_findings), rep.summary()
        http_findings = [
            f
            for f in rep.findings
            if f.check == "assets" and f.severity == "advisory" and "HTTP error" in f.message
        ]
        assert http_findings, rep.to_dict()
        assert any("404" in e for e in http_findings[0].detail["errors"])
    finally:
        server.shutdown()
        await close_browser_pool()


@_requires_chromium
@pytest.mark.asyncio
async def test_verify_light_brand_passes_a11y():
    # A light brand primary must still pass contrast (brand-ink text treatment).
    from src.services.browser_pool import close_browser_pool

    html = assemble_page(
        [
            {"type": "hero", "content": {"eyebrow": "NEW", "headline": "Hi"}},
            {"type": "stats", "content": {"items": [{"value": "99%", "label": "Uptime"}]}},
        ],
        brand={"name": "Acme", "primary_color": "#facc15"},
    ).html
    try:
        rep = await verify_html(html)
        assert rep.checks["a11y_serious_violations"] == 0, rep.checks.get("a11y_violation_ids")
    finally:
        await close_browser_pool()


# --------------------------------------------------------------------------- #
# An axe run that FAILED must not pass the hard a11y gate as "0 violations".
# --------------------------------------------------------------------------- #


def test_unmeasured_axe_is_a_hard_finding_not_a_silent_pass():
    """`_run_axe` returning None means UNMEASURED, which must fail the gate.

    Before the fix this block wrote `a11y_serious_violations: 0` and
    `0 > a11y_max_serious` was False, so a page nobody looked at was recorded
    byte-identically to a genuinely clean one and `status` became "passed".
    """
    checks: dict = {}
    findings: list[Finding] = []
    _apply_a11y_checks(None, 0, checks, findings)

    assert checks["a11y_probed"] is False
    # Absent is honest; a zero is a lie.
    assert "a11y_serious_violations" not in checks
    assert "a11y_violation_ids" not in checks
    assert "a11y_impact_counts" not in checks
    assert "accessibility_score" not in checks

    hard = [f for f in findings if f.severity == "hard" and f.check == "a11y"]
    assert len(hard) == 1
    assert "could not be measured" in hard[0].message


def test_measured_clean_axe_run_is_distinguishable_from_unmeasured():
    """A genuinely clean page still passes, and says so with `a11y_probed: True`."""
    checks: dict = {}
    findings: list[Finding] = []
    _apply_a11y_checks({"violations": []}, 0, checks, findings)

    assert checks["a11y_probed"] is True
    assert checks["a11y_serious_violations"] == 0
    assert checks["a11y_violation_ids"] == []
    assert checks["accessibility_score"] == 100.0
    assert findings == []
