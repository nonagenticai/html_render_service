"""Real headless-chromium render test for the lifted screenshot renderer.

Skips automatically when the Playwright chromium binary is not installed (e.g.
a CI runner that hasn't run ``playwright install chromium``), so it proves the
core capability locally / in the built image without reddening a browserless CI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.services.screenshot_renderer import render_from_html_string

_HTML = (
    "<html><body style='background:#2563EB;color:#fff;font:48px sans-serif'>"
    "<h1>Hello Render</h1></body></html>"
)


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        return bool(path) and Path(path).exists()
    except Exception:  # noqa: BLE001 -- availability probe: any failure => unavailable
        return False


pytestmark = pytest.mark.skipif(
    not _chromium_available(), reason="Playwright chromium not installed"
)


async def test_render_from_html_string_produces_pngs() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = await render_from_html_string(_HTML, Path(d), ["desktop", "mobile"], timeout_ms=15000)
        assert set(out.keys()) == {"desktop", "mobile"}
        for path in out.values():
            data = Path(path).read_bytes()
            assert data[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG signature
            assert len(data) > 1000
