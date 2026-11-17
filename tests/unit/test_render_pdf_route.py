"""Unit tests for POST /api/designs/render/pdf (stateless HTML-string → PDF)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.api.routes.artifacts import PdfRenderRequest, render_pdf
from tests.tenants import TENANT_1


@pytest.mark.asyncio
async def test_render_pdf_returns_pdf_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    async def fake_render(html: str, timeout_ms: int = 30000) -> bytes:
        captured["html"] = html
        return b"%PDF-1.4 fake-bytes"

    import src.services.screenshot_renderer as sr

    monkeypatch.setattr(sr, "render_pdf_from_html_string", fake_render)

    # The route now resolves a tenant (it was the only one in the module that did
    # not) — see test_pdf_render_subresource_ssrf for the 401 half.
    request = MagicMock()
    request.state.tenant_id = TENANT_1

    resp = await render_pdf(PdfRenderRequest(html="<html><body>Scorecard</body></html>"), request)

    assert resp.media_type == "application/pdf"
    assert resp.body == b"%PDF-1.4 fake-bytes"
    assert "attachment" in resp.headers["content-disposition"]
    # The exact HTML we passed reaches the renderer.
    assert captured["html"] == "<html><body>Scorecard</body></html>"


def test_empty_html_rejected() -> None:
    # min_length=1 → an empty body 422s at the door instead of rendering a blank PDF.
    with pytest.raises(ValidationError):
        PdfRenderRequest(html="")
