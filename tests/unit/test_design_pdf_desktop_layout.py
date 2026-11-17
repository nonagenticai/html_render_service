"""A design-concept PDF must print the DESKTOP layout.

``page.pdf(format="A4")`` lays the document out at ~794 CSS px, so every
``@media (min-width: 968px)`` desktop rule of the cliffordtraining.nl redesign was dead:
the nav wrapped and was clipped at the paper edge, the hero split collapsed to one
column and page 2 was 90% empty navy. The job PDF now lays out at 1440px and scales
onto A4, tightens an over-wide flex row instead of clipping it, and lets only sections
too tall to keep whole break across pages.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.api.routes.artifacts as artifacts
import src.services.screenshot_renderer as sr
from tests.tenants import TENANT_1


def test_plain_a4_by_default() -> None:
    assert sr._pdf_options(None) == {"format": "A4", "print_background": True}


def test_desktop_layout_scales_a4_to_the_layout_width() -> None:
    opts = sr._pdf_options(1440)
    assert opts["format"] == "A4"
    assert opts["scale"] == pytest.approx(793.7 / 1440)
    assert opts["margin"] == {"top": "0", "right": "0", "bottom": "0", "left": "0"}


async def test_job_pdf_route_renders_at_desktop_width(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_render(html: str, timeout_ms: int = 30000, **kwargs: object) -> bytes:
        captured.update(kwargs)
        return b"%PDF-1.4"

    monkeypatch.setattr(sr, "render_pdf_from_html_string", fake_render)
    job = SimpleNamespace(status="completed", redesigned_html_path="jobs/x/redesign.html")
    monkeypatch.setattr(artifacts.JobManager, "get_job", AsyncMock(return_value=job))
    storage = MagicMock()
    storage.retrieve_text = AsyncMock(return_value="<html></html>")
    monkeypatch.setattr(artifacts, "_get_storage", lambda _r: storage)
    monkeypatch.setattr(artifacts, "_get_tenant_id", lambda _r: TENANT_1)

    resp = await artifacts.download_pdf("job-1", MagicMock(), db=MagicMock())

    assert resp.body == b"%PDF-1.4"
    assert captured.get("layout_width_px") == sr.DESIGN_PDF_LAYOUT_WIDTH_PX == 1440


_PAGE = """<!doctype html><html><head><style>
body { margin: 0 }
[data-section] { break-inside: avoid }
.row { display: flex; max-width: 900px; margin: 0 auto; padding: 0 120px; gap: 260px }
.row a { white-space: nowrap }
.tall { height: 3000px } .short { height: 100px }
</style></head><body>
<section data-section="nav"><nav class="row">
  <a>Coaching</a><a>Training</a><a>Over Dave</a><a>Aanbevelingen</a><a>Contact</a>
</nav></section>
<section data-section="tall" class="tall"></section>
<section data-section="short" class="short"></section>
</body></html>"""


async def test_print_layout_is_desktop_fits_rows_and_relaxes_only_tall_sections() -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1024})
            await page.set_content(_PAGE)
            assert await page.evaluate(
                "() => { const r = document.querySelector('.row'); return r.scrollWidth > r.clientWidth }"
            ), "fixture must start overflowing"

            width = await sr._prepare_desktop_print_layout(page, 1440)

            assert width >= 1440
            state = await page.evaluate(
                """() => ({
                  desktop: matchMedia('print and (min-width: 968px)').matches,
                  rowOverflows: (() => { const r = document.querySelector('.row');
                                         return r.scrollWidth > r.clientWidth + 1 })(),
                  tall: document.querySelector('[data-section=tall]').style.breakInside,
                  short: document.querySelector('[data-section=short]').style.breakInside,
                })"""
            )
            assert state["desktop"] is True
            assert state["rowOverflows"] is False
            assert state["tall"] == "auto"
            assert state["short"] == ""
            pdf = await page.pdf(**sr._pdf_options(width))
            assert pdf.startswith(b"%PDF")
        finally:
            await browser.close()


# The recorded 2026-09-14 shapes (run 01a0a054-14fa): the LLM's own
# ``.testimonials-section``/``.testimonials-grid`` matched the old
# ``[class*="testimonial"]`` keep-whole hint, so the 2877px band jumped to page 5 and
# left page 4 70% empty; and equal-height cards opened ~250px of blank space above a
# ``margin-top: auto`` author block.
_RECORDED_SHAPES = """<!doctype html><html><head><style>
body { margin: 0 }
.testimonials-section { break-inside: avoid }
.testimonials-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;
  break-inside: avoid }
.testimonial { display: flex; flex-direction: column; border: 1px solid #ddd;
  break-inside: avoid; padding: 16px }
.testimonial .author { margin-top: auto }
.pad { height: 1400px }
</style></head><body>
<section data-section="short"><div class="pad"></div></section>
<section data-section="testimonials"><div class="testimonials-section">
  <div class="testimonials-grid">
    <div class="testimonial"><p style="height:500px">long</p><div class="author">A</div></div>
    <div class="testimonial"><p style="height:200px">short</p><div class="author">B</div></div>
    <div class="testimonial"><p style="height:480px">long</p><div class="author">C</div></div>
    <div class="testimonial"><p style="height:500px">long</p><div class="author">D</div></div>
  </div>
</div></section>
</body></html>"""


async def test_tall_inner_keep_whole_blocks_split_and_stretched_cards_close_their_gaps() -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1024})
            await page.set_content(_RECORDED_SHAPES)
            await sr._prepare_desktop_print_layout(page, 1440)
            state = await page.evaluate(
                """() => ({
                  band: document.querySelector('.testimonials-section').style.breakInside,
                  grid: document.querySelector('.testimonials-grid').style.breakInside,
                  card: document.querySelector('.testimonial').style.breakInside,
                  align: document.querySelector('.testimonials-grid').style.alignItems,
                  short: document.querySelectorAll('.testimonial')[1].getBoundingClientRect().height,
                })"""
            )
            assert state["band"] == "auto" and state["grid"] == "auto"
            assert state["card"] == ""  # a card still never splits
            assert state["align"] == "start"
            assert state["short"] < 400
        finally:
            await browser.close()


def test_the_page_css_keeps_only_atomic_pieces_whole() -> None:
    from src.services.html_gen.brand_css import _brand_tokens_css

    css = _brand_tokens_css(
        "#0E1B2C", "#B8501F", display_font=None, body_font=None, palette_known=True
    )
    rule = css.split("break-inside: avoid", 1)[0].rsplit("}", 1)[-1]
    assert "[data-section]" not in rule
    assert '[class*="testimonial"]' not in rule and '[class*="cta"]' not in rule
    assert '[class*="card"]' in rule and "button" in rule
