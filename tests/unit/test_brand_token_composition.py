"""compose_section_page must emit the brand-token CSS — never Tailwind/system-ui/#2563EB."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from src.services.context_resolver import resolve_context
from src.services.html_generator import HtmlGenerator

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

_SECTIONS = {"hero": "<section><h1>Acme</h1></section>"}


def _gen() -> HtmlGenerator:
    return HtmlGenerator(template_loader=None, wm_client=cast("WMClient", object()))


def test_composition_emits_real_palette_and_fonts() -> None:
    ctx = resolve_context(
        brand={
            "primary_color": "#0a1f44",
            "secondary_color": "#e8b923",
            "display_font": "Fraunces",
            "body_font": "IBM Plex Sans",
        }
    )
    page = _gen().compose_section_page(_SECTIONS, brand=None, context=ctx)
    assert "cdn.tailwindcss.com" not in page
    assert "system-ui, -apple-system, sans-serif" in page  # only as a fallback in the stack
    assert "--color-brand: #0a1f44" in page
    assert "--color-accent: #e8b923" in page
    assert '"Fraunces"' in page
    assert '"IBM Plex Sans"' in page
    assert "#2563EB" not in page


def test_composition_unknown_palette_has_no_blue_sentinel() -> None:
    page = _gen().compose_section_page(_SECTIONS, brand=None, context=None)
    assert "#2563EB" not in page
    assert "cdn.tailwindcss.com" not in page
    assert ":root" in page


def test_composition_pulls_fonts_from_html_when_context_lacks_them() -> None:
    html = (
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces">'
        "<style>:root{--font-display:'Fraunces',serif}</style>"
    )
    page = _gen().compose_section_page(_SECTIONS, brand=None, context=None, html=html)
    assert '"Fraunces"' in page
    assert "fonts.googleapis.com" in page


def test_composition_wraps_each_section_in_landmark() -> None:
    """Each logical section gets a real ``<section data-section>`` landmark so
    render_section_screenshots captures one full-width, meaningfully-named band per
    section (not the old ``<!-- Section: -->`` comment the selector can't see)."""
    sections = {
        "hero": "<div><h1>Acme</h1></div>",
        "features": "<div><h2>What we do</h2></div>",
        "footer": "<div>Contact</div>",
    }
    page = _gen().compose_section_page(sections, brand=None, context=None)
    for name in sections:
        assert f'data-section="{name}"' in page
        assert f'aria-label="{name}"' in page
    # The pre-#section-variety comment marker is gone — landmarks replace it.
    assert "<!-- Section:" not in page
    # Inner content is preserved verbatim inside the wrappers.
    assert "<h1>Acme</h1>" in page
    assert "<h2>What we do</h2>" in page


def test_composition_escapes_section_name_in_attribute() -> None:
    """A section key with quote/angle chars must not break out of the attribute."""
    page = _gen().compose_section_page({'a"><script>': "<div>x</div>"}, brand=None, context=None)
    assert '<section data-section="a"><script>' not in page
    assert "&quot;" in page and "&gt;" in page


def test_composition_balances_unclosed_fragment_so_siblings_dont_nest() -> None:
    """A section whose markup has an unclosed tag must not swallow the next section.

    Unbalanced fragments (an unclosed ``<section>``/``<div>``) composed flat let the
    dangling tag consume the next wrapper's ``</section>`` — every following section
    nests inside one greedy band at fractional width and never qualifies as a capture
    band (the live 019ea7e2 failure). compose_section_page balances each fragment, so
    every ``<section data-section>`` stays a DIRECT child of ``<body>``."""
    from bs4 import BeautifulSoup

    sections = {
        # 'main' leaves an unclosed inner <section> and <div> — the real failure shape.
        "main": '<main id="main"><section class="block"><div><h2>Projects</h2>',
        "contact": '<div class="contact"><h2>Contact</h2></div>',
        "footer": "<div>© Acme</div>",
    }
    page = _gen().compose_section_page(sections, brand=None, context=None)
    soup = BeautifulSoup(page, "html.parser")
    assert soup.body is not None
    direct = soup.body.find_all("section", attrs={"data-section": True}, recursive=False)
    labels = [s.get("data-section") for s in direct]
    assert labels == ["main", "contact", "footer"], (
        f"sibling sections nested instead of staying top-level: {labels}"
    )
