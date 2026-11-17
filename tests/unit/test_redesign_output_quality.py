"""The cliffordtraining.nl redesign, end to end, on recorded LLM replies.

The production artifact for https://www.cliffordtraining.nl/ shipped with: the hero
twice (``main`` AND its inner hero were both generated, duplicate ``id="hero-heading"``),
a services section cut mid-card, ``var(--step-6)``/``--paper-warm``/``--color-border``…
never defined, ``--color-accent:#0b1523`` on ``--color-brand:#0E1B2C``, body text in the
display face, a logo ``onerror`` mangled into a garbage attribute plus leaked text,
``lang="en"`` on a Dutch page, the raw URL as ``<title>``, relative links, and a
Cloudflare ``/cdn-cgi/`` script with an ``[email protected]`` placeholder.

This file drives the real ``HtmlGenerator.generate_page_chunked`` (production prompt
templates, real ``detect_sections``, real accessibility auto-theme) over a captured copy
of the page, with a fake worker_management that answers each section from
``fixtures/cliffordtraining_section_replies.json``. It deliberately imports only APIs
that also exist on the pre-fix ``main`` so the same file demonstrates the failure there.
No live LLM is called.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from bs4 import BeautifulSoup, Tag

from src.services.accessibility_theme import _contrast_ratio, build_accessibility_theme_plan
from src.services.context_resolver import detect_sections, resolve_context
from src.services.html_gen.rendered_checks import RenderedReport
from src.services.html_generator import HtmlGenerator
from src.services.prompt_template import PromptTemplateLoader

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

_FIXTURES = Path(__file__).parent / "fixtures"
_URL = "https://www.cliffordtraining.nl/"
_SOURCE = (_FIXTURES / "cliffordtraining_home.html").read_text(encoding="utf-8")
_REPLIES: dict[str, list[str]] = {
    k: v
    for k, v in json.loads(
        (_FIXTURES / "cliffordtraining_section_replies.json").read_text(encoding="utf-8")
    ).items()
    if not k.startswith("_")
}
# The brand dict the orchestrator sent for this run: secondary == primary, and the
# brand extractor's font guess (display face on body). Both are real failure inputs.
_BRAND = {
    "name": "Clifford Training & Coaching",
    "primary_color": "#0E1B2C",
    "secondary_color": "#0E1B2C",
    "font_family": "Fraunces",
    "body_font": "Fraunces",
}

# Which recorded reply answers a prompt, decided from the ORIGINAL SECTION MARKUP the
# prompt carries (first match wins, most specific first).
_MARKERS: tuple[tuple[str, str], ...] = (
    ("site-nav", "nav"),
    ("site-footer", "footer"),
    ("cta-section", "cta"),
    ("testimonials-section", "testimonials"),
    ("hero-dark", "hero"),
    ("trust-bar", "trust-bar"),
    ('id="diensten"', "diensten"),
    ("mentor-card", "section"),
    ("Hoe een traject begint", "section-2"),
    ("Over leiderschap, verandering", "section-3"),
)


def _section_markup(prompt: str) -> str:
    m = re.search(r"```html\n(.*?)```", prompt, re.DOTALL)
    return m.group(1) if m else prompt


class _Resp:
    def __init__(self, output: str) -> None:
        self.output = f"```html\n{output}\n```"
        self.usage_metrics = {"prompt_tokens": 100, "completion_tokens": 2000}


class _RecordedWM:
    def __init__(self, replies: dict[str, list[str]]) -> None:
        self._replies = replies
        self.calls: list[str] = []

    async def message(self, **kwargs: object) -> _Resp:
        markup = _section_markup(str(kwargs["message"]))
        key = next((k for marker, k in _MARKERS if marker in markup), None)
        assert key is not None, f"no recorded reply for prompt markup {markup[:200]!r}"
        self.calls.append(key)
        attempts = self._replies[key]
        return _Resp(attempts[min(self.calls.count(key), len(attempts)) - 1])


async def _no_rendered_defects(_page: str) -> RenderedReport:
    return RenderedReport()


async def _generate(replies: dict[str, list[str]] = _REPLIES) -> tuple[str, _RecordedWM]:
    ctx = resolve_context(brand=_BRAND, html=_SOURCE)
    # Mirrors phases._phase_resolve_theme with accessibility_first_auto_theme=True.
    plan = build_accessibility_theme_plan(
        style="modern",
        style_description=None,
        brand_info=_BRAND,
        context=ctx,
        critique_findings=None,
    )
    wm = _RecordedWM(replies)
    gen = HtmlGenerator(
        PromptTemplateLoader(),
        wm_client=cast("WMClient", wm),
        # These recordings are the wave-1 replies and pin the static properties below.
        # They also carry real WCAG AA failures (a pale lead on white, a 4.44:1 eyebrow)
        # that the rendered-page check now regenerates — and a recording cannot change,
        # so the rendered check is exercised in test_redesign_rendered_checks.py instead.
        # Passed explicitly: this module-scoped fixture runs outside the autouse stub.
        render_probe=_no_rendered_defects,
    )
    result = await gen.generate_page_chunked(
        url=_URL,
        html=_SOURCE,
        style=plan.style,
        brand=plan.brand_info,
        style_description=plan.style_description,
        context=plan.context,
    )
    return result.text, wm


@pytest.fixture(scope="module")
def page_and_calls() -> tuple[str, list[str]]:
    import asyncio

    page, wm = asyncio.run(_generate())
    return page, wm.calls


def _root_vars(page: str) -> dict[str, str]:
    m = re.search(r":root\s*\{(.*?)\}", page, re.DOTALL)
    assert m, "composed page has no :root token block"
    return {k: v.strip() for k, v in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", m.group(1))}


# ── detect_sections ────────────────────────────────────────────────────────────


def test_detect_sections_returns_non_overlapping_outermost_sections() -> None:
    sections = detect_sections(_SOURCE)
    tags = [s["tag"] for s in sections]
    assert "main" not in tags, "main is a wrapper — its child sections are the chunks"
    # Each real heading lives in exactly ONE section (no nesting/overlap).
    headings = [
        h.get_text(" ", strip=True)
        for h in BeautifulSoup(_SOURCE, "html.parser").find_all(["h1", "h2"])
    ]
    for heading in headings:
        owners = [
            s
            for s in sections
            if heading in BeautifulSoup(s["html"], "html.parser").get_text(" ", strip=True)
        ]
        assert len(owners) == 1, f"{heading!r} is in {len(owners)} sections"
    # Source order is preserved.
    order = [s["heading"] or "" for s in sections]
    assert order.index(next(h for h in order if h.startswith("Zes routes"))) < order.index(
        next(h for h in order if h.startswith("75 LinkedIn"))
    )


def test_every_services_card_is_complete_in_exactly_one_section() -> None:
    cards = BeautifulSoup(_SOURCE, "html.parser").select("#diensten .service-card")
    assert len(cards) == 6
    sections = [BeautifulSoup(s["html"], "html.parser") for s in detect_sections(_SOURCE)]
    for card in cards:
        text = card.get_text(" ", strip=True)
        holders = [s for s in sections if text in s.get_text(" ", strip=True)]
        assert len(holders) == 1, f"card {text[:40]!r} complete in {len(holders)} sections"


def test_section_keys_are_unique() -> None:
    keys = [s.get("key") for s in detect_sections(_SOURCE)]
    assert all(keys), keys
    assert len(set(keys)) == len(keys), keys


# ── the composed page ─────────────────────────────────────────────────────────


def test_hero_is_generated_once_and_ids_are_unique(page_and_calls: tuple[str, list[str]]) -> None:
    page, calls = page_and_calls
    soup = BeautifulSoup(page, "html.parser")
    ids = [str(t["id"]) for t in soup.find_all(id=True)]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"duplicate ids: {dupes}"
    assert calls.count("hero") == 1
    assert len(soup.find_all("h1")) == 1


def test_every_var_without_fallback_is_defined(page_and_calls: tuple[str, list[str]]) -> None:
    page, _ = page_and_calls
    declared = set(re.findall(r"(--[A-Za-z0-9_-]+)\s*:", page))
    used = set(re.findall(r"var\(\s*(--[A-Za-z0-9_-]+)\s*\)", page))
    assert not (used - declared), f"undefined tokens: {sorted(used - declared)}"


def test_css_less_section_was_retried_and_ships_styled(
    page_and_calls: tuple[str, list[str]],
) -> None:
    page, calls = page_and_calls
    assert calls.count("testimonials") == 2  # rejected once (no CSS), then accepted
    soup = BeautifulSoup(page, "html.parser")
    band = soup.find("section", attrs={"data-section": "testimonials"})
    assert isinstance(band, Tag)
    assert band.find("style") is not None


def test_accent_is_visibly_distinct_from_brand(page_and_calls: tuple[str, list[str]]) -> None:
    page, _ = page_and_calls
    tokens = _root_vars(page)
    brand, accent = tokens["--color-brand"], tokens["--color-accent"]
    assert _contrast_ratio(brand, accent) >= 3.0, f"{accent} on {brand}"
    # Taken from the SOURCE site's palette (its --accent), not invented.
    assert accent.upper() == "#B8501F"


def test_fonts_body_is_inter_and_no_bare_origin_stylesheet(
    page_and_calls: tuple[str, list[str]],
) -> None:
    page, _ = page_and_calls
    tokens = _root_vars(page)
    assert tokens["--font-body"].startswith('"Inter"')
    assert tokens["--font-display"].startswith('"Fraunces"')
    soup = BeautifulSoup(page, "html.parser")
    for link in soup.find_all("link"):
        rel = link.get("rel") or []
        if "stylesheet" in rel:
            assert "/css" in str(link.get("href")), f"non-CSS stylesheet link: {link}"


def test_document_lang_and_title_come_from_the_source(
    page_and_calls: tuple[str, list[str]],
) -> None:
    page, _ = page_and_calls
    soup = BeautifulSoup(page, "html.parser")
    assert soup.html is not None
    assert soup.html.get("lang") == "nl"
    title = soup.title.get_text() if soup.title else ""
    assert "http" not in title and "www." not in title, title
    assert "Clifford" in title


def test_no_inline_handlers_or_mangled_attributes(page_and_calls: tuple[str, list[str]]) -> None:
    page, _ = page_and_calls
    soup = BeautifulSoup(page, "html.parser")
    for tag in soup.find_all(True):
        for attr in tag.attrs:
            assert not attr.lower().startswith("on"), f"{tag.name} has {attr}"
            assert not re.search(r"[\"'\\]", attr), f"garbage attribute {attr!r} on {tag.name}"
    assert "this.parentNode.innerHTML" not in soup.get_text()


def test_cloudflare_artifacts_removed_and_email_decoded(
    page_and_calls: tuple[str, list[str]],
) -> None:
    page, _ = page_and_calls
    assert "/cdn-cgi/" not in page
    assert "email&#160;protected" not in page and "email\xa0protected" not in page
    assert 'href="mailto:info@cliffordtraining.nl"' in page


def test_relative_links_are_absolutized(page_and_calls: tuple[str, list[str]]) -> None:
    page, _ = page_and_calls
    soup = BeautifulSoup(page, "html.parser")
    body = soup.body
    assert body is not None
    hrefs = [str(a["href"]) for a in body.find_all("a", href=True)]
    assert hrefs
    ok = ("https://", "http://", "#", "mailto:", "tel:")
    bad = [h for h in hrefs if not h.startswith(ok)]
    assert not bad, f"relative hrefs: {bad[:5]}"
    assert f"{_URL}contact.html" in hrefs


def test_arrow_glyph_is_rendered_as_svg(page_and_calls: tuple[str, list[str]]) -> None:
    page, _ = page_and_calls
    soup = BeautifulSoup(page, "html.parser")
    body = soup.body
    assert body is not None
    assert "→" not in body.get_text()
    assert body.select("svg.hrs-arrow")


async def test_a_section_that_never_gets_css_fails_loudly() -> None:
    replies = dict(_REPLIES)
    replies["testimonials"] = [_REPLIES["testimonials"][0]]  # CSS-less on every attempt
    with pytest.raises(RuntimeError, match="testimonials"):
        await _generate(replies)


def test_section_css_is_scoped_to_its_own_band(page_and_calls: tuple[str, list[str]]) -> None:
    """The source reuses class names across sections (.stat-num is styled by the hero AND
    by the trust/services replies); unscoped, the later rule turned the hero statistics
    ink-on-navy in the rendered PDF."""
    page, _ = page_and_calls
    soup = BeautifulSoup(page, "html.parser")
    body = soup.body
    assert body is not None
    bands = body.find_all("section", attrs={"data-section": True}, recursive=False)
    assert bands
    for band in bands:
        for style in band.find_all("style"):
            assert f'@scope ([data-section="{band["data-section"]}"])' in style.get_text()
    assert 'loading="lazy"' not in page  # headless print never scrolls lazy images in


def test_light_on_dark_intent_survives_the_defined_text_token(
    page_and_calls: tuple[str, list[str]],
) -> None:
    """The CTA reply wrote ``color: var(--color-text, #ffffff)`` on a navy band. With
    ``--color-text`` now defined as ink, that rendered ink-on-navy (invisible) in the
    PDF until the contradicted fallback was honoured."""
    page, _ = page_and_calls
    soup = BeautifulSoup(page, "html.parser")
    cta = soup.find("section", attrs={"data-section": "cta"})
    assert isinstance(cta, Tag)
    css = " ".join(s.get_text() for s in cta.find_all("style"))
    assert "var(--color-text, #ffffff)" not in css
    assert re.search(r"\.cta-heading\s*\{[^}]*color:\s*#ffffff", css)
