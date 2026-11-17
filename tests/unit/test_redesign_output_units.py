"""Unit cover for the deterministic pieces behind the redesign-output fixes.

The end-to-end proof is ``test_redesign_output_quality.py``; these pin each mechanism
in isolation so a regression names its cause.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

from src.services.accessibility_theme import (
    _contrast_ratio,
    build_accessibility_theme_plan,
    ensure_distinct_accent,
)
from src.services.context_resolver import (
    extract_fonts,
    extract_page_meta,
    extract_palette,
    extract_source_styles,
    resolve_context,
    truncate_markup,
)
from src.services.design_builder.helpers import embed_source_styles
from src.services.html_gen.brand_css import (
    BRAND_TOKEN_NAMES,
    _brand_tokens_css,
    _google_fonts_head,
)
from src.services.html_gen.css_tokens import (
    css_defects,
    honor_contradicted_fallbacks,
    undefined_tokens,
)
from src.services.html_gen.prompt_builder import build_section_prompt
from src.services.html_gen.sanitize import (
    decode_cfemail,
    finalize_fragment,
    scope_css,
    strip_inline_event_handlers,
)

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

_SOURCE = (Path(__file__).parent / "fixtures" / "cliffordtraining_home.html").read_text(
    encoding="utf-8"
)

# The model's raw logo markup. html.parser turns exactly this into the shipped
# ``logo-text-fallback\"=""`` attribute + leaked text (verified against the artifact).
_RAW_LOGO = (
    '<a class="logo" href="/"><img src="assets/logo.svg" alt="Clifford" '
    "onerror=\"this.style.display='none';this.parentNode.innerHTML='<span "
    'class=\\"logo-text-fallback\\">Clifford <span class=\\"logo-ampersand\\">&amp;</span>'
    " Training';\" /></a>"
)


# ── truncation ────────────────────────────────────────────────────────────────


def test_truncate_markup_cuts_only_at_element_boundaries() -> None:
    items = "".join(f"<li><a href='/x{i}'>Item number {i} complete</a></li>" for i in range(80))
    out, truncated = truncate_markup(f"<section><ul>{items}</ul></section>", 1000)
    assert truncated
    assert len(out) <= 1000
    assert out.endswith("</ul></section>")
    # Every kept item is whole; none is cut mid-text.
    for li in BeautifulSoup(out, "html.parser").find_all("li"):
        assert li.get_text().endswith("complete")


def test_truncate_markup_untouched_when_it_fits() -> None:
    out, truncated = truncate_markup("<section><p>short</p></section>", 1000)
    assert (out, truncated) == ("<section><p>short</p></section>", False)


def _section_prompt(section: dict) -> str:
    return build_section_prompt(
        None,
        url="https://x.example/",
        html="",
        section=section,
        page_type=None,
        style="modern",
        findings=None,
        brand=None,
        site_analysis=None,
    )


def test_section_prompt_says_when_source_was_truncated() -> None:
    section = {"key": "s", "section_type": "section", "html": "<section>x</section>"}
    assert "TRUNCATED" not in _section_prompt(section)
    assert "TRUNCATED" in _section_prompt({**section, "truncated": True})


def test_section_prompt_lists_the_exact_token_contract() -> None:
    prompt = build_section_prompt(
        None,
        url="https://x.example/",
        html="",
        section={"key": "hero", "section_type": "hero", "html": "<section>x</section>"},
        page_type=None,
        style="modern",
        findings=None,
        brand=None,
        site_analysis=None,
    )
    for token in BRAND_TOKEN_NAMES:
        assert f"`{token}`" in prompt
    assert "onclick" in prompt  # the no-inline-JS rule is stated


# ── CSS validation ────────────────────────────────────────────────────────────


def test_undefined_tokens_respect_fallbacks_and_local_declarations() -> None:
    frag = (
        "<section class='h'><style>.h{--hero-bg:#000;background:var(--hero-bg);"
        "color:var(--x, red);padding:var(--space-2xl);margin:var(--paper-warm)}</style></section>"
    )
    assert undefined_tokens(frag, BRAND_TOKEN_NAMES) == {"--paper-warm"}


def test_css_defects_flags_markup_without_css() -> None:
    assert css_defects("<section><h2>x</h2></section>", BRAND_TOKEN_NAMES)
    assert not css_defects(
        "<section class='a'><style>.a{color:var(--color-text)}</style><h2>x</h2></section>",
        BRAND_TOKEN_NAMES,
    )


# ── sanitizer ─────────────────────────────────────────────────────────────────


def test_inline_handlers_are_removed_before_the_parser_can_mangle_them() -> None:
    stripped = strip_inline_event_handlers(_RAW_LOGO)
    assert "onerror" not in stripped
    img = BeautifulSoup(stripped, "html.parser").find("img")
    assert img is not None
    assert set(img.attrs) == {"src", "alt"}
    assert BeautifulSoup(stripped, "html.parser").get_text() == ""


def test_handler_text_outside_tags_is_left_alone() -> None:
    html = "<p>Use onclick=handler in docs</p><style>a[onclick]{x:y}</style>"
    assert strip_inline_event_handlers(html) == html


def test_decode_cfemail() -> None:
    encoded = "c0a9aea6af80a3aca9a6a6afb2a4b4b2a1a9aea9aea7eeaeac"  # pragma: allowlist secret
    assert decode_cfemail(encoded) == ("info@cliffordtraining.nl")
    assert decode_cfemail("zz") is None


def test_finalize_fragment_dedupes_ids_and_rewrites_references() -> None:
    seen: set[str] = set()
    finalize_fragment('<h2 id="t">A</h2>', base_url=None, seen_ids=seen)
    out = finalize_fragment(
        '<section aria-labelledby="t"><h2 id="t">B</h2><a href="#t">x</a></section>',
        base_url=None,
        seen_ids=seen,
    )
    assert 'id="t-2"' in out
    assert 'aria-labelledby="t-2"' in out
    assert 'href="#t-2"' in out


def test_finalize_fragment_keeps_fragments_mailto_and_drops_javascript_urls() -> None:
    out = finalize_fragment(
        '<a href="#x">a</a><a href="mailto:a@b.c">b</a><a href="javascript:void(0)">c</a>'
        '<a href="blog/">d</a>',
        base_url="https://site.example/nl/",
    )
    assert 'href="#x"' in out
    assert 'href="mailto:a@b.c"' in out
    assert "javascript:" not in out
    assert 'href="https://site.example/nl/blog/"' in out


# ── palette / fonts / meta ────────────────────────────────────────────────────


def test_accent_guard_prefers_the_sites_own_accent() -> None:
    palette = extract_palette(_SOURCE)
    assert {"name": "--accent", "value": "#B8501F"} in palette
    assert ensure_distinct_accent("#0E1B2C", "#0b1523", palette) == "#B8501F"


def test_accent_guard_derives_a_contrasting_accent_without_a_palette() -> None:
    for brand, accent in (("#0E1B2C", "#0b1523"), ("#FAFAFA", "#FFFFFF"), ("#777777", "#707070")):
        derived = ensure_distinct_accent(brand, accent, [])
        assert _contrast_ratio(brand, derived) >= 3.0, (brand, derived)


def test_accent_guard_keeps_an_accent_that_already_contrasts() -> None:
    assert ensure_distinct_accent("#0a1f44", "#e8b923", None) == "#e8b923"


def test_fonts_come_from_what_the_browser_applied() -> None:
    fonts = extract_fonts(_SOURCE)
    assert fonts["display"] == "Fraunces"
    assert fonts["body"] == "Inter"
    assert fonts["google_links"]
    assert all("/css" in h for h in fonts["google_links"])


def test_fonts_from_inline_css_rules_when_no_snapshot() -> None:
    html = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lora&family=Karla">'
        "<style>body{font-family:'Karla',sans-serif}h1,h2{font-family:'Lora',serif}</style>"
    )
    fonts = extract_fonts(html)
    assert fonts["google_links"] == ["https://fonts.googleapis.com/css2?family=Lora&family=Karla"]
    assert (fonts["display"], fonts["body"]) == ("Lora", "Karla")


def test_google_fonts_head_never_emits_the_bare_origin_as_a_stylesheet() -> None:
    head = _google_fonts_head(
        ["https://fonts.googleapis.com", "https://fonts.googleapis.com/css2?family=Inter"],
        None,
        None,
    )
    assert '<link rel="stylesheet" href="https://fonts.googleapis.com">' not in head
    assert '<link rel="preconnect" href="https://fonts.googleapis.com">' in head
    assert "css2?family=Inter" in head


def test_page_meta_reads_lang_and_title() -> None:
    meta = extract_page_meta(_SOURCE)
    assert meta["lang"] == "nl"
    assert meta["title"] and meta["title"].startswith("Executive Coach Arnhem")


def test_source_style_snapshot_round_trips_through_the_html() -> None:
    styles = {"body_font": "Inter", "heading_font": "Fraunces", "vars": [["--a", "#123456"]]}
    html = embed_source_styles("<html><head></head><body></body></html>", styles)
    assert extract_source_styles(html) == styles
    # A "</script>" inside the payload cannot terminate the block.
    hostile = embed_source_styles("<html><head></head></html>", {"body_font": "</script>x"})
    assert extract_source_styles(hostile) == {"body_font": "</script>x"}


def test_computed_font_roles_beat_google_url_order() -> None:
    styles = {"body_font": '"Inter", sans-serif', "heading_font": "Fraunces, serif"}
    html = embed_source_styles(
        '<html><head><link rel="stylesheet" '
        'href="https://fonts.googleapis.com/css2?family=Inter&family=Fraunces"></head></html>',
        styles,
    )
    fonts = extract_fonts(html)
    assert (fonts["display"], fonts["body"]) == ("Fraunces", "Inter")


def test_a_bare_origin_stylesheet_link_is_not_a_font_stylesheet() -> None:
    html = '<link rel="stylesheet" href="https://fonts.googleapis.com">'
    assert extract_fonts(html)["google_links"] == []


def test_computed_fonts_override_a_brand_font_guess() -> None:
    ctx = resolve_context(brand={"font_family": "Fraunces", "body_font": "Fraunces"}, html=_SOURCE)
    assert (ctx.display_font, ctx.body_font) == ("Fraunces", "Inter")


def test_brand_tokens_guard_the_accent_without_the_a11y_theme() -> None:
    css = _brand_tokens_css(
        "#0E1B2C",
        "#0E1B2C",
        display_font=None,
        body_font=None,
        palette_known=True,
        source_palette=extract_palette(_SOURCE),
    )
    assert "--color-accent: #B8501F;" in css


def test_a11y_theme_does_not_collapse_the_accent_onto_the_brand() -> None:
    ctx = resolve_context(
        brand={"primary_color": "#0E1B2C", "secondary_color": "#0E1B2C"}, html=_SOURCE
    )
    plan = build_accessibility_theme_plan(
        style="modern",
        style_description=None,
        brand_info=None,
        context=ctx,
        critique_findings=None,
    )
    primary, secondary = plan.context.primary_color, plan.context.secondary_color
    assert _contrast_ratio(primary, secondary) >= 3.0, (primary, secondary)


def test_an_explicit_url_title_is_replaced_by_the_source_title() -> None:
    from typing import cast

    from src.services.html_generator import HtmlGenerator

    gen = HtmlGenerator(template_loader=None, wm_client=cast("WMClient", object()))
    page = gen.compose_section_page(
        {"hero": "<section><h1>x</h1></section>"},
        title="https://www.cliffordtraining.nl/",
        html=_SOURCE,
    )
    title = BeautifulSoup(page, "html.parser").title
    assert title is not None
    assert title.get_text().startswith("Executive Coach Arnhem")
    assert "https://" not in title.get_text()


def test_scope_css_wraps_rules_and_hoists_unscopable_at_rules() -> None:
    css = (
        ":root{--a:#fff}.x > p{color:var(--a)}"
        "@keyframes k{from{opacity:0}to{opacity:1}}@media (min-width:40rem){.x{gap:1rem}}"
    )
    out = scope_css(css, '[data-section="hero"]')
    assert out.startswith("@keyframes k{")
    scoped = out[out.index("@scope") :]
    assert scoped.startswith('@scope ([data-section="hero"]) {')
    assert ":scope {--a:#fff}" in scoped
    assert ".x > p{color:var(--a)}" in scoped
    assert "@media (min-width:40rem){.x{gap:1rem}}" in scoped
    assert scope_css(out, '[data-section="hero"]') == out


def test_a_fallback_that_contradicts_a_fixed_token_is_the_authors_intent() -> None:
    css = (
        ".cta{background:var(--color-brand);color:var(--color-text, #ffffff)}"
        ".card{color:var(--color-text, #334155);background:var(--color-card, #000)}"
    )
    out = honor_contradicted_fallbacks(css)
    # Light text meant for a dark band keeps rendering light now that --color-text is ink.
    assert "color:#ffffff" in out
    # An agreeing fallback is untouched; a dark fallback on a light token is honoured.
    assert "var(--color-text, #334155)" in out
    assert "background:#000" in out
