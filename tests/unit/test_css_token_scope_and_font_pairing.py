"""The two durability gaps left open by the website_review artifact.

Both defects were fixed by luck in the recorded run and had NO structural guard:

1. a custom property declared in one ``@scope``d section and used in ANOTHER —
   ``.blog-section { background: var(--paper-warm); }`` against a ``--paper-warm``
   declared only in the hero — which ``undefined_tokens`` passed because it
   counted a declaration ANYWHERE as defined, while ``sanitize.scope_css`` had just
   made that untrue;
2. ``--font-body`` resolving to the same family as ``--font-display`` (the artifact
   shipped the serif ``Fraunces`` as body text with Inter loaded and unused).

Every assertion below was mutation-tested against the pre-fix source: each one goes
RED on ``origin/main`` and green with the fix. The fixtures are built with the REAL
``scope_css``, not a hand-typed imitation of its output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from src.services.font_pairing import ensure_distinct_body_font, normalize_font_family
from src.services.html_gen.brand_css import (
    BRAND_TOKEN_NAMES,
    _brand_tokens_css,
    _font_stack,
    _resolve_fonts,
)
from src.services.html_gen.css_tokens import (
    TokenFinding,
    css_token_findings,
    describe_tokens,
    undefined_tokens,
)
from src.services.html_gen.sanitize import scope_css
from src.services.html_gen.section_run import SectionRun
from src.services.html_generator import HtmlGenerator

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

_ROOT_CSS = ":root{--color-ink:#1a1a1a;--color-surface:#ffffff}"


def _section(name: str, css: str, markup: str) -> str:
    """One composed-page band, scoped exactly the way ``finalize_fragment`` scopes it."""
    scoped = scope_css(css, f'[data-section="{name}"]')
    return (
        f'<section data-section="{name}" aria-label="{name}">'
        f"<style>{scoped}</style>{markup}</section>"
    )


def _page(*sections: str) -> str:
    return (
        "<!DOCTYPE html><html><head><style>" + _ROOT_CSS + "</style></head>"
        "<body>" + "".join(sections) + "</body></html>"
    )


# The owner's defect shape: --paper-warm declared in the hero band, used in the blog band.
_BLEED_PAGE = _page(
    _section(
        "hero",
        ".hero{--paper-warm:#faf5ee;background:var(--paper-warm);color:var(--color-ink)}",
        '<div class="hero"><h1>Hero</h1></div>',
    ),
    _section(
        "blog",
        ".blog-section{background:var(--paper-warm);color:var(--color-ink)}",
        '<div class="blog-section"><h2>Blog</h2></div>',
    ),
)


def test_token_declared_in_another_scoped_section_is_undefined() -> None:
    """The exact artifact defect: the blog band's background resolved to nothing."""
    assert undefined_tokens(_BLEED_PAGE, BRAND_TOKEN_NAMES) == {"--paper-warm"}


def test_bleed_finding_names_the_consuming_section_and_is_fatal() -> None:
    findings = css_token_findings(_BLEED_PAGE, BRAND_TOKEN_NAMES)
    assert TokenFinding("--paper-warm", "error", "blog") in findings
    # The band that DECLARES it is fine — reporting the hero too would be noise.
    assert not [f for f in findings if f.scope == "hero"]


def test_page_level_root_tokens_still_satisfy_every_section() -> None:
    """Scope-awareness must not flag the tokens that DO reach everywhere."""
    page = _page(
        _section("hero", ".hero{color:var(--color-ink)}", '<div class="hero">x</div>'),
        _section("blog", ".blog{background:var(--color-surface)}", '<div class="blog">y</div>'),
    )
    assert css_token_findings(page, BRAND_TOKEN_NAMES) == []


def test_a_sections_own_token_resolves_within_that_section() -> None:
    page = _page(
        _section(
            "hero",
            ".hero{--hero-bg:#012;background:var(--hero-bg)}",
            '<div class="hero">x</div>',
        )
    )
    assert css_token_findings(page, BRAND_TOKEN_NAMES) == []


def test_unscoped_fragment_is_judged_exactly_as_before() -> None:
    """Section replies are checked BEFORE scoping; that path must not get stricter."""
    frag = (
        "<section class='h'><style>.h{--hero-bg:#000;background:var(--hero-bg);"
        "padding:var(--space-2xl);margin:var(--paper-warm)}</style></section>"
    )
    assert undefined_tokens(frag, BRAND_TOKEN_NAMES) == {"--paper-warm"}


# ── fallbacks are signal too, at a lower severity ─────────────────────────────


def test_undefined_token_with_fallback_is_a_warning_not_an_error() -> None:
    page = _page(
        _section(
            "blog",
            ".blog{background:var(--paper-warm, #faf5ee)}",
            '<div class="blog">y</div>',
        )
    )
    findings = css_token_findings(page, BRAND_TOKEN_NAMES)
    assert [(f.token, f.severity, f.scope) for f in findings] == [
        ("--paper-warm", "warning", "blog")
    ]
    # It renders, so it must NOT block composition.
    assert undefined_tokens(page, BRAND_TOKEN_NAMES) == set()


def test_fallback_usage_of_a_defined_token_is_not_reported() -> None:
    page = _page(
        _section("blog", ".blog{color:var(--color-ink, #000)}", '<div class="blog">y</div>')
    )
    assert css_token_findings(page, BRAND_TOKEN_NAMES) == []


def test_describe_tokens_names_the_section_a_human_has_to_look_in() -> None:
    """A bare ``--paper-warm`` on a ten-band page does not say which band went dark."""
    findings = [f for f in css_token_findings(_BLEED_PAGE, BRAND_TOKEN_NAMES)]
    assert describe_tokens(findings) == '--paper-warm in section "blog"'


# ── the COMPOSE call site: a token defect degrades, it does not destroy the page ──

_BUDGET_DEGRADED_SECTIONS = {
    # The shape _out_of_budget ships: a section whose only remaining defect is an
    # unresolved custom property. usable_fallback let it through; compose must too.
    "services": (
        "<style>.a{--card-hover:#eee}.b{background:var(--card-hover)}</style>"
        '<div class="a">A</div><div class="b">B</div>'
    ),
    "hero": "<style>.hero{color:var(--color-ink)}</style><div class='hero'><h1>Acme</h1></div>",
}


def _compose(sections: dict[str, str], run: SectionRun | None) -> str:
    gen = HtmlGenerator(template_loader=None, wm_client=cast("WMClient", object()))
    return gen.compose_section_page(sections, brand=None, context=None, run=run)


def test_compose_ships_a_budget_degraded_section_instead_of_destroying_the_page() -> None:
    """THE BLOCKING BEHAVIOUR: every LLM call is spent, so a token defect must not raise.

    ``_generate_section`` already decided this draw ships ("shipped a draft with
    unresolved defects"). Raising here would hand the tenant nothing instead of a page
    with one visibly broken band.
    """
    run = SectionRun()
    page = _compose(_BUDGET_DEGRADED_SECTIONS, run)

    # The deliverable survives, with its content intact.
    assert "<h1>Acme</h1>" in page
    assert 'data-section="services"' in page
    # …and the defect is RECORDED, naming the token AND the section it broke.
    assert any("--card-hover" in d for d in run.degradations), run.degradations
    assert any('section "services"' in d for d in run.degradations), run.degradations


def test_compose_records_the_fallback_tier_too() -> None:
    run = SectionRun()
    _compose({"blog": "<style>.b{color:var(--ghost, #333)}</style><div class='b'>B</div>"}, run)
    assert any("--ghost" in d and "fallback" in d for d in run.degradations), run.degradations


def test_compose_without_a_run_still_ships() -> None:
    """No SectionRun (direct callers, tests): the degradation goes to the log, not an exception."""
    assert "<h1>Acme</h1>" in _compose(_BUDGET_DEGRADED_SECTIONS, None)


def test_compose_still_fails_hard_when_no_section_has_content() -> None:
    """The one surviving hard failure: nothing to ship is a composition bug, not a defect.

    The guard must read the SECTIONS. Asking whether the composed DOCUMENT has markup
    can never be false — it always has <html>/<body> and a <section> wrapper per entry —
    and an unfalsifiable guard is the exact class of defect this whole change fixes.
    """
    with pytest.raises(RuntimeError, match="no section contributed any content"):
        _compose({"empty": "<style>.x{color:red}</style>"}, SectionRun())


def test_compose_ships_when_even_one_section_has_content() -> None:
    """The counterpart: the guard must not fire on a page that has something to show."""
    page = _compose(
        {"empty": "<style>.x{color:red}</style>", "hero": "<div><h1>Acme</h1></div>"},
        SectionRun(),
    )
    assert "<h1>Acme</h1>" in page


# ── display vs body font distinctness ─────────────────────────────────────────


_TWO_FAMILY_LINKS = [
    "https://fonts.googleapis.com/css2?family=Fraunces:wght@400;700&family=Inter:wght@400&display=swap"
]
_ONE_FAMILY_LINKS = ["https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap"]


@pytest.mark.parametrize(
    ("display", "body", "links", "expected"),
    [
        # The owner's artifact: the site LOADED Inter and never used it — repair from
        # that evidence rather than flattening to the generic stack.
        ("Fraunces", "Fraunces", _TWO_FAMILY_LINKS, "Inter"),
        ("Fraunces", '"Fraunces", serif', _TWO_FAMILY_LINKS, "Inter"),
        ("Fraunces", "fraunces", _TWO_FAMILY_LINKS, "Inter"),  # case is not distinctness
        # No alternative family anywhere: only then is the body face given up.
        ("Fraunces", "Fraunces", None, None),
        # ⛔ identity alone is NOT the bar. No display face resolved, body == the FIRST
        # loaded family: the artifact's serif body text, wearing a different hat.
        (None, "Fraunces", _TWO_FAMILY_LINKS, "Inter"),
        # A genuine single-family pairing (Inter-700 over Inter-400) must NOT be
        # degraded to system-ui — it is indistinguishable from the bug by identity.
        (None, "Inter", _ONE_FAMILY_LINKS, "Inter"),
        ("Inter", "Inter", _ONE_FAMILY_LINKS, None),
        # A real two-face pairing survives untouched.
        ("Fraunces", "Inter", _TWO_FAMILY_LINKS, "Inter"),
        (None, "Inter", _TWO_FAMILY_LINKS, "Inter"),  # body is the SECOND family: correct
        ("Fraunces", None, _TWO_FAMILY_LINKS, None),
        (None, "Inter", None, "Inter"),  # no evidence at all: leave it alone
    ],
)
def test_ensure_distinct_body_font(
    display: str | None, body: str | None, links: list[str] | None, expected: str | None
) -> None:
    assert ensure_distinct_body_font(display, body, links) == expected


def test_normalize_font_family_takes_the_first_family() -> None:
    assert normalize_font_family(' "Fraunces" , Georgia, serif ') == "fraunces"
    assert normalize_font_family("") is None


def test_brand_tokens_repair_a_body_face_equal_to_the_display_face() -> None:
    """``--font-body: "Fraunces"`` with Fraunces as the display face must not ship."""
    css = _brand_tokens_css(
        "#0E1B2C",
        "#B8501F",
        display_font="Fraunces",
        body_font="Fraunces",
        palette_known=True,
        google_links=_TWO_FAMILY_LINKS,
    )
    assert '--font-body: "Inter"' in css  # the loaded-but-unused face is put to work
    assert '--font-body: "Fraunces"' not in css
    assert '--font-display: "Fraunces"' in css  # the display face is still honoured


def test_brand_tokens_fall_back_to_the_generic_stack_with_no_alternative() -> None:
    css = _brand_tokens_css(
        "#0E1B2C", "#B8501F", display_font="Fraunces", body_font="Fraunces", palette_known=True
    )
    assert f"--font-body: {_font_stack(None, kind='body')};" in css
    assert '--font-body: "Fraunces"' not in css


def test_brand_tokens_keep_a_genuine_pairing() -> None:
    css = _brand_tokens_css(
        "#0E1B2C",
        "#B8501F",
        display_font="Fraunces",
        body_font="Inter",
        palette_known=True,
    )
    assert '--font-body: "Inter"' in css
    assert '--font-display: "Fraunces"' in css


def test_resolve_fonts_repairs_a_non_distinct_body_face_on_the_prompt_path() -> None:
    """The prompt directive must not tell the model to set body text in the display face."""
    head = (
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Fraunces:wght@400;700&family=Inter:wght@400&display=swap">'
        "<style>body{font-family:'Fraunces',serif}h1{font-family:'Fraunces',serif}</style>"
    )
    html = f"<html><head>{head}</head><body><h1>x</h1><p>y</p></body></html>"
    display, body, links = _resolve_fonts(None, html)
    assert display == "Fraunces"
    assert body == "Inter"
    assert links  # the source's own links are what made the repair possible
