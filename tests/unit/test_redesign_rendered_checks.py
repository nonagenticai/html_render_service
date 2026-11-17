"""Defects only the RENDERED page shows: text contrast, logo surfaces, print layout.

Found in the 2026-09-14 cliffordtraining.nl acceptance PDF (run 01a0a054-14fa, after
#126): p1 the dark logo on the dark nav, p3/p4 40%/70% empty (tall sections kept whole),
p5-6 testimonial cards with large blank gaps, p7 accent-orange "EERST GESPREK" on the
accent gradient. Everything here runs in real Chromium; no network (every http(s)
request is aborted, so the recorded image URLs never leave the machine).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from src.services.html_gen import generator as generator_module
from src.services.html_gen.brand_css import _contrast_ratio
from src.services.html_gen.rendered_checks import (
    RenderedReport,
    fix_logo_surfaces,
    fixed_text_color,
    measure_rendered_page,
)
from src.services.html_gen.section_run import GenerationBudget, SectionRun
from src.services.html_generator import HtmlGenerator

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

_RUN2: dict[str, str] = json.loads(
    (Path(__file__).parent / "fixtures" / "cliffordtraining_run2_section_replies.json").read_text(
        encoding="utf-8"
    )
)
_URL = "https://www.cliffordtraining.nl/"
# The run's real brand pair (navy brand, burnt-orange accent).
_BRAND = {"primary_color": "#0E1B2C", "secondary_color": "#B8501F"}
_DARK_LOGO = "https://www.cliffordtraining.nl/assets/logo-clifford.svg"
_WHITE_LOGO = "https://www.cliffordtraining.nl/assets/logo-clifford-white.svg"


async def _probe(html: str) -> RenderedReport:
    """``probe_rendered_page``'s measurement, offline (set_content + aborted network)."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1024})
            await page.route("http*://**/*", lambda route: route.abort())
            await page.set_content(html, wait_until="load")
            await page.emulate_media(media="print")
            return await measure_rendered_page(page)
        finally:
            await browser.close()


def _compose(sections: dict[str, str]) -> str:
    return HtmlGenerator(wm_client=cast("WMClient", object())).compose_section_page(
        sections, brand=_BRAND, base_url=_URL
    )


# ── contrast ────────────────────────────────────────────────────────────────


async def test_the_recorded_cta_eyebrow_is_accent_on_accent() -> None:
    """The gradient lives on ``.cta-section::before`` — a pseudo backdrop is measured."""
    report = await _probe(_compose({"cta": _RUN2["cta"]}))

    eyebrow = [f for f in report.contrast if f.text.lower() == "eerst gesprek"]
    assert eyebrow, [f.describe() for f in report.contrast]
    assert eyebrow[0].section == "cta" and eyebrow[0].ratio < 1.5
    assert "cta" in report.contrast_by_section()


@pytest.mark.parametrize(
    ("body", "fails"),
    [
        # muted grey on navy (the recorded hero lead + footer legal line)
        ("<div style='background:#0E1B2C'><p style='color:#555555'>Al 22 jaar</p></div>", True),
        ("<div style='background:#0E1B2C'><p style='color:#ffffff'>Al 22 jaar</p></div>", False),
        # large text needs only 3:1
        (
            "<div style='background:#fff'><h2 style='color:#949494;font-size:32px'>Kop</h2></div>",
            False,
        ),
        (
            "<div style='background:#fff'><p style='color:#949494;font-size:16px'>tekst</p></div>",
            True,
        ),
        # a round avatar: the circle's corners are not the text's backdrop
        (
            "<div style='background:#fff;padding:20px'><div style='width:48px;height:48px;"
            "border-radius:999px;background:#0E1B2C;color:#fff;display:grid;"
            "place-items:center;font-size:14px'>RB</div></div>",
            False,
        ),
        # a corner glow that never reaches the headline is not its backdrop (recorded hero)
        (
            "<div style='background:radial-gradient(40% 30% at 95% 5%, #B8501F, transparent 60%),"
            " #0E1B2C;height:900px;padding-top:600px'>"
            "<h1 style='color:#fff;font-size:48px;margin:0'>Leiders die bewegen</h1></div>",
            False,
        ),
        # a line that crosses onto a dark band fails even if most of it sits on white
        (
            "<div style='background:linear-gradient(90deg,#0E1B2C 30%,#fff 30%)'>"
            "<p style='color:#0E1B2C;margin:0;font-size:16px'>"
            + "Een eerste gesprek geeft helderheid over wat er speelt. " * 3
            + "</p></div>",
            True,
        ),
        # text over the matching half of a gradient fails (recorded CTA shape)
        (
            "<div style='background:linear-gradient(180deg,#B8501F,#0E1B2C);height:400px'>"
            "<p style='color:#B8501F;margin:0;padding:20px'>Eerst gesprek</p></div>",
            True,
        ),
        # a hairline ::before beside the text is not its backdrop
        (
            "<style>.e{position:relative;color:#0E1B2C}.e::before{content:'';position:absolute;"
            "left:0;top:0;width:30px;height:1px;background:#0E1B2C}</style>"
            "<div style='background:#fff'><p class='e'>eyebrow</p></div>",
            False,
        ),
        # semi-transparent white on navy composites before measuring
        (
            "<div style='background:#0E1B2C'><p style='color:rgb(255 255 255 / .35)'>x</p></div>",
            True,
        ),
    ],
)
async def test_contrast_shapes(body: str, fails: bool) -> None:
    report = await _probe(f"<body><section data-section='s'>{body}</section></body>")
    assert bool(report.contrast) is fails, [f.describe() for f in report.contrast]


# Text a reader cannot SEE, each in #999 on #aaa (1.2:1) — measured, every one of them
# failed a healthy job at 1.0:1. The last case is the control: the same colours, visible.
_LOW = "background:#aaaaaa;padding:20px"
_INK = "color:#999999"


@pytest.mark.parametrize(
    ("body", "measured"),
    [
        pytest.param(
            f"<div style='{_LOW}'><span style='{_INK};position:absolute;width:1px;height:1px;"
            "overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap'>Skip to content</span></div>",
            False,
            id="sr-only",
        ),
        pytest.param(
            f"<div style='{_LOW}'><div style='max-height:0;overflow:hidden'>"
            f"<p style='{_INK}'>Collapsed answer</p></div></div>",
            False,
            id="collapsed-accordion",
        ),
        pytest.param(
            f"<div style='{_LOW}'><div style='height:40px;overflow:hidden'>"
            f"<p style='{_INK};margin-top:80px'>Clipped away</p></div></div>",
            False,
            id="clipped-by-ancestor",
        ),
        pytest.param(
            f"<div style='{_LOW}'><p style='color:transparent'>Invisible ink</p></div>",
            False,
            id="color-transparent",
        ),
        pytest.param(
            f"<div style='{_LOW}'><span aria-hidden='true' style='{_INK}'>01</span></div>",
            False,
            id="aria-hidden-decoration",
        ),
        pytest.param(
            f"<div style='{_LOW}'><p style='{_INK};clip-path:inset(50%)'>Clip-path hidden</p></div>",
            False,
            id="clip-path-inset",
        ),
        pytest.param(
            f"<div style='{_LOW}'><p style='{_INK};position:absolute;left:-9999px'>Off page</p>"
            "</div>",
            False,
            id="off-page",
        ),
        pytest.param(
            f"<div style='{_LOW}'><p style='{_INK};visibility:hidden'>Hidden</p></div>",
            False,
            id="visibility-hidden",
        ),
        pytest.param(
            f"<div style='{_LOW}'><p style='{_INK}'>Plainly visible and unreadable</p></div>",
            True,
            id="control-visible",
        ),
    ],
)
async def test_only_visible_text_is_judged(body: str, measured: bool) -> None:
    report = await _probe(f"<body><section data-section='s'>{body}</section></body>")
    assert bool(report.contrast) is measured, [f.describe() for f in report.contrast]


# ── the deterministic colour fix ────────────────────────────────────────────


def test_the_fix_picks_the_nearest_page_token_that_reaches_aa() -> None:
    palette = {
        "--color-ink": "rgb(26, 26, 26)",
        "--color-on-brand": "rgb(255, 255, 255)",
        "--color-accent": "rgb(184, 80, 31)",
    }
    # accent-orange text on the navy brand band: white is the reachable token
    assert (
        fixed_text_color("rgb(184, 80, 31)", "rgb(14, 27, 44)", 4.5, palette)
        == "var(--color-on-brand)"
    )
    # grey on white: ink, not white (which does not reach AA on white at all)
    assert fixed_text_color("rgb(150, 150, 150)", "rgb(255, 255, 255)", 4.5, palette) == (
        "var(--color-ink)"
    )


def test_without_a_reachable_token_the_original_hue_is_darkened_just_enough() -> None:
    fixed = fixed_text_color("rgb(230, 120, 60)", "rgb(255, 255, 255)", 4.5, {})
    assert fixed is not None and fixed.startswith("#")
    assert _contrast_ratio(fixed, "#ffffff") >= 4.5
    assert _contrast_ratio("#E6783C", "#ffffff") < 4.5  # it needed a shift at all
    r, g, b = (int(fixed[i : i + 2], 16) for i in (1, 3, 5))
    assert r > g > b, "still an orange, not black"


def test_accent_text_token_stays_aa_on_the_recorded_warm_paper_band() -> None:
    """``--color-accent-text`` measured 4.44:1 on #F5F1EA (two eyebrows below AA)."""
    page = _compose({"s": "<section class='s'><style>.s{color:var(--color-ink)}</style></section>"})
    value = page.split("--color-accent-text:", 1)[1].split(";", 1)[0].strip()
    assert _contrast_ratio(value, "#F5F1EA") >= 4.5
    assert _contrast_ratio("#B8501F", "#F5F1EA") < 4.5  # the recorded value did not


# ── logo surfaces ───────────────────────────────────────────────────────────

_NAV = (
    "<nav class='n' style='background:#0E1B2C;padding:8px'>"
    f"<img src='{_DARK_LOGO}' alt='Clifford' width='191' height='70'></nav>"
)


async def test_the_dark_logo_on_the_dark_nav_is_swapped_to_the_sources_white_variant() -> None:
    report = await _probe(_compose({"nav": _NAV}))
    assert [(lg.section, lg.src) for lg in report.logos] == [("nav", _DARK_LOGO)]
    assert report.logos[0].surface_luminance < 0.05

    manifest = [{"role": "logo", "url": _DARK_LOGO}, {"role": "logo", "url": _WHITE_LOGO}]
    fixed, changes = fix_logo_surfaces({"nav": _NAV}, report, manifest)

    assert _WHITE_LOGO in fixed["nav"] and _DARK_LOGO not in fixed["nav"]
    assert changes and "swapped" in changes[0]


async def test_without_a_light_variant_the_logo_is_tinted_and_a_light_surface_is_left_alone() -> (
    None
):
    report = await _probe(_compose({"nav": _NAV}))
    fixed, _ = fix_logo_surfaces({"nav": _NAV}, report, [{"role": "logo", "url": _DARK_LOGO}])
    assert "brightness(0) invert(1)" in fixed["nav"]

    light = _NAV.replace("#0E1B2C", "#ffffff")
    report = await _probe(_compose({"nav": light}))
    assert fix_logo_surfaces({"nav": light}, report, [])[1] == []


# ── the repair loop ─────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, output: str) -> None:
        self.output = f"```html\n{output}\n```"
        self.usage_metrics = {"prompt_tokens": 100, "completion_tokens": 3000}


class _ScriptedWM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.prompts: list[str] = []

    async def message(self, **kwargs: object) -> _Resp:
        self.prompts.append(str(kwargs["message"]))
        return _Resp(self._replies[min(len(self.prompts), len(self._replies)) - 1])


# What a correct regeneration looks like: the eyebrow and the muted copy take the
# on-brand ink instead of the accent / a 72%-white mix.
_CTA_FIXED = (
    _RUN2["cta"]
    .replace(
        "text-transform: uppercase;\n      color: var(--cta-accent);",
        "text-transform: uppercase;\n      color: var(--color-on-brand);",
    )
    .replace(
        "--cta-muted: color-mix(in srgb, var(--color-on-brand) 72%, transparent);",
        "--cta-muted: var(--color-on-brand);",
    )
)


def _generator(replies: list[str]) -> tuple[HtmlGenerator, _ScriptedWM]:
    wm = _ScriptedWM(replies)
    return HtmlGenerator(wm_client=cast("WMClient", wm), render_probe=_probe), wm


async def _run(
    gen: HtmlGenerator, run: SectionRun | None = None
) -> tuple[dict[str, str], str, SectionRun]:
    run = run if run is not None else SectionRun()
    results = await gen.generate_sections(
        url=_URL,
        html="",
        sections=[{"section_type": "cta", "key": "cta", "html": "<section>cta</section>"}],
        brand=_BRAND,
        run=run,
    )
    results, page = await gen.repair_rendered_page(results, compose=_compose, run=run)
    return results, page, run


@pytest.fixture
def no_colour_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the computed colour fix, to exercise what happens when it cannot help."""
    monkeypatch.setattr(
        generator_module, "fix_contrast_css", lambda sections, report: (dict(sections), [], {})
    )


async def test_the_fixture_fix_really_passes() -> None:
    assert "72%, transparent" in _RUN2["cta"] and "72%, transparent" not in _CTA_FIXED
    assert "uppercase;\n      color: var(--color-on-brand);" in _CTA_FIXED
    report = await _probe(_compose({"cta": _CTA_FIXED}))
    assert report.contrast == [], [f.describe() for f in report.contrast]


async def test_the_recorded_cta_is_recoloured_without_another_llm_call() -> None:
    """Cheapest fix first: the accent-on-accent eyebrow gets a computed colour."""
    gen, wm = _generator([_RUN2["cta"]])

    results, page, run = await _run(gen)

    assert len(wm.prompts) == 1, "no regeneration was needed"
    assert "rendered contrast fix" in results["cta"]
    assert (await _probe(page)).contrast_by_section() == {}
    assert run.degradations == []


async def test_what_no_colour_fixes_is_regenerated_with_the_measurements(no_colour_fix) -> None:
    gen, wm = _generator([_RUN2["cta"], _CTA_FIXED])

    results, page, run = await _run(gen)

    assert len(wm.prompts) == 2
    assert "WCAG AA" in wm.prompts[1] and "Eerst gesprek" in wm.prompts[1]
    assert "cta" not in (await _probe(page)).contrast_by_section()
    assert "color: var(--color-on-brand);" in results["cta"]
    assert run.degradations == []


async def test_a_section_that_never_passes_contrast_degrades_the_job_instead_of_failing_it(
    no_colour_fix,
) -> None:
    gen, wm = _generator([_RUN2["cta"]])

    results, page, run = await _run(gen)  # no raise: a design with one flaw beats none

    assert len(wm.prompts) == 3  # first draw + 2 regeneration rounds
    assert "cta" in results and 'data-section="cta"' in page
    assert any("WCAG AA" in d and "section cta" in d for d in run.degradations), run.degradations


async def test_repair_stops_when_the_retry_budget_is_spent(no_colour_fix) -> None:
    gen, wm = _generator([_RUN2["cta"]])

    _, _, run = await _run(gen, SectionRun(budget=GenerationBudget(max_extra_calls=0)))

    assert len(wm.prompts) == 1, "the repair regeneration was not allowed to start"
    assert any("retry/repair budget spent" in d for d in run.degradations), run.degradations
    assert any("WCAG AA" in d for d in run.degradations), run.degradations


async def test_a_probe_error_is_advisory_never_job_fatal() -> None:
    async def _broken(_html: str) -> RenderedReport:
        raise RuntimeError("chromium crashed")

    wm = _ScriptedWM([_RUN2["cta"]])
    gen = HtmlGenerator(wm_client=cast("WMClient", wm), render_probe=_broken)

    results, page, run = await _run(gen)

    assert "cta" in results and page
    assert any("chromium crashed" in d for d in run.degradations), run.degradations


# ── 2026-10-18: white hero text recoloured to #555 on navy, three rounds running ──
# Job 01a14efd-59ac (nonagentic.ai redesign) completed DEGRADED with
# ``p.na-hero__sub … rgb(85, 85, 85) on rgb(11, 20, 55): contrast 2.41:1``. The model's
# hero CSS was correct (navy ``--color-brand``, white ``--color-on-brand`` text); the
# grey came from the deterministic repair. A ``<section hidden class="cb-panel">`` chat
# dialog whose own CSS says ``display: flex`` beat the UA's ``[hidden]`` rule and
# painted a fixed white panel over the hero, so the white text's worst backdrop sample
# was white; the fixer judged candidates against that ONE sample, picked
# ``--color-muted``, and the re-measure failed it on the navy everywhere else.

_NAVY = (11.0, 20.0, 55.0)
_WHITE = (255.0, 255.0, 255.0)
_LIVE_PALETTE = {
    "--color-ink": "rgb(26, 26, 26)",
    "--color-muted": "rgb(85, 85, 85)",
    "--color-on-brand": "rgb(255, 255, 255)",
    "--color-brand": "rgb(11, 20, 55)",
}


def test_the_fix_is_judged_against_the_whole_backdrop_not_one_sample() -> None:
    # 20% of the text box over the white panel, 80% over the navy hero.
    backdrop = [_WHITE] * 16 + [_NAVY] * 64
    fixed = fixed_text_color(
        "rgb(255, 255, 255)", "rgb(255, 255, 255)", 3.0, _LIVE_PALETTE, backdrop
    )
    assert fixed != "var(--color-muted)"  # the recorded regression
    assert fixed is not None and fixed.startswith("#")
    # whatever it picked passes on BOTH parts of the backdrop
    assert _contrast_ratio(fixed, "#ffffff") >= 3.0
    assert _contrast_ratio(fixed, "#0b1437") >= 3.0
    # without the backdrop (the old single-sample judgement) muted was chosen
    assert (
        fixed_text_color("rgb(255, 255, 255)", "rgb(255, 255, 255)", 3.0, _LIVE_PALETTE)
        == "var(--color-muted)"
    )


def test_no_colour_reaches_normal_text_aa_on_white_and_navy_so_nothing_is_recoloured() -> None:
    backdrop = [_WHITE] * 16 + [_NAVY] * 64
    assert fixed_text_color("rgb(255, 255, 255)", "rgb(255, 255, 255)", 4.5, {}, backdrop) is None


_HIDDEN_PANEL_PAGE = {
    "hero": (
        "<section class='hero'><style>.hero{background:var(--color-brand);"
        "color:var(--color-on-brand);padding:12rem 2rem}"
        ".hero p{color:color-mix(in srgb,var(--color-on-brand) 88%,transparent)}</style>"
        "<h1>Tools for the AI-native organization</h1>"
        "<p>The hard part of becoming AI-native isn't the AI.</p></section>"
    ),
    "cb-panel": (
        "<section class='cb-panel' hidden role='dialog'><style>.cb-panel{position:fixed;"
        "inset:auto 0 0 0;z-index:100;display:flex;height:85vh;"
        "background:var(--color-surface)}</style><div class='cb-msgs'></div></section>"
    ),
}


def test_the_base_stylesheet_makes_the_hidden_attribute_win() -> None:
    page = _compose(_HIDDEN_PANEL_PAGE)
    assert "[hidden]" in page and "display: none !important" in page


async def test_a_hidden_fixed_panel_does_not_paint_over_the_hero() -> None:
    report = await _probe(_compose(_HIDDEN_PANEL_PAGE))
    assert not [f for f in report.contrast if f.section == "hero"], [
        f.describe() for f in report.contrast
    ]
