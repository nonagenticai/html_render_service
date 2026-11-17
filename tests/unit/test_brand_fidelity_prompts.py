"""Brand-fidelity guarantees for the redesign prompts.

Regression cover for the "generic blue/white reskin" bug: the redesign prompts must
feed the model the original markup, preserve the existing brand, reuse original imagery,
and never default to a hardcoded blue when no palette was supplied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from src.services.context_resolver import (
    SECTION_MARKUP_BUDGET,
    SECTION_TEXT_PREVIEW_BUDGET,
    detect_sections,
)
from src.services.html_generator import (
    _DEFAULT_PRIMARY,
    HtmlGenerator,
    _resolve_palette,
)
from src.services.prompt_template import PromptTemplateLoader

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

# A dark-navy brand page — nothing blue/white — so a "generic" redesign would visibly
# drift away from these colours.
_BRAND_HTML = """<!DOCTYPE html><html><body>
<header class="bg-[#0a1f44]" style="color:#e8b923">
  <img src="https://cdn.example.com/logo.svg" alt="Acme">
  <h1>Acme Analytics</h1>
</header>
<main><section id="features"><h2>Features</h2><p>Fast dashboards.</p></section></main>
<footer>© Acme</footer>
</body></html>"""


def _gen(*, with_loader: bool) -> HtmlGenerator:
    loader = PromptTemplateLoader() if with_loader else None
    # Pass a sentinel wm_client so __init__ does not reach for a real WM client.
    return HtmlGenerator(template_loader=loader, wm_client=cast("WMClient", object()))


class TestDetectSections:
    def test_emits_markup_slice_and_text_preview(self) -> None:
        sections = detect_sections(_BRAND_HTML)
        assert sections, "expected at least one landmark section"
        header = next(s for s in sections if s["tag"] == "header")
        assert "logo.svg" in header["html"]
        assert "#0a1f44" in header["html"]
        assert "Acme Analytics" in header["text_preview"]

    def test_slices_are_capped_to_budget(self) -> None:
        big = "<section>" + ("<p>x</p>" * 5000) + "</section>"
        sections = detect_sections(big)
        assert sections
        assert all(len(s["html"]) <= SECTION_MARKUP_BUDGET for s in sections)
        assert all(len(s["text_preview"]) <= SECTION_TEXT_PREVIEW_BUDGET for s in sections)


class TestResolvePalette:
    def test_unknown_when_nothing_supplied(self) -> None:
        _aud, primary, _sec, known = _resolve_palette(context=None, brand=None, site_analysis=None)
        assert known is False
        assert primary == _DEFAULT_PRIMARY

    def test_known_from_brand(self) -> None:
        _aud, primary, secondary, known = _resolve_palette(
            context=None,
            brand={"primary_color": "#0a1f44", "secondary_color": "#e8b923"},
            site_analysis=None,
        )
        assert known is True
        assert primary == "#0a1f44"
        assert secondary == "#e8b923"

    def test_known_from_site_analysis_visual_identity(self) -> None:
        _aud, primary, _sec, known = _resolve_palette(
            context=None,
            brand=None,
            site_analysis={"visual_identity": {"primary_color": "#0a1f44"}},
        )
        assert known is True
        assert primary == "#0a1f44"


_SECTION = {
    "section_type": "hero",
    "heading": "Acme Analytics",
    "text_preview": "Fast dashboards.",
    "html": '<header class="bg-[#0a1f44]"><img src="https://cdn.example.com/logo.svg"></header>',
}


@pytest.mark.parametrize("with_loader", [False, True])
class TestSectionPrompt:
    def test_includes_original_markup_and_brand_directive(self, with_loader: bool) -> None:
        prompt = _gen(with_loader=with_loader)._build_section_prompt(
            url="https://acme.com",
            html=_BRAND_HTML,
            section=_SECTION,
            page_type="landing",
            style="modern",
            findings=None,
            brand=None,
            site_analysis=None,
        )
        assert "logo.svg" in prompt  # original markup is present
        assert "REFINEMENT of an existing brand" in prompt
        assert "reuse the exact absolute image" in prompt.lower()
        # new stack: brand-token CSS, never Tailwind / utility framework
        assert "cdn.tailwindcss.com" not in prompt
        assert "var(--color-brand)" in prompt
        # never invent SaaS social proof
        assert "Trusted by" not in prompt

    def test_no_blue_default_and_no_grey_placeholder_when_palette_unknown(
        self, with_loader: bool
    ) -> None:
        prompt = _gen(with_loader=with_loader)._build_section_prompt(
            url="https://acme.com",
            html=_BRAND_HTML,
            section=_SECTION,
            page_type="landing",
            style="modern",
            findings=None,
            brand=None,
            site_analysis=None,
        )
        assert _DEFAULT_PRIMARY not in prompt
        assert "placehold.co" not in prompt
        assert "derive" in prompt.lower()

    def test_prints_palette_when_supplied(self, with_loader: bool) -> None:
        prompt = _gen(with_loader=with_loader)._build_section_prompt(
            url="https://acme.com",
            html=_BRAND_HTML,
            section=_SECTION,
            page_type="landing",
            style="modern",
            findings=None,
            brand={"primary_color": "#0a1f44", "secondary_color": "#e8b923"},
            site_analysis=None,
        )
        assert "#0a1f44" in prompt
        assert "#e8b923" in prompt

    def test_falls_back_to_whole_page_when_no_section_slice(self, with_loader: bool) -> None:
        section = {"section_type": "hero", "heading": "", "text_preview": ""}
        prompt = _gen(with_loader=with_loader)._build_section_prompt(
            url="https://acme.com",
            html=_BRAND_HTML,
            section=section,
            page_type="landing",
            style="modern",
            findings=None,
            brand=None,
            site_analysis=None,
        )
        assert "logo.svg" in prompt  # whole-page excerpt used as fallback markup


@pytest.mark.parametrize("with_loader", [False, True])
class TestFullPagePrompt:
    def test_brand_directive_and_reuse_imagery(self, with_loader: bool) -> None:
        prompt = _gen(with_loader=with_loader)._build_full_page_prompt(
            url="https://acme.com",
            html=_BRAND_HTML,
            page_type="landing",
            style="modern",
            findings=None,
            brand=None,
            site_analysis=None,
        )
        assert "REFINEMENT of an existing brand" in prompt
        assert "reuse the exact absolute image" in prompt.lower()
        # the old grey-placeholder swap example must be gone
        assert "e2e8f0" not in prompt
        # new stack: self-contained brand-token CSS, never Tailwind
        assert "cdn.tailwindcss.com" not in prompt
        assert ":root" in prompt
        assert "--color-brand" in prompt
        assert "Trusted by" not in prompt

    def test_no_blue_default_when_palette_unknown(self, with_loader: bool) -> None:
        prompt = _gen(with_loader=with_loader)._build_full_page_prompt(
            url="https://acme.com",
            html="<html><body><h1>Plain</h1></body></html>",
            page_type="landing",
            style="modern",
            findings=None,
            brand=None,
            site_analysis=None,
        )
        assert _DEFAULT_PRIMARY not in prompt
        assert "derive" in prompt.lower()

    def test_prints_palette_when_supplied(self, with_loader: bool) -> None:
        prompt = _gen(with_loader=with_loader)._build_full_page_prompt(
            url="https://acme.com",
            html=_BRAND_HTML,
            page_type="landing",
            style="modern",
            findings=None,
            brand={"primary_color": "#0a1f44", "secondary_color": "#e8b923"},
            site_analysis=None,
        )
        assert "#0a1f44" in prompt


@pytest.mark.parametrize("with_loader", [True, False])
class TestGreenfieldPrompt:
    """scope=landing with no source HTML → design FROM the brief, not refine."""

    def test_greenfield_prompt_designs_from_scratch(self, with_loader: bool) -> None:
        prompt = _gen(with_loader=with_loader)._build_full_page_prompt(
            url="",
            html="",
            page_type="landing",
            style="bold",
            findings=None,
            brand={"primary_color": "#0a1f44", "secondary_color": "#e8b923"},
            site_analysis=None,
            greenfield=True,
        )
        # Greenfield language, and NONE of the refine-existing framing.
        assert "FROM SCRATCH" in prompt
        assert "Refine the following" not in prompt
        assert "Current HTML" not in prompt
        # Supplied palette is offered as the design's tokens.
        assert "#0a1f44" in prompt

    def test_refine_path_unchanged_when_html_present(self, with_loader: bool) -> None:
        prompt = _gen(with_loader=with_loader)._build_full_page_prompt(
            url="https://acme.com",
            html=_BRAND_HTML,
            page_type="landing",
            style="modern",
            findings=None,
            brand=None,
            site_analysis=None,
            greenfield=False,
        )
        assert "FROM SCRATCH" not in prompt


def test_greenfield_prompt_includes_the_brief_description() -> None:
    # Regression: the free-text brief (user_context.description) is the subject of a
    # greenfield design and MUST reach the prompt — it was being dropped, so the
    # model invented an unrelated business (live smoke: coffee brief → architecture).
    from src.services.context_resolver import ResolvedContext

    ctx = ResolvedContext(
        description="A landing page for Ember & Oak, a small-batch coffee roastery.",
        site_type="landing",
        target_audience=["coffee lovers"],
    )
    prompt = _gen(with_loader=True)._build_full_page_prompt(
        url="",
        html="",
        page_type="landing",
        style="warm editorial",
        findings=None,
        brand=None,
        site_analysis=None,
        context=ctx,
        greenfield=True,
    )
    assert "Ember & Oak" in prompt
    assert "coffee roastery" in prompt
