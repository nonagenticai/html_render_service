"""WCAG AA contrast gate for every brand-derived colour pair the assembler emits.

This is the deterministic replacement for catching ``color-contrast`` in axe after
the fact. The web-design-v2 E2E intermittently failed with
``a11y_violation_ids: ["color-contrast"]`` and the refine loop could never fix it:
refinement routes through ``generate_assembly_plan``, whose system prompt forbids
writing HTML or CSS, so all three generations emitted byte-identical colours.

Every assertion below computes a REAL contrast ratio from the CSS the service
actually emits. Nothing is mocked and no ratio is hard-coded as an expectation.

The palettes are not invented. ``_REAL_PRIMARIES`` is the full set of
``primary_color`` values present in the production ``analysis_service.brands``
table (queried 2026-08-27), plus the inline brand from the failing scenario
``w2v_orchestrator/scripts/e2e_scenarios/web_design_v2_happy_path.json``.
``#4a4a4a14`` is an 8-digit value the brand pipeline stored; ``_norm_hex``
rejects it and the theme falls back to ``_DEFAULT_PRIMARY``, which this suite
therefore also covers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.services.catalog.theme import (
    _AA_NORMAL,
    _BASE_100,
    _BASE_200,
    _DEFAULT_PRIMARY,
    build_theme_css,
    contrast_ratio,
)

# Production brand primaries (analysis_service.brands) + the E2E scenario brand.
_REAL_PRIMARIES: tuple[str, ...] = (
    "#4a4a4a14",  # Specra — invalid 8-digit; exercises the _DEFAULT_PRIMARY path
    "#333",  # Room 15 Amsterdam IJburg — 3-digit shorthand
    "#0B1437",  # Non-Agentic / Merqi
    "#c1c1c3",  # Example Domain — light grey, forces the dark foreground branch
    "#ac6600",  # Web typography (Wikipedia) — measured 3.99:1 before this fix
    "#141413",  # Newsroom \\ Anthropic
    "#1F6FEB",  # web_design_v2_happy_path.json — measured 4.07:1 before this fix
    _DEFAULT_PRIMARY,
)

_VAR_RE = re.compile(r"--([a-z0-9-]+):(#[0-9a-f]{6})")
_TEMPLATES = Path(__file__).resolve().parents[2] / "src/templates/web_design/v2"


def _vars(primary: str) -> dict[str, str]:
    css = build_theme_css({"primary_color": primary})
    return {name: value for name, value in _VAR_RE.findall(css)}


@pytest.mark.parametrize("primary", _REAL_PRIMARIES)
def test_content_colours_meet_aa_on_their_own_surface(primary: str) -> None:
    """``bg-primary``/``btn-primary`` label text (``*-content``) is AA on its surface."""
    v = _vars(primary)
    for role in ("primary", "secondary", "accent"):
        ratio = contrast_ratio(v[f"color-{role}-content"], v[f"color-{role}"])
        assert ratio >= _AA_NORMAL, f"{primary}: {role}-content is {ratio:.2f}:1"


@pytest.mark.parametrize("primary", _REAL_PRIMARIES)
def test_muted_cta_copy_meets_aa_on_the_primary_surface(primary: str) -> None:
    """The CTA subheading colour is AA *opaque* — the pair that was failing.

    The old markup used ``opacity-90``, which axe composites into the effective
    foreground: white on ``#1F6FEB`` fell from 4.63:1 to 4.07:1 and on ``#ac6600``
    from 4.51:1 to 3.99:1. The muted colour is now computed and clamped.
    """
    v = _vars(primary)
    ratio = contrast_ratio(v["brand-on-primary-muted"], v["color-primary"])
    assert ratio >= _AA_NORMAL, f"{primary}: muted CTA copy is {ratio:.2f}:1"


@pytest.mark.parametrize("primary", _REAL_PRIMARIES)
def test_brand_ink_meets_aa_on_both_light_surfaces(primary: str) -> None:
    """``.text-brand`` (hero eyebrow, stats figures) is AA on base-100 AND base-200."""
    ink = _vars(primary)["brand-ink"]
    for surface in (_BASE_100, _BASE_200):
        ratio = contrast_ratio(ink, surface)
        assert ratio >= _AA_NORMAL, f"{primary}: brand-ink on {surface} is {ratio:.2f}:1"


def test_no_brand_surface_is_rewritten() -> None:
    """The brand's own colour is emitted verbatim — only foregrounds are adjusted."""
    assert _vars("#1F6FEB")["color-primary"] == "#1f6feb"
    assert _vars("#ac6600")["color-primary"] == "#ac6600"


def test_cta_uses_the_computed_muted_colour_not_element_opacity() -> None:
    """Guard the markup half: no alpha may be re-introduced on the primary surface.

    An ``opacity-*`` utility inside ``bg-primary`` composites the foreground and
    silently drops it below the ratio the theme just guaranteed.
    """
    cta = (_TEMPLATES / "blocks/cta.html.j2").read_text(encoding="utf-8")
    assert "bg-primary" in cta
    assert "opacity-" not in cta
    assert "text-on-primary-muted" in cta
    shell = (_TEMPLATES / "page.html.j2").read_text(encoding="utf-8")
    assert ".text-on-primary-muted{color:var(--brand-on-primary-muted" in shell
