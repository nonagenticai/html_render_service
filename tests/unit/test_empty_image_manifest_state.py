"""An empty image manifest must be a STATED state, not an empty string.

``extract_image_manifest`` returning ``[]`` was unrecorded: ``_format_image_manifest``
returned ``""``, which deleted the "never blank out a real image" instruction
from the prompt — while the Requirements block still told the model to "reuse the
exact absolute image URLs listed above". Pointed at a list that was not there,
the model filled the gap with ``https://placehold.co/`` boxes.

Trigger: a real site whose ``<img src>`` extraction yields nothing — CSS
background images, ``data-src`` lazy-loading, ``srcset``-only ``<picture>``.

Compounding: ``verify_html`` counted placeholder LINKS and had no placeholder-
IMAGE check at all — guarded beside unguarded inside one function — so the
filler shipped clean-green.
"""

from __future__ import annotations

import pytest

from src.services.context_resolver import extract_image_manifest
from src.services.html_gen.images import _format_image_manifest
from src.services.html_gen.prompt_builder import build_full_page_prompt
from src.services.verification.verifier import (
    _BrowserProbe,
    _placeholder_images,
    _placeholder_links,
)

# --------------------------------------------------------------------------- #
# [] and None are different states
# --------------------------------------------------------------------------- #


def test_empty_manifest_states_itself_and_keeps_the_guardrail() -> None:
    """``[]`` = "a real page yielded no images" — say so, and forbid filler."""
    block = _format_image_manifest([])
    assert block, "an empty manifest rendered as an empty string — the gap is silent again"
    assert "placehold.co" in block
    assert "Do NOT invent" in block


def test_none_manifest_is_silent() -> None:
    """``None`` = "no manifest applies" (greenfield). Nothing to say is correct."""
    assert _format_image_manifest(None) == ""


def test_empty_and_none_are_distinguishable() -> None:
    """The whole defect was these two collapsing into the same value."""
    assert _format_image_manifest([]) != _format_image_manifest(None)


def test_populated_manifest_still_carries_the_original_instruction() -> None:
    """Regression guard on the path that already worked."""
    block = _format_image_manifest([{"role": "logo", "url": "https://x.com/a.png", "alt": "A"}])
    assert "https://x.com/a.png" in block
    assert "never blank out a real image" in block


# --------------------------------------------------------------------------- #
# The trigger, end to end through the real prompt builder
# --------------------------------------------------------------------------- #


LAZY_LOADED_PAGE = """
<html><body>
  <div class="hero" style="background-image:url('/img/hero.jpg')"></div>
  <img data-src="/img/product.png" alt="Product">
  <picture><source srcset="/img/wide.webp"></picture>
</body></html>
"""


def test_a_real_page_with_no_img_src_yields_an_empty_manifest() -> None:
    """The trigger is real, not hypothetical: this is a normal modern page."""
    assert extract_image_manifest(LAZY_LOADED_PAGE, "https://example.com/") == []


def test_refine_prompt_warns_against_placeholders_when_extraction_found_nothing() -> None:
    """The prompt the model actually receives must carry the warning.

    Without the fix this prompt tells the model to "reuse the exact absolute
    image URLs listed above" while listing nothing at all.
    """
    prompt = build_full_page_prompt(
        None,
        url="https://example.com/",
        html=LAZY_LOADED_PAGE,
        page_type="landing",
        style="modern",
        findings=None,
        brand=None,
        site_analysis=None,
        image_manifest=extract_image_manifest(LAZY_LOADED_PAGE, "https://example.com/"),
        greenfield=False,
    )
    assert "none could be extracted" in prompt
    assert "Do NOT invent" in prompt


def test_greenfield_prompt_is_not_given_the_empty_manifest_warning() -> None:
    """Greenfield is TOLD to use placehold.co — the warning would contradict it.

    This is why the fix distinguishes ``None`` from ``[]`` instead of treating
    every falsy manifest the same way.
    """
    prompt = build_full_page_prompt(
        None,
        url="",
        html="",
        page_type="landing",
        style="modern",
        findings=None,
        brand=None,
        site_analysis=None,
        image_manifest=[],
        greenfield=True,
    )
    assert "none could be extracted" not in prompt


# --------------------------------------------------------------------------- #
# The compounding half: the verifier's missing check
# --------------------------------------------------------------------------- #


def test_placeholder_images_are_counted_beside_placeholder_links() -> None:
    """The guarded-beside-unguarded pair, inside one function, now matched."""
    html = """
    <html><body>
      <a href="#">dead</a>
      <img src="https://placehold.co/600x400/png" alt="filler">
      <img src="https://dummyimage.com/600x400" alt="filler">
      <img src="https://example.com/real-logo.png" alt="real">
      <picture><source srcset="https://picsum.photos/800"></picture>
    </body></html>
    """
    assert _placeholder_images(html) == 3
    assert _placeholder_links(html) == 1


def test_placeholder_images_ignores_real_images() -> None:
    """Precision control — a real image must not be counted as filler."""
    html = '<html><body><img src="https://example.com/hero.jpg"></body></html>'
    assert _placeholder_images(html) == 0


async def test_verify_html_reports_the_placeholder_image_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check is wired into the report, not merely defined.

    The browser probe is stubbed rather than run. ``run_axe=False`` is NOT
    enough — ``verify_html`` calls ``_probe_in_browser`` unconditionally, so a
    "pure" static-HTML assertion would still launch Chromium on the shared,
    event-loop-bound pool and deadlock against the other browser-backed files in
    the same session. The property under test is the static check block, which
    needs no browser at all.
    """
    from src.services.verification import verifier as verifier_module
    from src.services.verification import verify_html

    async def _no_browser(html: str, *, timeout_ms: int, run_axe: bool) -> _BrowserProbe:
        return _BrowserProbe()

    monkeypatch.setattr(verifier_module, "_probe_in_browser", _no_browser)

    report = await verify_html(
        '<html><body><img src="https://placehold.co/600x400"></body></html>',
        run_axe=False,
    )
    assert report.checks["placeholder_images"] == 1
    assert any(f.check == "images" for f in report.findings)
