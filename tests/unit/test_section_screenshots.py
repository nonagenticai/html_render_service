"""Per-section redesign screenshot capture + asset emission.

The website_review after act keys each narration line to the redesign section it
discusses instead of panning one tall full-page scroll. html_render captures a
discrete still per landmark section inside the redesign job and surfaces them as
``redesign_section:<name>`` visual assets. These tests cover the pure naming /
asset-building logic always, and exercise the real headless capture (+ band
fallback) when chromium is installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

from src.api.routes._helpers import _build_visual_assets, _section_assets
from src.services.screenshot_renderer import (
    _MIN_SECTION_HEIGHT_PX,
    _SECTION_SPLIT_MIN_RATIO,
    CAPTURE_DEVICE_SCALE_FACTOR,
    VIEWPORT_SIZES,
    _pad_to_ratio,
    _parse_css_color,
    _plan_subbands,
    _resolve_bg,
    _section_name,
    _slugify_section,
    _split_section_png,
    render_section_screenshots,
)


class TestSectionNaming:
    def test_slugify_basic(self) -> None:
        assert _slugify_section("Chat Composer!") == "chat-composer"
        assert _slugify_section("  Pricing & Plans  ") == "pricing-plans"
        assert _slugify_section("") == ""

    def test_name_prefers_data_label_then_heading_then_id_then_tag(self) -> None:
        used: set[str] = set()
        assert _section_name({"dataLabel": "Hero", "heading": "x"}, 0, used) == "hero"
        assert _section_name({"heading": "Our Features"}, 1, used) == "our-features"
        assert _section_name({"id": "contact-us"}, 2, used) == "contact-us"
        assert _section_name({"tag": "footer"}, 3, used) == "footer"

    def test_name_dedupes_with_suffix(self) -> None:
        used: set[str] = set()
        assert _section_name({"heading": "Features"}, 0, used) == "features"
        assert _section_name({"heading": "Features"}, 1, used) == "features-2"
        assert _section_name({"heading": "Features"}, 2, used) == "features-3"

    def test_name_falls_back_to_tag_when_blank(self) -> None:
        # Empty identity → the literal "section" default (deduped on collision).
        used: set[str] = set()
        assert _section_name({}, 4, used) == "section"
        assert _section_name({}, 5, used) == "section-2"

    def test_name_falls_back_to_index_when_unsluggable(self) -> None:
        # A heading with no alphanumerics slugs to '' → index-based fallback.
        used: set[str] = set()
        assert _section_name({"heading": "!!!"}, 4, used) == "section-5"


class TestSectionAssets:
    def test_emits_one_asset_per_section_with_role(self) -> None:
        assets = _section_assets({"hero": "http://x/hero.png", "chat-composer": "http://x/cc.png"})
        roles = {a.role for a in assets}
        assert roles == {"redesign_section:hero", "redesign_section:chat-composer"}
        cc = next(a for a in assets if a.role == "redesign_section:chat-composer")
        assert cc.kind == "screenshot"
        assert cc.source == "redesign"
        assert cc.metadata == {"section": "chat-composer"}
        assert cc.title == "Redesign — Chat Composer"

    def test_skips_empty_urls(self) -> None:
        assert _section_assets({"hero": ""}) == []

    def test_none_yields_no_assets(self) -> None:
        assert _section_assets(None) == []

    def test_build_visual_assets_includes_sections(self) -> None:
        assets = _build_visual_assets(
            screenshots={"desktop": "http://x/d.png"},
            section_screenshots={"hero": "http://x/hero.png"},
        )
        roles = {a.role for a in assets}
        assert "redesign_desktop" in roles
        assert "redesign_section:hero" in roles


class TestSectionPadColour:
    def test_parses_rgb_and_rgba(self) -> None:
        assert _parse_css_color("rgb(18, 24, 64)") == (18, 24, 64)
        assert _parse_css_color("rgba(255, 0, 0, 1)") == (255, 0, 0)
        # Modern space/slash rgb() form is also accepted.
        assert _parse_css_color("rgb(10 20 30 / 0.8)") == (10, 20, 30)

    def test_transparent_and_blank_are_none(self) -> None:
        assert _parse_css_color("") is None
        assert _parse_css_color(None) is None
        assert _parse_css_color("transparent") is None
        assert _parse_css_color("rgba(0, 0, 0, 0)") is None  # fully transparent
        assert _parse_css_color("notacolor") is None

    def test_resolve_falls_through_to_first_opaque_then_white(self) -> None:
        # Element bg transparent → fall through to the body bg.
        assert _resolve_bg("rgba(0,0,0,0)", "rgb(12, 34, 56)") == (12, 34, 56)
        # Nothing opaque → white default.
        assert _resolve_bg("transparent", "") == (255, 255, 255)
        assert _resolve_bg() == (255, 255, 255)


class TestPadToRatio:
    def _png(self, path: Path, size: tuple[int, int]) -> None:
        from PIL import Image

        Image.new("RGB", size, (200, 100, 50)).save(path)

    def test_short_band_pads_to_16x9_centered(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "short.png"
            self._png(p, (1440, 360))
            _pad_to_ratio(p, (7, 8, 9))
            with Image.open(p) as img:
                assert img.size == (1440, 810)  # 16:9 of the capture width
                # Original content sits in the vertical middle; bars top & bottom.
                assert img.getpixel((720, 5)) == (7, 8, 9)  # pad bar
                assert img.getpixel((720, 405)) == (200, 100, 50)  # original

    def test_tall_band_is_untouched(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tall.png"
            self._png(p, (1440, 1000))  # already taller than 16:9 (810)
            _pad_to_ratio(p, (7, 8, 9))
            with Image.open(p) as img:
                assert img.size == (1440, 1000)  # unchanged


class TestPlanSubbands:
    def test_splits_tall_band_at_clean_heading_offsets(self) -> None:
        # Three section-level headings >= the min-height floor apart → three bands; the first
        # always starts at y=0 (the lead block) named by the first heading.
        used: set[str] = set()
        heads = [
            {"topCss": 20, "text": "Hero"},
            {"topCss": 540, "text": "Projects"},
            {"topCss": 1060, "text": "Writing"},
        ]
        bands = _plan_subbands(heads, 1440, 1560, 1440.0, used)
        assert bands == [(0, "hero"), (540, "projects"), (1060, "writing")]

    def test_no_split_when_headings_too_close(self) -> None:
        # Second heading within the min-height of the first → only one band → [].
        bands = _plan_subbands(
            [{"topCss": 10, "text": "A"}, {"topCss": 50, "text": "B"}], 1440, 1560, 1440.0, set()
        )
        assert bands == []

    def test_no_split_when_last_band_would_be_a_sliver(self) -> None:
        # A heading 100px from the bottom cannot anchor a full-height band → dropped.
        bands = _plan_subbands(
            [{"topCss": 20, "text": "A"}, {"topCss": 1460, "text": "B"}], 1440, 1560, 1440.0, set()
        )
        assert bands == []

    def test_empty_or_malformed_headings_no_split(self) -> None:
        assert _plan_subbands([], 1440, 1560, 1440.0, set()) == []
        assert _plan_subbands("nope", 1440, 1560, 1440.0, set()) == []
        assert _plan_subbands([{"topCss": 540, "text": "A"}], 1440, 1560, 0.0, set()) == []

    def test_scale_applied_from_png_width(self) -> None:
        # A 2x device scale: CSS offsets double in PNG space; min-height scales too.
        used: set[str] = set()
        heads = [{"topCss": 10, "text": "Hero"}, {"topCss": 600, "text": "Writing"}]
        bands = _plan_subbands(heads, 2880, 3120, 1440.0, used)
        assert bands == [(0, "hero"), (1200, "writing")]


class TestSplitSectionPng:
    def _png(self, path: Path, size: tuple[int, int]) -> None:
        from PIL import Image

        Image.new("RGB", size, (40, 40, 40)).save(path)

    def test_crops_and_pads_each_subband(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "00_main.png"
            self._png(p, (1440, 1560))
            heads = [
                {"topCss": 20, "text": "Hero"},
                {"topCss": 540, "text": "Projects"},
                {"topCss": 1060, "text": "Writing"},
            ]
            out = _split_section_png(p, heads, 1440.0, (9, 9, 9), set())
            assert [n for n, _ in out] == ["hero", "projects", "writing"]
            for _name, sp in out:
                with Image.open(sp) as img:
                    w, h = img.size
                    # Each crop is letterboxed up to at least 16:9 (the short final
                    # crops get padded exactly like a short band would).
                    assert w == 1440
                    assert h >= round(w * 9 / 16)

    def test_no_split_returns_empty_and_leaves_source(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "00_main.png"
            self._png(p, (1440, 1560))
            assert _split_section_png(p, [], 1440.0, (9, 9, 9), set()) == []
            assert p.exists()  # caller keeps the single band when no split


# ---------------------------------------------------------------------------
# Real headless-chromium capture (skipped when chromium is not installed)
# ---------------------------------------------------------------------------


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        return bool(path) and Path(path).exists()
    except Exception:  # noqa: BLE001 -- availability probe: any failure => unavailable
        return False


# Gate only the REAL-render tests on chromium — the pure naming/asset tests above
# must always run (a browserless CI must still cover them).
_needs_chromium = pytest.mark.skipif(
    not _chromium_available(), reason="Playwright chromium not installed"
)


_SECTIONED_HTML = """
<html><body style="margin:0;font:32px sans-serif">
  <header style="min-height:300px;background:#114"><h1>Acme</h1></header>
  <section data-section="hero" style="min-height:600px;background:#1a1"><h2>Hero</h2></section>
  <section style="min-height:600px;background:#a11"><h2>Pricing Plans</h2></section>
  <footer style="min-height:300px;background:#333"><h3>Contact</h3></footer>
</body></html>
"""

_NO_LANDMARK_HTML = """
<html><body style="margin:0">
  <div style="height:1800px;background:linear-gradient(#fff,#000)">tall page</div>
</body></html>
"""

# Mixes real full-width bands with the nested fragments the live redesign emits
# (a logo-only header, a chat card, a thin spacer) — those cover-fit to blank
# frames, so the capture must drop them and keep only the full-width band.
_FRAGMENT_HTML = """
<html><body style="margin:0;font:32px sans-serif">
  <header style="width:320px;height:110px;background:#114"><h1>Logo</h1></header>
  <section style="width:400px;height:397px;background:#1a1"><h2>Chat Card</h2></section>
  <section style="min-height:170px;background:#a11"><h2>Thin Spacer</h2></section>
  <section style="min-height:600px;background:#11a"><h2>Real Band</h2></section>
</body></html>
"""


# Geometry fixtures are expressed RELATIVE to the live desktop viewport, not as
# literals. They were written against 1440x900 with a 300px section floor; the
# delivery viewport is now 1920x1080 with a 400px floor, and hardcoded heights
# silently fell to the wrong side of it (a 360px "short" band stopped being
# captured at all, and the 1560px "greedy" band stopped exceeding the split
# threshold — both would have read as capture regressions).
_CSS_W = VIEWPORT_SIZES["desktop"]["width"]
_16_9 = round(_CSS_W * 9 / 16)  # the band height that is exactly 16:9
# A short band: comfortably over the floor, well under 16:9, so it must be
# letterboxed up rather than dropped.
_SHORT_BAND_CSS = round(_MIN_SECTION_HEIGHT_PX * 1.2)
# A tall band: past 16:9 (so it is NOT padded) but under the 1.6x split
# threshold (so it stays one still).
_TALL_BAND_CSS = round(_16_9 * 1.235)
# A greedy band: past the 1.6x split threshold, sliced into three equal sub-bands
# each comfortably over the floor.
_GREEDY_SUBBAND_CSS = round(_16_9 * _SECTION_SPLIT_MIN_RATIO / 3) + 100

# A short full-width band and a tall one. The short band must be padded up to
# 16:9 so the compile cover-fit doesn't over-zoom it; the tall band (already
# > 16:9) is left at its natural height.
_MIXED_HEIGHT_HTML = f"""
<html><body style="margin:0;font:32px sans-serif;background:#fff">
  <section data-section="short" style="height:{_SHORT_BAND_CSS}px;background:#123456"><h2>Short</h2></section>
  <section data-section="tall" style="height:{_TALL_BAND_CSS}px;background:#654321"><h2>Tall</h2></section>
</body></html>
"""

# One greedy landmark (a SPA <main> of plain divs) > 1.6x of 16:9 with three
# section-level headings — the capture must SPLIT it into three per-topic stills
# instead of one tall band, so the after act has variety to key narration to.
_TALL_MAIN_HTML = f"""
<html><body style="margin:0;font:32px sans-serif;background:#fff">
  <section data-section="main" style="background:#101820;color:#fff">
    <div style="height:{_GREEDY_SUBBAND_CSS}px"><h2>Hero</h2></div>
    <div style="height:{_GREEDY_SUBBAND_CSS}px"><h2>Projects</h2></div>
    <div style="height:{_GREEDY_SUBBAND_CSS}px"><h2>Writing</h2></div>
  </section>
</body></html>
"""

# A tall greedy band with NO inner headings degrades to a single still (no split,
# no regression) — only h1/h2 anchor a split.
_TALL_NO_HEADING_HTML = f"""
<html><body style="margin:0;background:#fff">
  <section data-section="solid" style="height:{_GREEDY_SUBBAND_CSS * 3}px;background:#202830"></section>
</body></html>
"""


# A THIN page (the example.com redesign shape): every landmark is a full-width
# data-section well under the 400px floor. All-or-nothing filtering used to drop
# both and fall back to anonymous ``band-NN`` viewport bands, leaving the
# orchestrator nothing to ground the after-act script on. Short-but-full-width
# data-sections are real sections (padded to 16:9), so they must come back named.
_SHORT_SECTION_PX = 200
_THIN_PAGE_HTML = f"""
<html><body style="margin:0;font:32px sans-serif;background:#fff">
  <section data-section="intro" style="height:{_SHORT_SECTION_PX}px;background:#123"><h1>Example</h1></section>
  <section data-section="more-info" style="height:{_SHORT_SECTION_PX}px;background:#321"><p>More</p></section>
</body></html>
"""

# Band fallback with named content: the data-sections are too NARROW to be stills
# (so both passes reject them and bands are cut), but each band must be named
# after the data-section at its vertical midpoint, de-duplicated with -2.
_VH = VIEWPORT_SIZES["desktop"]["height"]
_NARROW_SECTIONS_HTML = f"""
<html><body style="margin:0;background:#fff">
  <div data-section="alpha" style="width:40%;margin:0 auto;height:{_VH}px;background:#a00"></div>
  <div data-section="gamma" style="width:40%;margin:0 auto;height:{_VH * 2}px;background:#0a0"></div>
</body></html>
"""


# Second pass is TOP-LEVEL only. The nested data-section is pushed BELOW its
# parent's box (position:relative), so the overlap rule cannot drop it — only the
# top-level selector does. Without that filter it would come back as "inner".
_NESTED_SECTION_HTML = f"""
<html><body style="margin:0;font:32px sans-serif;background:#fff">
  <section data-section="outer" style="height:{_SHORT_SECTION_PX}px;background:#123">
    <h1>Outer</h1>
    <div data-section="inner"
         style="position:relative;top:{_SHORT_SECTION_PX}px;height:{_SHORT_SECTION_PX}px;background:#321">
      <p>Inner</p>
    </div>
  </section>
</body></html>
"""

# A 2px full-width data-section (a divider) is a sliver, not a section: the second
# pass keeps a small floor, so only the real short section is captured.
_SLIVER_SECTION_HTML = f"""
<html><body style="margin:0;font:32px sans-serif;background:#fff">
  <div data-section="divider" style="height:2px;background:#000"></div>
  <section data-section="intro" style="height:{_SHORT_SECTION_PX}px;background:#123"><h1>Example</h1></section>
</body></html>
"""


@pytest.fixture
def _isolate_browser_pool():
    # The process-wide browser pool is bound to the event loop that created it.
    # A prior real-chromium test (test_screenshot_renderer) leaves its pool open
    # bound to ITS loop; pytest-asyncio gives this test a fresh loop, so reusing
    # that pool would cross-loop hang. Drop the stale global ref (without awaiting
    # its loop-bound stop()) so get_browser_pool() builds a fresh pool in THIS loop;
    # stop OURS cleanly in teardown (same loop). See the cross-loop gotcha in
    # [[slide-service-html-chromium-renderer]].
    import src.services.browser_pool as bp

    bp._pool = None
    yield


@_needs_chromium
async def test_section_capture_and_band_fallback(_isolate_browser_pool) -> None:
    # ONE real-chromium test (the suite convention — see the pool fixture above).
    # Both the landmark-capture and the band-fallback paths are exercised here in
    # one loop, and the pool is dropped in `finally` so it doesn't leak onward.
    from src.services.browser_pool import close_browser_pool

    try:
        with tempfile.TemporaryDirectory() as d:
            out = await render_section_screenshots(
                _SECTIONED_HTML, Path(d), "desktop", timeout_ms=15000
            )
            names = [name for name, _ in out]
            assert "hero" in names  # data-section wins
            assert "pricing-plans" in names  # heading-derived
            for _name, path in out:
                assert Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

        with tempfile.TemporaryDirectory() as d:
            bands = await render_section_screenshots(
                _NO_LANDMARK_HTML, Path(d), "desktop", timeout_ms=15000
            )
            assert bands, "fallback must produce at least one band"
            assert all(name.startswith("band-") for name, _ in bands)
            for _name, path in bands:
                assert Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

        # Short full-width data-sections are sections, not bands (regression:
        # thin redesigns came back as band-01/band-02 and an empty script).
        with tempfile.TemporaryDirectory() as d:
            thin = await render_section_screenshots(
                _THIN_PAGE_HTML, Path(d), "desktop", timeout_ms=15000
            )
            assert [name for name, _ in thin] == ["intro", "more-info"]
            from PIL import Image as _Img

            for _name, path in thin:
                with _Img.open(path) as img:
                    w, h = img.size
                    assert h == round(w * 9 / 16)  # padded to 16:9, not a sliver

        # Top-level only: a nested named section never shadows its parent.
        with tempfile.TemporaryDirectory() as d:
            nested = await render_section_screenshots(
                _NESTED_SECTION_HTML, Path(d), "desktop", timeout_ms=15000
            )
            assert [name for name, _ in nested] == ["outer"]

        # The second pass still rejects decorative full-width slivers.
        with tempfile.TemporaryDirectory() as d:
            sliver = await render_section_screenshots(
                _SLIVER_SECTION_HTML, Path(d), "desktop", timeout_ms=15000
            )
            assert [name for name, _ in sliver] == ["intro"]

        # A single-call full-page redesign (round 4 example.com shape: full-width
        # header/footer + a centred ~832px content column, no data-section) comes
        # back NAMED after the generator's post-pass — not band-NN. The centred
        # column is only 43% of 1920, so this also pins the narrow-but-pads rule.
        from src.services.html_gen.section_landmarks import tag_data_sections
        from tests.unit.test_full_page_data_sections import MEASURED_FULL_PAGE_HTML

        with tempfile.TemporaryDirectory() as d:
            measured = await render_section_screenshots(
                tag_data_sections(MEASURED_FULL_PAGE_HTML), Path(d), "desktop", timeout_ms=15000
            )
            assert [name for name, _ in measured] == ["header", "example-domain", "footer"]

        # Band fallback names each band after the data-section at its midpoint.
        with tempfile.TemporaryDirectory() as d:
            named = await render_section_screenshots(
                _NARROW_SECTIONS_HTML, Path(d), "desktop", timeout_ms=15000
            )
            assert [name for name, _ in named] == ["alpha", "gamma", "gamma-2"]

        # Nested fragments (narrow logo header, narrow card, thin full-width
        # spacer) are rejected; only the full-width, tall-enough band survives.
        with tempfile.TemporaryDirectory() as d:
            frags = await render_section_screenshots(
                _FRAGMENT_HTML, Path(d), "desktop", timeout_ms=15000
            )
            names = [name for name, _ in frags]
            assert "real-band" in names
            assert "logo" not in names  # 320px-wide header dropped
            assert "chat-card" not in names  # 400px-wide card dropped
            assert "thin-spacer" not in names  # 170px-tall band dropped

        # End-to-end: the composite redesign (compose_section_page) must wrap each
        # logical section in a landmark so a real composite of N sections yields N
        # named bands — the variety fix. Previously these were comment-only and the
        # selector saw nothing, collapsing the after act onto ~2 nested landmarks.
        from src.services.html_generator import HtmlGenerator

        composed = HtmlGenerator(
            template_loader=None, wm_client=cast("WMClient", object())
        ).compose_section_page(
            {
                "hero": '<div style="min-height:600px"><h1>Acme</h1></div>',
                "features": '<div style="min-height:600px"><h2>Features</h2></div>',
                "footer": '<div style="min-height:400px">Contact</div>',
            },
            brand=None,
            context=None,
        )
        with tempfile.TemporaryDirectory() as d:
            comp = await render_section_screenshots(composed, Path(d), "desktop", timeout_ms=15000)
            comp_names = {name for name, _ in comp}
            assert {"hero", "features", "footer"} <= comp_names

        # A short full-width band is padded up to 16:9 (so the after-act cover-fit
        # doesn't over-zoom it); a tall one keeps its natural height.
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            mixed = await render_section_screenshots(
                _MIXED_HEIGHT_HTML, Path(d), "desktop", timeout_ms=15000
            )
            by_name = dict(mixed)
            assert {"short", "tall"} <= set(by_name)
            # PNG dimensions are DEVICE px: the page is captured at
            # device_scale_factor=2, so a band of N CSS px is a 2N px image.
            dsf = CAPTURE_DEVICE_SCALE_FACTOR
            with Image.open(by_name["short"]) as img:
                w, h = img.size
                assert w == _CSS_W * dsf  # supersampled, not captured at 1x
                assert h == round(w * 9 / 16)  # padded to exactly 16:9
                assert h > _SHORT_BAND_CSS * dsf  # taller than the captured band
            with Image.open(by_name["tall"]) as img:
                w2, h2 = img.size
                # Tall band untouched: kept at its natural height (modulo
                # sub-pixel capture rounding), NOT padded down/up to 16:9. It is
                # below the 1.6x split threshold, so it stays one band.
                assert abs(h2 - _TALL_BAND_CSS * dsf) <= 3 * dsf
                assert h2 != round(w2 * 9 / 16)

        # One greedy >16:9 landmark with three section-level headings is SPLIT into
        # three per-topic stills (the lever-2 variety fix), each letterboxed >= 16:9.
        with tempfile.TemporaryDirectory() as d:
            split = await render_section_screenshots(
                _TALL_MAIN_HTML, Path(d), "desktop", timeout_ms=15000
            )
            split_names = [name for name, _ in split]
            assert {"hero", "projects", "writing"} <= set(split_names)
            assert "main" not in split_names  # greedy parent replaced by its crops
            for _name, path in split:
                with Image.open(path) as img:
                    w, h = img.size
                    assert h >= round(w * 9 / 16)
                    assert Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

        # A tall band with NO inner headings degrades to exactly one still.
        with tempfile.TemporaryDirectory() as d:
            solid = await render_section_screenshots(
                _TALL_NO_HEADING_HTML, Path(d), "desktop", timeout_ms=15000
            )
            assert [name for name, _ in solid] == ["solid"]
    finally:
        await close_browser_pool()
