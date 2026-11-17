"""The capture-resolution contract (no chromium required).

Delivered stills are cover-fit to a 1920x1080 video frame downstream. Capturing
at 1440x900 therefore UPSCALED every slide 1.33x — the soft picture the owner
reported. The fix is two-part and both parts are load-bearing:

* the desktop viewport is the delivery frame (1920x1080), so no upscale happens;
* every captured page is supersampled at ``device_scale_factor=2``, so the
  1920x1080 frame is resampled DOWN from 3840x2160 instead of interpolated up.

These are source-level property tests on purpose. The defect they exist to
prevent is not "one call site is wrong today" — it is that
``device_scale_factor`` was set in exactly ZERO places across this repo, and a
newly added ``new_page`` would silently rejoin that state. A test that only
checked the four call sites that exist now would not notice.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from src.config import PlaywrightSettings
from src.services import screenshot_renderer as sr


def _module_tree() -> ast.Module:
    return ast.parse(Path(inspect.getfile(sr)).read_text(encoding="utf-8"))


def _calls(tree: ast.Module, attr: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
    ]


def _double_star_names(call: ast.Call) -> set[str]:
    """Names spread into ``call`` via ``**name`` / ``**fn(...)``."""
    out: set[str] = set()
    for kw in call.keywords:
        if kw.arg is not None:
            continue
        if isinstance(kw.value, ast.Name):
            out.add(kw.value.id)
        elif isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Name):
            out.add(kw.value.func.id)
    return out


class TestDeliveryViewport:
    def test_desktop_is_the_1920x1080_delivery_frame(self) -> None:
        assert sr.VIEWPORT_SIZES["desktop"] == {"width": 1920, "height": 1080}

    def test_device_viewports_are_untouched(self) -> None:
        # tablet/mobile are real device CSS sizes — they exist to exercise
        # responsive breakpoints, not to fill a video frame. Resizing them to
        # the delivery frame would make them meaningless.
        assert sr.VIEWPORT_SIZES["tablet"] == {"width": 768, "height": 1024}
        assert sr.VIEWPORT_SIZES["mobile"] == {"width": 390, "height": 844}

    def test_capture_is_supersampled(self) -> None:
        assert sr.CAPTURE_DEVICE_SCALE_FACTOR == 2

    def test_page_args_carry_both_viewport_and_scale(self) -> None:
        args = sr._page_args(sr.VIEWPORT_SIZES["desktop"])
        assert args == {
            "viewport": {"width": 1920, "height": 1080},
            "device_scale_factor": 2,
        }


class TestEveryCaptureSiteOptsIn:
    def test_no_new_page_hardcodes_a_viewport(self) -> None:
        """A literal ``viewport=`` re-pins the size and skips the scale factor."""
        offenders = [
            ast.unparse(c)
            for c in _calls(_module_tree(), "new_page")
            if any(kw.arg == "viewport" for kw in c.keywords)
        ]
        assert offenders == [], (
            f"new_page() must take **_page_args(size), not a literal viewport: {offenders}"
        )

    def test_every_sized_new_page_goes_through_page_args(self) -> None:
        tree = _module_tree()
        sized = [c for c in _calls(tree, "new_page") if c.args or c.keywords]
        # Every sized page in this module is a CAPTURE page.
        assert sized, "expected capture pages in screenshot_renderer"
        for call in sized:
            assert "_page_args" in _double_star_names(call), (
                f"capture page opened without _page_args: {ast.unparse(call)}"
            )

    def test_every_screenshot_call_is_stabilised(self) -> None:
        """Capture fires immediately after ``networkidle``, with no settle wait.

        Without ``animations='disabled'`` a CSS reveal is captured at whatever
        opacity/offset it happened to reach, and ``caret='hide'`` keeps a
        blinking text caret out of a delivered frame.
        """
        tree = _module_tree()
        shots = _calls(tree, "screenshot")
        assert len(shots) >= 6, f"expected every capture site to be covered, found {len(shots)}"
        for call in shots:
            assert "_SHOT_STABILITY" in _double_star_names(call), (
                f"screenshot() without **_SHOT_STABILITY: {ast.unparse(call)}"
            )

    def test_shot_stability_is_what_it_claims(self) -> None:
        assert sr._SHOT_STABILITY == {"animations": "disabled", "caret": "hide"}


class TestSectionThresholdsRederived:
    def test_min_height_tracks_the_desktop_width(self) -> None:
        """The floor is an ASPECT question, so it scales with the viewport width.

        At 1440 the tuned value was 300 (a 4.8:1 band, 63% letterbox once padded
        to 16:9). Carrying 300 over to a 1920-wide viewport would have admitted
        6.4:1 slivers at 72% letterbox.
        """
        assert sr._MIN_SECTION_HEIGHT_PX == 400
        assert (
            round(sr.VIEWPORT_SIZES["desktop"]["width"] * (300 / 1440)) == sr._MIN_SECTION_HEIGHT_PX
        )

    def test_min_width_is_relative_and_needs_no_rederivation(self) -> None:
        # This one is a FRACTION the caller multiplies by the live viewport
        # width, so it already means the same thing at 1440 and at 1920.
        assert sr._MIN_SECTION_WIDTH_FRAC == 0.6


class TestDeadViewportConfigIsGone:
    def test_playwright_settings_expose_no_viewport_scalars(self) -> None:
        """``PLAYWRIGHT_VIEWPORT_WIDTH``/``_HEIGHT`` were read by nothing.

        They defaulted to 1440x900 and so read, during the soft-picture
        investigation, exactly like the live capture size. The renderer sizes
        pages from ``VIEWPORT_SIZES``; gitops defines no ``PLAYWRIGHT_*`` key.
        """
        fields = set(PlaywrightSettings.model_fields)
        assert "viewport_width" not in fields
        assert "viewport_height" not in fields
