"""Unit tests for the full-page screenshot height cap (no chromium required).

A pathologically tall SPA captured with ``full_page=True`` makes Chromium
allocate the entire scroll-height bitmap — the real OOM source on the heavy
path. ``_capture`` clips to ``redesign.max_full_page_height`` when the page is
taller than the cap. These tests exercise that decision with a fake page.
"""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.services.screenshot_renderer import _capture


class _FakePage:
    def __init__(self, scroll_height: int, scroll_width: int = 1440) -> None:
        self._scroll_height = scroll_height
        self._scroll_width = scroll_width
        self.shots: list[dict[str, Any]] = []
        self.evaluated: list[str] = []

    async def evaluate(self, script: str) -> int:
        self.evaluated.append(script)
        if "scrollHeight" in script:
            return self._scroll_height
        if "scrollWidth" in script:
            return self._scroll_width
        return 0

    async def screenshot(self, **kwargs: Any) -> None:
        self.shots.append(kwargs)


_STABILITY = {"animations": "disabled", "caret": "hide"}


async def test_viewport_capture_never_measures_or_clips() -> None:
    page = _FakePage(scroll_height=50000)
    await _capture(page, "/tmp/x.png", full_page=False)
    assert page.evaluated == []  # no scrollHeight probe on a viewport-framed shot
    assert page.shots == [{"path": "/tmp/x.png", "full_page": False, **_STABILITY}]


async def test_short_page_uses_full_page() -> None:
    cap = get_settings().redesign.max_full_page_height
    page = _FakePage(scroll_height=cap // 4)
    await _capture(page, "/tmp/x.png", full_page=True)
    assert page.shots == [{"path": "/tmp/x.png", "full_page": True, **_STABILITY}]


async def test_tall_page_is_clipped_to_cap() -> None:
    cap = get_settings().redesign.max_full_page_height
    page = _FakePage(scroll_height=cap * 3, scroll_width=1440)
    await _capture(page, "/tmp/x.png", full_page=True)
    assert len(page.shots) == 1
    shot = page.shots[0]
    assert shot["clip"] == {"x": 0, "y": 0, "width": 1440, "height": cap}


async def test_clipped_capture_still_asks_for_full_page() -> None:
    """``clip`` WITHOUT ``full_page`` is intersected with the viewport.

    This is the regression that made the cap a lie: Chromium only enables
    capture-beyond-viewport when full-page is requested, so a clip rect taller
    than the viewport came back one viewport tall while the log claimed an
    8000px clip. Measured on a 12000px page: clip(h=8000) alone -> 1440x900;
    clip(h=8000) + full_page -> 1440x8000. Both keys must be present together.
    """
    cap = get_settings().redesign.max_full_page_height
    page = _FakePage(scroll_height=cap * 3, scroll_width=1440)
    await _capture(page, "/tmp/x.png", full_page=True)
    shot = page.shots[0]
    assert shot.get("full_page") is True, (
        "clip without full_page is silently intersected with the viewport — "
        "the tall page would be truncated to the fold"
    )


async def test_every_capture_path_disables_animations_and_caret() -> None:
    """Capture fires straight after ``networkidle`` with no settle wait.

    A hero fade-in or slide-up reveal is therefore caught mid-transition, at a
    nondeterministic opacity, and that frame is what ships in the video.
    """
    cap = get_settings().redesign.max_full_page_height
    for scroll_height, full_page in ((50000, False), (cap // 4, True), (cap * 3, True)):
        page = _FakePage(scroll_height=scroll_height)
        await _capture(page, "/tmp/x.png", full_page=full_page)
        shot = page.shots[0]
        assert shot["animations"] == "disabled"
        assert shot["caret"] == "hide"
