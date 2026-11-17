"""An unrecognised viewport must fail at the boundary, not silently yield 0 screenshots.

``screenshot_viewports`` was a bare ``list[str]``. ``["Desktop"]`` (wrong case) got
all the way to ``screenshot_renderer`` which logged "Unknown viewport, skipping",
``finalize_phases`` computed ``screenshots_count = 0`` and the job completed —
a design with no screenshots at all, reported as success.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import RedesignRequest
from src.config import DesignSettings
from src.services.screenshot_renderer import VIEWPORT_SIZES


@pytest.mark.parametrize("bad", [["Desktop"], ["desktop", "phone"], ["MOBILE"], [""]])
def test_request_rejects_unknown_viewport(bad: list[str]) -> None:
    with pytest.raises(ValidationError, match="viewport"):
        RedesignRequest(url="https://example.com", screenshot_viewports=bad)


@pytest.mark.parametrize("good", [None, ["desktop"], ["desktop", "tablet", "mobile"]])
def test_request_accepts_known_viewports(good: list[str] | None) -> None:
    req = RedesignRequest(url="https://example.com", screenshot_viewports=good)
    assert req.screenshot_viewports == good


def test_every_known_viewport_is_accepted() -> None:
    req = RedesignRequest(url="https://example.com", screenshot_viewports=sorted(VIEWPORT_SIZES))
    assert req.screenshot_viewports == sorted(VIEWPORT_SIZES)


def test_settings_reject_an_unknown_configured_viewport() -> None:
    """The same silent-zero-screenshots hole exists via config, not just the request."""
    with pytest.raises(ValidationError, match="viewport"):
        DesignSettings(screenshot_viewports=["Desktop"])


def test_screenshot_settings_reject_an_unknown_viewport() -> None:
    """``settings.screenshot.viewports`` is the list the renderer is actually handed."""
    from src.config import ScreenshotSettings

    with pytest.raises(ValidationError, match="viewport"):
        ScreenshotSettings(viewports=["desktop", "Tablet"])
    assert ScreenshotSettings().viewports == ["desktop", "tablet", "mobile"]
