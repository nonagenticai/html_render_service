"""Tests for the opt-in lean-render path.

Covers the new ``RedesignRequest`` screenshot-shaping flags and that
``_phase_screenshots`` honors them — skipping the live-original capture and
the before/after comparison, narrowing viewports, and propagating
``full_page`` to the renderer. These flags exist because rendering the full
set (6 full-page Chromium captures of a heavy live SPA + a PIL diff) is what
OOM-killed the pod; the promo only needs one viewport-framed hero shot.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.schemas import RedesignRequest
from src.config import get_settings
from src.services.design_builder.finalize_phases import RedesignFinalizeMixin
from src.services.design_builder.types import RenderOptions


class TestRequestRenderFlags:
    def test_defaults_reproduce_full_behavior(self) -> None:
        req = RedesignRequest(url="https://example.com", scope="hero")
        assert req.capture_original is True
        assert req.generate_comparison is True
        assert req.screenshot_viewports is None
        assert req.full_page is True

    def test_lean_overrides_accepted(self) -> None:
        req = RedesignRequest(
            url="https://example.com",
            scope="hero",
            capture_original=False,
            generate_comparison=False,
            screenshot_viewports=["desktop"],
            full_page=False,
        )
        assert req.capture_original is False
        assert req.generate_comparison is False
        assert req.screenshot_viewports == ["desktop"]
        assert req.full_page is False

    def test_section_screenshots_defaults_off(self) -> None:
        assert RedesignRequest(url="https://example.com").section_screenshots is False

    def test_section_screenshots_opt_in(self) -> None:
        req = RedesignRequest(url="https://example.com", section_screenshots=True)
        assert req.section_screenshots is True


@pytest.fixture
def finalize(tmp_path):
    """A bare RedesignFinalizeMixin with a stubbed storage service."""
    obj = RedesignFinalizeMixin.__new__(RedesignFinalizeMixin)
    storage = MagicMock()
    storage.get_job_dir.return_value = tmp_path
    storage.retrieve_text = AsyncMock(return_value="<html><body>hi</body></html>")
    # persist_paths is a no-op pass-through on the local backend; the minio
    # upload it performs is covered by test_storage_minio.
    storage.persist_paths = AsyncMock(side_effect=lambda paths: paths)
    obj._storage = storage  # type: ignore[attr-defined]
    return obj


@pytest.fixture
def render_mocks(tmp_path, monkeypatch):
    """Patch the package-level renderer fns; return the AsyncMocks."""
    cap = AsyncMock(return_value={"desktop": str(tmp_path / "orig.png")})
    rend = AsyncMock(return_value={"desktop": str(tmp_path / "redesign.png")})
    comp = AsyncMock(return_value=({"desktop": "c.png"}, {"desktop": "d.png"}))
    monkeypatch.setattr("src.services.design_builder.capture_url_screenshots", cap)
    monkeypatch.setattr("src.services.design_builder.render_from_html_string", rend)
    monkeypatch.setattr("src.services.design_builder.generate_comparisons", comp)
    return cap, rend, comp


def _job():
    job = MagicMock()
    job.url = "https://example.com"
    return job


async def test_lean_options_skip_original_and_comparison(finalize, render_mocks) -> None:
    cap, rend, comp = render_mocks
    opts = RenderOptions(
        capture_original=False,
        generate_comparison=False,
        viewports=("desktop",),
        full_page=False,
    )

    result = await finalize._phase_screenshots(
        _job(), "job1", "redesign.html", get_settings(), True, opts
    )

    cap.assert_not_called()
    comp.assert_not_called()
    rend.assert_awaited_once()
    # viewports narrowed to desktop, full_page propagated as False
    args, kwargs = rend.call_args
    assert args[2] == ["desktop"]
    assert kwargs["full_page"] is False
    assert result.original_paths is None
    assert result.comparison_paths is None
    assert result.screenshot_paths == {"desktop": str(rend.return_value["desktop"])}


async def test_greenfield_empty_url_skips_original_capture(finalize, render_mocks) -> None:
    # Greenfield design (scope=landing, no source URL): the original capture must
    # be skipped even with capture_original=True, else capture_url_screenshots does
    # Page.goto("") and Playwright's Error (not caught) fails the whole job.
    cap, rend, comp = render_mocks
    job = MagicMock()
    job.url = ""

    result = await finalize._phase_screenshots(
        job, "job1", "redesign.html", get_settings(), True, RenderOptions()
    )

    cap.assert_not_called()  # no original capture without a URL
    comp.assert_not_called()  # comparison needs an original
    rend.assert_awaited_once()  # the generated HTML is still rendered
    assert result.original_paths is None
    assert result.screenshot_paths == {"desktop": str(rend.return_value["desktop"])}


async def test_default_options_do_full_capture_and_comparison(finalize, render_mocks) -> None:
    cap, rend, comp = render_mocks

    await finalize._phase_screenshots(
        _job(), "job1", "redesign.html", get_settings(), True, RenderOptions()
    )

    cap.assert_awaited_once()
    rend.assert_awaited_once()
    comp.assert_awaited_once()
    # full multi-viewport, full-page by default
    assert cap.call_args.kwargs["full_page"] is True
    assert cap.call_args.args[2] == get_settings().screenshot.viewports


async def test_none_render_options_defaults_to_full(finalize, render_mocks) -> None:
    cap, rend, comp = render_mocks

    await finalize._phase_screenshots(_job(), "job1", "redesign.html", get_settings(), True, None)

    cap.assert_awaited_once()
    rend.assert_awaited_once()
    comp.assert_awaited_once()


async def test_generate_screenshots_false_short_circuits(finalize, render_mocks) -> None:
    cap, rend, comp = render_mocks

    result = await finalize._phase_screenshots(
        _job(), "job1", "redesign.html", get_settings(), False, RenderOptions()
    )

    cap.assert_not_called()
    rend.assert_not_called()
    comp.assert_not_called()
    assert result.screenshots_count == 0


@pytest.fixture
def section_mock(tmp_path, monkeypatch):
    """Patch the package-level section renderer; returns the AsyncMock."""
    sect = AsyncMock(
        return_value=[
            ("hero", str(tmp_path / "00_hero.png")),
            ("features", str(tmp_path / "01_features.png")),
        ]
    )
    monkeypatch.setattr("src.services.design_builder.render_section_screenshots", sect)
    return sect


async def test_section_screenshots_opt_in_captures_and_returns(
    finalize, render_mocks, section_mock
) -> None:
    opts = RenderOptions(section_screenshots=True, viewports=("desktop",))

    result = await finalize._phase_screenshots(
        _job(), "job1", "redesign.html", get_settings(), True, opts
    )

    section_mock.assert_awaited_once()
    # capture the FIRST viewport, not a viewport list
    assert section_mock.call_args.args[2] == "desktop"
    assert result.section_paths == {
        "hero": str(section_mock.return_value[0][1]),
        "features": str(section_mock.return_value[1][1]),
    }


async def test_section_screenshots_off_by_default(finalize, render_mocks, section_mock) -> None:
    result = await finalize._phase_screenshots(
        _job(), "job1", "redesign.html", get_settings(), True, RenderOptions()
    )

    section_mock.assert_not_called()
    assert result.section_paths is None


async def test_section_capture_failure_is_soft(finalize, render_mocks, section_mock) -> None:
    section_mock.side_effect = RuntimeError("browser crashed")
    opts = RenderOptions(section_screenshots=True, viewports=("desktop",))

    # The full-page screenshot still succeeds; section failure must not fail the job.
    result = await finalize._phase_screenshots(
        _job(), "job1", "redesign.html", get_settings(), True, opts
    )

    assert result.section_paths is None
    assert result.screenshot_paths is not None
