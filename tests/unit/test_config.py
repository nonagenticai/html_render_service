"""Tests for html_render_service configuration."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.config import (
    APISettings,
    DesignSettings,
    ScreenshotSettings,
    StorageSettings,
    WorkerManagementSettings,
    clear_settings_cache,
    get_settings,
)


@pytest.fixture(autouse=True)
def _isolate_settings_cache() -> Iterator[None]:
    """``get_settings`` is ``lru_cache``d, so a Settings built under a patched
    environment would otherwise leak into every later test in the session."""
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_api_port_default() -> None:
    """html_render_service overrides the API port to 6319."""
    assert APISettings().port == 6319
    assert get_settings().api.port == 6319


def test_storage_bucket_default() -> None:
    """Storage MinIO bucket defaults to the service name."""
    assert StorageSettings().minio_bucket == "html-render-service"


def test_design_section_defaults() -> None:
    """The web-design (REDESIGN_) section exposes expected defaults."""
    design = DesignSettings()
    assert design.scope == "full_page"
    assert design.style == "modern"
    assert design.max_html_chars == 100000
    assert design.max_variants == 4


def test_screenshot_section_defaults() -> None:
    """The screenshot section exposes expected defaults."""
    screenshot = ScreenshotSettings()
    assert screenshot.enabled is True
    assert screenshot.viewports == ["desktop", "tablet", "mobile"]


def test_wm_section_defaults() -> None:
    """The worker_management (WM_) gateway section exposes expected defaults."""
    wm = WorkerManagementSettings()
    assert wm.timeout_seconds == 600.0
    assert wm.temperature == 0.2


def test_settings_hierarchy() -> None:
    """Settings composes all the domain sub-configs."""
    settings = get_settings()
    assert settings.app_name == "HTML Render Service"
    assert settings.app_version == "0.1.0"
    assert isinstance(settings.api, APISettings)
    assert isinstance(settings.storage, StorageSettings)
    assert isinstance(settings.redesign, DesignSettings)
    assert isinstance(settings.screenshot, ScreenshotSettings)
    assert isinstance(settings.wm, WorkerManagementSettings)


def test_settings_helper_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings helper methods behave consistently with the environment.

    ``Settings`` reads the real process environment, and CI exports
    ``ENVIRONMENT=test`` at the job level — so asserting ``is_development() is
    True`` was really asserting whatever the runner happened to export. It went
    red on main, and a red ``Lint, types, tests`` job SKIPS build, promote and
    the Flux rollout, so the merge landed and nothing ever deployed.

    This test is about the DEFAULT, so the ambient value is removed rather than
    accommodated. ``test_environment_helpers_follow_the_ambient_env`` below pins
    the other direction, so the ``delenv`` can never quietly decay into "the
    setting is ignored".
    """
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    clear_settings_cache()

    settings = get_settings()
    assert settings.is_development() is True
    assert settings.is_production() is False
    assert settings.get_database_url().startswith("postgresql+psycopg://")
    assert settings.get_redis_url().startswith("redis://")


def test_environment_helpers_follow_the_ambient_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default must still be overridable by the environment."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    clear_settings_cache()

    settings = get_settings()
    assert settings.is_production() is True
    assert settings.is_development() is False


def test_get_settings_is_cached() -> None:
    """get_settings returns a cached singleton."""
    assert get_settings() is get_settings()


@pytest.mark.parametrize("scope", ["full_page", "hero", "section", "component", "landing"])
def test_design_scope_is_freeform_string(scope: str) -> None:
    """Scope is a free-form string field (validated downstream, not by Settings)."""
    assert DesignSettings(scope=scope).scope == scope
