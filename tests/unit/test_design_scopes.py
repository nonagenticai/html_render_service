"""Tests for the html_render_service additions on top of the lifted redesign core:

- the widened scope vocabulary + validators on ``RedesignRequest``
- absolute artifact-URL construction (``_publicize`` / ``_artifact_url``)
- the scope -> generator dispatch mapping
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.routes._helpers import _artifact_url, _publicize
from src.api.schemas import ComponentTargetInput, RedesignRequest


class TestScopeVocabulary:
    @pytest.mark.parametrize(
        "scope",
        ["full_page", "redesign", "section", "sections", "hero", "landing"],
    )
    def test_url_scopes_accept_a_url(self, scope: str) -> None:
        req = RedesignRequest(url="https://example.com", scope=scope)
        assert req.scope == scope

    def test_component_requires_target(self) -> None:
        with pytest.raises(ValidationError, match="component_target is required"):
            RedesignRequest(url="https://example.com", scope="component")

    def test_component_with_target_ok(self) -> None:
        req = RedesignRequest(
            url="https://example.com",
            scope="component",
            component_target=ComponentTargetInput(component_type="navbar"),
        )
        assert req.component_target is not None

    def test_hero_needs_no_target(self) -> None:
        # hero auto-targets the hero section in the dispatch — no target required.
        req = RedesignRequest(url="https://example.com", scope="hero")
        assert req.scope == "hero"
        assert req.component_target is None

    def test_landing_allows_empty_url(self) -> None:
        # from-scratch landing page: no source url required.
        req = RedesignRequest(scope="landing")
        assert req.url == ""

    def test_non_landing_requires_url_or_html(self) -> None:
        with pytest.raises(ValidationError, match=r"url .* is required unless scope is 'landing'"):
            RedesignRequest(scope="full_page")

    def test_non_landing_accepts_inline_html(self) -> None:
        req = RedesignRequest(scope="full_page", html_content="<html><body>hi</body></html>")
        assert req.url == ""

    def test_unknown_scope_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RedesignRequest(url="https://example.com", scope="banner")

    def test_too_many_styles_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Maximum 4 style variants"):
            RedesignRequest(url="https://x.com", styles=["a", "b", "c", "d", "e"])


class _LocalStorageStub:
    """Stands in for the local backend: no MinIO, so presign yields None and
    ``_publicize`` falls back to the in-service proxy route."""

    async def presigned_url(self, path: str, expiry_seconds: int | None = None) -> str | None:
        return None


class _MinioStorageStub:
    """Stands in for the minio backend: returns a presigned minio.test URL,
    the fleet-standard form video_compiler fetches unauthenticated."""

    async def presigned_url(self, path: str, expiry_seconds: int | None = None) -> str | None:
        key = path.lstrip("/")
        return f"http://minio.test:8389/w2v-html-render/{key}?X-Amz-Signature=deadbeef"


class TestAbsoluteArtifactUrls:
    def test_artifact_url_shape(self) -> None:
        url = _artifact_url("job123", "screenshots", "desktop")
        assert url.endswith("/api/designs/jobs/job123/screenshots/desktop")
        assert url.startswith("http://") or url.startswith("https://")

    async def test_publicize_local_backend_falls_back_to_proxy(self) -> None:
        out = await _publicize(
            _LocalStorageStub(),
            "job123",
            "screenshots",
            {"desktop": "/local/x.png", "mobile": "/y.png"},
        )
        assert out is not None
        assert set(out.keys()) == {"desktop", "mobile"}
        for viewport, url in out.items():
            assert url.endswith(f"/api/designs/jobs/job123/screenshots/{viewport}")
            # local filesystem path must NOT leak to consumers
            assert "/local/" not in url

    async def test_publicize_minio_backend_returns_presigned(self) -> None:
        # The fix: minio backend hands back a presigned minio.test URL so
        # video_compiler can fetch it with no license header — NOT the authed
        # in-service proxy route (which 401'd the website_review compile stage).
        out = await _publicize(_MinioStorageStub(), "job123", "screenshots", {"desktop": "/x.png"})
        assert out is not None
        assert out["desktop"].startswith("http://minio.test:8389/w2v-html-render/")
        assert "X-Amz-Signature" in out["desktop"]
        assert "/api/designs/jobs/" not in out["desktop"]

    async def test_publicize_passthrough_none_and_empty(self) -> None:
        assert await _publicize(_LocalStorageStub(), "j", "screenshots", None) is None
        assert await _publicize(_LocalStorageStub(), "j", "screenshots", {}) == {}

    async def test_publicize_respects_public_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import clear_settings_cache

        monkeypatch.setenv("API_PUBLIC_BASE_URL", "https://html.example.test")
        clear_settings_cache()
        try:
            out = await _publicize(_LocalStorageStub(), "jid", "comparison", {"desktop": "/p.png"})
            assert out == {
                "desktop": "https://html.example.test/api/designs/jobs/jid/comparison/desktop"
            }
        finally:
            clear_settings_cache()
