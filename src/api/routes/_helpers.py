"""Shared helpers, getters, asset builders, and background tasks for redesign routes."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.schemas import (
    ConceptImageAssetResponse,
    LLMConfigOverride,
    VariantResult,
    VisualAssetResponse,
)
from src.services.design_builder import RedesignService
from src.services.design_builder.types import RenderOptions
from src.services.job_manager import JobManager, get_claim_owner
from src.services.storage_service import StorageService
from src.services.upstream_failure import describe_failure, is_transient_upstream

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()


async def _record_background_failure(db: AsyncSession, job_id: str, exc: BaseException) -> None:
    """Persist the outcome of a background job that raised.

    The outermost of the two failure seams. The inner one (the pipeline's
    ``_phase_finalize_failed``) has usually already recorded the same exception
    and then re-raised it into here, so this must reach the SAME verdict — a
    plain ``mark_failed`` here would overwrite an inner requeue with a terminal
    ``failed`` and put back exactly the bug being fixed. ``mark_retryable`` is
    idempotent on an already-``pending`` row for that reason.
    """
    error_msg = describe_failure(exc)
    # Both writes are fenced on the claim this worker holds: by the time the
    # outer seam runs, the inner one may already have requeued the job and a
    # sibling may already have claimed it. Neither verdict is ours to write then.
    worker_id = get_claim_owner()
    if is_transient_upstream(exc) and await JobManager.mark_retryable(
        db, job_id, error_msg, worker_id=worker_id
    ):
        return
    await JobManager.mark_failed(db, job_id, error_msg, worker_id=worker_id)


def _artifact_url(job_id: str, kind: str, viewport: str) -> str:
    """Build an absolute, fetchable artifact URL served by this service.

    Stored artifact paths are local filesystem paths; downstream consumers
    (orchestrator promo_html / website_redesign) keep only ``http(s)`` URLs
    and drop relative paths. We expose the artifact-serving routes
    (``/api/designs/jobs/{id}/{kind}/{viewport}``) under an absolute base —
    the in-cluster service URL by default — so the URLs are fetchable by the
    orchestrator and by video_compiler cross-namespace.
    """
    from src.config import get_settings

    base = get_settings().api.public_base_url.rstrip("/")
    return f"{base}/api/designs/jobs/{job_id}/{kind}/{viewport}"


class _PresignProvider(Protocol):
    """The narrow slice of StorageService that ``_publicize`` depends on — keeps
    the helper testable with a duck-typed stub (no MinIO/settings needed)."""

    async def presigned_url(self, path: str, expiry_seconds: int | None = None) -> str | None: ...


async def _publicize(
    storage: _PresignProvider, job_id: str, kind: str, paths: dict[str, str] | None
) -> dict[str, str] | None:
    """Rewrite a viewport->stored-path map to viewport->fetchable URL.

    Prefers a presigned ``minio.test`` URL (the fleet convention): it's
    fetchable by video_compiler with no license header and re-presigned by the
    orchestrator's artifact proxy for the browser. Falls back to this service's
    authed proxy route only on the local backend (dev/tests), where there's no
    MinIO to sign and consumers stay in-process. This is the fix for the
    website_review compile 401 — video_compiler could not GET the authed proxy
    route, unlike every other (presigned) artifact in the fleet.
    """
    if not paths:
        return paths
    out: dict[str, str] = {}
    for viewport, path in paths.items():
        presigned = await storage.presigned_url(path)
        out[viewport] = presigned or _artifact_url(job_id, kind, viewport)
    return out


def _get_tenant_id(request: Request) -> str:
    """Return the caller's tenant as a canonical UUID string, or fail closed with 401.

    ``request.state.tenant_id`` is stamped by ``w2v_common.auth.middleware``
    (license path, Keycloak path, and the trusted-network path all set it, all as
    ``str(uuid)``). If it is absent, the request reached a route WITHOUT an
    established identity — a middleware ordering slip, a route wrongly listed
    public, or a directly-invoked handler.

    TWO FALLBACKS HAVE LIVED HERE AND BOTH WERE THE SAME BUG. The first returned
    the literal ``"anonymous"`` unconditionally: rows were created with
    ``tenant_id="anonymous"`` and the next unauthenticated caller — any other
    person on the planet — read, renamed and deleted them, because they landed in
    the same pooled scope. The second kept ``"anonymous"`` alive whenever
    ``settings.auth.enabled`` was False, on the argument that this was only the
    local/dev posture. It is not a safe carve-out: ``AUTH_ENABLED`` is a runtime
    env var, so one bad ConfigMap re-pools every tenant in prod, and the value
    still had to be a real tenant later — since w2v_common 0.67.0 a job enqueued
    under ``"anonymous"`` cannot even be dispatched, because the worker's
    ``set_identity_context`` rejects a non-UUID tenant. A dev convenience that
    creates rows no worker can process is not a convenience.

    So the tenant is now validated, not merely checked for presence:
    ``require_tenant`` proves it is a real, non-nil UUID and the canonical
    ``str(...)`` form is what every row and every RLS GUC binding sees. A missing
    OR unusable identity is not a tenant; it is a 401.

    Raises:
        HTTPException: 401 when no real tenant is resolvable for this request.
    """
    from w2v_common.tenant import TenantError

    from src.tenant import require_tenant

    raw = getattr(request.state, "tenant_id", None)
    try:
        return str(require_tenant(raw, source="request.state.tenant_id"))
    except TenantError as exc:
        # The message never echoes the value: an X-Tenant-Id field can carry a
        # ``Bearer <license token>`` and a 401 body is attacker-visible.
        logger.warning("Refusing request with no usable tenant: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="Authentication required: no tenant established for this request",
        ) from exc


def _get_redesign_service(request: Request) -> RedesignService:
    """Retrieve the RedesignService instance from application state."""
    return request.app.state.redesign_service


def _get_storage(request: Request) -> StorageService:
    """Retrieve the StorageService instance from application state."""
    return request.app.state.storage_service


def _get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    """Retrieve the async session factory from application state."""
    return request.app.state.session_factory


def _viewport_assets(
    items: dict[str, str] | None,
    *,
    asset_id_prefix: str,
    title_prefix: str,
    kind: str,
    role_prefix: str,
    source: str,
) -> list[VisualAssetResponse]:
    """Build viewport-keyed VisualAssetResponse entries (screenshot/comparison style)."""
    out: list[VisualAssetResponse] = []
    for viewport, url in (items or {}).items():
        if not url:
            continue
        out.append(
            VisualAssetResponse(
                asset_id=f"{asset_id_prefix}-{viewport}",
                title=f"{title_prefix} {viewport.title()}",
                kind=kind,
                role=f"{role_prefix}_{viewport}",
                url=url,
                source=source,
                metadata={"viewport": viewport},
            )
        )
    return out


def _concept_asset_metadata(asset: ConceptImageAssetResponse, idx: int) -> dict:
    metadata = dict(asset.metadata or {})
    metadata.setdefault("index", idx + 1)
    if asset.job_id:
        metadata.setdefault("job_id", asset.job_id)
    if asset.provider:
        metadata.setdefault("provider", asset.provider)
    if asset.download_url:
        metadata.setdefault("download_url", asset.download_url)
    if asset.presigned_url:
        metadata.setdefault("presigned_url", asset.presigned_url)
    if asset.image_url:
        metadata.setdefault("image_url", asset.image_url)
    _prov = metadata.get("provenance")
    provenance: dict[str, object] = dict(_prov) if isinstance(_prov, dict) else {}
    provenance.setdefault("asset_id", asset.asset_id)
    if asset.job_id:
        provenance.setdefault("job_id", asset.job_id)
    if asset.provider:
        provenance.setdefault("provider", asset.provider)
    if asset.target_usage:
        provenance.setdefault("target_usage", asset.target_usage)
    metadata["provenance"] = provenance
    return metadata


def _concept_image_assets(
    concept_image_assets: list[ConceptImageAssetResponse] | None,
) -> list[VisualAssetResponse]:
    out: list[VisualAssetResponse] = []
    for idx, asset in enumerate(concept_image_assets or []):
        metadata = _concept_asset_metadata(asset, idx)
        out.append(
            VisualAssetResponse(
                asset_id=asset.asset_id,
                title=str(metadata.get("title") or f"Concept {idx + 1}"),
                kind="generated_concept",
                role=f"concept_{idx + 1}",
                url=asset.url,
                source="content_generation",
                metadata=metadata,
            )
        )
    return out


def _variant_assets(variants: list[VariantResult] | None) -> list[VisualAssetResponse]:
    out: list[VisualAssetResponse] = []
    for variant in variants or []:
        for viewport, url in (variant.screenshot_paths or {}).items():
            if not url:
                continue
            out.append(
                VisualAssetResponse(
                    asset_id=f"variant-{variant.variant_id}-{viewport}",
                    title=f"{variant.style.title()} {viewport.title()}",
                    kind="variant_screenshot",
                    role=f"variant_{variant.style}_{viewport}",
                    url=url,
                    source="variant",
                    metadata={"variant_id": variant.variant_id, "viewport": viewport},
                )
            )
    return out


def _section_assets(
    section_screenshots: dict[str, str] | None,
) -> list[VisualAssetResponse]:
    """Build one VisualAssetResponse per discrete redesign-section still.

    Role is ``redesign_section:<name>`` (NOT viewport-keyed): the orchestrator's
    after-act visual director maps each narration line to the section it discusses
    by matching on this name/title, then compile shows that section settled.
    """
    out: list[VisualAssetResponse] = []
    for name, url in (section_screenshots or {}).items():
        if not url:
            continue
        title = name.replace("-", " ").replace("_", " ").title()
        out.append(
            VisualAssetResponse(
                asset_id=f"redesign-section-{name}",
                title=f"Redesign — {title}",
                kind="screenshot",
                role=f"redesign_section:{name}",
                url=url,
                source="redesign",
                metadata={"section": name},
            )
        )
    return out


def _build_visual_assets(
    *,
    screenshots: dict[str, str] | None = None,
    concept_image_assets: list[ConceptImageAssetResponse] | None = None,
    variants: list[VariantResult] | None = None,
    original_screenshots: dict[str, str] | None = None,
    comparisons: dict[str, str] | None = None,
    section_screenshots: dict[str, str] | None = None,
) -> list[VisualAssetResponse]:
    """Build structured visual asset metadata from redesign result fields."""
    assets: list[VisualAssetResponse] = []
    assets.extend(
        _viewport_assets(
            screenshots,
            asset_id_prefix="redesign",
            title_prefix="Redesign",
            kind="screenshot",
            role_prefix="redesign",
            source="redesign",
        )
    )
    assets.extend(_section_assets(section_screenshots))
    assets.extend(_concept_image_assets(concept_image_assets))
    assets.extend(
        _viewport_assets(
            original_screenshots,
            asset_id_prefix="original",
            title_prefix="Original",
            kind="screenshot",
            role_prefix="original",
            source="source_capture",
        )
    )
    assets.extend(
        _viewport_assets(
            comparisons,
            asset_id_prefix="comparison",
            title_prefix="Comparison",
            kind="comparison",
            role_prefix="comparison",
            source="comparison_generator",
        )
    )
    assets.extend(_variant_assets(variants))
    return assets


def _preferred_concept_image_url(payload: dict[str, object]) -> str:
    for key in ("url", "presigned_url", "download_url", "image_url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _concept_image_assets_from_payload(payload: object) -> list[ConceptImageAssetResponse]:
    assets: list[ConceptImageAssetResponse] = []
    for raw_item in payload if isinstance(payload, list) else []:
        if not isinstance(raw_item, dict):
            continue
        item: dict[str, object] = {str(k): v for k, v in raw_item.items()}
        url = _preferred_concept_image_url(item)
        asset_id = str(item.get("asset_id") or "").strip()
        if not url or not asset_id:
            continue
        _meta = item.get("metadata")
        metadata: dict[str, object] = (
            {str(k): v for k, v in _meta.items()} if isinstance(_meta, dict) else {}
        )
        assets.append(
            ConceptImageAssetResponse(
                prompt=str(item.get("prompt") or "") or None,
                job_id=str(item.get("job_id") or "") or None,
                asset_id=asset_id,
                provider=str(item.get("provider") or "") or None,
                status=str(item.get("status") or "completed"),
                url=url,
                download_url=str(item.get("download_url") or "") or None,
                presigned_url=str(item.get("presigned_url") or "") or None,
                image_url=str(item.get("image_url") or "") or None,
                target_usage=str(item.get("target_usage") or "") or None,
                usage_context=(
                    _uc if isinstance((_uc := item.get("usage_context")), dict) else None
                ),
                mime_type=str(item.get("mime_type") or "") or None,
                width=(_w if isinstance((_w := item.get("width")), int) else None),
                height=(_h if isinstance((_h := item.get("height")), int) else None),
                file_size_bytes=(
                    _fsb if isinstance((_fsb := item.get("file_size_bytes")), int) else None
                ),
                cost=(
                    float(_cost) if isinstance((_cost := item.get("cost")), int | float) else None
                ),
                error=str(item.get("error") or "") or None,
                created_at=str(item.get("created_at") or "") or None,
                completed_at=str(item.get("completed_at") or "") or None,
                metadata=metadata or None,
            )
        )
    return assets


async def _process_job_background(
    service: RedesignService,
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    tenant_id: str,
    *,
    run_accessibility_check: bool = False,
    auto_fix_accessibility: bool = False,
    accessibility_threshold: float = 70.0,
    accessibility_first_auto_theme: bool = False,
    generate_screenshots: bool = True,
    generate_concept_images: bool = False,
    render_options: RenderOptions | None = None,
    llm_config: LLMConfigOverride | None = None,
) -> None:
    """Background task to process a redesign job with an independent session."""
    try:
        async with session_factory() as db:
            await service.process_job(
                db,
                job_id,
                tenant_id,
                session_factory=session_factory,
                run_accessibility_check=run_accessibility_check,
                auto_fix_accessibility=auto_fix_accessibility,
                accessibility_threshold=accessibility_threshold,
                accessibility_first_auto_theme=accessibility_first_auto_theme,
                generate_screenshots=generate_screenshots,
                generate_concept_images=generate_concept_images,
                render_options=render_options,
                llm_config=llm_config,
            )
    except Exception as exc:
        logger.exception("Background redesign job %s failed", job_id)
        try:
            async with session_factory() as fallback_session, fallback_session.begin():
                await _record_background_failure(fallback_session, job_id, exc)
        except Exception:
            logger.exception("Failed to mark job %s as failed", job_id)


async def _process_refinement_background(
    service: RedesignService,
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    tenant_id: str,
    *,
    generate_screenshots: bool = True,
    generate_concept_images: bool = False,
    focus_areas: list[str] | None = None,
    llm_config: LLMConfigOverride | None = None,
) -> None:
    """Background task to process a refinement job with an independent session."""
    try:
        async with session_factory() as db:
            await service.process_refinement_job(
                db,
                job_id,
                tenant_id,
                generate_screenshots=generate_screenshots,
                generate_concept_images=generate_concept_images,
                focus_areas=focus_areas,
                llm_config=llm_config,
            )
    except Exception as exc:
        logger.exception("Background refinement job %s failed", job_id)
        try:
            async with session_factory() as fallback_session, fallback_session.begin():
                await _record_background_failure(fallback_session, job_id, exc)
        except Exception:
            logger.exception("Failed to mark job %s as failed", job_id)


async def _process_batch_background(
    service: RedesignService,
    session_factory: async_sessionmaker[AsyncSession],
    batch_id: str,
    tenant_id: str,
    job_ids: list[str] | None = None,
) -> None:
    """Background task to process pages in a batch.

    ``job_ids`` is passed straight through to ``process_batch``; see its
    docstring for why the retry path must name the pages it retries.
    """
    try:
        await service.process_batch(session_factory, batch_id, tenant_id, job_ids=job_ids)
    except Exception as exc:
        logger.exception("Background batch %s failed", batch_id)
        try:
            from src.services.batch_manager import BatchManager

            async with session_factory() as fallback_session, fallback_session.begin():
                await BatchManager.mark_batch_failed(fallback_session, batch_id, str(exc))
        except Exception:
            logger.exception("Failed to mark batch %s as failed", batch_id)
