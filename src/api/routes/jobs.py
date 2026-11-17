"""Redesign job lifecycle: create, status, result, cancel, list."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    AccessibilityCategoryResponse,
    AccessibilityFindingResponse,
    AccessibilityScoreResponse,
    AsyncJobResponse,
    CancelJobResponse,
    JobListItem,
    JobListResponse,
    JobStatusResponse,
    RedesignRequest,
    RedesignResultResponse,
    VariantResult,
)
from src.database.session import get_db
from src.services.design_builder.types import RenderOptions
from src.services.job_manager import JobManager
from src.services.job_worker import build_redesign_options

from ._helpers import (
    _build_visual_assets,
    _concept_image_assets_from_payload,
    _get_redesign_service,
    _get_storage,
    _get_tenant_id,
    _publicize,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# create_redesign uses path "" which collides with FastAPI's empty-prefix +
# empty-path validation when registered on a sub-router. It is therefore
# registered directly on the composite router in routes/__init__.py.
async def create_redesign(
    body: RedesignRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AsyncJobResponse:
    """Create a new redesign job (always async)."""
    tenant_id = _get_tenant_id(request)
    logger.info(
        "Received redesign request from tenant=%s url=%s scope=%s", tenant_id, body.url, body.scope
    )
    service = _get_redesign_service(request)

    job_id = await service.create_job(
        db,
        tenant_id=tenant_id,
        url=body.url,
        html_content=body.html_content,
        scope=body.scope,
        style=body.style,
        page_type=body.page_type,
        critique_findings=[f.model_dump() for f in body.critique_findings]
        if body.critique_findings
        else None,
        brand_info=body.brand.model_dump() if body.brand else None,
        site_analysis=body.site_analysis.model_dump() if body.site_analysis else None,
        user_context=body.user_context.model_dump(exclude_none=True) if body.user_context else None,
        styles=body.styles,
        component_target=body.component_target.model_dump() if body.component_target else None,
        webhook_url=body.webhook_url,
        webhook_secret=body.webhook_secret,
        mode=body.mode,
        interactivity=body.interactivity,
        component_set=body.component_set,
        embeds=[e.model_dump() for e in body.embeds] if body.embeds else None,
    )

    # Persist the full render/processing bundle on the row, then commit insert +
    # options atomically. A non-NULL job_options is the queue-ownership flag: the
    # worker pool only claims the job AFTER this commit makes it visible, so there
    # is no race with a half-initialized row. (Replaces the request-time
    # asyncio.create_task — the work now survives a pod restart.)
    render_options = RenderOptions(
        capture_original=body.capture_original,
        generate_comparison=body.generate_comparison,
        viewports=tuple(body.screenshot_viewports) if body.screenshot_viewports else None,
        full_page=body.full_page,
        section_screenshots=body.section_screenshots,
    )
    options = build_redesign_options(
        run_accessibility_check=body.run_accessibility_check,
        auto_fix_accessibility=body.auto_fix_accessibility,
        accessibility_threshold=body.accessibility_threshold,
        accessibility_first_auto_theme=body.accessibility_first_auto_theme,
        generate_screenshots=body.generate_screenshots,
        generate_concept_images=body.generate_concept_images,
        render_options=render_options,
        llm_config=body.llm_config,
    )
    await JobManager.set_job_options(db, job_id, options)
    await db.commit()

    logger.info("Enqueued redesign job %s for tenant=%s", job_id, tenant_id)
    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/designs/jobs/{job_id}",
        webhook_configured=body.webhook_url is not None,
    )


@router.get(
    "/jobs/{job_id}",
    responses={
        404: {"description": "Job not found"},
    },
    operation_id="get_redesign_job_status",
)
async def get_job_status(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    """Get status of a redesign job."""
    logger.debug("Status request for job %s", job_id)
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    result_url = f"/api/designs/jobs/{job_id}/result" if job.status == "completed" else None
    refine_url = f"/api/designs/jobs/{job_id}/refine" if job.status == "completed" else None

    token_usage = None
    if job.input_tokens or job.output_tokens:
        token_usage = {
            "prompt_tokens": job.input_tokens or 0,
            "completion_tokens": job.output_tokens or 0,
            "total_tokens": (job.input_tokens or 0) + (job.output_tokens or 0),
        }

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        url=job.url,
        scope=job.scope,
        style=job.style,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error_message,
        concept_image_degraded=job.concept_image_error is not None,
        concept_image_error=job.concept_image_error,
        result_url=result_url,
        iteration_number=job.iteration_number,
        parent_job_id=job.parent_job_id,
        root_job_id=job.root_job_id,
        refine_url=refine_url,
        accessibility_score=job.accessibility_score,
        batch_id=job.batch_id,
        token_usage=token_usage,
    )


@router.get(
    "/jobs/{job_id}/result",
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job not completed"},
    },
    operation_id="get_redesign_job_result",
)
async def get_job_result(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> RedesignResultResponse:
    """Get the full result of a completed redesign job.

    Served from a single builder off the authoritative DB row. A Redis result
    cache used to short-circuit this read, but it was a once-per-job endpoint
    whose cache was never invalidated and whose hand-maintained snapshot silently
    diverged from the DB serialization (it dropped the completion-gating
    ``verification_status``/``verification_report`` and ``variants``), so it was
    removed. There is exactly one serialization now — cache and DB cannot drift.
    """
    logger.debug("Result request for job %s", job_id)
    tenant_id = _get_tenant_id(request)
    job = await JobManager.get_job(db, job_id, tenant_id=tenant_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (status: {job.status})",
            headers={"Retry-After": "30"},
        )

    storage = _get_storage(request)

    variants = None
    if job.styles and len(job.styles) > 1:
        variant_records = await JobManager.get_variants(db, job_id, tenant_id=tenant_id)
        variants = [
            VariantResult(
                variant_id=v.variant_id,
                style=v.style,
                status=v.status,
                redesigned_html_path=v.redesigned_html_path,
                screenshot_paths=await _publicize(
                    storage, job_id, "screenshots", v.screenshot_paths
                ),
                error=v.error_message,
            )
            for v in variant_records
        ]

    a11y = None
    if job.accessibility_report:
        report = job.accessibility_report
        a11y = AccessibilityScoreResponse(
            overall_score=report.get("overall_score"),
            unscored_reason=report.get("unscored_reason"),
            categories=[
                AccessibilityCategoryResponse(
                    category=c.get("category", ""),
                    score=c.get("score"),
                    findings=[AccessibilityFindingResponse(**f) for f in c.get("findings", [])],
                )
                for c in report.get("categories", [])
            ],
            findings=[AccessibilityFindingResponse(**f) for f in report.get("findings", [])],
            html_was_fixed=report.get("html_was_fixed", False),
            auto_theme=report.get("auto_theme"),
            critique_adapter_validation=report.get("critique_adapter_validation"),
        )

    concept_image_assets = _concept_image_assets_from_payload(job.concept_image_assets)

    result_token_usage = None
    if job.input_tokens or job.output_tokens:
        result_token_usage = {
            "prompt_tokens": job.input_tokens or 0,
            "completion_tokens": job.output_tokens or 0,
            "total_tokens": (job.input_tokens or 0) + (job.output_tokens or 0),
        }

    screenshots_pub = await _publicize(storage, job_id, "screenshots", job.screenshot_paths)
    originals_pub = await _publicize(storage, job_id, "original", job.original_screenshot_paths)
    comparisons_pub = await _publicize(storage, job_id, "comparison", job.comparison_paths)
    diffs_pub = await _publicize(storage, job_id, "diff", job.diff_paths)
    sections_pub = await _publicize(storage, job_id, "sections", job.section_screenshot_paths)
    return RedesignResultResponse(
        job_id=job.job_id,
        url=job.url,
        scope=job.scope,
        style=job.style,
        redesigned_html_path=job.redesigned_html_path,
        section_html_paths=job.section_html_paths,
        screenshots=screenshots_pub,
        section_screenshots=sections_pub,
        concept_image_assets=concept_image_assets or None,
        concept_image_degraded=job.concept_image_error is not None,
        concept_image_error=job.concept_image_error,
        visual_assets=_build_visual_assets(
            screenshots=screenshots_pub,
            concept_image_assets=concept_image_assets,
            variants=variants,
            original_screenshots=originals_pub,
            comparisons=comparisons_pub,
            section_screenshots=sections_pub,
        ),
        variants=variants,
        component_html_path=job.component_html_path,
        accessibility=a11y,
        original_screenshots=originals_pub,
        comparisons=comparisons_pub,
        diffs=diffs_pub,
        html_validation=job.html_validation_report,
        verification_status=job.verification_status,
        verification_report=job.verification_report,
        token_usage=result_token_usage,
    )


@router.post(
    "/jobs/{job_id}/cancel",
    responses={
        409: {"description": "Job cannot be cancelled"},
    },
    operation_id="cancel_redesign_job",
)
async def cancel_job(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CancelJobResponse:
    """Cancel a pending or processing job."""
    logger.info("Cancel request for job %s", job_id)
    tenant_id = _get_tenant_id(request)
    cancelled = await JobManager.cancel_job(db, job_id, tenant_id=tenant_id)
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} cannot be cancelled (not found or already finished)",
        )
    await db.commit()

    job = await JobManager.get_job(db, job_id, tenant_id=tenant_id)
    return CancelJobResponse(
        job_id=job_id,
        status="cancelled",
        cancelled_at=job.cancelled_at.isoformat() if job and job.cancelled_at else "",
    )


@router.get("/jobs", operation_id="list_redesign_jobs")
async def list_jobs(
    request: Request,
    db: AsyncSession = Depends(get_db),
    status: str | None = Query(default=None, description="Filter by status"),
    page_type: str | None = Query(default=None, description="Filter by page type"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> JobListResponse:
    """List jobs for the current tenant with optional filters."""
    tenant_id = _get_tenant_id(request)
    logger.debug("List jobs request for tenant=%s status=%s limit=%d", tenant_id, status, limit)
    jobs, total = await JobManager.list_jobs(
        db,
        tenant_id,
        status=status,
        page_type=page_type,
        limit=limit,
        offset=offset,
    )
    return JobListResponse(
        jobs=[
            JobListItem(
                job_id=j.job_id,
                url=j.url,
                status=j.status,
                scope=j.scope,
                style=j.style,
                page_type=j.page_type,
                created_at=j.created_at.isoformat() if j.created_at else None,
                completed_at=j.completed_at.isoformat() if j.completed_at else None,
            )
            for j in jobs
        ],
        total=total,
        limit=limit,
        offset=offset,
    )
