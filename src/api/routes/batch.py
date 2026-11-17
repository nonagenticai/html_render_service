"""Batch redesign endpoints (Feature 10, Enhancement 11)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    BatchJobResponse,
    BatchListItem,
    BatchListResponse,
    BatchPageErrorInfo,
    BatchPageJobInfo,
    BatchPageStatusInfo,
    BatchRedesignRequest,
    BatchResultsResponse,
    BatchRetryResponse,
    BatchStatusResponse,
    RedesignResultResponse,
)
from src.database.session import get_db

from ._helpers import (
    _background_tasks,
    _build_visual_assets,
    _concept_image_assets_from_payload,
    _get_redesign_service,
    _get_session_factory,
    _get_tenant_id,
    _process_batch_background,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/batch",
    responses={
        422: {"description": "Validation error"},
    },
    operation_id="create_redesign_batch",
)
async def create_batch(
    body: BatchRedesignRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> BatchJobResponse:
    """Create a batch redesign job for multiple pages."""
    tenant_id = _get_tenant_id(request)
    logger.info("Batch redesign request from tenant=%s pages=%d", tenant_id, len(body.pages))
    service = _get_redesign_service(request)

    pages = [
        {
            "url": p.url,
            "page_type": p.page_type,
            "html_content": p.html_content,
        }
        for p in body.pages
    ]

    batch_id, page_jobs = await service.create_batch(
        db,
        tenant_id=tenant_id,
        pages=pages,
        style=body.style,
        scope=body.scope,
        generate_screenshots=body.generate_screenshots,
        generate_concept_images=body.generate_concept_images,
        critique_findings=[f.model_dump() for f in body.critique_findings]
        if body.critique_findings
        else None,
        brand_info=body.brand.model_dump() if body.brand else None,
        site_analysis=body.site_analysis.model_dump() if body.site_analysis else None,
        user_context=body.user_context.model_dump(exclude_none=True) if body.user_context else None,
        webhook_url=body.webhook_url,
        webhook_secret=body.webhook_secret,
    )
    await db.commit()

    session_factory = _get_session_factory(request)
    task = asyncio.create_task(
        _process_batch_background(service, session_factory, batch_id, tenant_id)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return BatchJobResponse(
        batch_id=batch_id,
        status="pending",
        total_pages=len(page_jobs),
        page_jobs=[
            BatchPageJobInfo(
                job_id=pj["job_id"],
                url=pj["url"],
                page_type=pj.get("page_type"),
                priority_score=pj.get("priority_score"),
            )
            for pj in page_jobs
        ],
        poll_url=f"/api/designs/batch/{batch_id}",
    )


@router.get(
    "/batch/{batch_id}",
    responses={
        404: {"description": "Batch not found"},
    },
    operation_id="get_redesign_batch_status",
)
async def get_batch_status(
    batch_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> BatchStatusResponse:
    """Get status of a batch job."""
    logger.debug("Batch status request for batch_id=%s", batch_id)
    from src.services.batch_manager import BatchManager

    tenant_id = _get_tenant_id(request)
    batch = await BatchManager.get_batch(db, batch_id, tenant_id=tenant_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    jobs = await BatchManager.get_batch_jobs(db, batch_id, tenant_id=tenant_id)

    return BatchStatusResponse(
        batch_id=batch.batch_id,
        status=batch.status,
        total_pages=batch.total_pages,
        completed_pages=batch.completed_pages,
        failed_pages=batch.failed_pages,
        error=batch.error_message,
        page_statuses=[
            BatchPageStatusInfo(
                job_id=j.job_id,
                url=j.url,
                status=j.status,
                error=j.error_message,
            )
            for j in jobs
        ],
    )


@router.get(
    "/batch/{batch_id}/results",
    responses={
        404: {"description": "Batch not found"},
    },
    operation_id="get_redesign_batch_results",
)
async def get_batch_results(
    batch_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> BatchResultsResponse:
    """Get results of a completed batch job."""
    logger.debug("Batch results request for batch_id=%s", batch_id)
    from src.services.batch_manager import BatchManager

    tenant_id = _get_tenant_id(request)
    batch = await BatchManager.get_batch(db, batch_id, tenant_id=tenant_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    jobs = await BatchManager.get_batch_jobs(db, batch_id, tenant_id=tenant_id)

    results = []
    errors = []
    for j in jobs:
        if j.status == "completed":
            concept_image_assets = _concept_image_assets_from_payload(j.concept_image_assets)
            results.append(
                RedesignResultResponse(
                    job_id=j.job_id,
                    url=j.url,
                    scope=j.scope,
                    style=j.style,
                    redesigned_html_path=j.redesigned_html_path,
                    section_html_paths=j.section_html_paths,
                    screenshots=j.screenshot_paths,
                    concept_image_assets=concept_image_assets or None,
                    visual_assets=_build_visual_assets(
                        screenshots=j.screenshot_paths,
                        concept_image_assets=concept_image_assets,
                        original_screenshots=j.original_screenshot_paths,
                        comparisons=j.comparison_paths,
                    ),
                )
            )
        elif j.status == "failed":
            errors.append(
                BatchPageErrorInfo(
                    job_id=j.job_id,
                    url=j.url,
                    error=j.error_message or "Unknown error",
                )
            )

    return BatchResultsResponse(
        batch_id=batch.batch_id,
        status=batch.status,
        total_pages=batch.total_pages,
        completed_pages=batch.completed_pages,
        failed_pages=batch.failed_pages,
        results=results,
        errors=errors,
    )


@router.get("/batch", operation_id="list_redesign_batches")
async def list_batches(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> BatchListResponse:
    """List all batches for the current tenant."""
    from src.services.batch_manager import BatchManager

    tenant_id = _get_tenant_id(request)
    logger.debug("List batches request for tenant=%s", tenant_id)
    batches = await BatchManager.list_batches(db, tenant_id)
    return BatchListResponse(
        batches=[
            BatchListItem(
                batch_id=b.batch_id,
                status=b.status,
                total_pages=b.total_pages,
                completed_pages=b.completed_pages,
                failed_pages=b.failed_pages,
                created_at=b.created_at.isoformat() if b.created_at else None,
            )
            for b in batches
        ],
        total=len(batches),
    )


@router.post(
    "/batch/{batch_id}/cancel",
    responses={
        409: {"description": "Batch cannot be cancelled"},
    },
    operation_id="cancel_redesign_batch",
)
async def cancel_batch(
    batch_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Cancel a batch and all its pending child jobs."""
    logger.info("Cancel batch request for batch_id=%s", batch_id)
    from src.services.batch_manager import BatchManager

    cancelled = await BatchManager.cancel_batch(db, batch_id, tenant_id=_get_tenant_id(request))
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail=f"Batch {batch_id} cannot be cancelled (not found or already finished)",
        )
    await db.commit()
    return {"cancelled": True, "batch_id": batch_id}


@router.post(
    "/batch/{batch_id}/retry-failed",
    responses={
        404: {"description": "Batch not found"},
        409: {"description": "No failed jobs to retry"},
    },
    operation_id="retry_redesign_batch_failed",
)
async def retry_failed_batch_pages(
    batch_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> BatchRetryResponse:
    """Re-queue failed pages in a batch for retry."""
    logger.info("Retry failed pages request for batch_id=%s", batch_id)
    from src.services.batch_manager import BatchManager

    tenant_id = _get_tenant_id(request)
    batch = await BatchManager.get_batch(db, batch_id, tenant_id=tenant_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    failed_jobs = await BatchManager.get_failed_jobs(db, batch_id, tenant_id=tenant_id)
    if not failed_jobs:
        raise HTTPException(status_code=409, detail="No failed jobs to retry")

    retried_ids: list[str] = []
    for job in failed_jobs:
        await BatchManager.reset_job_for_retry(db, job.job_id)
        retried_ids.append(job.job_id)

    from sqlalchemy import update as sa_update

    from src.database.models import BatchJob as BatchJobModel

    await db.execute(
        sa_update(BatchJobModel)
        .where(BatchJobModel.batch_id == batch_id)
        .values(
            failed_pages=batch.failed_pages - len(retried_ids),
            status="processing",
        )
    )
    await db.commit()

    service = _get_redesign_service(request)
    session_factory = _get_session_factory(request)
    # SCOPED to the jobs this endpoint just reset. Without `retried_ids` the
    # background run re-rendered EVERY page in the batch: 3 failures in a
    # 20-page batch overwrote 17 already-approved designs with fresh LLM output,
    # charged the tenant for 17 renders they did not request, and left
    # `completed_pages` at 37 of 20. The endpoint is called retry-FAILED; this
    # is what makes the name true.
    task = asyncio.create_task(
        _process_batch_background(
            service, session_factory, batch_id, tenant_id, job_ids=retried_ids
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return BatchRetryResponse(
        batch_id=batch_id,
        retried_jobs=retried_ids,
        count=len(retried_ids),
    )
