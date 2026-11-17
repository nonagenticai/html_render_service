"""Redesign iteration/refinement endpoints (Feature 5)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    AsyncJobResponse,
    IterationHistoryResponse,
    IterationSummary,
    RefineRequest,
)
from src.database.session import get_db
from src.services.job_manager import JobManager
from src.services.job_worker import build_refinement_options

from ._helpers import (
    _get_redesign_service,
    _get_tenant_id,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/jobs/{job_id}/refine",
    responses={
        400: {"description": "Invalid refinement request"},
        422: {"description": "Validation error"},
    },
    operation_id="refine_redesign",
)
async def refine_redesign(
    job_id: str,
    body: RefineRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AsyncJobResponse:
    """Refine an existing redesign with feedback."""
    tenant_id = _get_tenant_id(request)
    logger.info("Refinement request for parent job %s from tenant=%s", job_id, tenant_id)
    service = _get_redesign_service(request)

    try:
        new_job_id = await service.create_refinement_job(
            db,
            parent_job_id=job_id,
            feedback=body.feedback,
            generate_screenshots=body.generate_screenshots,
            generate_concept_images=body.generate_concept_images,
            tenant_id=tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Persist the refinement bundle + commit atomically so the worker pool picks
    # it up off the queue (durable across restarts), replacing the old
    # request-time asyncio.create_task.
    options = build_refinement_options(
        generate_screenshots=body.generate_screenshots,
        generate_concept_images=body.generate_concept_images,
        focus_areas=body.focus_areas,
        llm_config=body.llm_config,
    )
    await JobManager.set_job_options(db, new_job_id, options)
    await db.commit()

    return AsyncJobResponse(
        job_id=new_job_id,
        status="pending",
        poll_url=f"/api/designs/jobs/{new_job_id}",
    )


@router.get(
    "/jobs/{job_id}/history",
    responses={
        404: {"description": "Job not found"},
    },
    operation_id="list_redesign_iteration_history",
)
async def get_iteration_history(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> IterationHistoryResponse:
    """Get the full iteration history for a job."""
    logger.debug("Iteration history request for job %s", job_id)
    tenant_id = _get_tenant_id(request)
    job = await JobManager.get_job(db, job_id, tenant_id=tenant_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    root_id = job.root_job_id or job.job_id
    history = await JobManager.get_iteration_history(db, root_id, tenant_id=tenant_id)

    return IterationHistoryResponse(
        root_job_id=root_id,
        iterations=[
            IterationSummary(
                job_id=j.job_id,
                iteration_number=j.iteration_number or 0,
                status=j.status,
                feedback=j.feedback_text,
                created_at=j.created_at.isoformat() if j.created_at else None,
            )
            for j in history
        ],
        total_iterations=len(history),
    )
