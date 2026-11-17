"""Redesign variant comparison and download endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import VariantComparisonResponse, VariantResult
from src.database.session import get_db
from src.services.job_manager import JobManager

from ._helpers import _get_storage, _get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/jobs/{job_id}/compare",
    responses={
        404: {"description": "Job or variants not found"},
    },
    operation_id="compare_redesign_variants",
)
async def compare_variants(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> VariantComparisonResponse:
    """Compare all style variants for a job."""
    logger.debug("Variant comparison request for job %s", job_id)
    tenant_id = _get_tenant_id(request)
    job = await JobManager.get_job(db, job_id, tenant_id=tenant_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    variants = await JobManager.get_variants(db, job_id, tenant_id=tenant_id)
    if not variants:
        raise HTTPException(status_code=404, detail="No variants found for this job")

    return VariantComparisonResponse(
        job_id=job_id,
        variants=[
            VariantResult(
                variant_id=v.variant_id,
                style=v.style,
                status=v.status,
                redesigned_html_path=v.redesigned_html_path,
                screenshot_paths=v.screenshot_paths,
                error=v.error_message,
            )
            for v in variants
        ],
    )


@router.get(
    "/jobs/{job_id}/variants/{style}/html",
    responses={
        404: {"description": "Variant or HTML not found"},
        409: {"description": "Variant not completed"},
    },
    operation_id="download_redesign_variant_html",
)
async def download_variant_html(
    job_id: str,
    style: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download the HTML for a specific variant."""
    logger.debug("Variant HTML download for job %s style=%s", job_id, style)
    variants = await JobManager.get_variants(db, job_id, tenant_id=_get_tenant_id(request))
    variant = next((v for v in variants if v.style == style), None)
    if variant is None:
        raise HTTPException(status_code=404, detail=f"Variant not found: {style}")
    if variant.status != "completed" or not variant.redesigned_html_path:
        raise HTTPException(
            status_code=409,
            detail=f"Variant not completed (status: {variant.status})",
            headers={"Retry-After": "30"},
        )

    storage = _get_storage(request)
    html_data = await storage.retrieve_text(variant.redesigned_html_path)
    if html_data is None:
        raise HTTPException(status_code=404, detail="Variant HTML file not found")

    return Response(
        content=html_data,
        media_type="text/html",
        headers={
            "Content-Disposition": f'inline; filename="variant_{job_id}_{style}.html"',
        },
    )
