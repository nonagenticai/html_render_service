"""Redesign artifact downloads: HTML, screenshots, originals, comparisons, diffs, components.

Also exposes a stateless ``POST /render/pdf`` that renders an arbitrary self-contained
HTML string to a PDF via headless Chromium — used by callers that hold their own HTML
(e.g. the orchestrator's audit-report stage) rather than a stored redesign job.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.session import get_db
from src.services.job_manager import JobManager

from ._helpers import _get_storage, _get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter()


class PdfRenderRequest(BaseModel):
    """A self-contained HTML document to render to a PDF.

    "Self-contained" is enforced at the sink, not here: the render page refuses
    every http(s) subresource that is not public
    (``screenshot_renderer.block_private_subresources``). Inline/``data:``
    assets and public remote assets both render; an ``<iframe>`` or ``<img>``
    aimed at MinIO, a ClusterIP or cloud metadata is aborted before the request
    leaves the pod, so its bytes cannot appear in the returned PDF.
    """

    html: Annotated[str, Field(min_length=1)]


@router.get(
    "/jobs/{job_id}/html",
    responses={
        404: {"description": "Job or HTML not found"},
        409: {"description": "Job not completed"},
    },
    operation_id="download_redesign_html",
)
async def download_html(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download the redesigned HTML for a completed job."""
    logger.debug("HTML download request for job %s", job_id)
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (status: {job.status})",
            headers={"Retry-After": "30"},
        )
    if not job.redesigned_html_path:
        raise HTTPException(status_code=404, detail="Redesigned HTML not found")

    storage = _get_storage(request)
    html_data = await storage.retrieve_text(job.redesigned_html_path)
    if html_data is None:
        raise HTTPException(status_code=404, detail="HTML file not found on storage")

    return Response(
        content=html_data,
        media_type="text/html",
        headers={
            "Content-Disposition": f'inline; filename="redesign_{job_id}.html"',
            # Defense-in-depth for live-served artifacts (the frontend also renders
            # these in a sandboxed iframe). Assembled pages inline their own CSS/JS
            # (hence 'unsafe-inline'); no external script origin is allowed. Embeds
            # are same-origin-free sandboxed iframes → frame-src https:.
            "Content-Security-Policy": (
                "default-src 'none'; "
                "style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; "
                "img-src https: data:; "
                "font-src https: data:; "
                "frame-src https:; "
                "connect-src 'none'; "
                "base-uri 'none'; "
                "form-action 'none'"
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/jobs/{job_id}/pdf",
    responses={
        404: {"description": "Job or HTML not found"},
        409: {"description": "Job not completed"},
    },
    operation_id="download_redesign_pdf",
)
async def download_pdf(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Render the generated HTML to PDF on demand (headless Chromium print)."""
    logger.debug("PDF download request for job %s", job_id)
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (status: {job.status})",
            headers={"Retry-After": "30"},
        )
    if not job.redesigned_html_path:
        raise HTTPException(status_code=404, detail="Redesigned HTML not found")

    storage = _get_storage(request)
    html_data = await storage.retrieve_text(job.redesigned_html_path)
    if html_data is None:
        raise HTTPException(status_code=404, detail="HTML file not found on storage")

    from src.services.screenshot_renderer import (
        DESIGN_PDF_LAYOUT_WIDTH_PX,
        render_pdf_from_html_string,
    )

    # A design concept prints its DESKTOP layout, scaled onto A4.
    pdf_bytes = await render_pdf_from_html_string(
        html_data, layout_width_px=DESIGN_PDF_LAYOUT_WIDTH_PX
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="design_{job_id}.pdf"',
        },
    )


@router.post(
    "/render/pdf",
    responses={422: {"description": "Validation error (empty html)"}},
    operation_id="render_pdf_from_html",
)
async def render_pdf(body: PdfRenderRequest, request: Request) -> Response:
    """Render a self-contained HTML string to a PDF (stateless, headless Chromium print).

    Unlike ``/jobs/{job_id}/pdf`` this takes the HTML in the request body instead of
    loading a stored redesign job — for callers (e.g. the orchestrator audit-report
    stage) that build the HTML themselves. No persistence; the bytes are streamed back.

    Stateless does NOT mean anonymous. This was the only route in the module that
    never resolved a tenant, so it drove unattributable Chromium work — a full
    browser engine, on caller-supplied markup, inside the cluster — for anyone who
    could reach the port. ``_get_tenant_id`` fails closed with a 401, so the work
    is authenticated and attributable like every sibling route even though there is
    no row to scope. The orchestrator already sends ``X-Tenant-Id`` on this call
    (``BaseServiceClient``), so nothing legitimate loses access.
    """
    tenant_id = _get_tenant_id(request)
    logger.debug("Stateless PDF render for tenant %s (%d bytes of HTML)", tenant_id, len(body.html))

    from src.services.screenshot_renderer import render_pdf_from_html_string

    pdf_bytes = await render_pdf_from_html_string(body.html)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="report.pdf"'},
    )


@router.get(
    "/jobs/{job_id}/screenshots/{viewport}",
    responses={
        404: {"description": "Job or screenshot not found"},
        409: {"description": "Job not completed"},
    },
    operation_id="download_redesign_screenshot",
)
async def download_screenshot(
    job_id: str,
    viewport: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download a screenshot for a specific viewport."""
    logger.debug("Screenshot download request for job %s viewport=%s", job_id, viewport)
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (status: {job.status})",
            headers={"Retry-After": "30"},
        )
    if not job.screenshot_paths or viewport not in job.screenshot_paths:
        raise HTTPException(
            status_code=404, detail=f"Screenshot not found for viewport: {viewport}"
        )

    storage = _get_storage(request)
    image_data = await storage.retrieve(job.screenshot_paths[viewport])
    if image_data is None:
        raise HTTPException(status_code=404, detail="Screenshot file not found on storage")

    return Response(
        content=image_data,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="redesign_{job_id}_{viewport}.png"',
        },
    )


@router.get(
    "/jobs/{job_id}/sections/{section_name}",
    responses={
        404: {"description": "Job or section screenshot not found"},
        409: {"description": "Job not completed"},
    },
    operation_id="download_redesign_section_screenshot",
)
async def download_section_screenshot(
    job_id: str,
    section_name: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download a discrete per-section redesign still by section name."""
    logger.debug("Section screenshot download for job %s section=%s", job_id, section_name)
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (status: {job.status})",
            headers={"Retry-After": "30"},
        )
    if not job.section_screenshot_paths or section_name not in job.section_screenshot_paths:
        raise HTTPException(status_code=404, detail=f"Section screenshot not found: {section_name}")

    storage = _get_storage(request)
    image_data = await storage.retrieve(job.section_screenshot_paths[section_name])
    if image_data is None:
        raise HTTPException(status_code=404, detail="Section screenshot file not found on storage")

    return Response(
        content=image_data,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="redesign_{job_id}_{section_name}.png"',
        },
    )


@router.get(
    "/jobs/{job_id}/component-html",
    responses={
        404: {"description": "Job or component HTML not found"},
        409: {"description": "Job not completed"},
    },
    operation_id="download_redesign_component_html",
)
async def download_component_html(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download the redesigned component HTML."""
    logger.debug("Component HTML download for job %s", job_id)
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (status: {job.status})",
            headers={"Retry-After": "30"},
        )
    if not job.component_html_path:
        raise HTTPException(status_code=404, detail="Component HTML not found")

    storage = _get_storage(request)
    html_data = await storage.retrieve_text(job.component_html_path)
    if html_data is None:
        raise HTTPException(status_code=404, detail="Component HTML file not found on storage")

    return Response(
        content=html_data,
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="component_{job_id}.html"',
        },
    )


@router.get(
    "/jobs/{job_id}/original/{viewport}",
    responses={
        404: {"description": "Job or original screenshot not found"},
    },
    operation_id="download_redesign_original_screenshot",
)
async def download_original_screenshot(
    job_id: str,
    viewport: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download original screenshot for a viewport."""
    logger.debug("Original screenshot download for job %s viewport=%s", job_id, viewport)
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not job.original_screenshot_paths or viewport not in job.original_screenshot_paths:
        raise HTTPException(
            status_code=404, detail=f"Original screenshot not found for viewport: {viewport}"
        )

    storage = _get_storage(request)
    data = await storage.retrieve(job.original_screenshot_paths[viewport])
    if data is None:
        raise HTTPException(status_code=404, detail="Original screenshot file not found")

    return Response(content=data, media_type="image/png")


@router.get(
    "/jobs/{job_id}/comparison/{viewport}",
    responses={
        404: {"description": "Job or comparison not found"},
    },
    operation_id="download_redesign_comparison",
)
async def download_comparison(
    job_id: str,
    viewport: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download side-by-side comparison for a viewport."""
    logger.debug("Comparison download for job %s viewport=%s", job_id, viewport)
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not job.comparison_paths or viewport not in job.comparison_paths:
        raise HTTPException(
            status_code=404, detail=f"Comparison not found for viewport: {viewport}"
        )

    storage = _get_storage(request)
    data = await storage.retrieve(job.comparison_paths[viewport])
    if data is None:
        raise HTTPException(status_code=404, detail="Comparison file not found")

    return Response(content=data, media_type="image/png")


@router.get(
    "/jobs/{job_id}/diff/{viewport}",
    responses={
        404: {"description": "Job or diff not found"},
    },
    operation_id="download_redesign_diff",
)
async def download_diff(
    job_id: str,
    viewport: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download pixel diff for a viewport."""
    logger.debug("Diff download for job %s viewport=%s", job_id, viewport)
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not job.diff_paths or viewport not in job.diff_paths:
        raise HTTPException(status_code=404, detail=f"Diff not found for viewport: {viewport}")

    storage = _get_storage(request)
    data = await storage.retrieve(job.diff_paths[viewport])
    if data is None:
        raise HTTPException(status_code=404, detail="Diff file not found")

    return Response(content=data, media_type="image/png")
