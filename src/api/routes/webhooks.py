"""Webhook delivery status endpoint (Feature 9)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import WebhookDeliveryStatus
from src.database.session import get_db

from ._helpers import _get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/jobs/{job_id}/webhooks", operation_id="list_redesign_webhook_deliveries")
async def get_webhook_deliveries(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> list[WebhookDeliveryStatus]:
    """Get webhook delivery log for a job."""
    logger.debug("Webhook delivery log request for job %s", job_id)
    from sqlalchemy import select

    from src.database.models import RedesignJob, WebhookDeliveryLog

    # WebhookDeliveryLog carries no tenant_id of its own, so ownership is
    # established by joining back to the parent job row. A cross-tenant job id
    # yields an empty list — same as a job that never fired a webhook, so this is
    # not an existence oracle either.
    result = await db.execute(
        select(WebhookDeliveryLog)
        .join(RedesignJob, RedesignJob.job_id == WebhookDeliveryLog.job_id)
        .where(
            WebhookDeliveryLog.job_id == job_id,
            RedesignJob.tenant_id == _get_tenant_id(request),
        )
        .order_by(WebhookDeliveryLog.created_at.desc())
    )
    logs = result.scalars().all()

    return [
        WebhookDeliveryStatus(
            attempt_number=log.attempt_number,
            event_type=log.event_type,
            status=log.status,
            response_status=log.response_status,
            created_at=log.created_at.isoformat() if log.created_at else None,
        )
        for log in logs
    ]
