"""Data retention cleanup for expired jobs and orphan files."""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import (
    BatchJob,
    RedesignJob,
    RedesignVariant,
    WebhookDeliveryLog,
)
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class CleanupService:
    """Handles data retention cleanup for old jobs and files."""

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    async def cleanup_old_jobs(
        self,
        db: AsyncSession,
        retention_days: int = 30,
    ) -> int:
        """Delete jobs and associated data older than *retention_days*.

        Returns the number of deleted RedesignJob rows.
        """
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        logger.info(
            "Starting cleanup for jobs older than %s (%d-day retention)",
            cutoff.isoformat(),
            retention_days,
        )

        # Collect job_ids that will be removed so we can clean storage.
        stmt = select(RedesignJob.job_id).where(
            RedesignJob.created_at < cutoff,
        )
        result = await db.execute(stmt)
        expired_job_ids: list[str] = list(result.scalars().all())

        if not expired_job_ids:
            logger.info("No expired jobs found — nothing to clean up")
            return 0

        # 1. Webhook delivery logs
        wh_cursor = cast(
            CursorResult[Any],
            await db.execute(
                delete(WebhookDeliveryLog).where(
                    WebhookDeliveryLog.job_id.in_(expired_job_ids),
                )
            ),
        )
        wh_deleted = wh_cursor.rowcount if wh_cursor.rowcount is not None else 0
        logger.info("Deleted %d webhook delivery log(s)", wh_deleted)

        # 2. Redesign variants
        var_cursor = cast(
            CursorResult[Any],
            await db.execute(
                delete(RedesignVariant).where(
                    RedesignVariant.job_id.in_(expired_job_ids),
                )
            ),
        )
        var_deleted = var_cursor.rowcount if var_cursor.rowcount is not None else 0
        logger.info("Deleted %d variant(s)", var_deleted)

        # 3. Redesign jobs
        job_cursor = cast(
            CursorResult[Any],
            await db.execute(
                delete(RedesignJob).where(
                    RedesignJob.created_at < cutoff,
                )
            ),
        )
        deleted_jobs: int = job_cursor.rowcount if job_cursor.rowcount is not None else 0
        logger.info("Deleted %d redesign job(s)", deleted_jobs)

        # 4. Batch jobs
        batch_cursor = cast(
            CursorResult[Any],
            await db.execute(
                delete(BatchJob).where(BatchJob.created_at < cutoff),
            ),
        )
        batch_deleted = batch_cursor.rowcount if batch_cursor.rowcount is not None else 0
        logger.info("Deleted %d batch job(s)", batch_deleted)

        await db.flush()

        # Remove storage directories for each expired job.
        for job_id in expired_job_ids:
            job_dir = self._storage.get_job_dir(job_id)
            if job_dir.exists():
                shutil.rmtree(job_dir)
                logger.debug("Removed storage dir: %s", job_dir)

        logger.info("Cleanup complete — %d job(s) removed", deleted_jobs)
        return deleted_jobs

    async def cleanup_orphan_files(self, db: AsyncSession) -> int:
        """Remove storage directories that have no matching DB record.

        Returns the number of orphan directories removed.
        """
        storage_root = self._storage._local_path
        if not storage_root.exists():
            return 0

        storage_dirs = [d for d in storage_root.iterdir() if d.is_dir()]
        if not storage_dirs:
            return 0

        # Get all known job_ids from DB
        result = await db.execute(select(RedesignJob.job_id))
        known_job_ids = {row[0] for row in result.all()}

        removed = 0
        for job_dir in storage_dirs:
            if job_dir.name not in known_job_ids:
                shutil.rmtree(job_dir)
                logger.info("Removed orphan storage dir: %s", job_dir)
                removed += 1

        logger.info("Orphan cleanup complete — %d dir(s) removed", removed)
        return removed
