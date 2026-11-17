"""A DB-write failure while RECORDING a job failure must never replace the
real exception that caused the failure in the first place.

``RedesignService.process_job``'s only failure seam is:

    except Exception as exc:
        await self._phase_finalize_failed(db, job, job_id, tenant_id, exc)
        raise

``_phase_finalize_failed`` used to call ``JobManager.mark_failed`` and
``JobManager.record_usage`` unguarded. If either of those DB writes itself
raised (e.g. the backend was reaped -- a documented risk in this same file),
that new exception propagated out of ``_phase_finalize_failed`` instead of
returning, so the bare ``raise`` above never ran: the caller saw the DB
error instead of the real pipeline failure, and ``db.commit()`` /
the failure webhook were skipped too.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.design_builder.finalize_phases import RedesignFinalizeMixin


class _RealPipelineError(Exception):
    """Stand-in for the actual cause of a job failure (e.g. a Playwright error)."""


class _DBWriteError(Exception):
    """Stand-in for an unrelated failure while persisting the failure record."""


def _mixin(*, mark_failed_raises: bool) -> tuple[RedesignFinalizeMixin, MagicMock]:
    obj = RedesignFinalizeMixin.__new__(RedesignFinalizeMixin)
    obj._dispatch_webhook = AsyncMock()  # type: ignore[method-assign]

    fake_pkg = MagicMock()
    if mark_failed_raises:
        fake_pkg.JobManager.mark_failed = AsyncMock(side_effect=_DBWriteError("db down"))
    else:
        fake_pkg.JobManager.mark_failed = AsyncMock()
    fake_pkg.JobManager.record_usage = AsyncMock()
    return obj, fake_pkg


@pytest.mark.unit
class TestFinalizeFailedDoesNotMaskOriginalException:
    """⚠️ ONE test here, deliberately, and it is the END-TO-END one.

    A direct-call sibling was written first and REMOVED: it asserted await
    counts (the shape-only ratchet flagged it), and when rewritten to assert
    the webhook's payload it STILL stayed green under the mutation that
    removes the guard — i.e. it proved nothing about the thing it was named
    for. The end-to-end test below goes red under that same mutation, so it
    is the one carrying the guarantee. A test that cannot fail is worse than
    no test: it reads as cover.
    """

    async def test_original_exception_reaches_the_real_caller(self) -> None:
        """End-to-end through the ACTUAL `process_job` failure seam, with the
        real (unmocked) `_phase_finalize_failed`: patch `_phase_resolve_html`
        (the first phase) to raise the real pipeline error, and
        `JobManager.mark_failed` to raise a distinct DB error while
        recording it. Assert the ORIGINAL exception type is what
        `process_job` re-raises -- not the DB error.

        Mutation target: the `try/except` this PR adds around `mark_failed`
        + `record_usage` in `_phase_finalize_failed`. Remove it and this test
        goes red with `_DBWriteError` propagating instead of
        `_RealPipelineError`.
        """
        from src.services.design_builder.service import RedesignService

        service = RedesignService.__new__(RedesignService)
        service._dispatch_webhook = AsyncMock()  # type: ignore[method-assign]
        service._check_cancelled = AsyncMock(return_value=False)  # type: ignore[method-assign]
        original_exc = _RealPipelineError("the real pipeline failure")
        service._phase_resolve_html = AsyncMock(side_effect=original_exc)  # type: ignore[method-assign]

        job = MagicMock()
        job.scope = "redesign"
        job.user_context = None
        job.site_analysis = None
        job.brand_info = None

        fake_pkg = MagicMock()
        fake_pkg.JobManager.get_job = AsyncMock(return_value=job)
        fake_pkg.JobManager.mark_processing = AsyncMock()
        fake_pkg.JobManager.mark_failed = AsyncMock(side_effect=_DBWriteError("db down"))
        fake_pkg.JobManager.record_usage = AsyncMock()

        db = AsyncMock()

        with (
            patch("src.services.design_builder.service._pkg", return_value=fake_pkg),
            patch("src.services.design_builder.finalize_phases._pkg", return_value=fake_pkg),
            patch(
                "src.services.design_builder.service.get_settings",
                return_value=MagicMock(),
            ),
            pytest.raises(_RealPipelineError),
        ):
            await service.process_job(db, "job-1", "tenant-1")

        fake_pkg.JobManager.mark_failed.assert_awaited_once()
