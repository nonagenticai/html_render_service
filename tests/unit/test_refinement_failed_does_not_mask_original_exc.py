"""A refinement job's failure-recording except block must not let the
webhook (network I/O) or a DB-write failure replace the real exception, and
must not skip the ``db.commit()`` that makes ``mark_failed`` durable.

``RedesignLifecycleMixin.process_refinement_job`` previously called
``JobManager.mark_failed``, ``JobManager.record_usage``, and
``self._dispatch_webhook`` all unguarded, in that order, followed by
``db.commit()`` and a bare ``raise``. If the webhook (an httpx POST) raised,
that new exception propagated instead of the real cause AND ``db.commit()``
never ran -- so even though ``mark_failed`` executed its UPDATE, it was never
committed and rolled back with the session.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.design_builder.lifecycle import RedesignLifecycleMixin


class _RealPipelineError(Exception):
    """Stand-in for the actual cause (e.g. storage/LLM failure)."""


class _WebhookError(Exception):
    """Stand-in for the webhook's network error."""


def _mixin() -> RedesignLifecycleMixin:
    obj = RedesignLifecycleMixin.__new__(RedesignLifecycleMixin)
    obj._storage = MagicMock()  # type: ignore[attr-defined]
    obj._html_gen = MagicMock()  # type: ignore[attr-defined]
    obj._check_cancelled = AsyncMock(return_value=False)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return obj


@pytest.mark.unit
class TestRefinementFailedDoesNotMaskOriginalException:
    async def test_webhook_failure_does_not_replace_original_exception(self) -> None:
        """Mutation target: the ``try/except`` this PR adds around
        ``self._dispatch_webhook(..., "job.failed", ...)`` inside
        ``process_refinement_job``'s except block. Remove it and this test
        goes red with ``_WebhookError`` propagating instead of
        ``_RealPipelineError``, and ``db.commit()`` not being called for
        the failure path."""
        obj = _mixin()
        original_exc = _RealPipelineError("parent HTML fetch failed")
        obj._storage.retrieve_text = AsyncMock(side_effect=original_exc)

        # job.started succeeds, job.failed (the failure-path dispatch) raises.
        obj._dispatch_webhook = AsyncMock(side_effect=[None, _WebhookError("smtp down")])

        db = AsyncMock()
        job = MagicMock()
        job.parent_job_id = "parent-1"
        job.style = "modern"
        parent = MagicMock()
        parent.redesigned_html_path = "parent.html"

        fake_pkg = MagicMock()
        fake_pkg.JobManager.get_job = AsyncMock(side_effect=[job, parent])
        fake_pkg.JobManager.mark_processing = AsyncMock()
        fake_pkg.JobManager.mark_retryable = AsyncMock(return_value=False)
        fake_pkg.JobManager.mark_failed = AsyncMock()
        fake_pkg.JobManager.record_usage = AsyncMock()

        fake_settings = MagicMock()
        fake_settings.screenshot.enabled = False

        with (
            patch("src.services.design_builder.lifecycle._pkg", return_value=fake_pkg),
            patch(
                "src.services.design_builder.lifecycle.get_settings",
                return_value=fake_settings,
            ),
            pytest.raises(_RealPipelineError),
        ):
            await obj.process_refinement_job(db, "job-1", "tenant-1")

        fake_pkg.JobManager.mark_failed.assert_awaited_once()
        fake_pkg.JobManager.record_usage.assert_awaited_once()
        # mark_processing's earlier commit + the failure-path commit.
        assert db.commit.await_count == 2
