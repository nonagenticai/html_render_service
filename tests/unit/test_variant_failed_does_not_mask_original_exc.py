"""A DB-write failure while recording a variant's failure must not replace
the exception the parent job uses to classify its own verdict.

``_process_variant``'s failure recorder used to call
``JobManager.mark_variant_failed`` and ``db.commit()`` unguarded, inside its
own ``except`` block. If ``mark_variant_failed`` itself raised, that new
exception propagated out of ``_process_variant`` INSTEAD of the `return exc`
at the bottom. The outer ``_process_variants`` fan-out (``_run`` + a
``gather(..., return_exceptions=True)``) already contains that -- it does not
crash the batch or take out sibling variants -- but it still appends the
WRONG exception (the DB error, not the real cause) to the ``failures`` list
that the parent job reads to decide transient-vs-terminal.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.services.design_builder as design_builder
from src.services.design_builder.variants import RedesignVariantsMixin


class _RealUpstreamTimeoutError(Exception):
    """Stand-in for the actual cause (e.g. httpx.ReadTimeout)."""


class _DBWriteError(Exception):
    """Stand-in for an unrelated failure while persisting the variant's
    failure state."""


def _mixin() -> RedesignVariantsMixin:
    obj = RedesignVariantsMixin.__new__(RedesignVariantsMixin)
    obj._check_cancelled = AsyncMock(return_value=False)  # type: ignore[method-assign]
    return obj


def _job() -> MagicMock:
    job = MagicMock()
    job.styles = ["modern"]
    job.job_id = "job-x"
    job.tenant_id = "tenant-1"
    return job


@pytest.mark.unit
class TestVariantFailedDoesNotMaskOriginalException:
    async def test_direct_call_returns_original_exception_not_db_error(self, monkeypatch) -> None:
        """Mutation target: the ``try/except`` this PR adds around
        ``mark_variant_failed`` + ``db.commit()`` in ``_process_variant``'s
        except block. Remove it and this test goes red with
        ``_DBWriteError`` propagating (or the assertion on the returned
        value failing) instead of ``_process_variant`` returning the
        original ``_RealUpstreamTimeoutError``."""
        jm = MagicMock(
            create_variant=AsyncMock(return_value=MagicMock(variant_id="v1")),
            mark_variant_failed=AsyncMock(side_effect=_DBWriteError("db down")),
        )
        monkeypatch.setattr(design_builder, "JobManager", jm)

        obj = _mixin()
        original_exc = _RealUpstreamTimeoutError("upstream timed out")
        obj._html_gen = MagicMock(generate_page_chunked=AsyncMock(side_effect=original_exc))
        db = MagicMock(commit=AsyncMock(), rollback=AsyncMock())

        returned = await obj._process_variant(
            db, _job(), "<html></html>", "modern", MagicMock(), MagicMock(), False, None
        )

        assert returned is original_exc
        jm.mark_variant_failed.assert_awaited_once()

    async def test_fan_out_preserves_original_exception_type_for_classification(
        self, monkeypatch
    ) -> None:
        """Through the real caller (``_process_variants``, with a
        ``session_factory`` so the ``gather(return_exceptions=True)`` path
        runs): the ``failures`` list the parent reads for its
        transient/terminal verdict must contain the ORIGINAL exception type,
        not the DB-write error that happened while recording it."""

        class _FakeSessionCtx:
            async def __aenter__(self):
                return AsyncMock()

            async def __aexit__(self, *_exc):
                return False

        jm = MagicMock(
            create_variant=AsyncMock(return_value=MagicMock(variant_id="v1")),
            mark_variant_failed=AsyncMock(side_effect=_DBWriteError("db down")),
        )
        monkeypatch.setattr(design_builder, "JobManager", jm)

        obj = _mixin()
        original_exc = _RealUpstreamTimeoutError("upstream timed out")
        obj._html_gen = MagicMock(generate_page_chunked=AsyncMock(side_effect=original_exc))

        from src.config import get_settings

        failures = await obj._process_variants(
            session_factory=cast(Any, _FakeSessionCtx),
            db=MagicMock(),
            job=_job(),
            html="<html></html>",
            resolved_ctx=MagicMock(),
            settings=get_settings(),
        )

        assert len(failures) == 1
        assert isinstance(failures[0], _RealUpstreamTimeoutError), (
            f"expected the original upstream exception, got {type(failures[0]).__name__}"
        )
