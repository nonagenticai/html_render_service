"""The refinement job entrypoint creates and consults a job budget (#131).

THE DEFECT this locks. ``RedesignLifecycleMixin.process_refinement_job`` created no
``GenerationBudget`` at all, so ``refine_full_page``'s multi-minute LLM round trip ran
with no deadline and no call accounting — the seam every other LLM dispatch in this
service goes through (see ``test_assembly_and_component_are_budgeted.py`` and
``test_accessibility_scorer_is_budgeted.py``).

THE FIX's CONTRACT, locked here: the job constructs a budget exactly as
``RedesignService.process_job`` does (``service.py:207-230``) and hands it to
``refine_full_page``. A ``BudgetExhausted`` raised from that call is a TERMINAL
failure (not the transient-upstream retry path — see ``upstream_failure.py``): it is
recorded via ``JobManager.mark_failed``/``record_usage`` exactly like any other
permanent defect, and — the property that matters most — the PARENT design (the one
the refinement is based on, already generated and paid for) is never touched: only
the refinement job's own row is marked failed; nothing writes back to the parent's
stored HTML.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.design_builder.lifecycle import RedesignLifecycleMixin
from src.services.html_gen.section_run import BudgetExhausted, GenerationBudget


def _mixin() -> RedesignLifecycleMixin:
    obj = RedesignLifecycleMixin.__new__(RedesignLifecycleMixin)
    obj._storage = MagicMock()  # type: ignore[attr-defined]
    obj._html_gen = MagicMock()  # type: ignore[attr-defined]
    obj._check_cancelled = AsyncMock(return_value=False)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    obj._dispatch_webhook = AsyncMock()  # type: ignore[attr-defined]
    obj._emit_render_cost_event = MagicMock()  # type: ignore[attr-defined]
    return obj


def _job_and_parent() -> tuple[MagicMock, MagicMock]:
    job = MagicMock()
    job.parent_job_id = "parent-1"
    job.style = "modern"
    job.feedback_text = "make it pop"
    job.page_type = None
    job.brand_info = None
    job.site_analysis = None
    parent = MagicMock()
    parent.redesigned_html_path = "parent.html"
    return job, parent


def _fake_pkg(job: MagicMock, parent: MagicMock) -> MagicMock:
    fake_pkg = MagicMock()
    fake_pkg.JobManager.get_job = AsyncMock(side_effect=[job, parent])
    fake_pkg.JobManager.mark_processing = AsyncMock(return_value=True)
    fake_pkg.JobManager.mark_retryable = AsyncMock(return_value=False)
    fake_pkg.JobManager.mark_failed = AsyncMock(return_value=True)
    fake_pkg.JobManager.record_usage = AsyncMock()
    return fake_pkg


@pytest.mark.unit
class TestRefinementJobIsBudgeted:
    async def test_process_refinement_job_creates_and_passes_a_budget(self) -> None:
        """Mutation target: the ``budget = GenerationBudget.from_settings()`` +
        ``budget=budget`` kwarg this PR adds in ``process_refinement_job``. Revert
        either half and this test goes red (no budget kwarg reaches
        ``refine_full_page``, or it is ``None``)."""
        obj = _mixin()
        job, parent = _job_and_parent()
        obj._storage.retrieve_text = AsyncMock(return_value="<html>parent</html>")
        obj._storage.store_html = AsyncMock(return_value="refined.html")
        obj._html_gen.refine_full_page = AsyncMock(
            return_value=MagicMock(text="<html>new</html>", input_tokens=10, output_tokens=20)
        )
        fake_pkg = _fake_pkg(job, parent)
        fake_pkg.JobManager.mark_completed = AsyncMock(return_value=True)
        fake_settings = MagicMock()
        fake_settings.screenshot.enabled = False

        with (
            patch("src.services.design_builder.lifecycle._pkg", return_value=fake_pkg),
            patch("src.services.design_builder.lifecycle.get_settings", return_value=fake_settings),
        ):
            db = AsyncMock()
            await obj.process_refinement_job(db, "job-1", "tenant-1")

        obj._html_gen.refine_full_page.assert_awaited_once()
        assert obj._html_gen.refine_full_page.await_args is not None
        kwargs = obj._html_gen.refine_full_page.await_args.kwargs
        assert "budget" in kwargs, "refine_full_page was not handed a budget at all"
        assert isinstance(kwargs["budget"], GenerationBudget), (
            f"budget kwarg was not a GenerationBudget: {kwargs['budget']!r}"
        )

    async def test_budget_exhaustion_fails_the_job_without_touching_the_parent(self) -> None:
        """``BudgetExhausted`` out of ``refine_full_page`` must: (1) fail this
        refinement job terminally (not the transient-retry path), and (2) never
        write to the parent design's storage — only ``_load_parent_html``'s READ
        of it happens."""
        obj = _mixin()
        job, parent = _job_and_parent()
        obj._storage.retrieve_text = AsyncMock(return_value="<html>parent original</html>")
        obj._storage.store_html = AsyncMock()
        obj._html_gen.refine_full_page = AsyncMock(
            side_effect=BudgetExhausted("generation deadline reached (0s left)")
        )
        fake_pkg = _fake_pkg(job, parent)
        fake_settings = MagicMock()
        fake_settings.screenshot.enabled = False

        with (
            patch("src.services.design_builder.lifecycle._pkg", return_value=fake_pkg),
            patch("src.services.design_builder.lifecycle.get_settings", return_value=fake_settings),
            pytest.raises(BudgetExhausted),
        ):
            db = AsyncMock()
            await obj.process_refinement_job(db, "job-1", "tenant-1")

        # Terminal, not the transient-upstream retry path.
        fake_pkg.JobManager.mark_retryable.assert_not_awaited()
        fake_pkg.JobManager.mark_failed.assert_awaited_once()
        assert fake_pkg.JobManager.mark_failed.await_args is not None
        assert fake_pkg.JobManager.mark_failed.await_args.args[1] == "job-1", (
            "the FAILED row must be the refinement job, never the parent"
        )
        fake_pkg.JobManager.record_usage.assert_awaited_once()

        # The parent design is READ (to build the refine prompt) and never written.
        obj._storage.retrieve_text.assert_awaited_once_with(parent.redesigned_html_path)
        obj._storage.store_html.assert_not_awaited()
