"""``retry-failed`` retries the failed pages, and ONLY the failed pages.

DB-backed. The defect, in the shape that costs money:

``POST /api/batch/{id}/retry-failed`` reset only the failed jobs and then called
``process_batch`` with no scope at all. ``get_batch_jobs`` has no status filter
and ``process_job`` had no status guard, so **3 failures in a 20-page batch
re-rendered all 20**:

* 17 already-approved designs were overwritten with fresh LLM output — work the
  tenant had reviewed and accepted, destroyed;
* the tenant paid for 17 renders they never asked for;
* ``completed_pages`` finished at **37 of 20**, because every page that had
  already been counted was counted again.

The assertions below are all persisted state: the surviving artifact paths of
the untouched pages, and the counter. None of them is a call count — a call
count would be satisfied by a fix that re-rendered the pages but happened to
call something a different number of times.
"""

from __future__ import annotations

from sqlalchemy import select, update

from src.database.models import BatchJob, RedesignJob
from src.services.batch_manager import BatchManager
from src.services.design_builder.helpers import RedesignHelpersMixin
from src.services.design_builder.lifecycle import RedesignLifecycleMixin
from src.services.job_manager import JobManager
from tests.tenants import TENANT_1

TOTAL_PAGES = 20
FAILED_PAGES = 3
APPROVED = "jobs/approved/redesign.html"
RERENDERED = "jobs/RERENDERED/redesign.html"


class _Batcher(RedesignLifecycleMixin, RedesignHelpersMixin):
    """``process_batch`` composed with the real ``_dispatch_webhook``, as in prod."""


async def _seed(db) -> tuple[str, list[str], list[str]]:
    """A 20-page batch: 17 completed and approved, 3 failed. Returns ids."""
    batch = await BatchManager.create_batch(db, tenant_id=TENANT_1, total_pages=TOTAL_PAGES)
    done: list[str] = []
    failed: list[str] = []
    for i in range(TOTAL_PAGES):
        job = await JobManager.create_job(
            db,
            tenant_id=TENANT_1,
            url=f"https://example.com/page-{i}",
            batch_id=batch.batch_id,
        )
        if i < TOTAL_PAGES - FAILED_PAGES:
            await JobManager.mark_completed(db, job.job_id, redesigned_html_path=APPROVED)
            done.append(job.job_id)
        else:
            await JobManager.mark_failed(db, job.job_id, "upstream 502")
            failed.append(job.job_id)
    await db.execute(
        update(BatchJob)
        .where(BatchJob.batch_id == batch.batch_id)
        .values(
            status="completed_with_errors",
            completed_pages=TOTAL_PAGES - FAILED_PAGES,
            failed_pages=FAILED_PAGES,
        )
    )
    await db.commit()
    return batch.batch_id, done, failed


async def _run_retry(db, db_session_factory, batch_id: str, retried: list[str]) -> _Batcher:
    """Do exactly what the retry route does, then run the background batch."""
    for job_id in retried:
        await BatchManager.reset_job_for_retry(db, job_id)
    batch = await BatchManager.get_batch(db, batch_id, tenant_id=TENANT_1)
    assert batch is not None
    await db.execute(
        update(BatchJob)
        .where(BatchJob.batch_id == batch_id)
        .values(failed_pages=batch.failed_pages - len(retried), status="processing")
    )
    await db.commit()

    service = _Batcher()
    service._webhook = None  # type: ignore[attr-defined]

    async def _process_job(page_db, job_id, tenant_id, **kw):
        """Stand in for a paid render: it OVERWRITES the design, like the real one."""
        await JobManager.mark_completed(page_db, job_id, redesigned_html_path=RERENDERED)
        await page_db.commit()

    service.process_job = _process_job  # type: ignore[attr-defined,method-assign]
    await service.process_batch(db_session_factory, batch_id, TENANT_1, job_ids=retried)
    return service


async def test_retry_failed_does_not_re_render_approved_pages(
    db_session, db_session_factory
) -> None:
    """The 17 approved designs must still be the approved designs."""
    batch_id, done, failed = await _seed(db_session)
    await _run_retry(db_session, db_session_factory, batch_id, failed)

    db_session.expire_all()
    rows = (
        (await db_session.execute(select(RedesignJob).where(RedesignJob.job_id.in_(done))))
        .scalars()
        .all()
    )
    assert len(rows) == TOTAL_PAGES - FAILED_PAGES
    clobbered = [r.job_id for r in rows if r.redesigned_html_path != APPROVED]
    assert clobbered == [], (
        f"{len(clobbered)} already-approved designs were re-rendered and overwritten by a "
        f"retry of {FAILED_PAGES} failed pages"
    )


async def test_retry_failed_leaves_the_counter_at_the_page_count(
    db_session, db_session_factory
) -> None:
    """``completed_pages`` must land on 20 of 20 — it used to land on 37."""
    batch_id, _done, failed = await _seed(db_session)
    await _run_retry(db_session, db_session_factory, batch_id, failed)

    db_session.expire_all()
    row = (
        await db_session.execute(select(BatchJob).where(BatchJob.batch_id == batch_id))
    ).scalar_one()
    assert row.completed_pages == TOTAL_PAGES, (
        f"completed_pages={row.completed_pages} of total_pages={row.total_pages}"
    )
    assert row.failed_pages == 0
    assert row.status == "completed"


async def test_the_three_retried_pages_really_are_re_rendered(
    db_session, db_session_factory
) -> None:
    """The control.

    Without it, "nothing was overwritten" and "completed_pages == 20" are both
    satisfied by a scope so tight that the retry does nothing at all — which
    would be a different bug wearing the fix's clothes.
    """
    batch_id, _done, failed = await _seed(db_session)
    await _run_retry(db_session, db_session_factory, batch_id, failed)

    db_session.expire_all()
    rows = (
        (await db_session.execute(select(RedesignJob).where(RedesignJob.job_id.in_(failed))))
        .scalars()
        .all()
    )
    assert len(rows) == FAILED_PAGES
    assert all(r.status == "completed" for r in rows)
    assert all(r.redesigned_html_path == RERENDERED for r in rows), (
        "the pages the tenant asked to retry were not retried"
    )


async def test_an_unscoped_run_still_refuses_to_re_render_finished_pages(
    db_session, db_session_factory
) -> None:
    """The invariant does not depend on the caller remembering to pass a scope.

    ``retry-failed`` is not the only way to re-enter ``process_batch``, and the
    harm is not a stale row — it is a paid LLM render that destroys an approved
    design. So the terminal skip inside ``process_batch`` holds on its own.
    """
    batch_id, done, failed = await _seed(db_session)
    for job_id in failed:
        await BatchManager.reset_job_for_retry(db_session, job_id)
    await db_session.execute(
        update(BatchJob).where(BatchJob.batch_id == batch_id).values(status="processing")
    )
    await db_session.commit()

    service = _Batcher()
    service._webhook = None  # type: ignore[attr-defined]

    async def _process_job(page_db, job_id, tenant_id, **kw):
        await JobManager.mark_completed(page_db, job_id, redesigned_html_path=RERENDERED)
        await page_db.commit()

    service.process_job = _process_job  # type: ignore[attr-defined,method-assign]
    await service.process_batch(db_session_factory, batch_id, TENANT_1)  # no job_ids

    db_session.expire_all()
    rows = (
        (await db_session.execute(select(RedesignJob).where(RedesignJob.job_id.in_(done))))
        .scalars()
        .all()
    )
    clobbered = [r.job_id for r in rows if r.redesigned_html_path != APPROVED]
    assert clobbered == [], (
        f"{len(clobbered)} completed pages were re-rendered by an unscoped batch run"
    )
