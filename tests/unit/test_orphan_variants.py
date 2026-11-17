"""A variant is never left non-terminal under a terminal parent.

The defect (measured on prod 2026-09-08): **317 ``redesign_variants`` rows sat
``pending`` under an already-terminal parent job** — 225 ``failed`` parents, 92
``completed``, oldest 2026-06-07, zero under a live parent. They carried no
``error_message``, so no ranking of failures could see them; the variants
endpoint simply advertised work that was never coming.

Two layers are covered, deliberately:

* the REACHABILITY INVARIANT (:func:`test_every_non_terminal_variant_status_is_reachable`
  and friends) — set algebra over the status vocabulary, with no DB. This is the
  gate that makes a FUTURE regression fail here: add a fifth variant status, or
  narrow the settler's predicate, and the assertion breaks instead of the rows
  silently becoming immortal;
* the BEHAVIOUR — DB-backed, reading the row back. A mock's call count cannot
  satisfy these: the defect was that nothing ever wrote the row again, so only
  persisted state proves it fixed.
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select, update

from src.database.models import RedesignJob, RedesignVariant
from src.database.rls_resolvers import CLAIM_FN_SQL, REAP_BATCH_FN_SQL
from src.services.job_manager import (
    NON_TERMINAL_VARIANT_STATUSES,
    TERMINAL_JOB_STATUSES,
    TERMINAL_VARIANT_STATUSES,
    VARIANT_STATUSES,
    JobManager,
)
from tests.tenants import TENANT_1

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "src/database/migrations/versions/0012_settle_orphan_variants.py"
)


# ---------------------------------------------------------------------------
# Reachability invariant (no DB)
# ---------------------------------------------------------------------------


def test_the_two_variant_status_sets_partition_the_vocabulary() -> None:
    """Every declared status is classified exactly once, terminal or not."""
    assert set(TERMINAL_VARIANT_STATUSES) | set(NON_TERMINAL_VARIANT_STATUSES) == set(
        VARIANT_STATUSES
    )
    assert not set(TERMINAL_VARIANT_STATUSES) & set(NON_TERMINAL_VARIANT_STATUSES)
    assert NON_TERMINAL_VARIANT_STATUSES, "a vocabulary with no non-terminal state is a typo"


@pytest.mark.parametrize("status", NON_TERMINAL_VARIANT_STATUSES)
def test_every_non_terminal_variant_status_is_reachable(status: str) -> None:
    """The settler's predicate must select EVERY non-terminal status.

    This is the whole invariant. ``settle_orphan_variants`` filters on
    ``status.in_(NON_TERMINAL_VARIANT_STATUSES)``, so what is really asserted
    here is that the derivation is honest and nothing hard-codes a narrower
    literal list somewhere the derived tuple cannot see: the two SQL sweeps in
    ``rls_resolvers`` DO hard-code theirs, and they are checked below.
    """
    assert status in NON_TERMINAL_VARIANT_STATUSES
    assert status not in TERMINAL_VARIANT_STATUSES


@pytest.mark.parametrize("status", NON_TERMINAL_VARIANT_STATUSES)
def test_the_sql_sweeps_cover_every_non_terminal_variant_status(status: str) -> None:
    """``claim_next_render_job`` / ``reap_stale_batch_jobs`` must not lag behind.

    Both spell their variant predicate as a literal ``IN (...)`` inside PL/pgSQL,
    which no Python constant can reach. A status added to ``VARIANT_STATUSES``
    without also being added there would be settled by the Python path and
    stranded by the SQL one — the same hole, one table down.
    """
    assert f"'{status}'" in CLAIM_FN_SQL
    assert f"'{status}'" in REAP_BATCH_FN_SQL


@pytest.mark.parametrize("status", NON_TERMINAL_VARIANT_STATUSES)
def test_the_backfill_migration_covers_every_non_terminal_variant_status(status: str) -> None:
    """Migration 0012 duplicates the literals (it may not import app code)."""
    assert f"'{status}'" in _MIGRATION.read_text(encoding="utf-8")


@pytest.mark.parametrize("status", TERMINAL_JOB_STATUSES)
def test_every_terminal_job_status_has_a_settling_writer(status: str) -> None:
    """Each way a job reaches a verdict must settle its children.

    ``mark_completed`` / ``mark_failed`` / ``cancel_job`` are the three, and the
    ``parent_status=`` they pass is what the operator reads back off the row.
    """
    source_file = inspect.getsourcefile(JobManager)
    assert source_file is not None
    source = Path(source_file).read_text(encoding="utf-8")
    assert f'settle_orphan_variants(db, job_id, parent_status="{status}")' in source


# ---------------------------------------------------------------------------
# Behaviour (DB-backed)
# ---------------------------------------------------------------------------


async def _job_with_pending_variant(db, *, tenant_id: str = TENANT_1) -> tuple[str, str]:
    job = RedesignJob(
        job_id=str(uuid.uuid7()),
        tenant_id=tenant_id,
        status="processing",
        url="https://example.com",
        scope="full_page",
    )
    db.add(job)
    await db.flush()
    variant = await JobManager.create_variant(db, job_id=job.job_id, style="modern")
    await db.commit()
    return job.job_id, variant.variant_id


async def _variant(db, variant_id: str) -> RedesignVariant:
    db.expunge_all()
    row = (
        await db.execute(select(RedesignVariant).where(RedesignVariant.variant_id == variant_id))
    ).scalar_one_or_none()
    assert row is not None
    return row


@pytest.mark.parametrize(
    ("marker", "expected_parent_status"),
    [
        ("completed", "completed"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
    ],
)
async def test_a_terminal_parent_settles_its_pending_variants(
    db_session, marker: str, expected_parent_status: str
) -> None:
    """The row is READ BACK: it must be terminal and say why."""
    job_id, variant_id = await _job_with_pending_variant(db_session)
    assert (await _variant(db_session, variant_id)).status == "pending"

    if marker == "completed":
        assert await JobManager.mark_completed(db_session, job_id)
    elif marker == "failed":
        assert await JobManager.mark_failed(db_session, job_id, "upstream blew up")
    else:
        assert await JobManager.cancel_job(db_session, job_id, tenant_id=TENANT_1)
    await db_session.commit()

    row = await _variant(db_session, variant_id)
    assert row.status in TERMINAL_VARIANT_STATUSES
    assert row.completed_at is not None
    assert row.error_message is not None
    assert expected_parent_status in row.error_message
    # It must not blame the tenant for work the SERVICE dropped.
    assert "abandoned" in row.error_message


async def test_settling_does_not_overwrite_a_real_verdict(db_session) -> None:
    """A variant that reported its own outcome keeps it — and its error text."""
    job_id, variant_id = await _job_with_pending_variant(db_session)
    await JobManager.mark_variant_failed(db_session, variant_id, "the style itself is broken")
    await db_session.commit()

    assert await JobManager.mark_failed(db_session, job_id, "all variants failed")
    await db_session.commit()

    row = await _variant(db_session, variant_id)
    assert row.error_message == "the style itself is broken"


async def test_settling_is_idempotent(db_session) -> None:
    """Re-running matches zero rows — the backfill is safe to replay."""
    job_id, variant_id = await _job_with_pending_variant(db_session)
    first = await JobManager.settle_orphan_variants(db_session, job_id, parent_status="failed")
    second = await JobManager.settle_orphan_variants(db_session, job_id, parent_status="failed")
    await db_session.commit()

    assert first == 1
    assert second == 0
    assert (await _variant(db_session, variant_id)).error_message is not None


async def test_a_live_parent_leaves_its_variants_alone(db_session) -> None:
    """A losing terminal write must not settle the WINNER's children.

    ``mark_completed`` on a row already ``cancelled`` is rejected by the terminal
    fence; if the settle were unconditional it would still retire the variants of
    a job another worker owns.
    """
    job_id, variant_id = await _job_with_pending_variant(db_session)
    await db_session.execute(
        update(RedesignJob).where(RedesignJob.job_id == job_id).values(status="cancelled")
    )
    await db_session.commit()

    assert not await JobManager.mark_completed(db_session, job_id)
    await db_session.commit()

    assert (await _variant(db_session, variant_id)).status == "pending"
