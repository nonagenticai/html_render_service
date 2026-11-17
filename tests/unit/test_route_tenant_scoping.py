"""Every by-id job read/write is scoped to the caller's tenant.

``JobManager.get_job`` used to be ``WHERE job_id == :id`` with no tenant
predicate, so any authenticated caller who learned (or guessed) a job id could
read another tenant's job row, its HTML/screenshot artifacts, its variants, its
iteration history — and, worst of all, ``POST /jobs/{id}/refine`` cloned the
victim's job *into the victim's tenant*, billing them for the attacker's work.

Every case here is a PAIR on ONE seeded fixture: job A belongs to ``t1``, job B
to ``t2``, both live in the same session. t1 must get a real 200 (with an
asserted body field) on A **and** a 404 on B. The paired 200 is what stops an
empty/broken fixture from faking the whole suite green.

The mismatch response is 404, never 403: a 403 confirms the id exists, which is
an enumeration oracle over ids that double as artifact URLs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, update

from src.api.routes._helpers import _get_tenant_id
from src.api.routes.artifacts import download_html
from src.api.routes.batch import cancel_batch, get_batch_results, get_batch_status
from src.api.routes.jobs import cancel_job, get_job_result, get_job_status
from src.api.routes.refinement import get_iteration_history, refine_redesign
from src.api.routes.styles import delete_style, get_style, preview_style, update_style
from src.api.routes.variants import compare_variants, download_variant_html
from src.api.routes.webhooks import get_webhook_deliveries
from src.api.schemas import RefineRequest, StylePreviewRequest, UpdateCustomStyleRequest
from src.database.models import BatchJob, CustomStyle, RedesignJob, WebhookDeliveryLog
from src.services.batch_manager import BatchManager
from src.services.design_builder.service import RedesignService
from src.services.job_manager import JobManager
from src.services.style_manager import StyleManager
from tests.tenants import TENANT_1, TENANT_2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request(tenant_id: str, *, storage: Any = None, service: Any = None) -> Any:
    """Build a Request stand-in with a REAL string tenant on ``.state``.

    Deliberately not ``MagicMock()``: ``_get_tenant_id`` reads
    ``request.state.tenant_id``, and on a bare MagicMock that returns a
    *MagicMock*, not a string. Every scoped query would then miss and every
    assertion below would go green for entirely the wrong reason.

    The tenant must be a real UUID: ``_get_tenant_id`` now validates it, so a
    slug fixture would 401 before any scoping predicate was reached.
    """
    return SimpleNamespace(
        state=SimpleNamespace(tenant_id=tenant_id),
        app=SimpleNamespace(
            state=SimpleNamespace(
                storage_service=storage,
                redesign_service=service,
                session_factory=None,
            )
        ),
    )


def _storage(text: str = "<html>hi</html>") -> Any:
    """Storage stub whose reads always succeed, so a 404 can only be the tenant gate."""
    stub = MagicMock()
    stub.retrieve_text = AsyncMock(return_value=text)
    stub.retrieve = AsyncMock(return_value=b"png")
    stub.presigned_url = AsyncMock(return_value="https://minio.example/x")
    return stub


async def _seed_pair(db, **kwargs: Any) -> tuple[str, str]:
    """Seed job A (tenant t1) and job B (tenant t2) in the same session."""
    a = await JobManager.create_job(db, tenant_id=TENANT_1, url="https://a.example", scope="hero")
    b = await JobManager.create_job(db, tenant_id=TENANT_2, url="https://b.example", scope="hero")
    if kwargs.get("completed"):
        for jid in (a.job_id, b.job_id):
            await JobManager.mark_completed(db, jid, redesigned_html_path=f"/{jid}/redesign.html")
    await db.commit()
    return a.job_id, b.job_id


# ---------------------------------------------------------------------------
# 1. Paired read cases
# ---------------------------------------------------------------------------


async def test_get_job_status_is_tenant_scoped(db_session) -> None:
    a, b = await _seed_pair(db_session)

    ok = await get_job_status(a, _request(TENANT_1), db=db_session)
    assert ok.job_id == a  # paired 200 — proves the fixture is real

    with pytest.raises(HTTPException) as exc:
        await get_job_status(b, _request(TENANT_1), db=db_session)
    assert exc.value.status_code == 404
    assert exc.value.detail == f"Job not found: {b}"  # same string as a genuine miss


async def test_get_job_result_is_tenant_scoped(db_session) -> None:
    a, b = await _seed_pair(db_session, completed=True)

    ok = await get_job_result(a, _request(TENANT_1, storage=_storage()), db=db_session)
    assert ok.redesigned_html_path == f"/{a}/redesign.html"

    with pytest.raises(HTTPException) as exc:
        await get_job_result(b, _request(TENANT_1, storage=_storage()), db=db_session)
    assert exc.value.status_code == 404


async def test_download_html_is_tenant_scoped(db_session) -> None:
    a, b = await _seed_pair(db_session, completed=True)

    resp = await download_html(a, _request(TENANT_1, storage=_storage()), db=db_session)
    assert resp.body == b"<html>hi</html>"

    with pytest.raises(HTTPException) as exc:
        await download_html(b, _request(TENANT_1, storage=_storage()), db=db_session)
    assert exc.value.status_code == 404


async def test_compare_variants_404_is_the_job_guard_not_the_variant_lookup(db_session) -> None:
    """Isolates the ``get_job`` predicate on ``GET /jobs/{id}/compare``.

    The route has two guards in series and both 404, so a bare
    ``status_code == 404`` assertion is a CONJUNCTION: delete either predicate
    alone and the other still produces a 404, so neither is individually
    protected. (That is exactly what the previous single test did — it survived
    every single-predicate mutation.)

    The two 404s carry DIFFERENT detail strings, so the detail is what pins the
    job guard: drop it and B's job is found, the still-scoped variant lookup
    comes back empty, and the message flips to "No variants found for this job".
    """
    a, b = await _seed_pair(db_session)
    for jid in (a, b):
        await JobManager.create_variant(db_session, job_id=jid, style="modern")
    await db_session.commit()

    ok = await compare_variants(a, _request(TENANT_1), db=db_session)
    assert [v.style for v in ok.variants] == ["modern"]

    with pytest.raises(HTTPException) as exc:
        await compare_variants(b, _request(TENANT_1), db=db_session)
    assert exc.value.status_code == 404
    assert exc.value.detail == f"Job not found: {b}"


async def test_get_variants_is_scoped_independently_of_the_job_guard(db_session) -> None:
    """Isolates the ``get_variants`` predicate, which no route test can reach.

    On the route, ``get_job`` short-circuits first, so deleting the variant
    query's tenant predicate breaks nothing observable — untested belt-and-braces
    that will rot. So it is exercised DIRECTLY, the same way
    ``test_cancel_writes_are_scoped_independently_of_the_read_guard`` does.

    The failure mode here is a leaky list, not a 404, so the returned ids are
    asserted exactly and B really does own a variant — an empty result cannot
    pass vacuously.
    """
    _a, b = await _seed_pair(db_session)
    victim = await JobManager.create_variant(db_session, job_id=b, style="modern")
    await db_session.commit()

    leaked = await JobManager.get_variants(db_session, b, tenant_id=TENANT_1)
    assert [v.variant_id for v in leaked] == []

    theirs = await JobManager.get_variants(db_session, b, tenant_id=TENANT_2)
    assert [v.variant_id for v in theirs] == [victim.variant_id]


async def test_iteration_history_is_tenant_scoped(db_session) -> None:
    a, b = await _seed_pair(db_session)

    ok = await get_iteration_history(a, _request(TENANT_1), db=db_session)
    assert [i.job_id for i in ok.iterations] == [a]

    with pytest.raises(HTTPException) as exc:
        await get_iteration_history(b, _request(TENANT_1), db=db_session)
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# 2. Variant download — the variant table carries no tenant_id of its own
# ---------------------------------------------------------------------------


async def test_download_variant_html_is_tenant_scoped(db_session) -> None:
    """t1 gets 404 on B's variant; t2 gets the bytes — so the 404 isn't 'no variant'."""
    _a, b = await _seed_pair(db_session)
    variant = await JobManager.create_variant(db_session, job_id=b, style="bold")
    await JobManager.mark_variant_completed(
        db_session, variant.variant_id, redesigned_html_path=f"/{b}/bold.html"
    )
    await db_session.commit()

    with pytest.raises(HTTPException) as exc:
        await download_variant_html(
            b, "bold", _request(TENANT_1, storage=_storage()), db=db_session
        )
    assert exc.value.status_code == 404

    resp = await download_variant_html(
        b, "bold", _request(TENANT_2, storage=_storage("<html>bold</html>")), db=db_session
    )
    assert resp.body == b"<html>bold</html>"


# ---------------------------------------------------------------------------
# 3. Cancel — a refusal must also prove no write happened
# ---------------------------------------------------------------------------


async def test_cancel_job_is_tenant_scoped(db_session) -> None:
    _a, b = await _seed_pair(db_session)  # B is pending

    with pytest.raises(HTTPException) as exc:
        await cancel_job(b, _request(TENANT_1), db=db_session)
    # Falls through to the existing "not found or already finished" 409 — identical
    # to a genuine already-finished job, so it is not an existence oracle.
    assert exc.value.status_code == 409

    # A 409 alone does not prove the UPDATE was refused. Re-read the row.
    await db_session.rollback()
    row = (
        await db_session.execute(select(RedesignJob).where(RedesignJob.job_id == b))
    ).scalar_one()
    assert row.status == "pending"
    assert row.cancelled_at is None

    ok = await cancel_job(b, _request(TENANT_2), db=db_session)
    assert ok.status == "cancelled"
    await db_session.rollback()
    row = (
        await db_session.execute(select(RedesignJob).where(RedesignJob.job_id == b))
    ).scalar_one()
    assert row.status == "cancelled"


# ---------------------------------------------------------------------------
# 4. Refine — the privilege escalation: a cross-tenant clone billed to the victim
# ---------------------------------------------------------------------------


async def test_refine_cross_tenant_creates_no_child_job(db_session) -> None:
    """B is completed at iteration 0, so a failure cannot come from a status /
    max_iterations gate — only from the tenant predicate."""
    _a, b = await _seed_pair(db_session, completed=True)

    service = RedesignService(MagicMock(), MagicMock(), MagicMock())
    request = _request(TENANT_1, service=service)
    body = RefineRequest(feedback="make it pop")

    with pytest.raises(HTTPException) as exc:
        await refine_redesign(b, body, request, db=db_session)
    assert exc.value.status_code == 400

    await db_session.rollback()
    children = (
        await db_session.execute(
            select(func.count()).select_from(RedesignJob).where(RedesignJob.parent_job_id == b)
        )
    ).scalar_one()
    assert children == 0


# ---------------------------------------------------------------------------
# 5. Worker regression — the claim path holds NO tenant and must stay unscoped
# ---------------------------------------------------------------------------


async def test_worker_claim_path_is_unaffected_by_scoping(db_session) -> None:
    job = await JobManager.create_job(
        db_session, tenant_id=TENANT_1, url="https://w.example", scope="hero"
    )
    await JobManager.set_job_options(db_session, job.job_id, {"kind": "redesign"})
    await db_session.commit()

    claimed = await JobManager.claim_next_job(
        db_session, worker_id="pod", lease_seconds=600, max_attempts=3, max_lifetime_seconds=3600
    )
    assert claimed is not None
    assert claimed.job_id == job.job_id
    assert claimed.tenant_id == TENANT_1

    # ...and the max-lifetime branch (which does a get_job with no tenant bound)
    # still fires.
    row = await JobManager.get_job(db_session, job.job_id, tenant_id=None)
    assert row is not None
    row.started_at = datetime.now(UTC) - timedelta(seconds=7200)
    row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await db_session.commit()

    again = await JobManager.claim_next_job(
        db_session, worker_id="pod", lease_seconds=600, max_attempts=3, max_lifetime_seconds=3600
    )
    assert again is None
    row = await JobManager.get_job(db_session, job.job_id, tenant_id=None)
    assert row is not None
    assert row.status == "failed"
    assert "max job lifetime" in (row.error_message or "")


# ---------------------------------------------------------------------------
# 6. Batches — same contract, and cancel_batch is a cross-tenant WRITE
# ---------------------------------------------------------------------------


async def _seed_batch_pair(db) -> tuple[str, str, str, str]:
    """Seed batch A (t1) and batch B (t2), each with one completed child page.

    Returns ``(batch_a, job_a, batch_b, job_b)``.
    """
    out: list[str] = []
    for tenant, host in ((TENANT_1, "a"), (TENANT_2, "b")):
        batch = await BatchManager.create_batch(db, tenant_id=tenant, total_pages=1)
        job = await JobManager.create_job(
            db,
            tenant_id=tenant,
            url=f"https://{host}.example",
            scope="hero",
            batch_id=batch.batch_id,
        )
        await JobManager.mark_completed(
            db, job.job_id, redesigned_html_path=f"/{job.job_id}/redesign.html"
        )
        out.extend([batch.batch_id, job.job_id])
    await db.commit()
    return out[0], out[1], out[2], out[3]


async def _make_cancellable(db, batch_id: str, job_id: str) -> None:
    """Put a seeded batch + its child page back into a cancellable state.

    ``_seed_batch_pair`` completes the child so the results route has something to
    serve; cancel needs the opposite. Resetting it explicitly means a refusal can
    only come from the tenant gate, never from an already-finished status.
    """
    await db.execute(
        update(BatchJob).where(BatchJob.batch_id == batch_id).values(status="processing")
    )
    await db.execute(
        update(RedesignJob)
        .where(RedesignJob.job_id == job_id)
        .values(status="pending", completed_at=None)
    )
    await db.commit()


async def test_batch_status_is_tenant_scoped(db_session) -> None:
    batch_a, job_a, batch_b, _job_b = await _seed_batch_pair(db_session)

    ok = await get_batch_status(batch_a, _request(TENANT_1), db=db_session)
    assert ok.batch_id == batch_a
    assert [p.job_id for p in ok.page_statuses] == [job_a]  # only t1's child page

    with pytest.raises(HTTPException) as exc:
        await get_batch_status(batch_b, _request(TENANT_1), db=db_session)
    assert exc.value.status_code == 404
    assert exc.value.detail == f"Batch not found: {batch_b}"


async def test_batch_results_are_tenant_scoped(db_session) -> None:
    batch_a, job_a, batch_b, _job_b = await _seed_batch_pair(db_session)

    ok = await get_batch_results(batch_a, _request(TENANT_1), db=db_session)
    assert [r.job_id for r in ok.results] == [job_a]

    with pytest.raises(HTTPException) as exc:
        await get_batch_results(batch_b, _request(TENANT_1), db=db_session)
    assert exc.value.status_code == 404


async def test_cancel_batch_is_tenant_scoped(db_session) -> None:
    """The cross-tenant WRITE. A 409 does not prove the UPDATE was refused —
    re-read both the batch row and its child job and assert nothing moved."""
    _batch_a, _job_a, batch_b, job_b = await _seed_batch_pair(db_session)
    # Put B back in a cancellable state so the refusal can only be the tenant gate.
    await _make_cancellable(db_session, batch_b, job_b)

    with pytest.raises(HTTPException) as exc:
        await cancel_batch(batch_b, _request(TENANT_1), db=db_session)
    assert exc.value.status_code == 409

    await db_session.rollback()
    batch_row = (
        await db_session.execute(select(BatchJob).where(BatchJob.batch_id == batch_b))
    ).scalar_one()
    assert batch_row.status == "processing"  # the UPDATE was refused, not just the response
    job_row = (
        await db_session.execute(select(RedesignJob).where(RedesignJob.job_id == job_b))
    ).scalar_one()
    assert job_row.status == "pending"  # the child-cancel UPDATE was refused too

    ok = await cancel_batch(batch_b, _request(TENANT_2), db=db_session)
    assert ok == {"cancelled": True, "batch_id": batch_b}
    await db_session.rollback()
    batch_row = (
        await db_session.execute(select(BatchJob).where(BatchJob.batch_id == batch_b))
    ).scalar_one()
    assert batch_row.status == "cancelled"


async def test_cancel_writes_are_scoped_independently_of_the_read_guard(
    db_session, monkeypatch
) -> None:
    """The cancel UPDATEs carry the tenant predicate in their OWN WHERE clause.

    Deleting that predicate breaks no route test, because the scoped
    ``get_batch`` / ``get_job`` above it already returns None and short-circuits
    to False — the UPDATE is simply never reached with a mismatched tenant. That
    makes it belt-and-braces, and untested belt-and-braces rots.

    So the guard is deliberately defeated here (the read is stubbed to hand back
    the victim's row for the WRONG tenant) and the write is exercised directly.
    This is the test that fails if someone "simplifies" the UPDATE's WHERE clause
    down to the id, and it is what keeps the write safe if the read guard is ever
    weakened or refactored away.
    """
    _batch_a, _job_a, batch_b, job_b = await _seed_batch_pair(db_session)
    await _make_cancellable(db_session, batch_b, job_b)

    victim_batch = (
        await db_session.execute(select(BatchJob).where(BatchJob.batch_id == batch_b))
    ).scalar_one()
    victim_job = (
        await db_session.execute(select(RedesignJob).where(RedesignJob.job_id == job_b))
    ).scalar_one()

    # Defeat the read guard: pretend the lookup succeeded for the wrong tenant.
    monkeypatch.setattr(BatchManager, "get_batch", AsyncMock(return_value=victim_batch))
    monkeypatch.setattr(JobManager, "get_job", AsyncMock(return_value=victim_job))

    assert await BatchManager.cancel_batch(db_session, batch_b, tenant_id=TENANT_1) is True
    assert await JobManager.cancel_job(db_session, job_b, tenant_id=TENANT_1) is True
    await db_session.commit()

    # Both calls "succeeded" — and both wrote NOTHING, because the predicate is
    # on the UPDATE itself.
    await db_session.rollback()
    assert (
        await db_session.execute(select(BatchJob).where(BatchJob.batch_id == batch_b))
    ).scalar_one().status == "processing"
    assert (
        await db_session.execute(select(RedesignJob).where(RedesignJob.job_id == job_b))
    ).scalar_one().status == "pending"


# ---------------------------------------------------------------------------
# 7. Webhook delivery log — the failure mode is a LEAKY LIST, not a 404
# ---------------------------------------------------------------------------


async def test_webhook_deliveries_are_tenant_scoped(db_session) -> None:
    """``WebhookDeliveryLog`` has no ``tenant_id``; ownership comes from the JOIN.

    The rows are asserted exactly (event_type + attempt_number), not counted: a
    leak here does not raise, it silently appends another tenant's delivery
    history to a 200 response.
    """
    job_a, job_b = await _seed_pair(db_session)
    db_session.add_all(
        [
            WebhookDeliveryLog(
                job_id=job_a, attempt_number=1, event_type="t1.job.started", status="delivered"
            ),
            WebhookDeliveryLog(
                job_id=job_a, attempt_number=2, event_type="t1.job.completed", status="delivered"
            ),
            WebhookDeliveryLog(
                job_id=job_b, attempt_number=7, event_type="t2.job.failed", status="delivered"
            ),
        ]
    )
    await db_session.commit()

    own = await get_webhook_deliveries(job_a, _request(TENANT_1), db=db_session)
    assert {(d.event_type, d.attempt_number) for d in own} == {
        ("t1.job.started", 1),
        ("t1.job.completed", 2),
    }

    # t1 asking for t2's job id must see nothing — same shape as "never fired a
    # webhook", so it is not an existence oracle either.
    leaked = await get_webhook_deliveries(job_b, _request(TENANT_1), db=db_session)
    assert leaked == []

    # ...and t2 really does have a row there, so the empty list above is the
    # tenant gate and not an empty fixture.
    theirs = await get_webhook_deliveries(job_b, _request(TENANT_2), db=db_session)
    assert [(d.event_type, d.attempt_number) for d in theirs] == [("t2.job.failed", 7)]


# ---------------------------------------------------------------------------
# 8. The helper now fails CLOSED — the old fail-open default is gone
# ---------------------------------------------------------------------------


def test_get_tenant_id_fails_closed_when_auth_is_enabled(monkeypatch) -> None:
    """No tenant on ``request.state`` → 401, not a tenant.

    This test previously pinned the OPPOSITE: the helper returned the literal
    string ``"anonymous"`` and the comment above it argued that was safe because
    every read filters through the same helper. It is not. ``"anonymous"`` was
    used as a REAL tenant scope — rows were created under it — so every
    unauthenticated caller shared ONE bucket and could read, rename and delete
    what the previous one left there. Every tenant predicate in this file was
    intact and pointing at a pool.

    ``"anonymous"`` no longer survives ANYWHERE — the ``AUTH_ENABLED=false``
    carve-out this test used to assert is gone, because an env var is not a safe
    place to hide a tenant pool and (since w2v_common 0.67.0) a non-UUID tenant
    cannot be dispatched by the worker anyway. Full matrix, including the
    blank/non-string/slug and nil-UUID cases, lives in
    ``tests/unit/test_tenant_id_fail_closed.py``.
    """
    from src.config import clear_settings_cache

    bare: Any = SimpleNamespace(state=SimpleNamespace())
    clear_settings_cache()
    try:
        monkeypatch.setenv("AUTH_ENABLED", "true")
        with pytest.raises(HTTPException) as exc:
            _get_tenant_id(bare)
        assert exc.value.status_code == 401

        clear_settings_cache()
        monkeypatch.setenv("AUTH_ENABLED", "false")
        with pytest.raises(HTTPException) as exc:
            _get_tenant_id(bare)
        assert exc.value.status_code == 401
    finally:
        clear_settings_cache()


# ---------------------------------------------------------------------------
# 9. Predicates no route test can reach — exercised directly
# ---------------------------------------------------------------------------
#
# Each of these three manager queries sits BEHIND a scoped read in its only
# route (``get_job`` / ``get_batch``), which short-circuits on a mismatched
# tenant. So none of them is exploitable today — and deleting all three left the
# suite fully green, which is precisely why they will rot. They are called
# directly here, the same way the cancel-write test defeats its read guard.
#
# All three return LISTS: the failure mode is a leaky 200, never a 404. So the
# returned id set is asserted exactly, and the other tenant's rows are always
# seeded — an empty list must not be able to pass vacuously.


async def test_get_iteration_history_is_scoped_independently_of_the_read_guard(db_session) -> None:
    """``JobManager.get_iteration_history`` filters by tenant on its own.

    B is given a real second iteration so a leak would return TWO rows: the
    assertion cannot be satisfied by an empty or one-row fixture.
    """
    _a, b = await _seed_pair(db_session)
    child = await JobManager.create_job(
        db_session, tenant_id=TENANT_2, url="https://b.example", scope="hero"
    )
    await db_session.execute(
        update(RedesignJob)
        .where(RedesignJob.job_id == child.job_id)
        .values(root_job_id=b, parent_job_id=b, iteration_number=1)
    )
    await db_session.commit()

    leaked = await JobManager.get_iteration_history(db_session, b, tenant_id=TENANT_1)
    assert [j.job_id for j in leaked] == []

    theirs = await JobManager.get_iteration_history(db_session, b, tenant_id=TENANT_2)
    assert [j.job_id for j in theirs] == [b, child.job_id]


async def test_get_batch_jobs_is_scoped_independently_of_the_read_guard(db_session) -> None:
    """``BatchManager.get_batch_jobs`` filters by tenant on its own."""
    _batch_a, _job_a, batch_b, job_b = await _seed_batch_pair(db_session)

    leaked = await BatchManager.get_batch_jobs(db_session, batch_b, tenant_id=TENANT_1)
    assert [j.job_id for j in leaked] == []

    theirs = await BatchManager.get_batch_jobs(db_session, batch_b, tenant_id=TENANT_2)
    assert [j.job_id for j in theirs] == [job_b]


async def test_get_failed_jobs_is_scoped_independently_of_the_read_guard(db_session) -> None:
    """``BatchManager.get_failed_jobs`` filters by tenant on its own.

    It feeds the batch-retry path, so a leak here would let one tenant re-run
    (and be billed for) another tenant's failed pages.
    """
    _batch_a, _job_a, batch_b, job_b = await _seed_batch_pair(db_session)
    await db_session.execute(
        update(RedesignJob).where(RedesignJob.job_id == job_b).values(status="failed")
    )
    await db_session.commit()

    leaked = await BatchManager.get_failed_jobs(db_session, batch_b, tenant_id=TENANT_1)
    assert [j.job_id for j in leaked] == []

    theirs = await BatchManager.get_failed_jobs(db_session, batch_b, tenant_id=TENANT_2)
    assert [j.job_id for j in theirs] == [job_b]


# ---------------------------------------------------------------------------
# 10. Custom styles — the by-id CRUD that the job fix missed entirely
# ---------------------------------------------------------------------------
#
# ``CustomStyle.tenant_id`` has existed since the table was created, and
# ``list_styles`` / ``count_styles`` / ``get_style_by_slug`` all use it. The
# by-id trio (``get_style`` / ``update_style`` / ``delete_style``) did not, so
# every by-id styles route was id-only: a caller holding tenant A's token could
# READ another tenant's brand palette, RENAME it, and DELETE it, and every leg
# returned success rather than 404.


async def _seed_style_pair(db) -> tuple[str, str]:
    """Seed style A (tenant t1) and style B (tenant t2) in the same session."""
    a = await StyleManager.create_style(
        db, tenant_id=TENANT_1, name="T1 House", description="t1 house style"
    )
    b = await StyleManager.create_style(
        db,
        tenant_id=TENANT_2,
        name="T2 Confidential",
        description="secret brand",
        color_palette={"primary": "#c0ffee"},
    )
    await db.commit()
    return a.style_id, b.style_id


async def _style_row(db, style_id: str) -> CustomStyle:
    await db.rollback()
    return (
        await db.execute(select(CustomStyle).where(CustomStyle.style_id == style_id))
    ).scalar_one()


async def test_get_style_is_tenant_scoped(db_session) -> None:
    """The read leak: t1 could pull t2's description and colour dict verbatim."""
    a, b = await _seed_style_pair(db_session)

    ok = await get_style(a, _request(TENANT_1), db=db_session)
    assert ok.name == "T1 House"  # paired 200 — proves the fixture is real

    with pytest.raises(HTTPException) as exc:
        await get_style(b, _request(TENANT_1), db=db_session)
    assert exc.value.status_code == 404  # never 403 — that confirms the id exists
    assert exc.value.detail == f"Style not found: {b}"  # same string as a genuine miss


async def test_update_style_is_tenant_scoped_and_writes_nothing(db_session) -> None:
    """The rename leak. A 404 alone does not prove the UPDATE was refused —
    re-read the row and assert every field the request tried to move."""
    a, b = await _seed_style_pair(db_session)

    ok = await update_style(
        a, UpdateCustomStyleRequest(name="T1 Renamed"), _request(TENANT_1), db=db_session
    )
    assert ok.name == "T1 Renamed"
    assert ok.slug == "t1-renamed"

    with pytest.raises(HTTPException) as exc:
        await update_style(
            b, UpdateCustomStyleRequest(name="PwnedByT1"), _request(TENANT_1), db=db_session
        )
    assert exc.value.status_code == 404
    # The detail is asserted, not just the status: the route has a SECOND 404
    # ("Style not found after update") on the re-read below the write. If the
    # read guard is unscoped, the attacker's request reaches the UPDATE
    # statement and 404s there instead — same status, different message, and a
    # strictly weaker posture. Only the exact string pins the guard that is
    # supposed to stop it before the write.
    assert exc.value.detail == f"Style not found: {b}"

    row = await _style_row(db_session, b)
    assert row.name == "T2 Confidential"
    assert row.slug == "t2-confidential"
    assert row.tenant_id == TENANT_2

    # ...and t2 really can rename it, so the refusal above is the tenant gate.
    owner = await update_style(
        b, UpdateCustomStyleRequest(name="T2 Renamed"), _request(TENANT_2), db=db_session
    )
    assert owner.name == "T2 Renamed"


async def test_delete_style_is_tenant_scoped_and_deletes_nothing(db_session) -> None:
    """The destructive leg: the adversarial repro ended with ``{'deleted': True}``
    and the victim's row gone. Re-read and assert it is still there."""
    _a, b = await _seed_style_pair(db_session)

    with pytest.raises(HTTPException) as exc:
        await delete_style(b, _request(TENANT_1), db=db_session)
    assert exc.value.status_code == 404

    row = await _style_row(db_session, b)
    assert row.name == "T2 Confidential"  # the DELETE was refused, not just the response

    assert await delete_style(b, _request(TENANT_2), db=db_session) == {
        "deleted": True,
        "style_id": b,
    }
    await db_session.rollback()
    assert (
        await db_session.execute(select(CustomStyle).where(CustomStyle.style_id == b))
    ).scalar_one_or_none() is None


async def test_preview_style_is_tenant_scoped(db_session) -> None:
    """Preview composes the full LLM prompt from the style, so an unscoped read
    hands the victim's palette straight back in ``style_description``."""
    a, b = await _seed_style_pair(db_session)

    service = MagicMock()
    service._html_gen._call_claude = AsyncMock(
        return_value=SimpleNamespace(text="```html\n<p>styled</p>\n```")
    )
    body = StylePreviewRequest(html_snippet="<p>hi</p>")

    ok = await preview_style(a, body, _request(TENANT_1, service=service), db=db_session)
    assert "t1 house style" in ok.style_description

    with pytest.raises(HTTPException) as exc:
        await preview_style(b, body, _request(TENANT_1, service=service), db=db_session)
    assert exc.value.status_code == 404
    assert exc.value.detail == f"Style not found: {b}"


async def test_style_writes_are_scoped_independently_of_the_read_guard(db_session) -> None:
    """The style UPDATE/DELETE carry the tenant predicate in their OWN WHERE.

    Both routes do a scoped ``get_style`` first, so deleting the predicate from
    the write statement breaks no route test — the write is simply never reached
    with a mismatched tenant. This branch has already been bitten twice by
    exactly that, so the writes are exercised directly, with no read above them.
    """
    _a, b = await _seed_style_pair(db_session)

    assert (
        await StyleManager.update_style(db_session, b, tenant_id=TENANT_1, name="PwnedByT1")
        is False
    )
    assert await StyleManager.delete_style(db_session, b, tenant_id=TENANT_1) is False
    await db_session.commit()

    row = await _style_row(db_session, b)
    assert row.name == "T2 Confidential"

    # The owner's write does land — so `False` above is the predicate, not a
    # no-op signature.
    assert (
        await StyleManager.update_style(db_session, b, tenant_id=TENANT_2, name="T2 Renamed")
        is True
    )
    await db_session.commit()
    assert (await _style_row(db_session, b)).name == "T2 Renamed"
