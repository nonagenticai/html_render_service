"""Proves the Phase D RLS policies ISOLATE tenants for html_render_service, covering
the two novel mechanisms beyond the generic strict policy:

* ``redesign_jobs`` — strict own-only (the generic w2v_common policy).
* ``redesign_variants`` — no tenant_id; visibility derived from the parent
  ``redesign_jobs`` row via an EXISTS join.
* ``claim_next_render_job`` — the SECURITY DEFINER cross-tenant claim resolver still
  claims across tenants (owner bypasses ENABLE) and returns the owning tenant.

Connects AS a throwaway non-owner ``NOBYPASSRLS`` role (the runtime posture under
enforce). Needs a superuser test DB; skips loudly in CI where it can't CREATE ROLE.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from tests.tenants import TENANT_1, TENANT_2

TENANT_A = TENANT_1
TENANT_B = TENANT_2
_ROLE = "w2v_html_render_app_rls_test"

_INS_JOB = (
    "INSERT INTO redesign_jobs (job_id, tenant_id, url, status, scope, style) "
    "VALUES (:id, :t, 'http://x', :st, 'full_page', 'modern')"
)
_INS_VAR = (
    "INSERT INTO redesign_variants (variant_id, job_id, style, status) "
    "VALUES (:id, :job, 'modern', 'pending')"
)


@pytest.mark.asyncio
async def test_rls_isolates_tenants_under_restricted_role(db_engine) -> None:
    async with db_engine.begin() as conn:
        try:
            await conn.execute(text(f'CREATE ROLE "{_ROLE}" NOSUPERUSER NOBYPASSRLS NOINHERIT'))
        except ProgrammingError as exc:
            pytest.skip(f"cannot CREATE ROLE (needs superuser/CREATEROLE): {exc.orig}")
        for tbl in ("redesign_jobs", "redesign_variants"):
            await conn.execute(text(f'GRANT SELECT, INSERT, UPDATE, DELETE ON {tbl} TO "{_ROLE}"'))

        await conn.execute(text(_INS_JOB), {"id": "job-a", "t": TENANT_A, "st": "completed"})
        await conn.execute(text(_INS_JOB), {"id": "job-b", "t": TENANT_B, "st": "completed"})
        await conn.execute(text(_INS_VAR), {"id": "var-a", "job": "job-a"})
        await conn.execute(text(_INS_VAR), {"id": "var-b", "job": "job-b"})

        try:
            await conn.execute(text(f'SET LOCAL ROLE "{_ROLE}"'))

            for tbl in ("redesign_jobs", "redesign_variants"):
                n = (await conn.execute(text(f"SELECT count(*) FROM {tbl}"))).scalar()
                assert n == 0, f"{tbl} leaked {n} rows with no GUC"

            await conn.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": TENANT_A}
            )
            jobs = (await conn.execute(text("SELECT job_id FROM redesign_jobs"))).scalars().all()
            assert jobs == ["job-a"], jobs
            # child visibility derives from the parent job → only A's variant.
            vars_ = (
                (await conn.execute(text("SELECT variant_id FROM redesign_variants")))
                .scalars()
                .all()
            )
            assert vars_ == ["var-a"], vars_

            await conn.execute(text("SAVEPOINT s"))
            with pytest.raises(ProgrammingError):
                await conn.execute(text(_INS_JOB), {"id": "job-x", "t": TENANT_B, "st": "pending"})
            await conn.execute(text("ROLLBACK TO SAVEPOINT s"))
        finally:
            await conn.execute(text("RESET ROLE"))
            await conn.execute(text(f'DROP OWNED BY "{_ROLE}"'))
            await conn.execute(text(f'DROP ROLE IF EXISTS "{_ROLE}"'))


@pytest.mark.asyncio
async def test_claim_resolver_claims_cross_tenant(db_engine) -> None:
    """The SECURITY DEFINER claim resolver claims a queued job and returns its tenant."""
    async with db_engine.begin() as conn:
        # A queue-owned pending job (job_options NOT NULL) for tenant B.
        await conn.execute(
            text(
                "INSERT INTO redesign_jobs "
                "(job_id, tenant_id, url, status, scope, style, job_options, attempts) "
                "VALUES ('claim-b', :t, 'http://x', 'pending', 'full_page', 'modern', "
                '\'{"scope":"hero"}\'::jsonb, 0)'
            ),
            {"t": TENANT_B},
        )
        row = (
            await conn.execute(
                text(
                    "SELECT job_id, tenant_id, job_options "
                    "FROM claim_next_render_job('worker-1', 60, 3)"
                )
            )
        ).first()
        assert row is not None
        assert row.job_id == "claim-b"

        # THE RAW ROW CARRIES A ``uuid.UUID``, NOT A ``str`` — assert it explicitly.
        # Since 0010_tenant_id_uuid the resolver declares ``tenant_id uuid``, and this
        # is a raw ``text()`` query, so it bypasses the SQLAlchemy type layer and
        # psycopg hands back a UUID object. ``uuid.UUID(t) == t`` is False for every
        # string ``t``, so a bare ``row.tenant_id == TENANT_B`` silently fails — and in
        # production the same value would flow into ``ClaimedJob.tenant_id`` (declared
        # ``str``) and then into every per-job scoped read, matching nothing, with no
        # type error anywhere. Pinning the object type here is what makes the
        # ``str(...)`` coercion in ``JobManager.claim_next_job`` provably necessary
        # rather than defensive noise.
        assert isinstance(row.tenant_id, uuid.UUID)
        assert str(row.tenant_id) == TENANT_B
        # The job is now 'processing' with a lease + one attempt.
        st = (
            await conn.execute(
                text("SELECT status, attempts FROM redesign_jobs WHERE job_id = 'claim-b'")
            )
        ).first()
        assert st.status == "processing" and st.attempts == 1
