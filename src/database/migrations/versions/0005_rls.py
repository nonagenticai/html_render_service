"""Phase D RLS — enable row-level security across the tenant-scoped tables.

(1) Strict per-table policies on the four ``tenant_id`` tables:
    redesign_jobs, redesign_usage, custom_styles, batch_jobs.
(2) Parent-derived join policies on the child tables with no ``tenant_id`` of their
    own (redesign_variants, webhook_delivery_log → redesign_jobs by job_id).
(3) SECURITY DEFINER ``claim_next_render_job`` so the cross-tenant lease-queue worker
    keeps claiming jobs under enforce (a non-bypass role with no GUC would otherwise
    see 0 claimable rows). EXECUTE is REVOKEd from PUBLIC here and GRANTed to the app
    role out-of-band in Stage A (the app role does not exist in CI).

ENABLE-not-FORCE → inert until the runtime connects as the restricted
``w2v_html_render_app`` role with ``TENANT_RLS_MODE=enforce``. Fully reversible.

⚠️ THIS REVISION IS PINNED TO LITERAL, ERA-CORRECT DDL — DO NOT RE-RENDER IT.
It originally called ``w2v_common.upgrade_statements(Base.metadata)``,
``rls_policies.upgrade_statements()`` and ``rls_resolvers.CLAIM_FN_SQL``, i.e. it
rendered its DDL from the *live* modules. That is only correct while the live
modules still describe the schema as it stood AT THIS POINT in the chain, and
``0010_tenant_id_uuid`` ended that: those three now emit a ``::uuid`` GUC cast and a
``tenant_id uuid`` resolver return type. Replaying the chain from scratch therefore
died right here with

    (psycopg.errors.UndefinedFunction) operator does not exist: character varying = uuid
    [SQL: CREATE POLICY tenant_isolation ON batch_jobs USING (tenant_id = NULLIF(...)::uuid) ...]

— because at revision 0005 the column IS still ``character varying(255)`` (``0001``
creates it that way, as a literal). A fresh database could not be built at all, which
means CI, every ephemeral test DB and any disaster-recovery rebuild were broken, not
just the migration under development. A migration is immutable history: it must
describe the schema of its own era, so the DDL below is spelled out verbatim as this
revision originally produced it, and it no longer imports anything that can move.

``0010`` carries the same literals a second time (as ``_LEGACY_*``) for its
``downgrade()``. That duplication is deliberate and cannot be factored out: a version
module named ``0005_rls`` is not importable by normal Python syntax, and cross-
importing revisions would reintroduce exactly the coupling this pin removes.

Revision ID: 0005_rls
Revises: 0004_section_screenshot_paths
"""

from alembic import op

revision: str = "0005_rls"
down_revision: str | None = "0004_section_screenshot_paths"
branch_labels = None
depends_on = None

#: The transaction-scoped GUC, always TEXT. ``current_setting(..., true)`` is
#: missing-ok (unset → NULL); ``NULLIF(…, '')`` folds the empty string a pooled
#: PgBouncer connection leaves behind back to NULL, so the predicate is simply
#: false (0 rows) rather than raising.
_GUC = "NULLIF(current_setting('app.current_tenant', true), '')"

#: Era-correct: at this revision ``tenant_id`` is ``character varying(255)``, so the
#: comparison is text-to-text. ``0010_tenant_id_uuid`` is what adds the ``::uuid``.
_PREDICATE = f"tenant_id = {_GUC}"

#: The four tables carrying ``tenant_id`` directly, all NOT NULL → strict equality
#: (no nullable "global-or-own" variant). Order matches ``metadata.sorted_tables``.
_TENANT_TABLES: tuple[str, ...] = (
    "batch_jobs",
    "custom_styles",
    "redesign_jobs",
    "redesign_usage",
)

#: Children with no ``tenant_id`` of their own; isolation derives from the parent
#: ``redesign_jobs`` row via ``job_id``. The EXISTS sub-select runs under
#: ``redesign_jobs``' own policy — no recursion, since redesign_jobs never
#: references these children.
_CHILD_TABLES: tuple[str, ...] = ("redesign_variants", "webhook_delivery_log")

#: Policy names ``w2v_common`` may have created (strict + the nullable variants), so
#: the drop is exhaustive regardless of which shape was applied.
_ALL_POLICY_NAMES: tuple[str, ...] = (
    "tenant_isolation",
    "tenant_isolation_read",
    "tenant_isolation_ins",
    "tenant_isolation_upd",
    "tenant_isolation_del",
)


def _child_predicate(child: str) -> str:
    # Both interpolations are module-level literals: ``child`` comes from
    # ``_CHILD_TABLES`` and ``_GUC`` is a fixed SQL expression. A table IDENTIFIER
    # cannot be a bind parameter, so the f-string shape is unavoidable here.
    # ``noqa`` silences ruff ONLY -- bandit needs its own ``nosec``, and omitting it
    # is how a line that reads as handled leaves CI red.
    return (
        f"EXISTS (SELECT 1 FROM redesign_jobs j "  # noqa: S608  # nosec B608
        f"WHERE j.job_id = {child}.job_id AND j.tenant_id = {_GUC})"
    )


#: Era-correct resolver: ``tenant_id varchar``, matching the column at this point in
#: the chain. ``0010`` DROPs it and re-creates it returning ``uuid`` (``CREATE OR
#: REPLACE`` cannot change a return type).
_CLAIM_FN_SQL = """
CREATE OR REPLACE FUNCTION claim_next_render_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id varchar, job_options jsonb)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_job_id varchar; v_tenant varchar; v_options jsonb; v_attempts int;
  BEGIN
    SELECT rj.id, rj.job_id, rj.tenant_id, rj.job_options, rj.attempts
      INTO v_id, v_job_id, v_tenant, v_options, v_attempts
    FROM redesign_jobs rj
    WHERE rj.job_options IS NOT NULL
      AND (
        rj.status = 'pending'
        OR (rj.status = 'processing'
            AND (rj.lease_expires_at IS NULL OR rj.lease_expires_at < now()))
      )
    ORDER BY rj.created_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_id IS NULL THEN
      RETURN;  -- empty queue
    END IF;

    IF COALESCE(v_attempts, 0) >= p_max_attempts THEN
      UPDATE redesign_jobs
        SET status = 'failed',
            error_message = 'exceeded max claim attempts (' || p_max_attempts
                            || ') - failed as poison job',
            completed_at = now()
      WHERE id = v_id;
      RETURN;  -- poison job skipped
    END IF;

    UPDATE redesign_jobs
      SET status = 'processing',
          claimed_by = p_worker_id,
          started_at = COALESCE(started_at, now()),
          lease_expires_at = now() + make_interval(secs => p_lease_seconds),
          attempts = COALESCE(attempts, 0) + 1
    WHERE id = v_id;

    RETURN QUERY SELECT v_job_id, v_tenant, v_options;
  END
  $fn$
"""


def upgrade() -> None:
    for table in _TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})"
        )

    for child in _CHILD_TABLES:
        pred = _child_predicate(child)
        op.execute(f"ALTER TABLE {child} ENABLE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {child}")
        op.execute(f"CREATE POLICY tenant_isolation ON {child} USING ({pred}) WITH CHECK ({pred})")

    op.execute(_CLAIM_FN_SQL)
    op.execute("REVOKE EXECUTE ON FUNCTION claim_next_render_job(text, int, int) FROM PUBLIC")


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS claim_next_render_job(text, int, int)")
    for table in (*_CHILD_TABLES, *_TENANT_TABLES):
        for policy in _ALL_POLICY_NAMES:
            op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
