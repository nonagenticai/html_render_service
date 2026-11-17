"""Fleet ruling: ``tenant_id`` is a real ``uuid``, never text, never a sentinel.

All four tenant-scoped tables carried ``tenant_id`` as ``character varying(255)``.

MEASURED LIVE (prod, 2026-08-27) — every value already parses as a uuid:
``batch_jobs`` 0 rows, ``custom_styles`` 0 rows, ``redesign_jobs`` 967,
``redesign_usage`` 87; **0 dirty in all four**. So there is deliberately NO backfill
statement here — there is nothing to map. The fail-loud guard is still emitted for
every one of the four, because "0 dirty when I measured" is a statement about
yesterday: the migration runs against whatever prod holds at run time, and a row
written between the measurement and the deploy must abort the migration loudly
rather than be skipped, dropped, or discovered later as a NULL tenant.

Order is load-bearing. Four things must happen in this exact sequence:

1. **DROP ``claim_next_render_job`` FIRST — this is the trap in this repo.**
   The resolver declares ``RETURNS TABLE(job_id varchar, tenant_id varchar,
   job_options jsonb)``. ``ALTER COLUMN ... TYPE`` does **not** revalidate function
   bodies, so a resolver left in place survives the flip and then fails on every
   single call with ``structure of query does not match function result type``.
   That function IS the queue drain (``JobManager.claim_next_job``): the failure
   mode is not a broken query somewhere, it is that NOTHING EVER RENDERS AGAIN,
   with no schema error to point at. ``CREATE OR REPLACE`` cannot change a return
   type ("cannot change return type of existing function"), so it must be DROPped
   and re-created. The recreate renders from ``rls_resolvers.CLAIM_FN_SQL``, which
   now declares ``tenant_id uuid`` — single-sourced, so the migration (prod) and the
   ``create_all`` ``after_create`` event (tests) cannot drift.

2. **Guard, fail loud.** A ``DO`` block counts every remaining non-NULL, non-uuid
   value per table and ``RAISE EXCEPTION``s with the count and a sample.
   Deliberately NOT a ``WHERE tenant_id ~ ...`` filter on the ALTER: a filter would
   quietly leave unconvertible rows behind or drop them, and silent data loss on a
   tenant key is worse than a failed migration.

3. **Drop ALL SIX policies before the ALTER.** Postgres refuses ``ALTER TABLE ...
   ALTER COLUMN ... TYPE`` on a column referenced by a policy expression ("cannot
   alter type of a column used in a policy definition"). Two of the six are the
   subtle ones and are easy to miss because the tables they sit on have **no
   ``tenant_id`` column of their own**: ``redesign_variants`` and
   ``webhook_delivery_log`` isolate through an EXISTS sub-select against
   ``redesign_jobs.tenant_id`` (see ``rls_policies.py``). They reference the column
   being altered just as directly as the plain four do.

4. **ALTER, then re-create all six + the resolver.** The re-created predicates cast
   the GUC, not the column: the GUC ``app.current_tenant`` is always text, so
   against a ``uuid`` column the old text predicate raises ``operator does not
   exist: uuid = text``. Getting this half-right is worse than getting it wrong —
   ``j.tenant_id::text = <guc>`` *creates* fine and then returns ZERO rows SILENTLY.
   Both the plain four (rendered by ``w2v_common``, which keys its cast off the
   SQLAlchemy column type and now sees ``PgUUID``) and the two EXISTS policies
   (rendered by ``rls_policies``, which now appends ``::uuid``) are re-created here.

Nullability, indexes and constraints are preserved/rebuilt by ``ALTER COLUMN TYPE``
itself — including ``0009``'s UNIQUE ``ix_custom_styles_tenant_slug (tenant_id, slug)``
and ``uq_redesign_usage_tenant_date``. Nothing drops them, so nothing has to
re-create them; the migration test asserts they are still there afterwards.

The Python-side representation deliberately stays ``str``: ``models.TENANT_ID_TYPE``
is ``PgUUID(as_uuid=False)``. See the comment on that constant — ``as_uuid=True``
would return ``uuid.UUID`` objects and silently falsify every ``== tenant_id``
comparison in the service without tripping any type checker.

``downgrade()`` restores ``character varying(255)`` and the era-correct text
predicates + varchar-signature resolver. It CANNOT render either from the live
modules: those now describe uuid, so ``upgrade_statements`` would emit a ``::uuid``
predicate against a varchar column, and ``CLAIM_FN_SQL`` would declare a ``uuid``
return over a varchar column. Both are therefore spelled out as literals below —
same reason ``0005`` is pinned to what it originally produced.

Revision ID: 0010_tenant_id_uuid
Revises: 0009_style_tenant_slug_unique
"""

from collections.abc import Sequence

from alembic import op
from w2v_common.database.rls import downgrade_statements, upgrade_statements

from src.database import rls_policies
from src.database.models import Base
from src.database.rls_resolvers import CLAIM_FN_SQL

revision: str = "0010_tenant_id_uuid"
down_revision: str | None = "0009_style_tenant_slug_unique"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


#: The four tables carrying a ``tenant_id`` column, pinned in time. Deliberately a
#: literal tuple rather than ``tenant_table_names(Base.metadata)``: a migration is
#: immutable history, and a metadata-derived list would start emitting DDL for a
#: fifth tenant table the day one is added — a table that does not exist at this
#: point in the revision chain.
TENANT_TABLES: tuple[str, ...] = (
    "batch_jobs",
    "custom_styles",
    "redesign_jobs",
    "redesign_usage",
)

_UUID_RE = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

#: The GUC, exactly as ``w2v_common`` and ``rls_policies`` render it. Always text.
_GUC = "NULLIF(current_setting('app.current_tenant', true), '')"

#: What ``0005_rls`` emitted for a NOT NULL **varchar** tenant_id (text-to-text).
#: Re-created verbatim on downgrade — see the module docstring for why this cannot
#: be regenerated from ``w2v_common`` at downgrade time.
_LEGACY_TEXT_PREDICATE = f"tenant_id = {_GUC}"

#: What ``rls_policies`` emitted for the two child tables that have no ``tenant_id``
#: of their own, before this revision: EXISTS against the parent, compared as text.
_LEGACY_CHILD_TABLES: tuple[str, ...] = ("redesign_variants", "webhook_delivery_log")


def _legacy_child_predicate(child: str) -> str:
    # See 0005_rls._child_predicate: ``child`` and ``_GUC`` are module-level
    # literals and a table identifier cannot be bound. ``noqa`` is ruff-only, so
    # bandit gets its own ``nosec``.
    return (
        f"EXISTS (SELECT 1 FROM redesign_jobs j "  # noqa: S608  # nosec B608
        f"WHERE j.job_id = {child}.job_id AND j.tenant_id = {_GUC})"
    )


#: Era-correct (pre-0010) resolver signature, spelled out literally for the
#: downgrade. Identical to ``0005``'s body except the ``varchar`` return type and
#: ``v_tenant`` declaration this revision changed.
_LEGACY_CLAIM_FN_SQL = """
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

#: ``0005`` REVOKEs EXECUTE from PUBLIC after creating the function. ``DROP``
#: destroys the function's whole ACL, so this must be re-issued on BOTH sides of
#: this migration or the resolver comes back world-executable.
_REVOKE_CLAIM_FN = "REVOKE EXECUTE ON FUNCTION claim_next_render_job(text, int, int) FROM PUBLIC"

#: ...AND THE SAME DROP SILENTLY DESTROYS THE APP ROLE'S GRANT. Measured on the
#: throwaway 18 while verifying this migration: after the DROP/CREATE the restricted
#: role got ``ERROR: permission denied for function claim_next_render_job`` on the
#: very next claim. The grant was applied out-of-band in Stage A, so nothing in this
#: repo would have re-applied it and nothing would have said so — the queue would
#: simply stop draining, with a permissions error rather than a schema error to
#: chase. Re-granting is therefore part of this migration, not an operator's to
#: remember. Guarded by an EXISTS on ``pg_roles`` because the role does NOT exist in
#: CI or on a throwaway DB, where an unguarded GRANT would abort the migration.
_REGRANT_CLAIM_FN = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_html_render_app') THEN
    GRANT EXECUTE ON FUNCTION claim_next_render_job(text, int, int) TO w2v_html_render_app;
  END IF;
END $$
"""

_DROP_CLAIM_FN = "DROP FUNCTION IF EXISTS claim_next_render_job(text, int, int)"

#: Guard template. Interpolated ONLY with names from ``TENANT_TABLES`` (a
#: module-level literal tuple) and the uuid regex — never with anything caller- or
#: row-derived. A table identifier cannot be a bind parameter.
_GUARD_TEMPLATE = """
DO $$
DECLARE
  bad_count bigint;
  sample text;
BEGIN
  SELECT count(*), min(tenant_id) INTO bad_count, sample
    FROM {table}
   WHERE tenant_id IS NOT NULL AND tenant_id !~ '{uuid_re}';
  IF bad_count > 0 THEN
    RAISE EXCEPTION
      '{table}.tenant_id: % row(s) are not a parseable uuid (e.g. %). Refusing to '
      'ALTER to uuid. Map each value to its tenant uuid in licenses.tenants and add '
      'an explicit backfill above, then re-run. Do NOT skip them with a WHERE - a '
      'skipped row is a lost tenant key.',
      bad_count, sample;
  END IF;
END $$
"""


def _guard_sql(table: str) -> str:
    """FAIL LOUD if any non-NULL ``tenant_id`` is still not a parseable uuid."""
    return _GUARD_TEMPLATE.format(table=table, uuid_re=_UUID_RE)


def _drop_all_policies() -> None:
    """Drop all six ``tenant_isolation`` policies — the four plain, the two EXISTS."""
    for stmt in rls_policies.downgrade_statements():
        op.execute(stmt)
    for stmt in downgrade_statements(Base.metadata):
        op.execute(stmt)


def upgrade() -> None:
    # 1. The resolver's varchar return type would survive the ALTER and then kill
    #    every claim. DROP it (CREATE OR REPLACE cannot change a return type).
    op.execute(_DROP_CLAIM_FN)

    # 2. Fail loud on anything unparseable — never silently skipped or dropped.
    for table in TENANT_TABLES:
        op.execute(_guard_sql(table))

    # 3. Policies reference tenant_id, so they block the type change outright.
    _drop_all_policies()

    # 4. The type flip. Preserves nullability; rebuilds indexes + unique constraints
    #    (0009's UNIQUE ix_custom_styles_tenant_slug, uq_redesign_usage_tenant_date).
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid")

    # 5. Re-create all six policies with the GUC cast to uuid (w2v_common renders
    #    ::uuid now that models.py declares PgUUID; rls_policies appends it for the
    #    two EXISTS children), then the resolver with ``tenant_id uuid``.
    for stmt in upgrade_statements(Base.metadata):
        op.execute(stmt)
    for stmt in rls_policies.upgrade_statements():
        op.execute(stmt)
    op.execute(CLAIM_FN_SQL)
    op.execute(_REVOKE_CLAIM_FN)
    op.execute(_REGRANT_CLAIM_FN)


def downgrade() -> None:
    # Drop the uuid-shaped resolver + all six uuid-shaped policies first.
    op.execute(_DROP_CLAIM_FN)
    _drop_all_policies()

    for table in TENANT_TABLES:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN tenant_id TYPE character varying(255) "
            f"USING tenant_id::text"
        )

    # Restore the text-comparison policies 0005 originally created. All four
    # tenant_id columns are NOT NULL -> strict equality (no global-or-own variant).
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING ({_LEGACY_TEXT_PREDICATE}) WITH CHECK ({_LEGACY_TEXT_PREDICATE})"
        )

    # ...and the two EXISTS child policies, comparing the parent tenant as text.
    for child in _LEGACY_CHILD_TABLES:
        pred = _legacy_child_predicate(child)
        op.execute(f"ALTER TABLE {child} ENABLE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {child}")
        op.execute(f"CREATE POLICY tenant_isolation ON {child} USING ({pred}) WITH CHECK ({pred})")

    # The resolver goes back to its ``tenant_id varchar`` signature, re-REVOKEd.
    op.execute(_LEGACY_CLAIM_FN_SQL)
    op.execute(_REVOKE_CLAIM_FN)
    op.execute(_REGRANT_CLAIM_FN)
