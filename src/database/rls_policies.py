"""Phase D RLS — join policies for the child tables with no tenant_id of their own.

``redesign_variants`` and ``webhook_delivery_log`` both reference ``redesign_jobs``
by ``job_id`` (varchar) but carry no ``tenant_id`` column, so the generic
w2v_common per-table policy can't cover them. Their isolation is derived from the
parent job: a row is visible/writable iff the parent ``redesign_jobs`` row is visible
to the current tenant. The EXISTS sub-select runs under ``redesign_jobs``' own (strict)
RLS — no recursion, since redesign_jobs never references these children.

Single-sourced for both the migration (prod) and the ``create_all`` ``after_create``
event (tests). Fully reversible.

THE GUC IS ALWAYS TEXT AND ``redesign_jobs.tenant_id`` IS A ``uuid`` (migration
``0010_tenant_id_uuid``), so the tenant side of the predicate MUST carry an explicit
``::uuid`` cast. Without it Postgres raises ``operator does not exist: uuid =
text`` at ``CREATE POLICY`` time; with the cast on the WRONG side (``j.tenant_id::text
= <guc>``) the policy is created happily and then compares a canonical uuid rendering
against whatever the caller bound — which returns ZERO rows SILENTLY rather than
erroring. ``NULLIF(..., '')`` still folds the empty string PgBouncer leaves behind
back to NULL, so an unbound tenant is a false predicate (0 rows) and never
``''::uuid`` raising a 500. The ``job_id`` join stays a plain varchar-to-varchar
comparison — it is not a tenant column and nothing about it changed.
"""

from __future__ import annotations

from sqlalchemy import Connection, MetaData, event, text

POLICY_NAME = "tenant_isolation"
_GUC = "NULLIF(current_setting('app.current_tenant', true), '')"

_CHILD_TABLES = ("redesign_variants", "webhook_delivery_log")


def _parent_visible(child: str) -> str:
    return (
        f"EXISTS (SELECT 1 FROM redesign_jobs j "  # nosec B608
        f"WHERE j.job_id = {child}.job_id AND j.tenant_id = {_GUC}::uuid)"
    )


def upgrade_statements() -> list[str]:
    out: list[str] = []
    for t in _CHILD_TABLES:
        pred = _parent_visible(t)
        out += [
            f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY",
            f"DROP POLICY IF EXISTS {POLICY_NAME} ON {t}",
            f"CREATE POLICY {POLICY_NAME} ON {t} USING ({pred}) WITH CHECK ({pred})",
        ]
    return out


def downgrade_statements() -> list[str]:
    out: list[str] = []
    for t in _CHILD_TABLES:
        out.append(f"DROP POLICY IF EXISTS {POLICY_NAME} ON {t}")
        out.append(f"ALTER TABLE {t} DISABLE ROW LEVEL SECURITY")
    return out


def register_child_policies(metadata: MetaData) -> None:
    """Apply the child join policies after create_all (tests)."""

    def _apply(target: object, connection: Connection, **kw: object) -> None:
        for stmt in upgrade_statements():
            connection.execute(text(stmt))

    event.listen(metadata, "after_create", _apply)
