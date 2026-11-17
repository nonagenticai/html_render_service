"""Phase D RLS — re-bind the tenant GUC at the start of every transaction.

The job-queue worker claims a job cross-tenant (SECURITY DEFINER
``claim_next_render_job``), then ``job_worker._run_claimed`` calls
``set_identity_context(claimed.tenant_id)`` and runs the render. But the heartbeat
(``renew_lease``) and the background processors open RAW ``app.state.session_factory()``
sessions (NOT w2v_common ``get_db_session``), which do not auto-bind the GUC — and
those sessions commit multiple times, dropping the transaction-scoped GUC each time.

This installs an ``after_begin`` listener that re-applies ``app.current_tenant`` from
the identity ContextVar (which ``_run_claimed`` sets) at the start of EVERY
transaction. Inert unless ``TENANT_RLS_MODE=enforce`` and a tenant is in context, so
the pre-claim poll session (no tenant yet) is a no-op and the claim itself goes
through the RLS-bypassing SECURITY DEFINER function. Scope is this process only.
"""

from __future__ import annotations

from sqlalchemy import Connection, event
from sqlalchemy.orm import Session, SessionTransaction
from w2v_common.auth.identity_context import get_tenant_id_from_context
from w2v_common.database.rls import tenant_rls_mode

_installed = False


def install_rls_tenant_rebind() -> None:
    """Register the after_begin GUC re-bind listener once per process."""
    global _installed
    if _installed:
        return

    @event.listens_for(Session, "after_begin")
    def _rebind_tenant(
        session: Session, transaction: SessionTransaction, connection: Connection
    ) -> None:
        if tenant_rls_mode() != "enforce":
            return
        tenant = get_tenant_id_from_context()
        if tenant is None:
            return
        # ``str(...)`` is load-bearing, not cosmetic. Since w2v_common 0.67.0
        # this ContextVar holds a ``uuid.UUID``, and psycopg adapts a UUID to
        # the postgres ``uuid`` type — so the bare object would render as
        # ``set_config(unknown, uuid, boolean)``, which does not exist, and
        # every enforce-mode transaction would die on the GUC re-bind. The GUC
        # is ALWAYS text — that is unchanged by ``0010_tenant_id_uuid``, which
        # made the tenant COLUMNS ``uuid``: the policy predicates now cast the
        # GUC (``NULLIF(current_setting(...), '')::uuid``) rather than the
        # column, so the tenant must still cross this seam as text.
        connection.exec_driver_sql(
            "SELECT set_config('app.current_tenant', %s, true)", (str(tenant),)
        )

    _installed = True
