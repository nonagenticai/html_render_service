"""Tenant-id coercion for this service — the one place a ``str`` becomes a ``uuid.UUID``.

w2v_common 0.67.0 made the tenant a real ``uuid.UUID`` everywhere it crosses a
library boundary (``set_identity_context``, ``get_identity_headers``,
``build_downstream_tenant_headers``) and DELETED every fallback: there is no
``config.auth.default_tenant_id``, no ``UNATTRIBUTED_TENANT_ID``, no nil-UUID
substitution. An absent or unparseable tenant now RAISES.

Since ``0010_tenant_id_uuid`` this service stores ``tenant_id`` as a native
Postgres ``uuid`` (``RedesignJob.tenant_id`` et al). The IN-REPO representation is
still a ``str``, on purpose: the columns are mapped ``PgUUID(as_uuid=False)`` so
every existing ``== tenant_id`` comparison, cache key and pydantic ``str`` field
keeps working unchanged (see ``models.TENANT_ID_TYPE``). That makes exactly one
thing necessary: a single, named seam where a string is proven to be a real
tenant before it is handed to w2v_common or written to a row. This module is that
seam.
"""

from __future__ import annotations

import uuid

from w2v_common.tenant import InvalidTenantIdError, parse_tenant_id

__all__ = ["require_tenant"]


def require_tenant(value: object, *, source: str = "tenant_id") -> uuid.UUID:
    """Return ``value`` as a real, non-nil ``uuid.UUID`` or raise.

    Raises ``w2v_common.tenant.MissingTenantError`` when there is no tenant at
    all and ``InvalidTenantIdError`` when there is one but it is not a usable
    UUID — including the all-zeros "nil" UUID.

    WHY THE NIL CHECK IS BY VALUE AND NOT BY SPELLING: ``uuid.UUID`` accepts the
    dashed (``00000000-0000-0000-0000-000000000000``), undashed
    (``00000000000000000000000000000000``), braced (``{0000...}``) and
    ``urn:uuid:`` forms of the SAME value, and normalises them all to one
    ``UUID`` object. w2v_common's older trusted-network path wrote exactly that
    value as a stand-in for "no tenant", and it is registered in
    ``licenses.tenants`` as ``default-anon``, so it passes every type check and
    every dirty-row audit while silently mis-attributing cost. A guard that only
    rejected ``None`` — or one that string-matched a single spelling — would
    leave the hole open. Comparing ``parsed.int == 0`` closes all four at once.
    """
    parsed = parse_tenant_id(value, source=source)
    if parsed.int == 0:
        raise InvalidTenantIdError(
            f"{source}: nil UUID is not a tenant — every row must carry a real registered tenant"
        )
    return parsed
