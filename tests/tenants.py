"""Real tenant ids for the test suite.

Since w2v_common 0.67.0 a tenant is a ``uuid.UUID`` at every library boundary
and there is no fallback: ``set_identity_context``, ``get_identity_headers`` and
``build_downstream_tenant_headers`` all RAISE on an absent or non-UUID tenant,
and this service's own ``src.tenant.require_tenant`` additionally rejects the
nil UUID.

The suite used to scope its fixtures with slugs (``"t1"``, ``"t2"``,
``"tenant-abc"``). Those are not tenants, and a test that scopes with one is
proving isolation against a value production can no longer produce. Every
fixture below is a real, distinct, non-nil UUID, so the scoping assertions
prove the same thing against the shape the service actually handles.
"""

from __future__ import annotations

#: Two distinct real tenants — the ones every cross-tenant isolation assertion
#: uses. Every character differs between them, so a swapped value in an error
#: message is obvious at a glance.
TENANT_1 = "11111111-1111-4111-8111-111111111111"
TENANT_2 = "22222222-2222-4222-8222-222222222222"

#: Every spelling ``uuid.UUID`` accepts for the all-zeros value. They normalise
#: to ONE ``UUID`` object, which is exactly why ``require_tenant`` rejects by
#: VALUE (``parsed.int == 0``) and not by string match — a guard written against
#: the dashed form alone would let the other three straight through.
NIL_TENANT_SPELLINGS = (
    "00000000-0000-0000-0000-000000000000",
    "00000000000000000000000000000000",
    "{00000000-0000-0000-0000-000000000000}",
    "urn:uuid:00000000-0000-0000-0000-000000000000",
)

__all__ = ["NIL_TENANT_SPELLINGS", "TENANT_1", "TENANT_2"]
