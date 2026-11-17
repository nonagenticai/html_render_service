"""``_get_tenant_id`` fails CLOSED: anything that is not a real tenant is a 401.

The helper used to be::

    return getattr(request.state, "tenant_id", "anonymous")

and the returned string was then used as a REAL tenant scope by all 19 call
sites across ``variants``, ``batch``, ``refinement``, ``jobs``, ``styles`` and
``webhooks``. Rows were created with ``tenant_id="anonymous"``, and the next
caller who also failed to authenticate landed in the SAME pooled scope — able to
list, read, rename and delete everything the previous one had left there. Every
tenant-scoping guard in this service was intact and still pointed at a shared
bucket.

The first repair moved the fix into the HELPER (one place to be right, no route
can forget it) but kept ``"anonymous"`` alive behind ``AUTH_ENABLED=false``, on
the argument that this was only the local/dev posture. That carve-out is gone
now, for two reasons:

* ``AUTH_ENABLED`` is a runtime env var, so the "only local/dev" premise is one
  bad ConfigMap away from being false in prod, and the failure mode there is
  re-pooling every tenant.
* Since w2v_common 0.67.0 a tenant is a ``uuid.UUID`` with NO fallback. A job
  enqueued under ``"anonymous"`` could not be dispatched at all — the worker's
  ``set_identity_context`` rejects a non-UUID tenant — so the "dev convenience"
  only bought rows that no worker could ever process.

So the tenant is now VALIDATED, not merely checked for presence: it must parse
as a real, non-nil UUID. The nil UUID gets its own matrix below because it is
the substitution w2v_common itself used to make, it is a *valid* UUID (so it
passes every type check), and it is registered in ``licenses.tenants`` as
``default-anon`` (so it passes a dirty-row audit too).
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException, Request

from src.api.routes._helpers import _get_tenant_id
from src.config import clear_settings_cache
from tests.tenants import NIL_TENANT_SPELLINGS, TENANT_1


@pytest.fixture(autouse=True)
def _isolate_settings_cache() -> Iterator[None]:
    """``get_settings`` is ``lru_cache``d — a Settings built under a patched
    AUTH_ENABLED must not leak into (or out of) these tests."""
    clear_settings_cache()
    yield
    clear_settings_cache()


def _request(tenant_id: object | None = None) -> Request:
    """Request stand-in. Not a MagicMock: ``getattr(state, "tenant_id")`` on one
    returns a truthy auto-attribute, so the fail-closed branch would never be
    reached and every assertion here would pass for the wrong reason.

    ``cast`` rather than a per-call-site ``# type: ignore``: ty does not honour
    mypy's ``[arg-type]`` code, so those suppressions were silently inert and
    ``ty check`` (which CI runs) failed on every call below."""
    state = SimpleNamespace() if tenant_id is None else SimpleNamespace(tenant_id=tenant_id)
    return cast(Request, SimpleNamespace(state=state))


def test_authenticated_request_returns_its_tenant(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    assert _get_tenant_id(_request(TENANT_1)) == TENANT_1


def test_a_tenant_is_returned_in_canonical_uuid_form(monkeypatch) -> None:
    """The middleware stamps ``str(uuid)``, but a braced/undashed spelling of the
    SAME tenant must not create a second, non-matching scope in the varchar
    ``tenant_id`` columns. Normalising here is what keeps one tenant one row set.
    """
    monkeypatch.setenv("AUTH_ENABLED", "true")
    assert _get_tenant_id(_request(TENANT_1.replace("-", ""))) == TENANT_1
    assert _get_tenant_id(_request("{" + TENANT_1 + "}")) == TENANT_1


def test_missing_tenant_with_auth_enabled_is_401(monkeypatch) -> None:
    """The pooled-tenant bug: this used to silently return ``"anonymous"``."""
    monkeypatch.setenv("AUTH_ENABLED", "true")

    with pytest.raises(HTTPException) as exc:
        _get_tenant_id(_request())
    assert exc.value.status_code == 401
    assert "anonymous" not in str(exc.value.detail)


@pytest.mark.parametrize("value", ["", "   ", None, 12345, "anonymous", "tenant-abc"])
def test_blank_non_string_or_slug_tenant_is_401(monkeypatch, value) -> None:
    """An empty / whitespace-only / non-string / slug ``tenant_id`` is not an
    identity. ``""`` is the dangerous one: it is a perfectly valid SQL value, so
    it would scope every query to one shared empty-string tenant exactly like
    ``"anonymous"`` did."""
    monkeypatch.setenv("AUTH_ENABLED", "true")

    with pytest.raises(HTTPException) as exc:
        _get_tenant_id(_request(value))
    assert exc.value.status_code == 401


@pytest.mark.parametrize("spelling", NIL_TENANT_SPELLINGS)
def test_the_nil_uuid_is_refused_in_every_spelling(monkeypatch, spelling) -> None:
    """The nil UUID is NOT a tenant, however it is written.

    ``uuid.UUID`` accepts the dashed, undashed, braced and ``urn:uuid:`` forms
    and normalises all four to the same value, so a guard that string-matched
    only the dashed form would leave three doors open. ``require_tenant`` checks
    the VALUE (``parsed.int == 0``), which closes all four at once. This matters
    because the nil UUID is precisely what w2v_common's old trusted-network path
    substituted for a missing tenant.
    """
    monkeypatch.setenv("AUTH_ENABLED", "true")

    with pytest.raises(HTTPException) as exc:
        _get_tenant_id(_request(spelling))
    assert exc.value.status_code == 401


@pytest.mark.parametrize("spelling", NIL_TENANT_SPELLINGS)
def test_the_nil_uuid_is_refused_with_auth_disabled_too(monkeypatch, spelling) -> None:
    """Turning auth off does not re-open the substitution.

    ``AUTH_ENABLED=false`` used to be the carve-out that kept ``"anonymous"``
    alive. If the nil UUID were accepted here, the old pooled-tenant bug would
    simply have moved to a value that LOOKS like a real tenant everywhere it is
    later inspected.
    """
    monkeypatch.setenv("AUTH_ENABLED", "false")

    with pytest.raises(HTTPException) as exc:
        _get_tenant_id(_request(spelling))
    assert exc.value.status_code == 401


def test_missing_tenant_with_auth_disabled_is_also_401(monkeypatch) -> None:
    """The ``AUTH_ENABLED=false`` carve-out is GONE.

    This test previously pinned the opposite (``== "anonymous"``). It is
    rewritten rather than deleted: the behaviour it guarded is exactly what must
    now be refused, and an env var is not a safe place to hide a tenant pool.
    """
    monkeypatch.setenv("AUTH_ENABLED", "false")

    with pytest.raises(HTTPException) as exc:
        _get_tenant_id(_request())
    assert exc.value.status_code == 401


def test_auth_enabled_is_the_default_posture(monkeypatch) -> None:
    """``AuthSettings.enabled`` defaults to True in ``w2v_common``, so a
    deployment that never sets AUTH_ENABLED gets the 401, not the pool."""
    monkeypatch.delenv("AUTH_ENABLED", raising=False)

    with pytest.raises(HTTPException) as exc:
        _get_tenant_id(_request())
    assert exc.value.status_code == 401
