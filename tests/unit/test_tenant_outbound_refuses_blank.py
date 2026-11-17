"""The outbound tenant header REFUSES; it never emits a blank ``X-Tenant-Id``.

This is the seam ``ty check .`` pointed at when the 0.67.0 bump turned main red:
a ``uuid.UUID | None`` being fed into a ``dict[str, str]`` of headers. The type
error was the symptom. The defect it named is older and worse: the ``None`` half
of that union used to be written into the header as ``""``, so a tenant-less S2S
call went out looking well-formed and the receiver either 401'd with a confusing
"no valid tenant" or — on a lane that tolerated blanks — attributed the work to
nobody.

Every assertion here is about a REFUSAL. There is deliberately no test that a
missing tenant produces some safe placeholder, because there is no safe
placeholder: the nil UUID was the previous answer to that question and it is the
bug this whole change exists to remove.
"""

from __future__ import annotations

import uuid

import pytest
from w2v_common.auth.identity_context import (
    clear_identity_context,
    get_identity_headers,
    set_identity_context,
)
from w2v_common.auth.outbound import build_downstream_tenant_headers
from w2v_common.tenant import InvalidTenantIdError, MissingTenantError, TenantSource

from src.tenant import require_tenant
from tests.tenants import NIL_TENANT_SPELLINGS, TENANT_1


@pytest.fixture(autouse=True)
def _no_identity_leak():
    """ContextVars live on the shared pytest event loop — a bound tenant leaking
    out of one test would make the "unbound" assertions below pass vacuously."""
    clear_identity_context()
    yield
    clear_identity_context()


def test_identity_headers_refuse_when_no_tenant_is_bound() -> None:
    """No tenant → raise. Emitting ``{"X-Tenant-Id": ""}`` is the old bug."""
    with pytest.raises(MissingTenantError):
        get_identity_headers()


def test_identity_headers_carry_the_real_tenant_as_a_string() -> None:
    set_identity_context(TENANT_1, "lic-1", "subj-1")
    headers = get_identity_headers()

    assert headers["X-Tenant-Id"] == TENANT_1
    # A header dict is ``dict[str, str]``: a bare UUID here would be a TypeError
    # inside httpx at send time, i.e. an outage rather than a type error.
    assert all(isinstance(v, str) for v in headers.values())


def test_downstream_headers_refuse_a_missing_tenant() -> None:
    """``build_downstream_tenant_headers`` is the new helper for hand-built
    header dicts. ``None`` raises rather than producing a blank header."""
    with pytest.raises(MissingTenantError):
        build_downstream_tenant_headers(None, "html_render_service", TenantSource.LICENSE_TOKEN)


def test_downstream_headers_to_a_plain_backend_carry_the_tenant_verbatim() -> None:
    headers = build_downstream_tenant_headers(
        uuid.UUID(TENANT_1), "html_render_service", TenantSource.LICENSE_TOKEN
    )
    assert headers == {"X-Tenant-Id": TENANT_1}


@pytest.mark.parametrize("spelling", NIL_TENANT_SPELLINGS)
def test_require_tenant_refuses_the_nil_uuid_in_every_spelling(spelling) -> None:
    """All four spellings normalise to the same value, so the guard must compare
    the VALUE. This is the mutation-sensitive assertion: flipping
    ``if parsed.int == 0`` to a constant-false makes every parameter here fail.
    """
    with pytest.raises(InvalidTenantIdError):
        require_tenant(spelling, source="test")


def test_require_tenant_refuses_absent_and_slug_tenants() -> None:
    with pytest.raises(MissingTenantError):
        require_tenant(None, source="test")
    with pytest.raises(MissingTenantError):
        require_tenant("", source="test")
    with pytest.raises(InvalidTenantIdError):
        require_tenant("anonymous", source="test")


def test_require_tenant_returns_a_uuid_for_a_real_tenant() -> None:
    assert require_tenant(TENANT_1, source="test") == uuid.UUID(TENANT_1)
    assert require_tenant(uuid.UUID(TENANT_1), source="test") == uuid.UUID(TENANT_1)
