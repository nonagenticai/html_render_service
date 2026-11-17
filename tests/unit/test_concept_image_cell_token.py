"""P7: the html_render -> ai_content_generation concept-image call is a CELL call.

It used to be a raw ``httpx.AsyncClient`` with a static ``Authorization`` header
and no identity hook: no tenant on the wire, no ``X-W2V-Cell-Token``. Once ACG
runs ``CELL_AUTH_MODE=enforce`` that call is refused. These tests pin:

* a queued render job, dispatched by the REAL ``job_worker._run_claimed``, sends
  a cell token whose signature verifies, ``aud`` = ACG's in-cell identity (its
  discovery Service name), ``iss`` = this service, ``tenant`` = the JOB ROW's
  tenant, ``run`` = the w2v job the caller stamped — on every send (submit AND
  each poll), each freshly minted;
* the cell token is the ONLY credential: the shared ``DOWNSTREAM_LICENSE_TOKEN``
  (and the retired ``CONTENT_GENERATION_API_TOKEN``) is neither read into the
  client nor sent, so ACG's ``dual`` receiver takes the cell lane proper — and
  ACG's real verifier accepts the token;
* no tenant bound -> the generator RAISES before any request leaves; it never
  degrades into an anonymous call.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from w2v_common.auth import (
    W2V_JOB_ID_HEADER,
    bind_attribution_from_headers,
    clear_identity_context,
    set_identity_context,
)
from w2v_common.auth.cell_jwt import CELL_TOKEN_HEADER, reset_cell_state, verify_cell_token
from w2v_common.auth.identity_context import mark_inbound_request
from w2v_common.clients import base as client_base
from w2v_common.discovery import cell_services
from w2v_common.tenant import MissingTenantError, TenantSource

from src.services import job_worker
from src.services.concept_image_generator import CellCredentialError, ConceptImageGenerator
from src.services.job_manager import ClaimedJob
from tests.tenants import TENANT_1, TENANT_2

ACG_AUD = cell_services()["ai_content_generation"]  # "ai-content-generation-service"
SELF_ISS = cell_services()["html_render"]
CELL_ID = "test-cell"
SA_LICENCE = "opaque-service-licence"  # pragma: allowlist secret
LEGACY_API_TOKEN = "opaque-legacy-acg-token"  # pragma: allowlist secret
W2V_RUN = "d4d4d4d4-4444-4444-8444-444444444444"

_REAL_ASYNC_CLIENT = httpx.AsyncClient


@pytest.fixture
def signing_key(monkeypatch) -> Iterator[Ed25519PrivateKey]:
    """Configure this pod as a cell signer (what gitops G1 provides in-cluster)."""
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", pem)
    monkeypatch.setenv("CELL_JWT_KID", "html-test-kid")
    monkeypatch.setenv("CELL_SERVICE_NAME", SELF_ISS)
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", CELL_ID)
    monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", SA_LICENCE)
    # Both legacy credentials are SET, so a regression that reads either shows up.
    monkeypatch.setenv("CONTENT_GENERATION_API_TOKEN", LEGACY_API_TOKEN)
    monkeypatch.setattr(client_base, "_RETRY_DELAY", 0.0)
    reset_cell_state()
    clear_identity_context()
    yield key
    clear_identity_context()
    reset_cell_state()


@pytest.fixture
def acg(monkeypatch):
    """Route the generator's REAL BaseServiceClient (hooks intact) to a fake ACG."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"batch_id": "batch-1"})
        return httpx.Response(200, json={"status": "completed", "results": []})

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(client_base.httpx, "AsyncClient", _client)
    return seen


def _generator() -> ConceptImageGenerator:
    gen = ConceptImageGenerator()
    gen._enabled = True
    gen._poll_interval_s = 0.0
    return gen


def _decode(token: str, key: Ed25519PrivateKey) -> dict:
    return jwt.decode(
        token,
        key.public_key(),
        algorithms=["EdDSA"],
        audience=ACG_AUD,
        issuer=SELF_ISS,
    )


def _enqueue_snapshot(tenant: str) -> dict:
    """``_capture_identity`` inside the enqueue request (licence-lane caller)."""
    set_identity_context(tenant, "lic", "sub", tenant_source=TenantSource.LICENSE_TOKEN)
    mark_inbound_request()
    bind_attribution_from_headers({W2V_JOB_ID_HEADER: W2V_RUN}, trusted=True)
    ident = job_worker._capture_identity()
    clear_identity_context()
    return ident


async def _run_job_calling_acg(monkeypatch, gen: ConceptImageGenerator, claimed: ClaimedJob):
    """Run the real worker binding, with ``_dispatch`` replaced by the ACG call."""
    results: list = []

    async def _dispatch(app, job) -> None:
        results.append(await gen.generate(site_description="a site", output_dir=Path("/tmp")))

    monkeypatch.setattr(job_worker, "_dispatch", _dispatch)
    app = SimpleNamespace(state=SimpleNamespace(session_factory=None))
    await job_worker._run_claimed(app, claimed, lease_seconds=30, worker_id="pod")
    return results[0]


async def test_job_acg_call_carries_cell_token_for_the_job_tenant(
    monkeypatch, signing_key, acg
) -> None:
    gen = _generator()
    claimed = ClaimedJob(
        job_id="job-1", tenant_id=TENANT_1, options={"identity": _enqueue_snapshot(TENANT_1)}
    )
    try:
        result = await _run_job_calling_acg(monkeypatch, gen, claimed)
    finally:
        await gen.aclose()

    assert result.degraded is False
    assert [r.method for r in acg] == ["POST", "GET"]
    jtis = set()
    for request in acg:
        assert request.url.host == f"{ACG_AUD}.w2v"
        claims = _decode(request.headers[CELL_TOKEN_HEADER], signing_key)
        assert claims["aud"] == ACG_AUD == "ai-content-generation-service"
        assert claims["tenant"] == TENANT_1
        assert claims["cell"] == CELL_ID
        assert claims["scope"] == "tenant"
        assert claims["run"] == W2V_RUN
        jtis.add(claims["jti"])
        _assert_cell_token_is_the_only_credential(request)
        # Plain tenant scope, no service-account delegation.
        assert request.headers["X-Tenant-Id"] == TENANT_1
        assert "X-Effective-Tenant-Id" not in request.headers
    assert len(jtis) == 2, "every send (submit + poll) must mint its own token"


def _assert_cell_token_is_the_only_credential(request: httpx.Request) -> None:
    for name, value in request.headers.items():
        assert SA_LICENCE not in value, f"DOWNSTREAM_LICENSE_TOKEN leaked in {name}"
        assert LEGACY_API_TOKEN not in value, f"CONTENT_GENERATION_API_TOKEN leaked in {name}"
    assert "Authorization" not in request.headers
    # ACG's AuthMiddleware counts a non-empty ``Bearer`` in X-Tenant-Id or
    # Authorization as a legacy credential; in ``dual`` that keeps the call on
    # the licence lane. None may be present so the cell lane authenticates.
    for name in ("X-Tenant-Id", "Authorization"):
        assert not request.headers.get(name, "").lower().startswith("bearer"), name


def test_client_holds_no_licence_even_with_the_env_token_set(monkeypatch, signing_key) -> None:
    """The env token is not READ: the constructed client carries no licence at all."""
    gen = ConceptImageGenerator()
    assert gen._client.license_token is None
    assert not hasattr(gen, "license_token")


async def test_acg_receiver_verifier_accepts_the_token(
    monkeypatch, tmp_path, signing_key, acg
) -> None:
    """ACG's own verifier (aud/allowed-callers/keyring as deployed) admits the token."""
    pub = signing_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    jwks = tmp_path / "jwks.json"
    jwks.write_text(
        json.dumps(
            {
                "keys": [
                    {
                        "kid": "html-test-kid",
                        "iss": SELF_ISS,
                        "kty": "OKP",
                        "crv": "Ed25519",
                        "x": base64.urlsafe_b64encode(pub).rstrip(b"=").decode(),
                    }
                ]
            }
        )
    )
    gen = _generator()
    claimed = ClaimedJob(
        job_id="job-4", tenant_id=TENANT_1, options={"identity": _enqueue_snapshot(TENANT_1)}
    )
    try:
        await _run_job_calling_acg(monkeypatch, gen, claimed)
    finally:
        await gen.aclose()
    assert acg, "no request reached ACG"

    # Become the receiver: ACG's Service name, its caller allow-list, the keyring.
    monkeypatch.setenv("CELL_SERVICE_NAME", ACG_AUD)
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", f"w2v-orchestrator,{SELF_ISS}")
    monkeypatch.setenv("CELL_JWKS_PATH", str(jwks))
    reset_cell_state()
    for request in acg:
        claims = verify_cell_token(request.headers[CELL_TOKEN_HEADER])
        assert claims.issuer == SELF_ISS
        assert str(claims.tenant_id) == TENANT_1


async def test_job_row_tenant_wins_over_the_snapshot(monkeypatch, signing_key, acg) -> None:
    """The worker binds the JOB ROW's tenant; the cell token must carry exactly that."""
    gen = _generator()
    claimed = ClaimedJob(
        job_id="job-2", tenant_id=TENANT_2, options={"identity": _enqueue_snapshot(TENANT_1)}
    )
    try:
        await _run_job_calling_acg(monkeypatch, gen, claimed)
    finally:
        await gen.aclose()

    assert acg, "no request reached ACG"
    for request in acg:
        assert _decode(request.headers[CELL_TOKEN_HEADER], signing_key)["tenant"] == TENANT_2


async def test_no_tenant_bound_fails_loudly_and_sends_nothing(signing_key, acg) -> None:
    """No tenant -> raise; never an anonymous call riding the service licence."""
    gen = _generator()
    try:
        with pytest.raises(MissingTenantError):
            await gen.generate(site_description="a site", output_dir=Path("/tmp"))
    finally:
        await gen.aclose()
    assert acg == []


async def test_unvouched_job_tenant_fails_loudly_not_degraded(
    monkeypatch, signing_key, acg
) -> None:
    """A legacy job row with no stored provenance: the hook refuses; it must not be swallowed.

    ``UnvouchedTenantError`` is a ``ValueError``; the generator's broad
    ``except ValueError`` used to turn it into an innocuous degraded result.
    """
    from w2v_common.tenant import UnvouchedTenantError

    gen = _generator()
    legacy = {"tenant_id": TENANT_1, "license_id": "lic", "subject_id": "sub"}
    claimed = ClaimedJob(job_id="job-3", tenant_id=TENANT_1, options={"identity": legacy})
    try:
        with pytest.raises(UnvouchedTenantError):
            await _run_job_calling_acg(monkeypatch, gen, claimed)
    finally:
        await gen.aclose()
    assert acg == []


async def test_no_signing_key_fails_loudly_and_sends_nothing(monkeypatch, signing_key, acg) -> None:
    """No signer -> no cell token -> the request must not leave credential-less."""
    monkeypatch.delenv("CELL_JWT_SIGNING_KEY")
    reset_cell_state()
    gen = _generator()
    claimed = ClaimedJob(
        job_id="job-5", tenant_id=TENANT_1, options={"identity": _enqueue_snapshot(TENANT_1)}
    )
    try:
        with pytest.raises(CellCredentialError):
            await _run_job_calling_acg(monkeypatch, gen, claimed)
    finally:
        await gen.aclose()
    assert acg == []
