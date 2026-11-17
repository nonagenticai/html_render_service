"""ConceptImageGenerator distinguishes a real failure from a legitimately-empty result.

Concept images are optional enrichment: ``generate`` never raises into the redesign.
But an empty asset list is ambiguous — it can mean "generation failed" OR "feature
off / zero usable images". ``generate`` now returns a :class:`ConceptImageResult`
whose ``degraded`` flag disambiguates the two, so the job can surface a genuine
failure instead of it being indistinguishable from "none requested".
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from w2v_common.auth import clear_identity_context, set_identity_context
from w2v_common.auth.cell_jwt import reset_cell_state
from w2v_common.clients import base as client_base
from w2v_common.tenant import TenantSource

from src.services.concept_image_generator import ConceptImageGenerator
from tests.tenants import TENANT_1

_REAL_ASYNC_CLIENT = httpx.AsyncClient


@pytest.fixture(autouse=True)
def _job_tenant(monkeypatch):
    """Every concept-image call acts for a tenant; bind one (as the job worker does).

    The ACG hop's only credential is the cell token, so this pod must be a cell
    signer (as in-cluster) or the client refuses to send at all.
    """
    pem = (
        Ed25519PrivateKey.generate()
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", pem)
    monkeypatch.setenv("CELL_JWT_KID", "html-degrade-kid")
    monkeypatch.setenv("CELL_SERVICE_NAME", "html-render-service")
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", "test-cell")
    reset_cell_state()
    # BaseServiceClient replays a 5xx GET up to 3x with linear sleeps; not under test here.
    monkeypatch.setattr(client_base, "_RETRY_DELAY", 0.0)
    set_identity_context(TENANT_1, "lic", "sub", tenant_source=TenantSource.LICENSE_TOKEN)
    yield
    clear_identity_context()
    reset_cell_state()


@pytest.fixture
def _gen_with_transport(monkeypatch):
    """A generator whose REAL ``BaseServiceClient`` (identity hooks intact) talks to ``handler``."""

    def _build(handler) -> ConceptImageGenerator:
        transport = httpx.MockTransport(handler)

        def _client(*args, **kwargs):
            kwargs["transport"] = transport
            return _REAL_ASYNC_CLIENT(*args, **kwargs)

        monkeypatch.setattr(client_base.httpx, "AsyncClient", _client)
        gen = ConceptImageGenerator()
        # Force the "available" path regardless of settings flags.
        gen._enabled = True
        gen._poll_interval_s = 0.0
        return gen

    return _build


@pytest.mark.asyncio
async def test_submit_http_500_is_degraded_not_silent_empty(_gen_with_transport) -> None:
    """A 5xx on the batch-submit POST → degraded=True (redesign continues), not a raise."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    gen = _gen_with_transport(handler)
    try:
        result = await gen.generate(site_description="a site", output_dir=Path("/tmp"))
    finally:
        await gen.aclose()

    assert result.assets == []
    assert result.degraded is True
    assert result.reason and "submit failed" in result.reason


@pytest.mark.asyncio
async def test_poll_http_500_is_degraded(_gen_with_transport) -> None:
    """POST accepted, but a 5xx while polling the batch → degraded=True."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"batch_id": "batch-1"})
        return httpx.Response(500, json={"detail": "boom"})

    gen = _gen_with_transport(handler)
    try:
        result = await gen.generate(site_description="a site", output_dir=Path("/tmp"))
    finally:
        await gen.aclose()

    assert result.assets == []
    assert result.degraded is True


@pytest.mark.asyncio
async def test_missing_batch_id_is_degraded(_gen_with_transport) -> None:
    """A submit that returns no batch_id is a failure, not a legitimate empty."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    gen = _gen_with_transport(handler)
    try:
        result = await gen.generate(site_description="a site", output_dir=Path("/tmp"))
    finally:
        await gen.aclose()

    assert result.assets == []
    assert result.degraded is True


@pytest.mark.asyncio
async def test_batch_completed_zero_assets_is_not_degraded(_gen_with_transport) -> None:
    """A batch that completes with zero usable images is empty but NOT degraded."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"batch_id": "batch-1"})
        return httpx.Response(200, json={"status": "completed", "results": []})

    gen = _gen_with_transport(handler)
    try:
        result = await gen.generate(site_description="a site", output_dir=Path("/tmp"))
    finally:
        await gen.aclose()

    assert result.assets == []
    assert result.degraded is False


@pytest.mark.asyncio
async def test_feature_disabled_is_not_degraded() -> None:
    """When image generation is disabled, an empty result is not a degradation."""
    gen = ConceptImageGenerator()
    gen._enabled = False
    try:
        result = await gen.generate(site_description="a site", output_dir=Path("/tmp"))
    finally:
        await gen.aclose()

    assert result.assets == []
    assert result.degraded is False


@pytest.mark.asyncio
async def test_completed_batch_whose_items_are_all_unnormalizable_is_degraded(
    _gen_with_transport,
) -> None:
    """N items back, 0 usable → degraded, with the counts in the reason.

    The batch reports ``completed`` and returns a NON-empty ``results`` list, but
    every item is missing the ``asset_id``/url pair ``_normalize_asset_payload``
    requires, so ``assets`` comes out empty. That used to be reported as
    ``degraded=False`` — byte-identical to a provider that legitimately produced
    nothing — even though the distinguishing datum (len(results) > 0 while
    len(assets) == 0) was in hand and simply discarded. It is a downstream
    contract failure and must say so.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"batch_id": "batch-1"})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                # no asset_id on either item → both fail normalisation
                "results": [{"url": "http://img.test/a.png"}, {"url": "http://img.test/b.png"}],
            },
        )

    gen = _gen_with_transport(handler)
    try:
        result = await gen.generate(site_description="a site", output_dir=Path("/tmp"))
    finally:
        await gen.aclose()

    assert result.assets == []
    assert result.degraded is True, (
        "a batch that returned items none of which are usable is a failure, not a "
        "legitimately-empty result"
    )
    assert result.reason and "2 item(s) returned, 0 normalizable" in result.reason


@pytest.mark.asyncio
async def test_completed_batch_with_a_malformed_results_field_is_degraded(
    _gen_with_transport,
) -> None:
    """``results`` that is not a list is a malformed payload, not an empty one."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"batch_id": "batch-1"})
        return httpx.Response(200, json={"status": "completed", "results": {"oops": 1}})

    gen = _gen_with_transport(handler)
    try:
        result = await gen.generate(site_description="a site", output_dir=Path("/tmp"))
    finally:
        await gen.aclose()

    assert result.assets == []
    assert result.degraded is True
    assert result.reason and "malformed batch payload" in result.reason
