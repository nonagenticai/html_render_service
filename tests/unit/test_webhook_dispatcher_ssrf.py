"""SSRF guard + IP-pinning / redirect / bounded-body behaviour on the
tenant-configured webhook target (webhook_dispatcher).

These tests drive the *real* ``w2v_common.net.safe_post`` code path: the
dispatcher's ``httpx.AsyncClient`` is replaced with a genuine client bound to an
``httpx.MockTransport``, so every layer below ``dispatch()`` — validation, the
IP pin, redirect handling, the response-body cap — actually executes. Nothing
here touches the network: ``socket.getaddrinfo`` is stubbed and the transport is
in-process.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services.webhook_dispatcher import WebhookDispatcher

# Captured before any patching so the client factory below can build a REAL
# client while ``httpx.AsyncClient`` is monkeypatched (otherwise: recursion).
_REAL_ASYNC_CLIENT = httpx.AsyncClient

_PUBLIC_IP = "93.184.216.34"
_WEBHOOK_HOST = "hooks.example.com"
_WEBHOOK_URL = f"https://{_WEBHOOK_HOST}/cb"


def _fake_db() -> AsyncMock:
    """An AsyncSession stub whose add() is sync and flush() awaits."""
    db = AsyncMock()
    db.add = MagicMock()
    return db


def _public_getaddrinfo(_host: str, *_a: object, **_k: object) -> list:
    """Resolve any host to a public IP so validate_public_url passes."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (_PUBLIC_IP, 0))]


_Handler = Callable[[httpx.Request], httpx.Response]


def _client_factory(handler: _Handler) -> Callable[..., httpx.AsyncClient]:
    """Build a drop-in replacement for ``httpx.AsyncClient`` that routes every
    request through ``handler`` via ``httpx.MockTransport``.

    ``safe_post`` receives this client and calls ``client.stream(...)`` on it for
    real, so the pin/redirect/cap logic is genuinely exercised.
    """

    def _factory(
        *_args: object, timeout: float | None = None, **_kwargs: object
    ) -> httpx.AsyncClient:
        return _REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(handler),
            timeout=timeout if timeout is not None else 10.0,
        )

    return _factory


def _patch_client(handler: _Handler) -> AbstractContextManager[object]:
    """Patch the dispatcher module's httpx.AsyncClient with the mock-transport factory."""
    return patch("src.services.webhook_dispatcher.httpx.AsyncClient", _client_factory(handler))


# ---------------------------------------------------------------------------
# Guard: targets that must never be POSTed to at all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data",  # cloud metadata (literal IP)
        "http://10.0.0.5/hook",  # RFC1918 (literal IP)
        "ftp://example.com/hook",  # non-http scheme
    ],
)
async def test_dispatch_blocks_internal_or_bad_webhook(url: str) -> None:
    """A webhook pointed at an internal address (or bad scheme) is refused — no POST,
    a 'blocked' delivery row is logged, and dispatch returns False."""
    db = _fake_db()
    dispatcher = WebhookDispatcher()

    with patch("src.services.webhook_dispatcher.httpx.AsyncClient") as mock_client:
        delivered = await dispatcher.dispatch(
            db,
            job_id="job-1",
            webhook_url=url,
            event_type="job.completed",
            payload={"ok": True},
        )

    assert delivered is False
    mock_client.assert_not_called()  # never attempted the POST
    # a blocked delivery row was recorded
    assert db.add.called
    logged = db.add.call_args[0][0]
    assert logged.status == "blocked"


@pytest.mark.asyncio
async def test_dispatch_blocks_cgnat_tailnet_target() -> None:
    """A CGNAT/tailnet address (100.64.0.0/10 — the estate's Tailscale range, e.g.
    the postgres primary) is internal and must be refused like any RFC1918 host."""
    db = _fake_db()
    dispatcher = WebhookDispatcher()

    with patch("src.services.webhook_dispatcher.httpx.AsyncClient") as mock_client:
        delivered = await dispatcher.dispatch(
            db,
            job_id="job-cgnat",
            webhook_url="http://100.64.0.5/hook",
            event_type="job.completed",
            payload={"ok": True},
        )

    assert delivered is False
    mock_client.assert_not_called()
    logged = db.add.call_args[0][0]
    assert logged.status == "blocked"


# ---------------------------------------------------------------------------
# The fix: the outbound connection is pinned to the validated IP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_webhook_request_is_ip_pinned_not_rehresolved() -> None:
    """THE regression test for the DNS-rebinding TOCTOU this change closes.

    The old code validated the URL (resolving the host once) and then handed the
    *hostname* to ``client.stream()``, which resolved it a **second** time — a
    rebinding attacker could answer the first lookup with a public IP and the
    second with an internal one. ``safe_post`` instead rewrites the request URL's
    netloc to the literal validated IP while preserving the original hostname in
    the ``Host`` header and the TLS SNI, so there is no second resolution to
    poison.
    """
    db = _fake_db()
    dispatcher = WebhookDispatcher(max_retries=1)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"ok")

    with _patch_client(handler), patch("socket.getaddrinfo", _public_getaddrinfo):
        delivered = await dispatcher.dispatch(
            db,
            job_id="job-pin",
            webhook_url=_WEBHOOK_URL,
            event_type="job.completed",
            payload={"ok": True},
        )

    assert delivered is True
    assert len(seen) == 1
    request = seen[0]

    # The connection target is the literal validated IP — NOT the hostname.
    assert request.url.host == _PUBLIC_IP
    assert str(request.url) == f"https://{_PUBLIC_IP}/cb"
    # ...while the origin server still sees the real vhost, and TLS still
    # verifies against the hostname (not the IP).
    assert request.headers["Host"] == _WEBHOOK_HOST
    assert request.extensions.get("sni_hostname") == _WEBHOOK_HOST


# ---------------------------------------------------------------------------
# Happy path + signing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_allows_public_webhook() -> None:
    """A public webhook target passes the guard, is POSTed, and is logged delivered."""
    db = _fake_db()
    dispatcher = WebhookDispatcher(max_retries=1)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"ok")

    with _patch_client(handler), patch("socket.getaddrinfo", _public_getaddrinfo):
        delivered = await dispatcher.dispatch(
            db,
            job_id="job-2",
            webhook_url=_WEBHOOK_URL,
            event_type="job.completed",
            payload={"ok": True},
        )

    assert delivered is True
    assert len(seen) == 1
    assert seen[0].method == "POST"
    logged = db.add.call_args[0][0]
    assert logged.status == "delivered"
    assert logged.response_text == "ok"
    assert logged.response_status == 200


@pytest.mark.asyncio
async def test_dispatch_signs_body_with_hmac_header() -> None:
    """When a secret is configured the POST carries an X-Webhook-Signature over the
    exact bytes sent — the pin must not disturb the signed body or the headers."""
    db = _fake_db()
    dispatcher = WebhookDispatcher(max_retries=1)
    payload = {"ok": True, "job": "job-hmac"}
    secret = "s3cret"  # pragma: allowlist secret
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"ok")

    with _patch_client(handler), patch("socket.getaddrinfo", _public_getaddrinfo):
        delivered = await dispatcher.dispatch(
            db,
            job_id="job-hmac",
            webhook_url=_WEBHOOK_URL,
            webhook_secret=secret,
            event_type="job.completed",
            payload=payload,
        )

    assert delivered is True
    request = seen[0]
    body = json.dumps(payload, default=str).encode("utf-8")
    assert request.content == body
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    assert request.headers["X-Webhook-Signature"] == f"sha256={expected}"
    assert request.headers["X-Webhook-Event"] == "job.completed"
    assert request.headers["X-Webhook-Delivery"]


# ---------------------------------------------------------------------------
# A signed body is never replayed to a redirect target
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_does_not_follow_redirect_from_webhook_target() -> None:
    """The POST body carries an HMAC signature, so it must never be replayed to a
    location the webhook target chooses. ``max_redirects=0`` means the 302 is not
    followed: exactly one request leaves the process.

    The resulting SSRFError must be handled INSIDE dispatch — it is a ValueError,
    so it is not caught by the transport-error tuple, and letting it propagate
    would crash the caller instead of failing this one delivery. A redirecting
    target is terminal (retrying cannot help), so it logs ``blocked`` and returns
    False without burning the remaining attempts.
    """
    db = _fake_db()
    dispatcher = WebhookDispatcher(max_retries=1)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(302, headers={"location": "https://evil.example.com/steal"})

    with _patch_client(handler), patch("socket.getaddrinfo", _public_getaddrinfo):
        delivered = await dispatcher.dispatch(
            db,
            job_id="job-redirect",
            webhook_url=_WEBHOOK_URL,
            webhook_secret="s3cret",  # pragma: allowlist secret
            event_type="job.completed",
            payload={"ok": True},
        )

    assert delivered is False
    assert db.add.call_args[0][0].status == "blocked"

    # The signed body was sent exactly once, and never to the redirect target.
    assert len(seen) == 1
    assert seen[0].url.host == _PUBLIC_IP
    assert seen[0].headers["Host"] == _WEBHOOK_HOST


# ---------------------------------------------------------------------------
# Transport errors are caught, retried, and logged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_handles_httpx_connect_error() -> None:
    """A webhook to a resolvable-but-down host raises httpx.ConnectError inside the
    delivery loop; it must be caught (not propagate), retried, logged as 'error',
    and dispatch returns False after retries."""
    db = _fake_db()
    # backoff_base=0.0 so the retry sleep is instant.
    dispatcher = WebhookDispatcher(max_retries=2, backoff_base=0.0)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    with _patch_client(handler), patch("socket.getaddrinfo", _public_getaddrinfo):
        delivered = await dispatcher.dispatch(
            db,
            job_id="job-3",
            webhook_url=_WEBHOOK_URL,
            event_type="job.completed",
            payload={"ok": True},
        )

    assert delivered is False  # did not propagate
    assert len(attempts) == 2  # retried
    logged = db.add.call_args[0][0]
    assert logged.status == "error"
    assert "connection refused" in (logged.response_text or "")


# ---------------------------------------------------------------------------
# The attacker-controlled response body is bounded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_rejects_oversized_content_length_without_reading_body() -> None:
    """An oversized declared Content-Length is refused BEFORE the body is read —
    the (attacker-controlled) stream is never consumed. The resulting
    ResponseTooLargeError is caught by the delivery loop and logged as 'error'."""
    db = _fake_db()
    dispatcher = WebhookDispatcher(max_retries=1)

    async def _exploding_body() -> AsyncIterator[bytes]:
        raise AssertionError("body was read despite an oversized Content-Length")
        yield b""  # pragma: no cover - unreachable, makes this an async generator

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": str(200 * 1024)},  # 200 KB > the 64 KB cap
            content=_exploding_body(),
        )

    with _patch_client(handler), patch("socket.getaddrinfo", _public_getaddrinfo):
        delivered = await dispatcher.dispatch(
            db,
            job_id="job-4",
            webhook_url=_WEBHOOK_URL,
            event_type="job.completed",
            payload={"ok": True},
        )

    assert delivered is False
    logged = db.add.call_args[0][0]
    assert logged.status == "error"
    assert "Content-Length exceeds" in (logged.response_text or "")


@pytest.mark.asyncio
async def test_dispatch_aborts_streamed_body_once_it_crosses_the_cap() -> None:
    """A body that lies about (or omits) its Content-Length is aborted the moment the
    running total crosses the 64 KB cap — it is never fully materialised."""
    db = _fake_db()
    dispatcher = WebhookDispatcher(max_retries=1)
    chunk = b"A" * 32_768
    total_chunks = 64  # 2 MB if fully read
    yielded: list[int] = []

    async def _huge_body() -> AsyncIterator[bytes]:
        for _ in range(total_chunks):
            yielded.append(len(chunk))
            yield chunk

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_huge_body())

    with _patch_client(handler), patch("socket.getaddrinfo", _public_getaddrinfo):
        delivered = await dispatcher.dispatch(
            db,
            job_id="job-5",
            webhook_url=_WEBHOOK_URL,
            event_type="job.completed",
            payload={"ok": True},
        )

    assert delivered is False
    # Bounded at roughly the cap plus one chunk — nowhere near the full 2 MB.
    assert sum(yielded) <= 64 * 1024 + len(chunk)
    assert len(yielded) < total_chunks
    logged = db.add.call_args[0][0]
    assert logged.status == "error"
    assert "exceeds" in (logged.response_text or "")


@pytest.mark.asyncio
async def test_dispatch_truncates_logged_response_text() -> None:
    """A body that fits under the byte cap is still truncated to a short snippet in
    the delivery log — the log is not a dumping ground for attacker content."""
    db = _fake_db()
    dispatcher = WebhookDispatcher(max_retries=1)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"B" * 4096)

    with _patch_client(handler), patch("socket.getaddrinfo", _public_getaddrinfo):
        delivered = await dispatcher.dispatch(
            db,
            job_id="job-6",
            webhook_url=_WEBHOOK_URL,
            event_type="job.completed",
            payload={"ok": True},
        )

    assert delivered is True
    logged = db.add.call_args[0][0]
    assert logged.status == "delivered"
    assert len(logged.response_text or "") == 1000
