"""Resolver-level SSRF pin for the render target — the browser must be forced to
the validated IP, not left free to re-resolve.

WHY
---
``capture_url_screenshots`` drives a full browser engine against a
tenant-supplied URL. ``validate_public_url`` resolves the hostname once and
returns the vetted public IP, but **Chromium resolves the hostname itself** when
it navigates, so validating-then-``goto``ing the same string leaves a DNS-rebind
TOCTOU window: validation sees a public address, the browser re-resolves
milliseconds later and can land on an internal one (tailnet ``100.64/10``, a
ClusterIP, cloud metadata). The httpx-layer IP-pin used everywhere else in the
fleet cannot reach a browser — it pins a transport we do not own.

The only correct pin for a browser is at its resolver:
``--host-resolver-rules=MAP <host> <ip>``.

WHAT THESE TESTS PROVE
----------------------
Most of these drive **real Chromium** against **real local HTTP/HTTPS servers**,
because the failure modes this change can cause are all invisible to a mock:

* ``test_rebound_target_cannot_reach_the_private_address`` — the load-bearing
  RED. Fails on unmodified ``origin/main``.
* ``test_pinned_render_loads_cross_origin_subresources_and_is_not_blank`` — the
  anti-regression that matters most. A pin that blanks the page by blocking CDN
  assets is worse than the hole it closes, and only a **pixel** assertion can
  catch that.
* ``test_pin_preserves_the_hostname_for_tls_sni`` — reads the real SNI off a
  real TLS handshake, proving the mapped connection still presents the
  *hostname* (so certificate validation is unaffected).
* the redirect pair — a cross-host redirect escapes a single-host MAP rule, so
  it is re-validated; a same-host redirect must still render.

THE DNS STUB
------------
There is deliberately **no** ``socket.getaddrinfo`` monkeypatch. The "two
different answers" of a rebind are modelled with the two resolvers that really
exist: ``validate_public_url`` is stubbed to return the *public* first answer
(``8.8.8.8``), while Chromium's own resolver independently supplies the
*private* second answer — natively, with no patching at all, because Chromium
resolves ``localhost`` to ``127.0.0.1`` itself. Literal IPs are untouched and
resolve to themselves throughout.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import http.server
import io
import socketserver
import ssl
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from w2v_common.net import SSRFError, validate_public_url

from src.services.browser_pool import BrowserPool, _Slot, host_resolver_rule
from src.services.screenshot_renderer import (
    _assert_landed_host_is_public,
    capture_url_screenshots,
)


def _chromium_available() -> bool:
    """Whether a Playwright Chromium binary is installed (CI sets ``needs_chromium``)."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        return bool(path) and Path(path).exists()
    except Exception:  # noqa: BLE001 -- availability probe: any failure => unavailable
        return False


needs_chromium = pytest.mark.skipif(
    not _chromium_available(), reason="Playwright chromium not installed"
)

# A 1x1 red PNG — the smallest thing that proves an <img> sub-resource was
# actually fetched and decoded rather than merely requested. The hex trips
# detect-secrets' "Hex High Entropy String" plugin; it is image bytes, not a
# credential, hence the pragmas.
_DOT_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"  # pragma: allowlist secret
    "01f15c4890000000d4944415478da63f8cfc0f00f0004010180a2d5c8"  # pragma: allowlist secret
    "3d0000000049454e44ae426082"  # pragma: allowlist secret
)
_CSS = b"body{background:#2563EB;margin:0}h1{color:#fff;font:48px sans-serif;margin:0}"
_BG_RGB = (0x25, 0x63, 0xEB)


class _Recorder:
    """Records every request a test server saw, so 'never reached' is assertable."""

    def __init__(self) -> None:
        self.hits: list[tuple[str, str]] = []

    def content(self) -> list[tuple[str, str]]:
        """Hits excluding Chromium's automatic ``/favicon.ico`` probe.

        The favicon fetch is browser chrome, not page content, and it arrives
        non-deterministically. It is filtered for the *positive* assertions only
        — the security assertion in the RED test still requires the recorder to
        be **completely** empty.
        """
        return [(host, path) for host, path in self.hits if path != "/favicon.ico"]


def _handler(rec: _Recorder, body_for: Any) -> type[http.server.BaseHTTPRequestHandler]:
    """Build a request handler that records hits and delegates bodies to ``body_for``.

    Args:
        rec: Collector for ``(Host header, path)`` of every request served.
        body_for: ``path -> (status, content_type, body, extra_headers)``.

    Returns:
        A handler class for ``socketserver.TCPServer``.
    """

    class _H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 -- base class name
            """Silence the stderr access log (pytest captures it as noise)."""

        def do_GET(self) -> None:
            """Record the hit and serve whatever ``body_for`` decides."""
            rec.hits.append((self.headers.get("Host", ""), self.path))
            status, ctype, body, extra = body_for(self.path)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

    return _H


@contextlib.contextmanager
def _server(rec: _Recorder, body_for: Any, tls_ctx: ssl.SSLContext | None = None) -> Iterator[int]:
    """Run a threaded HTTP(S) server on an ephemeral 127.0.0.1 port; yield the port."""

    # Threaded, not the plain TCPServer: with HTTP/1.1 keep-alive a single-threaded
    # server holds the first connection open forever, and Chromium — which opens
    # several in parallel and fetches /favicon.ico — then never reaches
    # ``networkidle``, so ``goto`` times out. That reads exactly like "the pin
    # broke rendering". It is a test-harness bug; fix it in the harness.
    class _Threaded(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = _Threaded(("127.0.0.1", 0), _handler(rec, body_for))
    if tls_ctx is not None:
        srv.socket = tls_ctx.wrap_socket(srv.socket, server_side=True)
    port = int(srv.server_address[1])
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _plain(body: bytes, ctype: str = "text/html") -> Any:
    """A ``body_for`` that serves one fixed body for every path."""

    def _body_for(_path: str) -> tuple[int, str, bytes, dict[str, str]]:
        return (200, ctype, body, {})

    return _body_for


async def _real_pool(monkeypatch: pytest.MonkeyPatch) -> BrowserPool:
    """Patch the renderer's pool getter to a fresh real ``BrowserPool`` and return it."""
    pool = BrowserPool(size=1, max_renders=1)
    monkeypatch.setattr("src.services.browser_pool.get_browser_pool", AsyncMock(return_value=pool))
    return pool


def _pin_validation_to(monkeypatch: pytest.MonkeyPatch, ip: str) -> MagicMock:
    """Stub ``validate_public_url`` at the renderer to return ``ip`` (the FIRST DNS answer)."""
    stub = MagicMock(return_value=ip)
    monkeypatch.setattr("src.services.screenshot_renderer.validate_public_url", stub)
    return stub


# ---------------------------------------------------------------------------
# The RED: a rebound host must not be reachable
# ---------------------------------------------------------------------------


@needs_chromium
async def test_rebound_target_cannot_reach_the_private_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host that validates public but resolves private in the browser must not be fetched.

    This is the defect, reproduced end-to-end with two genuinely independent
    resolvers — exactly the production shape:

    * **Validation-time answer (public):** ``validate_public_url`` is stubbed to
      return ``8.8.8.8``, i.e. the attacker's first, benign DNS response. (The
      real validator has its own suite; stubbing it here is what makes the
      *rebind* deterministic instead of timing-dependent.)
    * **Browser-time answer (private):** the URL's host is ``localhost``, which
      Chromium resolves to ``127.0.0.1`` **on its own**, with no patching. A
      local HTTP server sits there recording every request — it stands in for
      the metadata endpoint / ClusterIP an attacker is aiming at.

    On unmodified ``origin/main`` the pooled browser re-resolves and hits that
    server, so ``rec.hits`` is non-empty and this assertion FAILS. With the
    resolver pin, Chromium is launched with ``MAP localhost 8.8.8.8`` and can
    never reach 127.0.0.1, so the recorder stays empty.

    Args:
        tmp_path: Screenshot output directory parent.
        monkeypatch: Stubs the validator's answer and the pool getter.
    """
    rec = _Recorder()
    with _server(rec, _plain(b"<html><body>INTERNAL SECRETS</body></html>")) as port:
        _pin_validation_to(monkeypatch, "8.8.8.8")
        pool = await _real_pool(monkeypatch)
        try:
            # Navigation is EXPECTED to fail once pinned (8.8.8.8 does not serve
            # this port). The assertion below — not the exception — is the proof.
            with contextlib.suppress(Exception):
                await capture_url_screenshots(
                    f"http://localhost:{port}/",
                    tmp_path / "shots",
                    ["desktop"],
                    timeout_ms=4000,
                    full_page=False,
                )
        finally:
            await pool.stop()

    assert rec.hits == [], (
        "DNS-rebind reached the private address: the browser re-resolved the host "
        f"instead of being pinned to the validated IP. Requests seen: {rec.hits}"
    )


# ---------------------------------------------------------------------------
# The anti-regression: pinning must not blank the page
# ---------------------------------------------------------------------------


@needs_chromium
async def test_pinned_render_loads_cross_origin_subresources_and_is_not_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned render still loads CDN sub-resources and produces real pixels.

    The rule is scoped to the ONE host being rendered (``MAP <host> <ip>``, no
    ``MAP *``, no ``~NOTFOUND``), so every other origin resolves normally. This
    test serves the document from a *pinned hostname* and its stylesheet and
    image from a *different origin*, then asserts on the decoded PNG:

    * correct dimensions (the viewport), so the capture is not a stub;
    * the stylesheet's background colour is the dominant pixel — only true if
      the **cross-origin CSS actually loaded and applied**;
    * more than one distinct colour, i.e. the page is not a blank fill.

    A pixel assertion is the point. This fleet has repeatedly shipped green
    tests over blank output; "the screenshot file exists" proves nothing.

    Args:
        tmp_path: Screenshot output directory parent.
        monkeypatch: Stubs the validator's answer and the pool getter.
    """
    from PIL import Image

    cdn_rec = _Recorder()

    def cdn_bodies(path: str) -> tuple[int, str, bytes, dict[str, str]]:
        if path.endswith(".css"):
            return (200, "text/css", _CSS, {})
        return (200, "image/png", _DOT_PNG, {})

    with _server(cdn_rec, cdn_bodies) as cdn_port:
        page_html = (
            f'<html><head><link rel="stylesheet" href="http://127.0.0.1:{cdn_port}/s.css">'
            f"</head><body><h1>Pinned</h1>"
            f'<img src="http://127.0.0.1:{cdn_port}/d.png" width="80" height="80">'
            f"</body></html>"
        ).encode()
        page_rec = _Recorder()
        with _server(page_rec, _plain(page_html)) as page_port:
            _pin_validation_to(monkeypatch, "127.0.0.1")
            pool = await _real_pool(monkeypatch)
            try:
                out = await capture_url_screenshots(
                    f"http://pinned.test:{page_port}/",
                    tmp_path / "shots",
                    ["desktop"],
                    timeout_ms=15000,
                    full_page=False,
                )
            finally:
                await pool.stop()

    # The pinned host was reached, with its hostname intact in the Host header —
    # MAP rewrites the address, never the name.
    assert page_rec.content() == [(f"pinned.test:{page_port}", "/")]
    # Both cross-origin sub-resources loaded despite the pin on the page host.
    assert sorted(path for _host, path in cdn_rec.content()) == ["/d.png", "/s.css"]

    shot = Path(out["desktop"])
    with Image.open(io.BytesIO(shot.read_bytes())) as img:
        rgb = img.convert("RGB")
        # 1920x1080 CSS delivery frame supersampled at device_scale_factor=2.
        # This is the live proof of BOTH halves of the sharpness fix: a wrong
        # viewport shows up as a wrong CSS size, a missing/ignored scale factor
        # as a missing 2x. Anything else is an upscale downstream.
        assert rgb.size == (3840, 2160), f"wrong capture dimensions: {rgb.size}"
        colors = rgb.getcolors(maxcolors=1_000_000) or []
        assert len(colors) > 1, "render is a flat fill — the page is blank"
        dominant = max(colors)[1]
        assert dominant == _BG_RGB, (
            f"dominant pixel {dominant} is not the stylesheet background {_BG_RGB} — "
            "the cross-origin CSS did not apply, i.e. the pin blocked sub-resources"
        )


# ---------------------------------------------------------------------------
# HTTPS: the hostname survives for SNI / certificate validation
# ---------------------------------------------------------------------------


def _self_signed(hostname: str, tmp_path: Path) -> ssl.SSLContext:
    """A TLS server context with a throwaway self-signed cert for ``hostname``."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)
    return ctx


@needs_chromium
async def test_pin_preserves_the_hostname_for_tls_sni(tmp_path: Path) -> None:
    """Over HTTPS, the pinned connection still presents the hostname in the TLS SNI.

    This is *why* ``--host-resolver-rules`` is the right lever rather than
    rewriting the URL to the IP: it changes only the address the socket connects
    to, leaving the hostname in place for SNI and therefore for certificate
    validation. Rewriting the URL to ``https://<ip>/`` would instead break every
    HTTPS target's cert check (and its virtual hosting).

    Proven directly: a TLS server on ``127.0.0.1`` records the SNI value out of
    the real handshake via ``sni_callback``, and the pinned browser is asked for
    ``https://sni.test:<port>/``. ``ignore_https_errors`` is set only because the
    cert is self-signed — it does not affect which name Chromium *sends*.

    Args:
        tmp_path: Cert/key scratch directory.
    """
    seen: list[str | None] = []
    ctx = _self_signed("sni.test", tmp_path)

    def _record_sni(_sock: Any, name: str | None, _ctx: ssl.SSLContext) -> None:
        seen.append(name)

    ctx.sni_callback = _record_sni  # type: ignore[assignment]

    rec = _Recorder()
    with _server(rec, _plain(b"<html><body>tls</body></html>"), ctx) as port:
        pool = BrowserPool(size=1, max_renders=1)
        try:
            async with pool.lease_pinned("sni.test", "127.0.0.1") as browser:
                page = await browser.new_page(ignore_https_errors=True)
                try:
                    await page.goto(f"https://sni.test:{port}/", timeout=15000)
                finally:
                    await page.close()
        finally:
            await pool.stop()

    assert seen, "no TLS handshake was observed"
    # Chromium may open more than one connection (favicon, preconnect); every one
    # of them must carry the ORIGINAL hostname, never the mapped address.
    assert set(seen) == {"sni.test"}, f"SNI was not the original hostname: {seen}"
    assert rec.content() == [(f"sni.test:{port}", "/")]


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------


@needs_chromium
async def test_cross_host_redirect_to_internal_is_refused_before_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 30x onto an internal host is caught by re-validation, and nothing is captured.

    A single-host ``MAP`` rule names one host; Chromium resolves a redirect to a
    *different* host normally, so the pin alone does not cover it (verified
    against real Chromium: a pinned host redirecting to ``127.0.0.1`` was
    followed). ``_assert_landed_host_is_public`` re-validates ``page.url`` after
    navigation, before any capture — so an internal page's content can never
    reach a tenant-visible artifact.

    The validator stub answers the *pin* call with the vetted IP and then hands
    every later call to the REAL ``validate_public_url``, so the refusal comes
    from production logic rather than a hard-coded test expectation.

    Args:
        tmp_path: Screenshot output directory parent.
        monkeypatch: Stubs the validator's first answer and the pool getter.
    """
    target_rec = _Recorder()
    with _server(target_rec, _plain(b"<html><body>METADATA</body></html>")) as target_port:
        src_rec = _Recorder()

        def redirect(_path: str) -> tuple[int, str, bytes, dict[str, str]]:
            return (302, "text/html", b"", {"Location": f"http://127.0.0.1:{target_port}/creds"})

        with _server(src_rec, redirect) as src_port:
            answers = iter(["127.0.0.1"])

            def _validate(url: str, **kwargs: Any) -> str | None:
                return next(answers, None) or validate_public_url(url, **kwargs)

            monkeypatch.setattr("src.services.screenshot_renderer.validate_public_url", _validate)
            pool = await _real_pool(monkeypatch)
            out_dir = tmp_path / "shots"
            try:
                with pytest.raises(SSRFError):
                    await capture_url_screenshots(
                        f"http://redir.test:{src_port}/",
                        out_dir,
                        ["desktop"],
                        timeout_ms=15000,
                        full_page=False,
                    )
            finally:
                await pool.stop()

    assert list(out_dir.glob("*.png")) == [], "an internal page was screenshotted"


@needs_chromium
async def test_same_host_redirect_still_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-host redirect must NOT be blocked — over-blocking would blank real pages.

    Redirects within a host (``/`` → ``/home``, ``http`` → ``https``, trailing
    slash normalisation) are ubiquitous on real sites. A naive "the landed URL
    must equal the requested URL" rule would refuse them and turn this hardening
    into an outage, so the check compares *hosts* and short-circuits when they
    match.

    Args:
        tmp_path: Screenshot output directory parent.
        monkeypatch: Stubs the validator's answer and the pool getter.
    """
    rec = _Recorder()

    def bodies(path: str) -> tuple[int, str, bytes, dict[str, str]]:
        if path == "/":
            return (302, "text/html", b"", {"Location": "/home"})
        return (200, "text/html", b"<html><body><h1>HOME</h1></body></html>", {})

    with _server(rec, bodies) as port:
        _pin_validation_to(monkeypatch, "127.0.0.1")
        pool = await _real_pool(monkeypatch)
        try:
            out = await capture_url_screenshots(
                f"http://same.test:{port}/",
                tmp_path / "shots",
                ["desktop"],
                timeout_ms=15000,
                full_page=False,
            )
        finally:
            await pool.stop()

    assert [path for _h, path in rec.content()] == ["/", "/home"]
    assert Path(out["desktop"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cross_host_redirect_to_a_public_host_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """``example.com`` → ``www.example.com`` passes: re-validated, not refused.

    The apex-to-www redirect is the single most common cross-host hop on the
    web. It must survive, so the check re-validates rather than refusing on any
    host change. Note the documented consequence: the redirect target is
    validated but NOT pinned — it keeps the narrow rebind window that this
    change closes for the initial host.

    Args:
        monkeypatch: Stubs the validator so the assertion is about the branch.
    """
    stub = _pin_validation_to(monkeypatch, "93.184.216.34")
    _assert_landed_host_is_public("http://example.com/", "https://www.example.com/", "example.com")
    stub.assert_called_once_with("https://www.example.com/")


def test_same_host_landing_skips_revalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The common case does no extra DNS work: same host → no second validation."""
    stub = _pin_validation_to(monkeypatch, "93.184.216.34")
    _assert_landed_host_is_public("http://example.com/", "http://example.com/x", "example.com")
    stub.assert_not_called()


# ---------------------------------------------------------------------------
# Rule construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "ip", "expected"),
    [
        ("example.com", "93.184.216.34", "MAP example.com 93.184.216.34"),
        ("example.com", "8.8.8.8", "MAP example.com 8.8.8.8"),
        ("example.com", "2606:2800:220:1::1", "MAP example.com [2606:2800:220:1::1]"),
        ("example.com", "::1", "MAP example.com [::1]"),
    ],
)
def test_host_resolver_rule_form(host: str, ip: str, expected: str) -> None:
    """The rule is single-host, port-less, and brackets IPv6.

    Bracketing is not cosmetic: Chromium **discards the entire rule** for a bare
    IPv6 literal, which would silently un-pin (and then fail to navigate) every
    AAAA-resolved target. Verified against real Chromium — ``MAP h ::1`` fails
    to navigate, ``MAP h [::1]`` loads.

    The absence of a trailing ``:port`` is equally load-bearing: it makes
    Chromium preserve the request's own port, so non-443/80 targets still work.

    Args:
        host: Hostname to pin.
        ip: Validated address.
        expected: Exact switch value.
    """
    assert host_resolver_rule(host, ip) == expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("bücher.example", "MAP xn--bcher-kva.example 8.8.8.8"),
        # Already-encoded A-labels and plain ASCII pass straight through.
        ("xn--bcher-kva.example", "MAP xn--bcher-kva.example 8.8.8.8"),
        ("example.com", "MAP example.com 8.8.8.8"),
        # A fully-qualified trailing root dot is left alone: Chromium matches it
        # as written (verified), so "normalising" it would be a behaviour change
        # with no benefit.
        ("example.com.", "MAP example.com. 8.8.8.8"),
    ],
)
def test_host_resolver_rule_punycodes_idn(host: str, expected: str) -> None:
    """An IDN host is encoded to its A-label, because a Unicode rule silently un-pins.

    This is the fail-*open* case, and the dangerous one. ``urlparse().hostname``
    returns the Unicode form while Chromium resolves the A-label, so a Unicode
    MAP rule does not match and the browser resolves the host freely — a guard
    that is present in the args and protects nothing. Verified against real
    Chromium in ``test_idn_pin_actually_binds_in_chromium``.

    Args:
        host: Hostname as ``urlparse`` would report it.
        ip: (via parametrize) the vetted address.
        expected: Exact switch value.
    """
    assert host_resolver_rule(host, "8.8.8.8") == expected


def test_host_resolver_rule_fails_closed_on_unencodable_host() -> None:
    """An unencodable non-ASCII host raises rather than rendering unpinned.

    Failing closed matters more than the render here: the alternative is an
    unpinned browser navigation, i.e. exactly the hole this change exists to
    close. The caller degrades a ``ValueError`` to "no before image" — a missing
    optional screenshot, never a failed paid job and never a blank artifact.
    """
    # A 100-character u-umlaut label: valid Unicode, but its A-label blows the
    # 63-byte DNS label limit, so the idna codec refuses it. Written as an escape
    # so the source stays pure ASCII and the intent is unambiguous.
    with pytest.raises(ValueError, match="Cannot IDNA-encode"):
        host_resolver_rule("\u00fc" * 100 + ".example", "8.8.8.8")


@needs_chromium
async def test_idn_pin_actually_binds_in_chromium(tmp_path: Path) -> None:
    """The punycode rule really binds Chromium's resolver for a Unicode URL.

    The unit test above asserts the *string*; this asserts the **effect**, which
    is the part that was silently broken. A browser is pinned with the A-label
    rule and asked for the Unicode URL: the request must land on the pinned
    address, carrying the A-label in its Host header.

    Args:
        tmp_path: Unused; keeps the signature uniform with its neighbours.
    """
    rec = _Recorder()
    with _server(rec, _plain(b"<html><body>idn</body></html>")) as port:
        pool = BrowserPool(size=1, max_renders=1)
        try:
            async with pool.lease_pinned("bücher.test", "127.0.0.1") as browser:
                page = await browser.new_page()
                try:
                    await page.goto(f"http://bücher.test:{port}/", timeout=15000)
                finally:
                    await page.close()
        finally:
            await pool.stop()

    assert rec.content() == [(f"xn--bcher-kva.test:{port}", "/")], (
        f"the IDN pin did not bind — Chromium resolved freely. Hits: {rec.hits}"
    )


def test_host_resolver_rule_is_scoped_to_one_host() -> None:
    """No wildcard and no ``~NOTFOUND``: other origins must keep resolving.

    A blanket ``MAP * ...`` or ``MAP * ~NOTFOUND`` would harden the target host
    at the cost of blocking every CDN the page depends on, producing a blank
    render — strictly worse than the vulnerability being closed. The pixel test
    above proves sub-resources load; this pins the intent in the rule itself.
    """
    rule = host_resolver_rule("example.com", "93.184.216.34")
    assert "*" not in rule
    assert "NOTFOUND" not in rule
    assert rule.count("MAP") == 1


# ---------------------------------------------------------------------------
# Wiring (fast, no browser): the renderer must actually pass the rule
# ---------------------------------------------------------------------------


def _mock_pool() -> tuple[BrowserPool, AsyncMock, MagicMock, MagicMock]:
    """A started ``BrowserPool`` whose Chromium launch is a capturing mock."""
    page = MagicMock()
    # An int status, because `require_ok_navigation` compares it against 400 —
    # a bare AsyncMock return value raises TypeError there. Real Playwright
    # Responses always carry an int; the fake should too.
    _ok_response = MagicMock()
    _ok_response.status = 200
    page.goto = AsyncMock(return_value=_ok_response)
    page.screenshot = AsyncMock()
    page.close = AsyncMock()
    page.url = "https://example.com/"

    browser = MagicMock()
    browser.new_page = AsyncMock(return_value=page)
    browser.close = AsyncMock()

    launch = AsyncMock(return_value=browser)
    pw = MagicMock()
    pw.chromium = MagicMock()
    pw.chromium.launch = launch

    pool = BrowserPool(size=1, max_renders=100)
    pool._pw = pw
    idle: asyncio.Queue[_Slot] = asyncio.Queue()
    # Seed one idle slot. The pinned path uses ``lease_pinned`` and never touches
    # this queue, but if the pin is reverted the renderer falls back to the pooled
    # ``lease``, which blocks on ``idle.get()``. With a slot present that fallback
    # LAUNCHES (with the base args only, no MAP rule) so the arg assertion below
    # fails with a clean AssertionError instead of hanging the test on an empty queue.
    idle.put_nowait(_Slot())
    pool._idle = idle  # non-None → start() is skipped
    return pool, launch, page, browser


async def test_capture_url_launches_chromium_with_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact ``--host-resolver-rules`` switch reaches ``chromium.launch``.

    A fast, browser-free companion to the RED above: it names the switch, so a
    refactor that keeps the other tests green by accidentally dropping the pin
    (e.g. reverting to the shared pool) still fails here, with an obvious
    message.

    Args:
        tmp_path: Screenshot output directory parent.
        monkeypatch: Stubs the pool getter and the validator.
    """
    pool, launch, page, browser = _mock_pool()
    monkeypatch.setattr("src.services.browser_pool.get_browser_pool", AsyncMock(return_value=pool))
    _pin_validation_to(monkeypatch, "93.184.216.34")

    await capture_url_screenshots(
        "https://example.com/", tmp_path / "shots", ["desktop"], full_page=False
    )

    launch.assert_awaited_once()
    assert launch.await_args is not None
    args = launch.await_args.kwargs.get("args", [])
    assert "--host-resolver-rules=MAP example.com 93.184.216.34" in args
    # The base hardening flags must survive alongside the pin.
    assert "--no-sandbox" in args
    page.goto.assert_awaited_once()
    # Ephemeral: a tenant-specific resolver rule must never re-enter the pool.
    browser.close.assert_awaited()
