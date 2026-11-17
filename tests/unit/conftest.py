"""Hermetic DNS for the unit suite, made necessary by the render-target SSRF guard.

WHY THIS FILE EXISTS
--------------------
``RedesignRequest.url``, ``BatchPageInput.url``, ``EmbedSpec.artifact_url``,
``_fetch_page_html`` and ``capture_url_screenshots`` now all run
``w2v_common.net.validate_public_url``. That function **resolves the hostname**
before deciding, so from this commit onwards a large slice of the unit suite
performs a real DNS lookup for whatever placeholder host the test happened to
use (``https://example.com``, ``https://x.example/s.html``,
``https://cdn.example/scene.html``, ``http://x``, ``https://acme.test`` ...).

That is unacceptable for three separate reasons, only the first of which is
about speed:

1. It is not hermetic. The suite must not touch the network or a resolver.
2. It is **non-deterministic**. ``validate_public_url`` treats
   ``socket.gaierror`` as "allow, let the transport surface it"
   (``safe_fetch.py:164-168``), so a reserved-TLD placeholder like
   ``cdn.example`` passes on a machine whose resolver NXDOMAINs it — and
   *fails* on a machine behind a captive portal, a wildcard/ISP-hijacked
   resolver, or macOS mDNS, any of which can answer a bare label like ``x`` or
   ``test`` with an RFC1918 address. Same code, opposite result, depending on
   the developer's Wi-Fi.
3. It would make the *blocked* assertions in
   ``test_render_target_ssrf.py`` meaningless: ``registry.internal`` does not
   resolve off-tailnet, so on a CI runner it would NXDOMAIN, fall into that same
   allow-on-gaierror path, and the "this is blocked" test would go green for
   entirely the wrong reason.

So DNS is stubbed rather than avoided, and the stub is an explicit, readable
model of the estate's name space.

THE PATCH TARGET, AND WHY IT IS THIS ONE
----------------------------------------
``w2v_common/net/safe_fetch.py`` does a module-level ``import socket``
(line 32) and then calls::

    addrinfo = socket.getaddrinfo(hostname, None)   # line 163

The name is looked up through the ``socket`` *module object* at call time, not
bound into the function at import time. The attribute that must change is
therefore ``w2v_common.net.safe_fetch.socket.getaddrinfo``, which is what this
module patches. Patching a per-service alias (e.g.
``src.api.schemas.socket.getaddrinfo``) is dead — this service never imports
``socket`` and never calls the resolver itself; the only resolver call in the
whole guarded path is the one above, inside the installed ``w2v_common``.

Note that ``w2v_common.net.safe_fetch.socket`` *is* the stdlib ``socket``
module, so this patch is global for the duration of each test. That is
deliberate: it means nothing else in the unit suite can leak a lookup either.
The stub honours the requested port and passes IP literals through untouched,
so the two DB-backed unit tests (``test_job_queue``,
``test_result_response_from_db``) still reach ``127.0.0.1:5474`` exactly as they
did before.

THE MAP, AND WHY THE DEFAULT IS *PUBLIC*
----------------------------------------
Unknown hostnames resolve to a public address. That is the safe default here
because it preserves the pre-existing meaning of every legacy test: those tests
were written about escaping, embed injection, viewport plumbing and scope
rules — the URL was always an inert placeholder, never the subject. Failing
them closed would be changing what they assert.

The hosts that must stay **blocked** are therefore mapped *explicitly*, into
the ranges they really occupy on this fleet, so a blocked-assertion is proved by
a resolved internal address rather than by a resolution failure:

* ``registry.internal``  -> 100.64.0.6  (worker3, real tailnet address)
* ``postgres.internal``  -> 100.64.0.5 (the prod primary, real)
* ``minio.internal``     -> 10.43.0.21    (in-cluster ClusterIP range)
* ``localhost``          -> 127.0.0.1
* :data:`INTERNAL_REBIND_HOST` -> 100.64.0.5

RFC 6598 ``100.64.0.0/10`` is this fleet's Tailscale tailnet and
``ipaddress.ip_address("100.64.0.5").is_private`` is **False**, which is the
entire reason the guard has to be ``validate_public_url`` and not a hand-rolled
private-range check.

:data:`INTERNAL_REBIND_HOST` is the one entry that exists purely for the new
test: an innocuous-looking public-shaped name whose DNS answer is a tailnet
address. It is the case a literal-IP denylist cannot catch, and it only has
teeth because this stub gives it a deterministic answer.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Final

import pytest
import w2v_common.net.safe_fetch as _safe_fetch

#: Address every *unmapped* hostname resolves to. Public, so legacy placeholder
#: URLs keep passing the guard and keep asserting what they were written to
#: assert.
PUBLIC_STUB_IP: Final[str] = "93.184.216.34"

#: The prod Postgres primary. RFC 6598 CGNAT — ``is_private`` is False for it.
TAILNET_POSTGRES_IP: Final[str] = "100.64.0.5"

#: worker3, which hosts ``registry.internal:5000``.
TAILNET_REGISTRY_IP: Final[str] = "100.64.0.6"

#: A public-*looking* hostname whose stubbed DNS answer is inside the tailnet.
#: This is the bypass that a literal-IP check misses, and the reason the guard
#: has to resolve rather than pattern-match.
INTERNAL_REBIND_HOST: Final[str] = "cdn-lookalike.example"

#: Hosts with a deliberate, non-default answer. Everything here resolves into a
#: range ``validate_public_url`` blocks, so tests asserting "this is refused"
#: are proved by a resolved internal address, not by a lookup failure.
STUB_DNS: Final[dict[str, str]] = {
    "localhost": "127.0.0.1",
    "registry.internal": TAILNET_REGISTRY_IP,
    "postgres.internal": TAILNET_POSTGRES_IP,
    "minio.internal": "10.43.0.21",
    INTERNAL_REBIND_HOST: TAILNET_POSTGRES_IP,
}

# (family, socktype, proto, canonname, sockaddr) — the stdlib getaddrinfo tuple.
_SockAddr = tuple[str, int] | tuple[str, int, int, int]
_AddrInfo = tuple[int, int, int, str, _SockAddr]


def _answer_for(host: str) -> str:
    """Resolve ``host`` against the stub's view of the estate.

    Args:
        host: Hostname or IP literal, exactly as ``urlparse().hostname`` yields
            it (already lower-cased, IPv6 brackets already stripped).

    Returns:
        The IP address string this host resolves to. IP literals are returned
        unchanged so a test can assert on a specific address without the stub
        rewriting it; mapped hosts get their estate address; everything else
        gets :data:`PUBLIC_STUB_IP`.
    """
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        # Already an address — never remap, or the literal-IP cases in
        # test_render_target_ssrf would be testing the stub, not the guard.
        return host
    return STUB_DNS.get(host.lower(), PUBLIC_STUB_IP)


def _stub_getaddrinfo(
    host: str | bytes | None,
    port: str | int | None = None,
    *_args: object,
    **_kwargs: object,
) -> list[_AddrInfo]:
    """Drop-in replacement for ``socket.getaddrinfo`` backed by :data:`STUB_DNS`.

    Accepts and ignores the trailing ``family``/``type``/``proto``/``flags``
    arguments so it is call-compatible with every caller in the stack
    (``validate_public_url`` passes ``(host, None)``; psycopg and asyncio pass
    the full positional set).

    Args:
        host: Hostname, IP literal, bytes, or ``None`` (wildcard).
        port: Port number; echoed into the returned ``sockaddr`` so real
            connections (the DB-backed unit tests) still target the right port.
        *_args: Ignored ``family``/``type``/``proto``/``flags`` positionals.
        **_kwargs: Ignored keyword forms of the same.

    Returns:
        A single-entry addrinfo list, IPv4 or IPv6 depending on the answer.
    """
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    resolved = _answer_for(host or "localhost")
    port_num = port if isinstance(port, int) else 0

    if isinstance(ipaddress.ip_address(resolved), ipaddress.IPv6Address):
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (resolved, port_num, 0, 0))]
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (resolved, port_num))]


@pytest.fixture(autouse=True)
def _hermetic_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route every ``socket.getaddrinfo`` in the unit suite through :data:`STUB_DNS`.

    Patches ``w2v_common.net.safe_fetch.socket.getaddrinfo`` — see this module's
    docstring for why that is the correct and only target.

    Args:
        monkeypatch: pytest's patcher; undoes the stub after each test.
    """
    monkeypatch.setattr(_safe_fetch.socket, "getaddrinfo", _stub_getaddrinfo)


@pytest.fixture(autouse=True)
def _no_rendered_page_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the rendered-page contrast/logo probe for generator tests that do not target it.

    ``HtmlGenerator.repair_rendered_page`` renders every composed page in Chromium, and
    the recorded fixtures reference real public image URLs — left live, every
    ``generate_page_chunked`` test would launch a browser and fetch from the internet.
    The probe itself is exercised against REAL Chromium in
    ``test_redesign_rendered_checks.py``, which passes ``render_probe=`` explicitly (an
    explicit probe is not affected by this stub).
    """
    import src.services.html_gen.generator as generator
    from src.services.html_gen.rendered_checks import RenderedReport

    async def _clean(_html: str) -> RenderedReport:
        return RenderedReport()

    monkeypatch.setattr(generator, "probe_rendered_page", _clean)
