"""Classify and describe the failure that ended a job.

Two jobs, both learned from one live incident: the nightly orchestrator E2E of
2026-08-24, where ``web-design-v1`` came back as::

    RuntimeError: design job 01a031e6-… failed: ReadTimeout: (no message)

**Classification.** An ``httpx`` read timeout on the ``worker_management`` LLM
call is the upstream not answering *in time* — it says nothing about whether the
request was well-formed, so it is transient. It was nonetheless persisted as a
terminal ``status='failed'``, a verdict the job could never come back from, and
the orchestrator (whose ``retry_on=()`` for a server-marked failure is correct)
propagated it immediately. Only *timeouts* are widened here: a 4xx, a schema
violation or a verification failure is a real permanent defect and must stay
terminal, because retrying it burns a paid 16k-token render to reach the same
answer while hiding the bug.

Note the division of labour with ``w2v_common.clients.base``: the WMClient POST
already carries an ``idempotency_key``, so a timeout is retried there up to
``_MAX_RETRIES`` at the *request* level. What reaches this module has already
exhausted that budget, so the retry offered here is the coarser job-level one —
bounded by the claim queue's ``attempts``/``max_attempts`` poison gate, never
unbounded. (``base`` deliberately excludes ``ReadTimeout`` from its
``_CONNECTION_RESET_ERRORS`` "safe for ANY method" set; that exclusion is about
replaying a *non-idempotent* request, which does not apply to re-running a whole
job keyed by its own ``job_id``.)

**Description.** ``str(httpx.ReadTimeout())`` is the empty string — httpx maps
the httpcore error and the message frequently comes through blank — so the
previous ``str(exc) or f"{type(exc).__name__}: (no message)"`` fallback produced
a record with no diagnostic content at all. Every description now carries the
exception type, a non-empty detail (``repr`` when ``str`` is blank), and the
upstream endpoint when the exception knows it.
"""

from __future__ import annotations

import httpx

# Transient upstream failures: the request was sent and no answer arrived in
# time. Every httpx timeout (Connect/Read/Write/Pool) derives from
# ``httpx.TimeoutException``.
#
# Deliberately NOT here:
#   * ``httpx.HTTPStatusError`` — a 4xx is a permanent contract failure and a
#     5xx has already been retried by BaseServiceClient's per-status budget.
#   * ``httpx.ConnectError``/``ReadError`` and friends — already retried inside
#     BaseServiceClient as connection resets; a job-level replay adds nothing.
#   * the builtin ``TimeoutError`` — this service raises it itself from its own
#     concept-image waits and ``asyncio.wait_for`` guards, which are local
#     budgets, not an upstream failing to answer.
#   * ``ValueError``/``RuntimeError`` — schema violations and the verification
#     gate. Permanent by construction.
_TRANSIENT_UPSTREAM_ERRORS: tuple[type[BaseException], ...] = (httpx.TimeoutException,)


def is_transient_upstream(exc: BaseException) -> bool:
    """Return True when *exc* is an upstream timeout worth re-running the job for."""
    return isinstance(exc, _TRANSIENT_UPSTREAM_ERRORS)


def upstream_target(exc: BaseException) -> str | None:
    """Return ``METHOD scheme://host[:port]/path`` for *exc*, or None.

    Rebuilt field by field from :class:`httpx.URL` rather than stringified, so
    the query string and any ``user:password@`` userinfo are structurally
    unreachable — this string is persisted on the job row and echoed to the
    caller, and a presigned/token-bearing URL must never land there. Path is
    kept because it names the worker (``/workers/<name>/message``), which is the
    whole diagnostic value.
    """
    if not isinstance(exc, httpx.HTTPError):
        return None
    try:
        request = exc.request
    except RuntimeError:
        # httpx raises this when the error was constructed without a request
        # (e.g. hand-built in a test, or a failure before the request existed).
        return None
    url = request.url
    netloc = f"{url.host}:{url.port}" if url.port else url.host
    return f"{request.method} {url.scheme}://{netloc}{url.path}"


def describe_failure(exc: BaseException) -> str:
    """Build the durable, non-empty error string persisted on the job row."""
    detail = str(exc).strip()
    if not detail:
        # ``repr`` is the last resort that still says something: for a bare
        # ``httpx.ReadTimeout`` it renders ``ReadTimeout('')`` — which at least
        # proves the message really was empty rather than lost on the way here.
        detail = repr(exc)
    described = f"{type(exc).__name__}: {detail}"
    target = upstream_target(exc)
    return f"{described} [upstream={target}]" if target else described
