"""Health check endpoints for html_render_service."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.cache.redis_client import get_redis_client
from src.database.connection import get_engine
from src.services.browser_pool import pool_status

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# The library exception trees below are NOT optional and must never be
# "simplified" away. None of these inherit from the builtins whose names they
# shadow::
#
#     sqlalchemy.exc.OperationalError.__mro__
#         -> DatabaseError -> DBAPIError -> StatementError -> SQLAlchemyError
#            -> HasDescriptionCode -> Exception
#     issubclass(sqlalchemy.exc.OperationalError, OSError | RuntimeError) -> False
#     issubclass(redis.exceptions.RedisError,     OSError)                -> False
#
# A ``psycopg.OperationalError`` from a Postgres/pgbouncer blip surfaces as
# ``sqlalchemy.exc.OperationalError``, so a builtins-only ``except (OSError,
# RuntimeError)`` let every REAL database fault escape these probes and FastAPI
# turned it into a 500. That is exactly what happened on 2026-08-15 during a
# ~45s pgbouncer blip: ``/health/ready`` logged "unhandled exception in GET
# /health/ready" and answered 500 with no body.
#
# On /health/ready (and its /readyz alias) that breaks the Kubernetes readiness
# contract: a transient DB fault must be a 503 ("not ready"), never a 500 ("the
# app is broken").
#
# Probe failure = the dependency is down, not the process. The builtins stay in
# the tuples because a socket-level error can still surface as one.
_DB_PROBE_ERRORS = (OSError, ConnectionError, TimeoutError, RuntimeError, SQLAlchemyError)
_REDIS_PROBE_ERRORS = (OSError, ConnectionError, TimeoutError, RuntimeError, RedisError)


def _browser_pool_component() -> tuple[str, str]:
    """Report the Chromium pool: does a render slot actually exist on this pod?

    ``pool_status`` reads the module global without going through the
    start-on-demand accessor, so this reports the pool the render path would
    find — not one the probe just conjured.
    """
    status = pool_status()
    if not status.started or status.capacity == 0:
        return (
            "unhealthy: browser pool not started -- this pod has no Chromium render slot",
            "unhealthy",
        )
    return (
        f"healthy: {status.capacity} slot(s), {status.idle} idle, {status.live} browser(s) live",
        "healthy",
    )


def _job_workers_component(request: Request) -> tuple[str, str]:
    """Report the claim-queue worker tasks: is anything draining the queue?

    This is the component that does the work. Jobs arrive as ``pending`` rows and
    are pulled by these tasks; if they are all gone the queue silently stops
    draining while DB, Redis and every HTTP route keep answering perfectly.
    """
    pool = getattr(request.app.state, "job_worker_pool", None)
    status = pool.status() if pool is not None else None
    if status is None or not status.started:
        return (
            "unhealthy: job worker pool not started -- the claim queue is not being drained",
            "unhealthy",
        )
    if status.alive == 0:
        return (
            f"unhealthy: all {status.workers} worker task(s) have exited -- "
            "the claim queue is not being drained",
            "unhealthy",
        )
    if status.alive < status.workers:
        # Partial die-off: throughput is cut but the queue still drains, so this
        # is degraded rather than a full stop.
        return (
            f"degraded: only {status.alive}/{status.workers} worker task(s) alive",
            "degraded",
        )
    return f"healthy: {status.alive}/{status.workers} worker task(s) alive", "healthy"


def _batch_progress_component() -> tuple[str, str]:
    """Report whether the batch progress clock is actually being written.

    ``report_progress`` swallows its exceptions on purpose — a liveness signal
    must never take down a render that is succeeding. But a reporter that fails
    every time is indistinguishable from having no clock at all, and the
    consequence is silent and expensive: the stale-batch reaper moves a LIVE
    batch (and its child jobs and variants) to ``failed``. Counting the failures
    is what makes that visible before it happens rather than after.

    Degraded, never unhealthy: the renders themselves are fine, and this must not
    take a pod out of rotation.
    """
    from src.services.batch_progress import (
        CONSECUTIVE_FAILURE_ALERT,
        progress_report_failures,
    )

    total, consecutive = progress_report_failures()
    if consecutive >= CONSECUTIVE_FAILURE_ALERT:
        return (
            f"degraded: batch progress clock not being written "
            f"({consecutive} consecutive failures, {total} total) -- "
            "a live batch may be reaped as abandoned",
            "degraded",
        )
    if total:
        return f"healthy: {total} transient progress-report failure(s)", "healthy"
    return "healthy", "healthy"


@router.get("/health", response_model=None, operation_id="health_check")
async def health_check(request: Request) -> dict[str, Any] | JSONResponse:
    """Combined health check for all components.

    Reports the two components that actually do the work — the claim-queue worker
    tasks and the Chromium pool — alongside the DB/Redis dependencies. Neither is
    wired into readiness; see :func:`health_ready` for why.
    """
    try:
        db_status = "healthy"
        try:
            engine = await get_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except _DB_PROBE_ERRORS as e:
            db_status = f"unhealthy: {e}"

        redis_status = "healthy"
        try:
            redis_client = get_redis_client()
            if redis_client:
                healthy = await redis_client.ping()
                if not healthy:
                    redis_status = "unhealthy: ping failed"
            else:
                redis_status = "not configured"
        except _REDIS_PROBE_ERRORS as e:
            redis_status = f"unhealthy: {e}"

        workers_status, workers_level = _job_workers_component(request)
        browser_status, browser_level = _browser_pool_component()
        progress_status, progress_level = _batch_progress_component()

        deps_ok = db_status == "healthy" and redis_status == "healthy"
        if "unhealthy" in (workers_level, browser_level, progress_level):
            # The thing that does the work is stopped. Distinguished from
            # "degraded" so an alert can tell "one dependency is flaky" from
            # "this pod renders nothing at all".
            overall = "unhealthy"
        elif not deps_ok or "degraded" in (workers_level, browser_level, progress_level):
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "status": overall,
            "components": {
                "database": db_status,
                "redis": redis_status,
                "api": "healthy",
                "job_workers": workers_status,
                "browser_pool": browser_status,
                "batch_progress": progress_status,
            },
        }
    except (*_DB_PROBE_ERRORS, *_REDIS_PROBE_ERRORS) as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)},
        )


@router.get("/health/live", response_model=None, operation_id="health_live")
async def health_live() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "ok"}


@router.get("/health/ready", response_model=None, operation_id="health_ready")
async def health_ready() -> dict[str, Any] | JSONResponse:
    """Kubernetes readiness probe -- checks database connectivity.

    The worker pool and the browser pool are deliberately NOT checked here, and
    that is a decision, not an omission:

    * **Readiness gates HTTP, but work does not arrive over HTTP.** Jobs are
      *pulled* off a shared Postgres claim queue, not pushed by the Service. So
      evicting a pod whose workers died stops none of the damage — it only
      removes a submit/status/artifact surface that is still perfectly correct.
    * **The realistic causes are shared, not per-pod.** A bad image or config
      kills the workers on every replica at once. Readiness would then evict all
      of them simultaneously, so on top of a stalled queue clients could no
      longer submit jobs or poll the ones already queued. That is strictly worse
      than the blindness it replaces.
    * **Liveness was considered and rejected** for the same reason: a restart
      would fix a locally-crashed pool, but a shared cause turns it into a
      fleet-wide crashloop.

    So the signal lands on ``/health`` as ``status: unhealthy`` — loud to an
    operator and to alerting, inert to the scheduler.
    """
    try:
        engine = await get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "healthy", "ready": True, "checks": {"database": {"status": "up"}}}
    except _DB_PROBE_ERRORS as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "ready": False,
                "checks": {"database": {"status": "down", "error": str(e)}},
            },
        )


@router.get("/healthz", response_model=None, operation_id="health.live")
async def healthz() -> dict[str, str]:
    """Modern K8s liveness probe path (alias of /health/live)."""
    return await health_live()


@router.get("/readyz", response_model=None, operation_id="health.ready")
async def readyz() -> dict[str, Any] | JSONResponse:
    """Modern K8s readiness probe path (alias of /health/ready)."""
    return await health_ready()
