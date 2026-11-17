"""Tests for health check endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError

from src.api.health import router
from src.services.browser_pool import PoolStatus
from src.services.job_worker import WorkerPoolStatus

_HEALTHY_POOL = PoolStatus(started=True, capacity=2, idle=2, live=1)


class _FakeWorkerPool:
    """Stands in for JobWorkerPool: only ``status()`` is on the health path."""

    def __init__(self, status: WorkerPoolStatus) -> None:
        self._status = status

    def status(self) -> WorkerPoolStatus:
        return self._status


@pytest.fixture(autouse=True)
def _healthy_browser_pool():
    """Default the Chromium pool to started, so each test names its own fault."""
    with patch("src.api.health.pool_status", return_value=_HEALTHY_POOL):
        yield


def _make_health_app(
    *,
    workers: WorkerPoolStatus | None = None,
    attach_worker_pool: bool = True,
) -> FastAPI:
    """Build the health app. Defaults to a fully-alive worker pool."""
    app = FastAPI(title="health-test")
    app.include_router(router)
    if attach_worker_pool:
        app.state.job_worker_pool = _FakeWorkerPool(
            workers or WorkerPoolStatus(started=True, workers=2, alive=2)
        )
    return app


@pytest.mark.asyncio
async def test_health_live():
    """Liveness probe always returns ok."""
    app = _make_health_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_ready_success():
    """Readiness probe returns ready when DB is up."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()

    mock_engine = AsyncMock()
    # Make connect() return an async context manager yielding mock_conn
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    mock_engine.connect = MagicMock(return_value=cm)

    with patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine):
        app = _make_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_ready_db_down():
    """Readiness probe returns 503 when DB is down."""
    with patch(
        "src.api.health.get_engine",
        new_callable=AsyncMock,
        side_effect=ConnectionError("no db"),
    ):
        app = _make_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["ready"] is False
    assert data["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_full_health_check_healthy():
    """Full health endpoint returns healthy when DB and Redis are up."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()

    mock_engine = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    mock_engine.connect = MagicMock(return_value=cm)

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)

    with (
        patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
        patch("src.api.health.get_redis_client", return_value=mock_redis),
    ):
        app = _make_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["components"]["database"] == "healthy"
    assert data["components"]["redis"] == "healthy"
    assert data["components"]["api"] == "healthy"


@pytest.mark.asyncio
async def test_full_health_check_no_redis():
    """Full health returns not configured when Redis client is None."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()

    mock_engine = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    mock_engine.connect = MagicMock(return_value=cm)

    with (
        patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
        patch("src.api.health.get_redis_client", return_value=None),
    ):
        app = _make_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["components"]["redis"] == "not configured"


@pytest.mark.asyncio
async def test_full_health_check_db_down():
    """Full health returns degraded when DB is unhealthy."""
    with (
        patch(
            "src.api.health.get_engine",
            new_callable=AsyncMock,
            side_effect=ConnectionError("db down"),
        ),
        patch("src.api.health.get_redis_client", return_value=None),
    ):
        app = _make_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert "unhealthy" in data["components"]["database"]


@pytest.mark.asyncio
async def test_full_health_check_redis_ping_fails():
    """Full health returns degraded when Redis ping fails."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()

    mock_engine = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    mock_engine.connect = MagicMock(return_value=cm)

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=False)

    with (
        patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
        patch("src.api.health.get_redis_client", return_value=mock_redis),
    ):
        app = _make_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["components"]["redis"] == "unhealthy: ping failed"


# ---------------------------------------------------------------------------
# The component that does the WORK: claim-queue workers + Chromium pool.
# Each of these is a state in which the old health check said "healthy" while
# the pod rendered nothing at all.
# ---------------------------------------------------------------------------


def _up_engine():
    """An engine whose connect() context manager yields a working connection."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    engine = AsyncMock()
    engine.connect = MagicMock(return_value=cm)
    return engine


async def _get_health(app: FastAPI) -> dict:
    with (
        patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=_up_engine()),
        patch("src.api.health.get_redis_client", return_value=None),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_health_unhealthy_when_all_worker_tasks_have_died():
    """Every worker task ended -> the claim queue is not drained by this pod."""
    app = _make_health_app(workers=WorkerPoolStatus(started=True, workers=2, alive=0))
    data = await _get_health(app)

    assert data["status"] == "unhealthy"
    assert "not being drained" in data["components"]["job_workers"]
    # The dependencies are fine; the fault must be attributed to the workers.
    assert data["components"]["database"] == "healthy"


@pytest.mark.asyncio
async def test_health_unhealthy_when_worker_pool_was_never_started():
    """Startup never reached JobWorkerPool.start() -- no attribute on app.state."""
    app = _make_health_app(attach_worker_pool=False)
    data = await _get_health(app)

    assert data["status"] == "unhealthy"
    assert "not started" in data["components"]["job_workers"]


@pytest.mark.asyncio
async def test_health_flags_a_partial_worker_die_off():
    """A partial die-off halves throughput silently; surface it."""
    app = _make_health_app(workers=WorkerPoolStatus(started=True, workers=4, alive=1))
    data = await _get_health(app)

    # Degraded, not unhealthy: the queue still drains, just slower.
    assert data["status"] == "degraded"
    assert "1/4" in data["components"]["job_workers"]


@pytest.mark.asyncio
async def test_health_unhealthy_when_browser_pool_never_started():
    """No Chromium slot exists -> every render blocks or fails."""
    app = _make_health_app()
    with patch(
        "src.api.health.pool_status",
        return_value=PoolStatus(started=False, capacity=0, idle=0, live=0),
    ):
        data = await _get_health(app)

    assert data["status"] == "unhealthy"
    assert "no Chromium render slot" in data["components"]["browser_pool"]


@pytest.mark.asyncio
async def test_health_unhealthy_when_browser_pool_has_no_capacity():
    """Started but zero slots is capacity zero -- 'the attribute is set' is not a signal."""
    app = _make_health_app()
    with patch(
        "src.api.health.pool_status",
        return_value=PoolStatus(started=True, capacity=0, idle=0, live=0),
    ):
        data = await _get_health(app)

    assert data["status"] == "unhealthy"
    assert data["components"]["browser_pool"].startswith("unhealthy")


@pytest.mark.asyncio
async def test_readiness_stays_green_when_the_worker_pool_is_dead():
    """Deliberate: the queue is shared and pulled, so eviction fixes nothing.

    A shared-cause die-off would otherwise evict every replica at once and take
    job submission down on top of an already-stalled queue.
    """
    app = _make_health_app(workers=WorkerPoolStatus(started=True, workers=2, alive=0))
    with (
        patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=_up_engine()),
        patch(
            "src.api.health.pool_status",
            return_value=PoolStatus(started=False, capacity=0, idle=0, live=0),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["ready"] is True


# ---------------------------------------------------------------------------
# 2026-08-15 pgbouncer blip: the probes caught only builtins, so a
# ``sqlalchemy.exc.OperationalError`` escaped to the ASGI layer and readiness
# answered **500 with no body** instead of a clean 503.
# ---------------------------------------------------------------------------


def _sqlalchemy_operational_error() -> SQLAlchemyOperationalError:
    """The exact exception a Postgres/pgbouncer blip raises through SQLAlchemy.

    ``psycopg.OperationalError`` is wrapped into
    ``sqlalchemy.exc.OperationalError``, whose MRO stops at ``SQLAlchemyError ->
    Exception``. It is NOT an ``OSError`` and NOT a ``RuntimeError``.
    """
    return SQLAlchemyOperationalError(
        "SELECT 1",
        {},
        Exception("server closed the connection unexpectedly"),
    )


def test_sqlalchemy_operational_error_is_not_a_builtin():
    """The premise of the bug: the narrow tuple could never have fired."""
    assert not issubclass(SQLAlchemyOperationalError, OSError)
    assert not issubclass(SQLAlchemyOperationalError, RuntimeError)
    assert not issubclass(SQLAlchemyOperationalError, ConnectionError)
    assert not issubclass(SQLAlchemyOperationalError, TimeoutError)


@pytest.mark.asyncio
async def test_health_ready_returns_503_not_500_on_sqlalchemy_operational_error():
    """Readiness must degrade to 503, never raise into the ASGI layer."""
    with patch(
        "src.api.health.get_engine",
        new_callable=AsyncMock,
        side_effect=_sqlalchemy_operational_error(),
    ):
        app = _make_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")

    assert response.status_code != 500, "the exception escaped to the ASGI layer"
    assert response.status_code == 503
    data = response.json()
    assert data["ready"] is False
    assert data["status"] == "unhealthy"
    # The 503 body must name WHICH check failed.
    assert data["checks"]["database"]["status"] == "down"
    assert "server closed the connection unexpectedly" in data["checks"]["database"]["error"]


@pytest.mark.asyncio
async def test_health_ready_still_returns_200_on_the_happy_path():
    """The widened tuple must not change the healthy response."""
    with patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=_up_engine()):
        app = _make_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["checks"]["database"]["status"] == "up"


@pytest.mark.asyncio
async def test_health_aggregate_attributes_a_sqlalchemy_fault_to_the_db():
    """The /health per-component DB probe had the identical narrow tuple."""
    with (
        patch(
            "src.api.health.get_engine",
            new_callable=AsyncMock,
            side_effect=_sqlalchemy_operational_error(),
        ),
        patch("src.api.health.get_redis_client", return_value=None),
    ):
        app = _make_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code != 500, "the exception escaped to the ASGI layer"
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert "unhealthy" in data["components"]["database"]


@pytest.mark.asyncio
async def test_health_aggregate_returns_503_not_500_on_escaping_sqlalchemy_error():
    """The /health OUTER handler had the same too-narrow tuple.

    Anything raising a SQLAlchemy error outside the per-component blocks (here:
    the browser-pool component) must land on the 503 fallback, not ASGI's 500.
    """
    with (
        patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=_up_engine()),
        patch("src.api.health.get_redis_client", return_value=None),
        patch("src.api.health.pool_status", side_effect=_sqlalchemy_operational_error()),
    ):
        app = _make_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code != 500, "the exception escaped the aggregate handler"
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert "server closed the connection unexpectedly" in data["error"]


def test_browser_pool_status_does_not_start_the_pool(monkeypatch: pytest.MonkeyPatch):
    """The probe must report the pool the render path sees, not create one."""
    import src.services.browser_pool as bp

    monkeypatch.setattr(bp, "_pool", None)
    status = bp.pool_status()
    assert status == PoolStatus(started=False, capacity=0, idle=0, live=0)
    assert bp._pool is None, "pool_status() must never construct the pool"


# --- batch_progress component: pinned at the ENDPOINT, not just the helper ---
# test_batch_reaper calls ``_batch_progress_component()`` directly, so dropping
# the component from the /health payload (or from the overall-status roll-up)
# would leave every existing test green while ops lost the only signal that a
# dead progress clock is about to get a LIVE batch reaped.


async def _get_health_all_deps_up(app: FastAPI) -> dict:
    """GET /health with DB and Redis both healthy, so only the component under test can degrade."""
    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    with (
        patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=_up_engine()),
        patch("src.api.health.get_redis_client", return_value=redis),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_health_exposes_the_batch_progress_component() -> None:
    from src.services.batch_progress import reset_progress_report_failures

    reset_progress_report_failures()
    data = await _get_health_all_deps_up(_make_health_app())

    assert "batch_progress" in data["components"], data["components"]
    assert data["components"]["batch_progress"] == "healthy"
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_a_dead_progress_clock_degrades_health_but_never_marks_it_unhealthy() -> None:
    from src.services.batch_progress import CONSECUTIVE_FAILURE_ALERT

    with patch(
        "src.services.batch_progress.progress_report_failures",
        return_value=(CONSECUTIVE_FAILURE_ALERT, CONSECUTIVE_FAILURE_ALERT),
    ):
        data = await _get_health_all_deps_up(_make_health_app())

    component = data["components"]["batch_progress"]
    assert component.startswith("degraded:"), component
    assert f"{CONSECUTIVE_FAILURE_ALERT} consecutive failures" in component
    # Degraded, never unhealthy: the renders are fine; this must not page as an outage.
    assert data["status"] == "degraded"


def test_the_reaper_window_setting_is_named_batch_no_progress_seconds() -> None:
    """The reaper's window is NO-PROGRESS, not age; the field name says so and is what config sets."""
    from src.config import DesignSettings

    fields = DesignSettings.model_fields
    assert "batch_no_progress_seconds" in fields
    assert fields["batch_no_progress_seconds"].default == 3600
