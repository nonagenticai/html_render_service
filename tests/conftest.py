"""Shared test fixtures for html_render_service tests."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

# Set test environment variables before importing any app code.
# Port 5474, not 5432: a 5432 default resolves to whatever postgres the dev
# happens to be running, which answers the TCP connect and THEN fails auth —
# disguising a config error as a connection error. ci_test /
# html_render_service_ci so conftest, scripts/ci-local.sh and the
# `html_render_service_test_postgres` compose service all agree.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://ci_test:ci_test_password_secure@127.0.0.1:5474/html_render_service_ci",  # pragma: allowlist secret
)
# w2v_common >= 0.15.0 dropped the localhost defaults on RedisSettings.url and
# StorageSettings.minio_endpoint; seed module-level so Settings() instantiation
# during pytest collection doesn't fail. The autouse _clear_settings fixture
# re-applies them after its env-clearing sweep.
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("STORAGE_MINIO_ENDPOINT", "localhost:9000")

# Capture the test DB URL BEFORE the autouse _clear_settings fixture
# monkeypatches DATABASE_URL to its in-test default. CI sets DATABASE_URL
# to its html_render_service_ci postgres; locally falls back to the default
# above (developer must run a local postgres to execute db_engine tests).
_TEST_DATABASE_URL = os.environ["DATABASE_URL"]


# ---------------------------------------------------------------------------
# Settings cache management
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch):
    """Remove CI env vars that interfere with config-default tests."""
    for key in list(os.environ):
        if any(
            key.startswith(p)
            for p in (
                "DATABASE_",
                "AUTH_",
                "REDIS_",
                "API_",
                "STORAGE_",
                "WM_",
                "SCREENSHOT_",
                "REDESIGN_",
                "CACHE_",
                "LOG_",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    # Restore w2v_common 0.15.0 fail-fast fields with placeholders so
    # config-default tests work without hitting ValidationError on missing url
    # / minio_endpoint env.
    monkeypatch.setenv("DATABASE_URL", _TEST_DATABASE_URL)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "localhost:9000")


# ---------------------------------------------------------------------------
# Real database fixtures (postgres+psycopg, TRUNCATE-based isolation)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_engine():
    """Per-test async postgres engine with TRUNCATE-based isolation.

    Connects to _TEST_DATABASE_URL captured at module load (CI postgres in
    CI, localhost default locally). Runs Base.metadata.create_all (idempotent)
    then TRUNCATEs every model table so each test starts from a clean slate.

    Uses NullPool: the CI postgres is fronted by pgbouncer, which drops idle
    connections; pooling a dead conn across tests yields ProtocolViolation.
    """
    from src.database.models import Base

    engine = create_async_engine(_TEST_DATABASE_URL, echo=False, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # A prior test (e.g. the concurrent-claim race) can leave a connection
        # idle-in-transaction still holding a row lock. With FOR UPDATE SKIP
        # LOCKED that makes the next test's claim intermittently return None, and
        # it would make TRUNCATE block. Terminate any such leaked backend and cap
        # lock waits so each test starts from a genuinely clean, lock-free slate.
        await conn.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid() "
                "AND state IN ('idle in transaction', 'idle in transaction (aborted)')"
            )
        )
        await conn.execute(text("SET lock_timeout = '10s'"))
        if Base.metadata.sorted_tables:
            tables = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
            await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session_factory(db_engine):
    """Real async_sessionmaker for html_render_service tests."""
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db_session(db_session_factory):
    """Real AsyncSession for a single test."""
    async with db_session_factory() as session:
        yield session
