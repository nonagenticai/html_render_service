"""Lifespan (startup) tests for the ASGI app.

REGRESSION: ``_startup`` logged ``settings.database.url`` raw -- the full
postgresql+psycopg://<user>:<password>@host:port/db DSN -- at INFO on the
connect path, so the live html_render_service password landed in this pod's
stdout and in anything shipping those logs, indefinitely.

The test drives the REAL ``_startup`` rather than the redactor, because the
defect was at the CALL SITE: a test that only exercises
``w2v_common.observability.database_endpoint`` passes for the whole duration of
the leak. That helper has its own unit tests in w2v_common; this one exists to
fail if anyone hands the raw url back to the logger here.
"""

from __future__ import annotations

import io
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

# Wholly synthetic and shares no prefix with any real credential -- this string
# is committed, printed by pytest on failure, and ends up in CI logs, i.e.
# exactly the places this change exists to keep secrets out of.
_FAKE_PASSWORD = "not-a-real-password-xyzzy"  # pragma: allowlist secret
_FAKE_DSN = (
    f"postgresql+psycopg://html_render_app:{_FAKE_PASSWORD}"
    "@pgbouncer.pgbouncer:6432/html_render_service"
)


@pytest.mark.asyncio
async def test_startup_db_log_does_not_leak_the_dsn_password(monkeypatch: pytest.MonkeyPatch):
    """The emitted startup log record must not contain the DSN password."""
    from src import asgi

    monkeypatch.setattr(asgi.settings.database, "url", _FAKE_DSN)

    # A handler attached directly to the module logger, NOT ``caplog``. This
    # service calls ``configure_logging()``, which re-points the root handlers,
    # so caplog's root handler observes nothing and every assertion below would
    # pass vacuously.
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    asgi.logger.addHandler(handler)
    # ``setLevel`` and NOT ``logger.level = ...``: ``logging.Logger`` memoizes
    # ``isEnabledFor`` results in ``Logger._cache``, and only ``setLevel()``
    # invalidates that cache. Assigning ``.level`` directly leaves a stale entry
    # behind, so a logger already probed while it was at WARNING would keep
    # suppressing the record this test needs to see.
    previous_level = asgi.logger.level
    asgi.logger.setLevel(logging.INFO)

    app = FastAPI(title="lifespan-test")
    try:
        with (
            patch("src.asgi.get_engine", new_callable=AsyncMock, return_value=AsyncMock()),
            patch("src.asgi.get_session_maker", new_callable=AsyncMock, return_value=MagicMock()),
            patch("src.database.rls_events.install_rls_tenant_rebind"),
            # Redis + the whole render stack come AFTER the DB block and open
            # their own connections (storage, Chromium, the job-worker pool),
            # burning ~60s on retry ladders. Irrelevant to this assertion.
            patch("src.asgi.init_redis", new_callable=AsyncMock),
            patch("src.asgi.get_redis_client", return_value=None),
            patch("src.asgi.WMClient", return_value=MagicMock()),
            patch("src.asgi._init_render_stack", new_callable=AsyncMock),
        ):
            await asgi._startup(app)
    finally:
        asgi.logger.removeHandler(handler)
        asgi.logger.setLevel(previous_level)

    captured = buffer.getvalue()
    # Guard against a vacuous pass: if the line never emitted, the assertions
    # below would hold for the wrong reason.
    assert "Database connected" in captured, (
        "the 'Database connected' line never emitted; the assertions would be vacuous"
    )
    assert _FAKE_PASSWORD not in captured, "DSN password leaked into the startup log"
    assert "html_render_app" not in captured, "DSN username leaked into the startup log"
    # Still useful, or the next person deletes the log line instead of keeping it.
    assert "pgbouncer.pgbouncer:6432/html_render_service" in captured


# ── The startup "Redis connected" line must never carry the Redis password ──
#
# THE SIBLING LEAK. The DSN test above closed `Database connected`; four lines
# below it in `_startup`, `Redis connected` logged `settings.redis.url` raw. A
# Redis URL carries its password in exactly the same `user:pass@` slot
# (`redis://[<user>]:<password>@host:port/db`), so the guarded line and the
# unguarded one sat in the same function -- guarded beside unguarded.
_FAKE_REDIS_URL = f"redis://default:{_FAKE_PASSWORD}@redis.w2v:6379/0"  # pragma: allowlist secret


@pytest.mark.asyncio
async def test_startup_redis_log_does_not_leak_the_redis_password(monkeypatch: pytest.MonkeyPatch):
    """The emitted "Redis connected" record must not contain the Redis password.

    Drives the REAL `_startup`, not `w2v_common.observability.redis_endpoint`:
    the defect is at the CALL SITE, so a test of the helper alone stays green for
    the entire duration of the leak. The helper has its own unit tests upstream.
    """
    from src import asgi

    monkeypatch.setattr(asgi.settings.database, "url", _FAKE_DSN)
    monkeypatch.setattr(asgi.settings.redis, "url", _FAKE_REDIS_URL)

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    asgi.logger.addHandler(handler)
    # ``setLevel`` and NOT ``logger.level = ...`` / ``monkeypatch.setattr(logger,
    # "level", ...)``: ``logging.Logger`` memoizes ``isEnabledFor`` in
    # ``Logger._cache`` and only ``setLevel()`` invalidates it. A stale "INFO
    # disabled" entry would stop the record ever reaching this handler, and then
    # the negative assertions below would pass on EMPTY output with the leak
    # fully intact. The "Redis connected" positive control makes that impossible.
    previous_level = asgi.logger.level
    asgi.logger.setLevel(logging.INFO)

    app = FastAPI(title="lifespan-test-redis")
    try:
        with (
            patch("src.asgi.get_engine", new_callable=AsyncMock, return_value=AsyncMock()),
            patch("src.asgi.get_session_maker", new_callable=AsyncMock, return_value=MagicMock()),
            patch("src.database.rls_events.install_rls_tenant_rebind"),
            patch("src.asgi.init_redis", new_callable=AsyncMock),
            patch("src.asgi.get_redis_client", return_value=None),
            patch("src.asgi.WMClient", return_value=MagicMock()),
            patch("src.asgi._init_render_stack", new_callable=AsyncMock),
        ):
            await asgi._startup(app)
    finally:
        asgi.logger.removeHandler(handler)
        asgi.logger.setLevel(previous_level)

    captured = buffer.getvalue()
    # POSITIVE CONTROL, asserted FIRST. The Redis block is fail-open: on any
    # exception it logs a warning instead, and then every negative assertion
    # below would hold for the wrong reason.
    assert "Redis connected" in captured, (
        "the 'Redis connected' line never emitted; the assertions would be vacuous"
    )
    assert _FAKE_PASSWORD not in captured, "Redis password leaked into the startup log"
    assert "default@" not in captured, "Redis userinfo leaked into the startup log"
    # Still useful, or the next person deletes the log line instead of keeping it.
    assert "redis.w2v:6379" in captured
