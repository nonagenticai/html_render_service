"""ASGI application for html_render_service."""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from w2v_common.asgi import create_app
from w2v_common.clients import WMClient
from w2v_common.observability import database_endpoint, redis_endpoint

from src.cache.redis_client import close_redis, get_redis_client, init_redis
from src.config import get_settings
from src.database.connection import close_engine, get_engine
from src.database.session import get_session_maker
from src.logging_config import configure_logging

load_dotenv()

settings = get_settings()
configure_logging(json_logs=settings.logging.json_logs, level=settings.logging.level)
logger = logging.getLogger(__name__)


class HealthCheckLoggingFilter(logging.Filter):
    """Suppress noisy health-check requests from uvicorn access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(
            path in message
            for path in ["/health", "/health/ready", "/health/live", "/healthz", "/readyz"]
        )


logging.getLogger("uvicorn.access").addFilter(HealthCheckLoggingFilter())


async def _init_render_stack(app: FastAPI, wm_client: WMClient) -> None:
    """Build storage, the generative/render services, and the two pools.

    Extracted from ``lifespan`` to keep that function's branch-count in check;
    starts (in order) storage, the RedesignService, the shared Chromium browser
    pool, then the durable job-queue worker pool.
    """
    # Storage (local or MinIO; artifacts served back as absolute URLs).
    from src.services.storage_service import StorageService

    storage = StorageService()
    await storage.initialize()
    app.state.storage_service = storage

    from src.services.accessibility_scorer import AccessibilityScorer
    from src.services.concept_image_generator import ConceptImageGenerator
    from src.services.design_builder import RedesignService
    from src.services.hero_text_check import HeroTextChecker
    from src.services.html_generator import HtmlGenerator
    from src.services.prompt_template import PromptTemplateLoader
    from src.services.webhook_dispatcher import WebhookDispatcher

    template_loader = PromptTemplateLoader(
        version=settings.redesign.template_version,
        template_dir=settings.redesign.template_directory,
    )
    html_gen = HtmlGenerator(template_loader=template_loader, wm_client=wm_client)
    concept_gen = ConceptImageGenerator(template_loader=template_loader)
    a11y_scorer = AccessibilityScorer(wm_client=wm_client)
    webhook_dispatcher = (
        WebhookDispatcher(
            max_retries=settings.webhook.max_retries, timeout=settings.webhook.timeout
        )
        if settings.webhook.enabled
        else None
    )

    cache_service = None
    try:
        from src.services.cache_service import CacheService

        redis = getattr(app.state, "redis_client", None)
        if redis and getattr(redis, "is_available", False):
            cache_service = CacheService(redis)
            logger.info("CacheService initialized")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("CacheService not available: %s", exc)
    app.state.cache_service = cache_service

    # Cost-bus redis: a SEPARATE connection from settings.redis.url. Unset
    # METERING_REDIS_URL disables emission silently — a render must never fail
    # because billing is unreachable.
    from src.services.metering import init_metering_redis, resolve_stream

    app.state.metering_redis = await init_metering_redis(os.getenv("METERING_REDIS_URL", ""))
    # Resolve the cost-bus stream ONCE, at startup, and carry it on app.state so
    # the name this process bills to is a single visible fact rather than an
    # implicit library default re-read on every emit. ``resolve_stream()`` reads
    # METERING_STREAM (the env var the gitops manifests actually set) and falls
    # back to w2v_common's canonical ``billing:costs``.
    app.state.metering_stream = resolve_stream()
    logger.info("Cost-bus stream resolved: %s", app.state.metering_stream)

    app.state.redesign_service = RedesignService(
        storage=storage,
        html_generator=html_gen,
        concept_generator=concept_gen,
        accessibility_scorer=a11y_scorer,
        webhook_dispatcher=webhook_dispatcher,
        cache_service=cache_service,
        metering_redis=app.state.metering_redis,
        metering_stream=app.state.metering_stream,
        hero_text_checker=HeroTextChecker(wm_client),
    )
    logger.info("Services initialized: RedesignService, HtmlGenerator, StorageService, WMClient")

    # Shared Chromium browser pool — long-lived browsers reused across renders
    # (replaces launch-per-request). Started before the worker pool so the first
    # claimed job renders without paying a cold launch.
    from src.services.browser_pool import init_browser_pool

    await init_browser_pool()

    # Durable job-queue worker pool. Drains `pending`/expired-lease rows from
    # redesign_jobs and reclaims jobs orphaned by a crashed pod (the lease-based
    # successor to the old startup reap). Started last so all deps are ready.
    from src.services.job_worker import JobWorkerPool

    app.state.job_worker_pool = JobWorkerPool(app)
    await app.state.job_worker_pool.start()


async def _startup(app: FastAPI) -> None:
    """Init DB, Redis, WMClient, and the design/render stack (create_app on_startup)."""
    logger.info("=" * 70)
    logger.info("Starting HTML Render Service")
    logger.info("=" * 70)

    # Database — schema is owned by alembic (initContainer runs `alembic upgrade
    # head` before this app starts). Do NOT call Base.metadata.create_all here.
    # Phase D RLS: install the after_begin GUC re-bind listener before any session is
    # opened, so the worker's raw session_factory() heartbeat/processor sessions keep
    # app.current_tenant bound (from the ContextVar _run_claimed sets) under enforce.
    from src.database.rls_events import install_rls_tenant_rebind

    install_rls_tenant_rebind()
    try:
        app.state.db_engine = await get_engine()
        app.state.session_factory = await get_session_maker()
        logger.info(
            "Database connected (schema managed by alembic): %s",
            database_endpoint(settings.database.url),
        )
    except Exception:
        logger.exception("Failed to connect to database")
        raise

    # Redis (fail-open — caching is optional)
    try:
        redis_client = await init_redis()
        await redis_client.ping()
        app.state.redis_client = get_redis_client()
        logger.info("Redis connected: %s", redis_endpoint(settings.redis.url))
    except Exception:  # fail-open: caching is optional, never block startup
        app.state.redis_client = None
        logger.warning("Continuing without Redis -- caching will be disabled", exc_info=True)

    # One shared WMClient per service (sole LLM gateway; identity via ContextVars).
    wm_client = WMClient(timeout=settings.wm.timeout_seconds)
    app.state.wm_client = wm_client

    await _init_render_stack(app, wm_client)

    logger.info("HTML Render Service started — API %s:%d", settings.api.host, settings.api.port)
    logger.info("=" * 70)


async def _shutdown(app: FastAPI) -> None:
    """Stop the worker + browser pools, then close render/redis/db/WM (create_app on_shutdown)."""
    logger.info("Shutting down HTML Render Service")
    # Stop the worker pool first: cancels worker loops and expires this pod's
    # in-flight leases so a sibling/restart reclaims them immediately.
    pool = getattr(app.state, "job_worker_pool", None)
    if pool is not None:
        try:
            await pool.stop()
        except Exception:
            logger.exception("Error stopping job worker pool")
    # Close the browser pool after the workers that use it have stopped.
    try:
        from src.services.browser_pool import close_browser_pool

        await close_browser_pool()
    except Exception:
        logger.exception("Error stopping browser pool")
    wm_client = getattr(app.state, "wm_client", None)
    redesign_service = getattr(app.state, "redesign_service", None)
    metering_redis = getattr(app.state, "metering_redis", None)

    async def _close_metering() -> None:
        from src.services.metering import close_metering_redis

        await close_metering_redis(metering_redis)

    for name, closer in (
        ("RedesignService", getattr(redesign_service, "aclose", None)),
        ("MeteringRedis", _close_metering),
        ("Redis", close_redis),
        ("Database", close_engine),
        ("WMClient", getattr(wm_client, "aclose", None)),
    ):
        if closer is None:
            continue
        try:
            await closer()
        except Exception:
            logger.exception("Error closing %s", name)
    logger.info("HTML Render Service shutdown complete")


from src.api.health import router as health_router  # noqa: E402
from src.api.routes import router as api_router  # noqa: E402

app = create_app(
    settings,
    routers=[health_router, api_router],  # api_router → /api/designs (canonical)
    title=settings.app_name,
    description="Web-design microservice: LLM HTML generation + Playwright screenshot rendering",
    version=settings.app_version,
    on_startup=[_startup],
    on_shutdown=[_shutdown],
    redirect_slashes=False,
    swagger_ui_parameters={"persistAuthorization": True},
    generate_unique_id_function=lambda route: (
        f"{route.tags[0] if route.tags else 'default'}_{route.name}"
    ),
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, dependency probes."},
        {
            "name": "designs",
            "description": (
                "Async web-design jobs (HTML generation across scopes + screenshot "
                "artifacts). Canonical surface."
            ),
        },
    ],
)


@app.get("/", operation_id="service_root", tags=["health"], summary="Service banner")
async def root() -> dict[str, Any]:
    """Return service info and available endpoint listing."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Web-design microservice (HTML generation + screenshot rendering)",
        "endpoints": {"health": "/health", "docs": "/docs", "designs": "/api/designs"},
        "status": "running",
        "environment": settings.environment,
    }


def main() -> None:
    """Main entry point for running the ASGI application."""
    import uvicorn

    uvicorn.run(
        "src.asgi:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
        reload=settings.api.reload,
    )


if __name__ == "__main__":
    main()
