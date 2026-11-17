"""html-render cost-metering identity + thin delegation to ``w2v_common.metering``.

Marshalling, retry, idempotency-key minting and stream resolution all live in
the shared cost-bus emitter ``w2v_common.metering`` (one source of truth for
every w2v service). This module keeps only what is html-render-specific: the
routing identity, and a single import site re-exporting the Redis lifecycle.

Why this exists at all: before it, a web-design/HTML render emitted **no cost
event of any kind**. A 30-day sweep of ``billing_cost_events`` carried keys for
mlx-backend, inference-router, kag, w2v-tts, w2v-video-analyzer,
mlx-ingestion-backend, ingestion, w2v-video-compiler, w2v-image, kag-graph,
karma and hirag — and no render key whatsoever. Screenshots, the Playwright
render, concept images and the render job itself billed nobody; only the LLM
slice billed, indirectly, as ``inference-router``.

The metering Redis is a SEPARATE connection from ``settings.redis.url`` (the
cache/queue Redis), taken from ``METERING_REDIS_URL``. When that is unset,
``init_metering_redis`` returns ``None`` and emission is disabled silently —
the service renders normally. A billing failure must never fail a render.
"""

from __future__ import annotations

from w2v_common.metering import (
    close_metering_redis,
    init_metering_redis,
    resolve_stream,
)

__all__ = [
    "COST_SOURCE",
    "SERVICE_KEY",
    "close_metering_redis",
    "init_metering_redis",
    "resolve_stream",
]

# COST_SOURCE identifies WHERE the spend came from — the platform-local
# Playwright/Chromium render lane in this service.
COST_SOURCE = "w2v-html-render"
# SERVICE_KEY is the billable PRODUCT slug, the key billing groups spend under.
# It MUST stay in step with the licenses ``App.key`` for the web-design product
# so a single per-(tenant, product) markup rule prices this lane; a divergent
# slug leaves the render silently unpriced, which is the exact defect this
# module was added to end.
SERVICE_KEY = "w2v-html-render"  # pragma: allowlist secret -- routing slug, not a credential
