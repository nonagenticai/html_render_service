# HTML Render Service

Generates web-design HTML via an LLM and rasterises it into multi-viewport screenshots and PDFs with headless Chromium. Port 6319.

## What HTML Render Service Does

When `w2v_orchestrator` runs a `website_review` or `web_design` pipeline (the `website_redesign` and `promo_html` stages), it POSTs a design request to this service describing what to build — a scope (`hero`, `section`, `component`, `landing`, `full_page`, or a `redesign` of an existing page), brand tokens, and optional feedback. Rather than block the HTTP call for the tens of seconds a full-page generation plus render takes, the service **enqueues** the work: it inserts a `pending` row into `redesign_jobs` carrying a JSONB `job_options` bundle (including the caller's tenant/license identity), returns `202` with a job ID, and hands the caller a poll URL. Everything downstream happens on a detached worker.

Generation runs through `worker_management` — the sole LLM gateway — never a provider or the inference-router directly. Two modes are available. In **`freeform`** mode the `web-design` worker hand-writes a self-contained HTML+CSS page. In **`assembly`** mode the LLM only emits a JSON *page plan* selecting vetted catalog blocks (navbar, hero, features, stats, testimonials, pricing, FAQ, CTA, footer) and filling content slots; a deterministic assembler then renders those blocks from autoescaping Jinja partials with vendored Tailwind/DaisyUI CSS and JS **inlined** into the artifact. The philosophy is "everything that can be computed, shall be computed" — layout, markup, theming, and assets are code, not model output, so the result is fully self-contained and injection-safe. Once HTML exists, a shared Chromium browser pool captures screenshots across desktop/tablet/mobile viewports (full-page captures are clipped at 8000px to bound Chromium's bitmap allocation) and produces a PDF on demand.

The durable claim-queue is the defining architectural choice and the reason the service runs **replicas=2 behind an HPA**. A per-pod pool of worker loops drains the queue by calling the `SECURITY DEFINER` function `claim_next_render_job(worker_id, lease_secs, max_attempts)`, which atomically selects one pending or expired-lease row with `FOR UPDATE SKIP LOCKED`, fails poison jobs past their attempt ceiling, moves the row to `processing` under a fresh lease, and returns `(job_id, tenant_id, job_options)`. Because a claimed job carries a time-bounded lease renewed by a heartbeat, a pod that crashes mid-render simply drops its lease — a sibling pod reclaims the orphaned job once the lease expires, so no work is silently lost and horizontal scaling requires no external broker.

Generated HTML and screenshots are written to local scratch and then persisted to **MinIO** (bucket `html-render-service`) so that any replica can serve any artifact. Consumers never touch MinIO directly: the service proxy-serves every artifact through its own routes as **absolute** http(s) URLs (`API_PUBLIC_BASE_URL` defaults to the in-cluster service URL), so the orchestrator and `video_compiler` can fetch screenshots cross-namespace with a plain GET. All persistence is Postgres 18 via psycopg3, and the runtime connects as a restricted RLS role (`TENANT_RLS_MODE=enforce`) with an `after_begin` listener that re-binds `app.current_tenant` so tenant isolation survives the multi-commit background work a job performs.

## Features

- **Durable claim-queue** — POST enqueues a `pending` job and returns immediately; a per-pod worker pool claims rows via `FOR UPDATE SKIP LOCKED` so requests never block on generation and a crashed pod's leased jobs are reclaimed instead of lost
- **Lease-based crash recovery** — each claimed job holds a renewable lease (`REDESIGN_LEASE_SECONDS`); expired leases are reclaimed by sibling pods, making the service safely horizontally scalable (replicas=2 + HPA) with no external broker
- **Freeform + assembly generation** — the `web-design` worker either hand-writes a full page or emits a JSON page plan the deterministic assembler renders from vetted catalog blocks, so layout and theming are code, not fragile LLM output
- **Self-contained artifacts** — assembly mode inlines vendored Tailwind/DaisyUI CSS and JS and autoescapes all slot content, so the HTML renders identically under Playwright, in a live-preview iframe, and as a download with no CSP or asset-reach concerns
- **Shared Chromium browser pool** — long-lived browsers are reused across renders and relaunched after `REDESIGN_BROWSER_MAX_RENDERS` leases to bound the memory ratchet, replacing the old launch-per-request cost
- **Multi-viewport screenshots** — captures desktop/tablet/mobile PNGs plus on-demand PDF, clipping full-page height at 8000px to prevent the giant-bitmap OOM on tall marketing pages
- **MinIO-backed artifact serving** — HTML and screenshots persist to a shared bucket and are proxy-served as absolute URLs so any replica serves any artifact and downstream services fetch cross-namespace
- **A/B style variants** — generates multiple styled variants per job (`REDESIGN_MAX_VARIANTS`) with bounded concurrency so callers can compare treatments
- **Iterative refinement** — a job can be refined with feedback text, chaining a lineage of iterations up to `REDESIGN_MAX_ITERATIONS` so designs improve without starting over
- **Batch design jobs** — accepts multi-page batch requests that fan out under a concurrency cap so a whole site can be requested in one call
- **Accessibility verification gate** — assembly designs run an axe-based verification lane that refines up to a bound and fails loud on serious violations
- **License-token multi-tenancy** — every job is tenant-isolated via introspected license tokens and Postgres RLS enforce, with the caller identity persisted into `job_options` so the detached worker re-binds the correct tenant

## Authentication

Auth enabled by default (`AUTH_ENABLED=true`). Inbound requests carry the license token as `X-Tenant-Id: Bearer <license_token>` — **not** the `Authorization` header, which is reserved for the SPA's Keycloak JWT. The `AuthMiddleware` introspects the token against the licenses service and caches the result for 5 minutes. Because the enqueue path detaches work onto a worker that does not inherit request ContextVars, the caller identity is persisted into the job's `job_options` and re-applied via `set_identity_context()` before the worker dispatches. RLS is **enforce-capable**: the runtime connects as the restricted `w2v_html_render_app` role with `TENANT_RLS_MODE=enforce`, and `rls_events.py` re-binds `app.current_tenant` on `after_begin` so the GUC survives multi-commit background work.

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | `X-Tenant-Id: Bearer <token>` introspected against the licenses service, cached 5 min |
| 2 | Trusted network | Synthetic in-cluster identity for trusted CIDRs — OFF by default (`trust_incluster_identity=false`, fail-closed) |

Outbound service-to-service calls forward `X-Tenant-Id`, `X-License-Id`, and `X-Subject-Id` as plain headers (no `Bearer` prefix). Exempt paths (from `asgi.py`): `/`, `/health`, `/health/live`, `/health/ready`, `/healthz`, `/readyz`, `/ready`, `/docs`, `/redoc`, `/openapi.json`, `/favicon.ico`, `/metrics`.

## Quick Start

```bash
uv sync
uv run uvicorn src.asgi:app --host 0.0.0.0 --port 6319
uv run pytest tests/ -v --cov=src
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

All domain routes are under the `/api/designs` prefix (canonical surface).

### Designs `/api/designs`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/` | Enqueue a design/redesign job (returns 202 + job ID) |
| GET | `/jobs` | List design jobs |
| GET | `/jobs/{job_id}` | Poll job status |
| GET | `/jobs/{job_id}/result` | Final result payload |
| POST | `/jobs/{job_id}/cancel` | Cancel a queued or running job |
| POST | `/jobs/{job_id}/refine` | Iterate on a job with feedback |
| GET | `/jobs/{job_id}/history` | Iteration lineage |

### Artifacts `/api/designs`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/jobs/{job_id}/html` | Generated HTML |
| GET | `/jobs/{job_id}/pdf` | Rendered PDF |
| GET | `/jobs/{job_id}/screenshots/{viewport}` | Screenshot artifact for a viewport |
| GET | `/jobs/{job_id}/sections/{section_name}` | Per-section screenshot |
| GET | `/jobs/{job_id}/component-html` | Extracted component HTML |
| GET | `/jobs/{job_id}/original/{viewport}` | Original-page screenshot (redesign) |
| GET | `/jobs/{job_id}/comparison/{viewport}` | Side-by-side comparison image |
| GET | `/jobs/{job_id}/diff/{viewport}` | Visual diff image |

### Variants `/api/designs`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/jobs/{job_id}/compare` | Compare A/B variants |
| GET | `/jobs/{job_id}/variants/{style}/html` | HTML for a specific variant |

### Styles `/api/designs/styles`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/styles` | List tenant custom styles |
| POST | `/styles` | Create a custom style |
| GET | `/styles/{style_id}` | Get a custom style |
| PUT | `/styles/{style_id}` | Update a custom style |
| DELETE | `/styles/{style_id}` | Delete a custom style |
| POST | `/styles/{style_id}/preview` | Render a style preview |

Behaviour changes (2026-08-08): `POST /styles/{style_id}/preview` now returns **404 for a deactivated style** (it is a consumption route and matches the generation path, which has always skipped inactive styles); `GET /styles` and `GET /styles/{style_id}` now **include deactivated styles** (each carries `is_active`) because deactivated rows still consume the per-tenant quota and must stay deletable.

### Batch `/api/designs/batch`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/batch` | Create a multi-page batch job |
| GET | `/batch` | List batch jobs |
| GET | `/batch/{batch_id}` | Get batch status |
| GET | `/batch/{batch_id}/results` | Batch results |
| POST | `/batch/{batch_id}/cancel` | Cancel a batch |
| POST | `/batch/{batch_id}/retry-failed` | Retry failed pages in a batch |

### Webhooks `/api/designs`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/jobs/{job_id}/webhooks` | List webhook delivery attempts for a job |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Health check |
| GET | `/health/live` · `/healthz` | Liveness probe |
| GET | `/health/ready` · `/readyz` | Readiness probe (checks DB) |

## Configuration

All configuration via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Listen host |
| `API_PORT` | `6319` | Listen port |
| `API_DEBUG` | `false` | Enable /docs and /redoc |
| `API_PUBLIC_BASE_URL` | `http://html-render-service.w2v:6319` | Absolute base URL for fetchable artifact URLs |
| `AUTH_ENABLED` | `true` | Toggle authentication middleware |
| `AUTH_CACHE_TTL` | `300` | License introspection cache TTL (seconds) |
| `TENANT_RLS_MODE` | `enforce` (prod) | RLS mode; enforce binds `app.current_tenant` per transaction |
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_JSON_LOGS` | `false` | Structured JSON logging |
| `DATABASE_URL` | -- | PostgreSQL (psycopg3) connection URL |
| `DATABASE_POOL_SIZE` | `10` | Database connection pool size |
| `DATABASE_POOL_MIN_SIZE` | `2` | Minimum pool connections |
| `DATABASE_MAX_OVERFLOW` | `20` | Max connections beyond pool size |
| `REDIS_URL` | -- | Redis URL (cache/locks; fail-open) |
| `STORAGE_BACKEND` | `local` | Storage backend: `local` or `minio` |
| `STORAGE_MINIO_ENDPOINT` | -- | MinIO endpoint (required when backend=minio) |
| `STORAGE_MINIO_BUCKET` | `html-render-service` | MinIO artifact bucket |
| `STORAGE_MINIO_EXTERNAL_ENDPOINT` | -- | External MinIO host for presigned URLs |
| `WM_URL` | in-cluster | worker_management base URL (LLM gateway) |
| `WM_TIMEOUT_SECONDS` | `600` | Per-message timeout against worker_management |
| `PLAYWRIGHT_HEADLESS` | `true` | Run Chromium headless |
| `PLAYWRIGHT_TIMEOUT_MS` | `30000` | Page load timeout (ms) |
| `PLAYWRIGHT_VIEWPORT_WIDTH` | `1440` | Browser viewport width |
| `PLAYWRIGHT_VIEWPORT_HEIGHT` | `900` | Browser viewport height |
| `PLAYWRIGHT_WAIT_UNTIL` | `networkidle` | Playwright wait strategy |
| `SCREENSHOT_ENABLED` | `true` | Enable screenshot rendering |
| `SCREENSHOT_VIEWPORTS` | `desktop,tablet,mobile` | Viewports to render |
| `REDESIGN_RENDER_MAX_CONCURRENCY` | `2` | Process-wide Chromium concurrency cap (browser pool size) |
| `REDESIGN_BROWSER_MAX_RENDERS` | `200` | Leases before a pooled Chromium is relaunched |
| `REDESIGN_MAX_FULL_PAGE_HEIGHT` | `8000` | Full-page screenshot clip height (px) |
| `REDESIGN_WORKER_POOL_SIZE` | `1` | Per-pod job-worker loops draining the queue |
| `REDESIGN_QUEUE_POLL_INTERVAL_S` | `2.0` | Idle poll interval when queue is empty |
| `REDESIGN_LEASE_SECONDS` | `180` | Job lease TTL (renewed by heartbeat) |
| `REDESIGN_MAX_ATTEMPTS` | `3` | Claim attempts before a job is failed as poison |
| `REDESIGN_MAX_VARIANTS` | `4` | Max A/B style variants per job |
| `CATALOG_INLINE_ASSETS` | `true` | Inline vendored CSS/JS into the artifact |
| `CATALOG_DEFAULT_THEME` | `corporate` | Default DaisyUI theme before brand overrides |
| `CATALOG_MAX_BLOCKS` | `12` | Max section blocks the assembler composes per page |
| `CONTENT_GENERATION_ENABLED` | `true` | Route AI concept images to ai_content_generation_service |

## Architecture

```
src/
├── asgi.py                          # FastAPI app + lifespan (storage → services → browser pool → worker pool)
├── config.py                        # Pydantic settings (API:6319, Storage/MinIO, WM, Playwright, REDESIGN_, Catalog)
├── logging_config.py                # Structured logging setup
├── llm_client.py                    # Legacy LLM client shim
├── llm_credentials.py               # LLM credential resolution
├── api/
│   ├── health.py                    # Liveness/readiness/dependency probes
│   ├── schemas.py                   # Request/response DTOs
│   └── routes/
│       ├── __init__.py              # Composes /api/designs router from submodules
│       ├── jobs.py                  # Enqueue + status + result + cancel + list
│       ├── artifacts.py             # HTML/PDF/screenshot/section/comparison/diff serving
│       ├── variants.py              # A/B variant compare + HTML
│       ├── refinement.py            # Refine-with-feedback + iteration history
│       ├── styles.py                # Custom style CRUD + preview
│       ├── batch.py                 # Multi-page batch jobs
│       ├── webhooks.py              # Webhook delivery log
│       └── _helpers.py              # Shared route helpers
├── auth/
│   ├── middleware.py                # License-token AuthMiddleware
│   └── models.py                    # Auth context models
├── middleware/
│   └── request_id.py                # Request-ID correlation middleware
├── cache/
│   ├── redis_client.py              # Redis connection (fail-open)
│   ├── cache_keys.py                # Cache key builders
│   └── job_lock.py                  # Distributed job locks
├── database/
│   ├── connection.py                # Async engine (psycopg3)
│   ├── session.py                   # Session maker
│   ├── models.py                    # redesign_jobs, redesign_usage, variants, custom_styles, batch_jobs, webhook_delivery_log
│   ├── rls_resolvers.py             # claim_next_render_job SECURITY DEFINER claim fn
│   ├── rls_events.py                # after_begin tenant GUC re-bind
│   ├── rls_policies.py              # RLS child-table policy DDL
│   └── migrations/versions/         # Alembic (0001 initial … 0007 verification_report)
└── services/
    ├── job_worker.py                # Per-pod queue worker pool (claim → render)
    ├── browser_pool.py              # Shared long-lived Chromium pool
    ├── storage_service.py           # Local-scratch → MinIO persist; absolute-URL serving
    ├── html_generator.py            # Freeform + assembly-plan generation via worker_management
    ├── html_validator.py            # Generated-HTML validation
    ├── screenshot_renderer.py       # Multi-viewport Chromium screenshots
    ├── comparison_generator.py      # Comparison/diff image generation (Pillow)
    ├── component_extractor.py       # Component HTML extraction
    ├── concept_image_generator.py   # Optional AI concept images (ai_content_generation)
    ├── accessibility_scorer.py      # Accessibility scoring via worker_management
    ├── accessibility_theme.py       # Accessible theme adjustments
    ├── assembly_planner.py          # Assembly page-plan schema/prompt/parse
    ├── prompt_template.py           # Prompt template loader (web_design v1/v2)
    ├── page_selector.py             # Page selection for multi-page inputs
    ├── context_resolver.py          # Tenant/identity context resolution
    ├── job_manager.py               # Job lifecycle helpers
    ├── batch_manager.py             # Batch fan-out orchestration
    ├── style_manager.py             # Custom style persistence
    ├── cache_service.py             # Result caching (Redis)
    ├── cleanup_service.py           # Retention/cleanup
    ├── webhook_dispatcher.py        # Webhook delivery with retries
    ├── catalog/                     # Assembly catalog: registry, assembler, theme
    ├── design_builder/              # RedesignService: lifecycle, phases, variants, finalize
    └── verification/                # Assembly accessibility verification gate
```

## Design Principles

1. **Enqueue, never block** — the API persists a `pending` job and returns 202; all generation and rendering happens on a detached worker, so slow LLM+Chromium work never holds an HTTP connection open.
2. **Lease-based durability** — jobs are claimed under a renewable lease with `FOR UPDATE SKIP LOCKED`; a crashed pod's orphaned jobs are reclaimed by siblings, making the service horizontally scalable (replicas=2 + HPA) without an external queue.
3. **Compute what can be computed** — assembly mode limits the LLM to selecting catalog blocks and filling slots; layout, markup, theming, and inlined assets are deterministic code, producing self-contained, injection-safe artifacts.
4. **Bounded memory** — the shared browser pool relaunches Chromium after N leases and clips full-page captures at 8000px, containing the two OOM sources (browser ratchet, giant bitmaps) on a memory-limited pod.
5. **Absolute-URL artifact contract** — artifacts persist to MinIO but are always served as absolute http(s) URLs so any replica serves any artifact and cross-namespace consumers fetch with a plain GET.

## Testing

```bash
uv run pytest tests/ -v --cov=src          # All tests with coverage
uv run pytest tests/unit/ -v               # Unit tests
uv run pytest tests/integration/ -v        # Integration tests (require local Postgres)
uv run ruff check src/ tests/              # Lint
uv run ruff format --check src/ tests/     # Format check
uv run ty check                            # Type check
```

Coverage threshold: 70% (enforced in pyproject.toml). `db_*` fixtures require a local Postgres; unit config/health tests run without one.

## CI/CD

`.github/workflows/ci-public.yml` is self-contained and runs on GitHub-hosted runners: a betterleaks secret scan, the `no-private-refs` guard, then `ruff check`, `ruff format --check`, `pytest` against a Postgres 18 service container, and `scripts/pip-audit.sh`.

No image is published from this repository; build it locally with the Dockerfile.

## Docker

```bash
docker build -t html-render-service .
docker run -p 6319:6319 html-render-service
docker run -p 6319:6319 -e LOG_LEVEL=DEBUG -e AUTH_ENABLED=false -e DATABASE_URL=postgresql+psycopg://host/db html-render-service
```

The runtime stage runs `playwright install --with-deps chromium` — the Chromium system libraries are required for screenshot rendering.

## Dependencies

- **worker_management** (port 8887) — sole LLM gateway; workers `web-design` (generation) and `redesign-html` (rework), called via `WMClient` with per-call structured-output schemas
- **ai_content_generation_service** — optional AI concept images (resolved via `w2v_common.discovery`); in turn calls `image-gen.test:9204`
- **w2v_orchestrator** — upstream caller (`website_review` / `web_design` product families)
- **video_compiler** — cross-namespace consumer of screenshot artifacts
- **PostgreSQL 18** — durable claim-queue + job storage via psycopg3, through PgBouncer (`pgbouncer.pgbouncer:6432`) in-cluster
- **Redis** — result cache and distributed job locks (fail-open)
- **MinIO** — shared artifact storage (bucket `html-render-service`) so any replica serves any artifact
- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **Pydantic v2 / pydantic-settings** — request/response validation and settings
- **SQLAlchemy (async) + psycopg3** — ORM and Postgres 18 wire driver
- **Alembic** — schema migrations (initContainer)
- **Playwright** — headless Chromium screenshot/PDF rendering
- **Jinja2** — autoescaping block partial rendering (assembly mode)
- **Pillow** — comparison/diff image generation
- **beautifulsoup4 / lxml** — HTML parsing and extraction
- **miniopy-async** — MinIO client
- **w2v-common** — shared config, auth middleware, clients, RLS, error envelopes

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. Tenant and licence identity is forwarded from the caller. It runs with
no licence switch set; unset means off.
