"""Configuration management for html_render_service using Pydantic Settings.

html_render_service is the contained web-design capability lifted out of
analysis_service's redesign domain: it generates HTML (any scope — hero,
section, component, landing, full_page, redesign) via worker_management and
rasterises it with Playwright. The settings sections below preserve the env
surface the lifted generative/render code reads (``wm``, ``screenshot``,
``playwright``, ``redesign``, ``flux``, ``dalle``, ``content_generation``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from w2v_common.config import (
    APISettings as BaseAPISettings,
)
from w2v_common.config import (
    AuthSettings,
    DatabaseSettings,
    LoggingSettings,
    RedisSettings,
)
from w2v_common.config import (
    CacheSettings as BaseCacheSettings,
)
from w2v_common.config import (
    StorageSettings as BaseStorageSettings,
)


class APISettings(BaseAPISettings):
    """API settings with html_render_service default port (6319)."""

    port: int = Field(default=6319, description="API server port")
    public_base_url: str = Field(
        default="http://html-render-service.w2v:6319",
        description=(
            "Absolute base URL used to build fetchable artifact URLs (screenshots, "
            "comparisons) in job results. Defaults to the in-cluster service URL — "
            "reachable by the orchestrator and by video_compiler cross-namespace. "
            "Override with API_PUBLIC_BASE_URL for local dev."
        ),
    )


class StorageSettings(BaseStorageSettings):
    """Storage settings with html_render_service defaults.

    Artifacts (generated HTML + screenshots) are stored locally by default —
    the service serves them back as absolute URLs via its own
    ``/api/designs/jobs/{id}/artifact/...`` routes (see ``design_builder``),
    so downstream consumers (orchestrator promo_html / website_redesign) get
    fetchable http(s) URLs rather than relative paths.
    """

    minio_bucket: str = Field(default="html-render-service", description="MinIO bucket name")
    minio_external_endpoint: str | None = Field(
        default=None,
        description="External MinIO endpoint for presigned URLs (replaces internal host when set)",
    )


class WorkerManagementSettings(BaseSettings):
    """Worker-management gateway configuration (sole LLM gateway)."""

    url: str | None = Field(
        default=None,
        description="Override for the worker_management base URL (defaults to in-cluster).",
    )
    timeout_seconds: float = Field(
        default=600.0,
        description=(
            "Per-message HTTP timeout against worker_management. Must cover the "
            "slowest worker — a full-page web-design generation (~16k tokens)."
        ),
    )
    max_tokens: int = Field(
        default=16000,
        description=(
            "Per-call output-token budget sent to worker_management. This IS used: "
            "WMClient.message() accepts max_tokens and puts it in the request body, "
            "and WM honours it per call. The previous description claimed "
            "'Unused: WM's MessageRequest forbids max_tokens' — that was FALSE and was "
            "copy-pasted between repos, pushing budgets DOWN. The model behind WM is a "
            "reasoning model whose thinking is charged against this budget; exhausting it "
            "mid-thought yields a terminal (never-retried) 422 'Model token limit exceeded "
            "before any response was generated'. 4000 and 6000 are recorded production 422 "
            "failures. 8000 has no recorded 422, but it is HALF the ~16k this service's "
            "own timeout_seconds documents for a full-page web-design generation, and "
            "web_design_v1 shipped TRUNCATED pages under it — a truncation, not a 422, "
            "which is why it never showed up as a failure. w2v_common's "
            "WM_MIN_OUTPUT_TOKENS (8000) is only "
            "a FLOOR raised when a caller sends less — it applies no ceiling, so this value "
            "reaches worker_management unclamped by the client."
        ),
    )
    output_token_ceiling: int = Field(
        default=12069,
        ge=1000,
        description=(
            "The output budget a call can ACTUALLY get, whatever max_tokens requests. "
            "worker_management clamps a self-hosted provider's max_tokens to a wall-clock "
            "time budget — provider_timeout_s x decode_tokens_per_s x headroom = "
            "1100 x 11.55 x 0.95 = 12069 (worker_management settings "
            "self_hosted_max_tokens_ceiling) — SILENTLY: finish_reason is not on the wire. "
            "Per-section generation caps its request here and tells the model the budget, "
            "so a section is sized to fit instead of being cut off (2026-09-14: section-3 "
            "stopped at 11225 of a requested 16000, mid-CSS)."
        ),
    )
    temperature: float = Field(default=0.2, description="Sampling temperature hint")
    # NO RETRY KNOBS HERE, DELIBERATELY. ``max_retries=1`` / ``retry_delay=2.0``
    # used to sit on this class and were DEAD: ``llm_client.get_wm_client`` builds
    # ``WMClient(timeout=...)`` and passes nothing else, and nothing else read
    # them. Retries belong to ``w2v_common.clients.base`` — ``_MAX_RETRIES = 3``,
    # ``_RETRY_DELAY = 1.0``, backing off ``_RETRY_DELAY * attempt`` — and
    # ``httpx.TimeoutException`` is a subclass of ``httpx.TransportError``, so a
    # read timeout IS replayed. They are deleted rather than wired up because
    # wiring them would silently CHANGE retry behaviour (1 attempt instead of 3),
    # which is not this change's business.
    #
    # This is not pedantry: those two fields are exactly why the batch reaper's
    # window was first derived as 600*2+2 = 1202s instead of the true 600*3+3 =
    # 1803s. Silently-ignored knobs are how the next person derives another wrong
    # number. Read the real values from w2v_common, never from here.

    model_config = SettingsConfigDict(
        env_prefix="WM_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class CacheSettings(BaseCacheSettings):
    """Result caching configuration with design-specific TTL."""

    design_ttl: int = Field(default=3600, description="Design job result cache TTL in seconds")
    ttl: int = Field(default=3600, description="Default cache TTL in seconds")


class PlaywrightSettings(BaseSettings):
    """Playwright browser rendering configuration."""

    headless: bool = Field(default=True, description="Run browser in headless mode")
    timeout_ms: int = Field(default=30000, description="Page load timeout in milliseconds")

    # NOTE: there is deliberately no ``viewport_width``/``viewport_height`` here.
    # Both existed (defaulting to 1440x900) and were read by NOTHING: the renderer
    # sizes every page from ``screenshot_renderer.VIEWPORT_SIZES``, and no
    # ``PLAYWRIGHT_*`` key is defined in gitops either, so the pair was pure
    # misdirection — during the A/V soft-picture investigation they read exactly
    # like the live capture size. Deleted rather than wired up, because a single
    # scalar pair cannot express the thing the renderer actually needs: a NAMED
    # map of three viewports (desktop/tablet/mobile) whose tablet and mobile
    # entries are fixed real-device CSS sizes and must not be operator-tunable.
    # If desktop ever needs to be tunable, add one ``desktop_viewport`` override
    # that feeds ``VIEWPORT_SIZES``, and make the renderer read it.
    wait_until: Literal["commit", "domcontentloaded", "load", "networkidle"] = Field(
        default="networkidle", description="Playwright wait_until strategy"
    )

    model_config = SettingsConfigDict(
        env_prefix="PLAYWRIGHT_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class ScreenshotSettings(BaseSettings):
    """Screenshot rendering configuration."""

    enabled: bool = Field(default=True, description="Enable screenshot rendering")
    viewports: list[str] = Field(
        default=["desktop", "tablet", "mobile"],
        description="Viewport sizes to render",
    )
    timeout_ms: int = Field(default=30000, description="Page load timeout in milliseconds")

    @field_validator("viewports")
    @classmethod
    def _known_viewports(cls, v: list[str]) -> list[str]:
        """This is the list the renderer actually receives (``settings.screenshot``)."""
        from src.services.screenshot_renderer import validate_viewport_names

        return validate_viewport_names(v) or []

    model_config = SettingsConfigDict(
        env_prefix="SCREENSHOT_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class DalleSettings(BaseSettings):
    """DALL-E concept image generation configuration (legacy; ACG preferred)."""

    enabled: bool = Field(default=False, description="Enable AI concept image generation")
    api_key: str = Field(default="", description="OpenAI API key for DALL-E")
    model: str = Field(default="dall-e-3", description="DALL-E model")
    size: str = Field(default="1792x1024", description="Image size")
    quality: str = Field(default="hd", description="Image quality")
    count: int = Field(default=2, description="Number of concept images to generate")

    model_config = SettingsConfigDict(
        env_prefix="DALLE_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class ContentGenerationSettings(BaseSettings):
    """Downstream ai_content_generation service config (AI concept images).

    The peer URL is resolved via :func:`w2v_common.discovery.service_url`
    using the canonical ``ai_content_generation`` logical name.
    """

    enabled: bool = Field(default=True, description="Enable downstream content generation")
    timeout: float = Field(default=30.0, description="HTTP request timeout")
    poll_interval_s: float = Field(default=2.0, description="Polling interval")
    poll_timeout_s: float = Field(default=180.0, description="Max time to wait for image jobs")
    default_provider: str | None = Field(default=None, description="Preferred image provider")

    model_config = SettingsConfigDict(
        env_prefix="CONTENT_GENERATION_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class FluxSettings(BaseSettings):
    """FLUX local image generation configuration (off by default)."""

    enabled: bool = Field(default=False, description="Enable local FLUX image generation")
    model_id: str = Field(
        default="black-forest-labs/FLUX.2-klein-4B", description="HuggingFace model ID"
    )
    device: str = Field(default="auto", description="Device: auto, cuda, mps, or cpu")
    num_inference_steps: int = Field(default=4, description="Number of diffusion steps")
    guidance_scale: float = Field(default=3.5, description="Classifier-free guidance scale")
    width: int = Field(default=1024, description="Image width in pixels")
    height: int = Field(default=768, description="Image height in pixels")
    cpu_offload: bool = Field(default=False, description="Enable CPU offloading for low VRAM")
    remote_url: str = Field(default="", description="URL of external FLUX inference server")

    model_config = SettingsConfigDict(
        env_prefix="FLUX_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class WebhookSettings(BaseSettings):
    """Webhook notification configuration."""

    enabled: bool = Field(default=True, description="Enable webhook notifications")
    max_retries: int = Field(default=3, description="Maximum webhook delivery retries")
    timeout: int = Field(default=10, description="Webhook delivery timeout in seconds")

    model_config = SettingsConfigDict(
        env_prefix="WEBHOOK_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class RetentionSettings(BaseSettings):
    """Data retention configuration."""

    enabled: bool = Field(default=False, description="Enable data retention cleanup")
    job_retention_days: int = Field(default=90, description="Days to keep completed/failed jobs")
    cleanup_interval_hours: int = Field(default=24, description="Hours between cleanup runs")

    model_config = SettingsConfigDict(
        env_prefix="RETENTION_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class DesignSettings(BaseSettings):
    """Web-design domain settings (formerly RedesignSettings).

    Env prefix kept as ``REDESIGN_`` so existing deploy env carries over
    during the analysis_service → html_render_service cutover.
    """

    scope: str = Field(default="full_page", description="Default scope")
    style: str = Field(default="modern", description="Default design style")
    max_pages: int = Field(default=5, description="Maximum pages to design per request")
    max_html_chars: int = Field(
        default=100000, description="Max page-HTML chars sent to the design LLM (~25k tokens)."
    )

    # Custom styles
    max_custom_styles_per_tenant: int = Field(
        default=20, description="Max custom styles per tenant"
    )
    preview_max_tokens: int = Field(default=8192, description="Max tokens for style preview")

    # A/B variants
    max_variants: int = Field(default=4, description="Maximum style variants per job")
    variant_concurrency: int = Field(default=3, description="Max concurrent variant processing")

    # Iteration/refinement
    max_iterations: int = Field(
        default=10, description="Maximum refinement iterations per root job"
    )
    max_feedback_chars: int = Field(default=2000, description="Maximum feedback text length")

    # Batch processing
    batch_max_concurrency: int = Field(default=3, description="Max concurrent pages in a batch")

    # Generation bound for ONE job (per-section drafts, retries, rendered repair).
    #
    # DERIVED from the caller, not guessed (w2v_orchestrator origin/main,
    # blueprints/website_review.py `website_redesign` + stages/website_review/redesign.py).
    # ONE attempt of the stage that waits on this job is default_timeout_seconds=7200,
    # and it is spent as: start 60 + orchestrator claim queue 1810 + wait-for-job
    # (1+7) x (590+10) + backoff 159 = 4959 + result 120. So this job must finish
    # within BOTH, with >= 15% headroom:
    #   * the wait ladder:   job <= 0.85 x 4959               = 4215s
    #   * the whole attempt: job <= 0.85 x 7200 - 60-1810-120 = 4130s
    # The job spends time outside generation: before it (webhook + page fetch + theme,
    # ~60s) and after it (screenshots, concept images, finalize — ~500s, see
    # batch_no_progress_seconds' derivation). 60 + 3500 + 500 = 4060s:
    #   4060 / 4959 = 81.9% (18.1% headroom); (60+1810+4060+120) / 7200 = 84.0% (16.0%).
    # Every LLM call and page probe runs under the REMAINING time, so the deadline is a
    # hard bound, not a check between calls. Past it, no retry or repair starts;
    # sections not yet drafted are dropped and the job completes with the degradations
    # recorded (html_validation_report.quality_warnings).
    generation_deadline_seconds: int = Field(
        default=3500,
        ge=60,
        description="Wall-clock budget for a job's HTML generation + rendered repair",
    )
    # Retries and repair regenerations a job may spend BEYOND one first draft per
    # section. First drafts are never refused by this (a page needs every section);
    # only extra calls are. Worst case LLM calls per job = sections + this.
    max_extra_llm_calls: int = Field(
        default=12,
        ge=0,
        description="LLM calls a job may spend on section retries/repairs beyond first drafts",
    )

    # Process-wide Chromium concurrency cap. Bounds the number of browser
    # instances alive at once across ALL jobs/variants/viewports so concurrent
    # fan-out can't exceed the pod memory limit (the OOMKill root cause).
    render_max_concurrency: int = Field(
        default=2, description="Max concurrent Chromium render instances (process-wide)"
    )
    # Relaunch a pooled browser after this many leases to bound the long-lived
    # Chromium memory ratchet (Phase 3 browser pool).
    browser_max_renders: int = Field(
        default=200, description="Leases before a pooled Chromium is relaunched"
    )

    # Hard cap on full-page screenshot height. A pathologically tall marketing
    # SPA (10k-20k px) captured with full_page=True makes Chromium allocate the
    # entire scroll-height bitmap and write a giant PNG — the real remaining OOM
    # source on the heavy path (the PIL down-scale only applies AFTER that PNG
    # exists). When the page is taller than this, we clip the screenshot to this
    # height instead of capturing the full scroll height.
    max_full_page_height: int = Field(
        default=8000, description="Max full-page screenshot height in px before clipping"
    )

    # Durable claim-queue (Phase 1). Jobs are inserted as `pending` rows and
    # picked up by an in-process pool of worker loops that claim via
    # FOR UPDATE SKIP LOCKED, so a job survives a pod restart (its expired lease
    # is reclaimed) instead of being lost with the in-process task.
    worker_pool_size: int = Field(
        default=1, description="Per-pod concurrent job-worker loops draining the queue"
    )
    queue_poll_interval_s: float = Field(
        default=2.0, description="Idle poll interval when the queue is empty"
    )
    lease_seconds: int = Field(
        default=180, description="Job lease TTL; renewed by a heartbeat while running"
    )
    max_attempts: int = Field(
        default=3, description="Max claim attempts before a job is failed as poison"
    )
    # A graceful handoff (rollout / HPA scale-down) refunds its attempt, so churn
    # can no longer poison a healthy job — but that also removes the only bound on
    # how many times a job may be re-claimed, and each re-claim replays the whole
    # render (a 16k-token WM call). This wall-clock ceiling is that bound. Measured
    # from first claim, so a backlogged (never-started) job is never aged out.
    max_job_lifetime_seconds: int = Field(
        default=3600,
        description="Max wall-clock a job may spend running before it is failed outright",
    )
    # The batch arm of the same sweep. A batch is NOT queue-owned — it has no
    # lease, no claimer, and its orchestration lives in an in-process asyncio
    # task — so a dead pod strands the row forever.
    #
    # A NO-PROGRESS WINDOW, not a wall-clock ceiling on the batch: how long a
    # batch may go without a single page completing or failing before it is
    # declared abandoned. `batch_jobs.last_progress_at` is the clock, bumped by
    # `increment_completed`/`increment_failed`.
    #
    # It is a no-progress window because a wall-clock ceiling could not be set
    # safely. `started_at` is written once and never again, so a ceiling built on
    # it has to exceed the longest legitimate batch: `BatchRedesignRequest.pages`
    # allows 20 pages, `batch_max_concurrency` is 3, so 7 sequential waves — and
    # `max_job_lifetime_seconds` does NOT bound an in-process batch child, only
    # the queue path, so there is no per-page cap to multiply. 7 waves of
    # unbounded page renders has no honest ceiling.
    #
    # THE NUMBER IS DERIVED, and this is the derivation. It must exceed the
    # longest gap between two consecutive progress signals, where the clock is
    # bumped after every SECTION (`batch_progress.report_progress`, called from
    # `generate_sections`).
    #
    # ONE WM CALL (the inter-section gap):
    #     w2v_common `_MAX_RETRIES` = 3 attempts x `wm.timeout_seconds` 600s
    #     + backoff `_RETRY_DELAY * attempt` = 1.0 + 2.0
    #     = 1803s
    # NOT 1202s. An earlier revision computed 600*2+2 from `wm.max_retries=1` /
    # `wm.retry_delay=2.0` — two fields that were never read by anything and have
    # since been deleted (see WorkerManagementSettings). Retries live in
    # `w2v_common.clients.base`, and `httpx.TimeoutException` is a
    # `httpx.TransportError`, so a 600s read timeout is genuinely replayed 3x.
    #
    # THE LONGEST GAP IS NOT AN INTER-SECTION GAP, it is the one BEFORE the first
    # section, from the reporter being bound (`lifecycle.process_page`) to the
    # first `report_progress` (`generator.generate_sections`):
    #     _dispatch_webhook("job.started")        <= ~60s
    #   + _fetch_page_html: pool.lease() wait + goto 30s (`screenshot_timeout_ms`)
    #   + first _call_claude                       1803s
    #     ~= 1893s
    # The tail (last section -> `increment_completed`) is far shorter, ~500s:
    # 2 viewports x 30s x 2 captures + PIL, concept images at `poll_timeout_s`
    # 180s, plus the finalize webhook.
    #
    # So the bound to clear is ~1893s. 3600 is ~1.9x it. The previous 2400 did
    # clear it, but only by 1.27x while its comment claimed "~2x" — the margin
    # was smaller than it read, on a number that was itself wrong.
    #
    # An even earlier 7200 was a GUESS. It assumed the gap was one PAGE, but a
    # page is unbounded: `generate_sections` makes one sequential LLM call per DOM
    # landmark from an uncapped `detect_sections`, so 13 landmarks is 13 * 1803s
    # of healthy work with no signal — a live page, reaped. Instrumenting per
    # section is what made a derived number possible at all.
    #
    # The resolver additionally clamps anything below its own floor — config is a
    # preference, not the safety bound. 0 disables the reaper entirely.
    batch_no_progress_seconds: int = Field(
        default=3600,
        description="Seconds a batch may make no section progress before it is reaped (0=off)",
    )
    # How often a worker loop runs the batch sweep. Independent of the ~2s claim
    # poll: this is a whole-table sweep, not a single-row claim, and a stranded
    # batch has already been stranded for hours by the time it qualifies.
    batch_reap_interval_s: float = Field(
        default=300.0, description="Seconds between stale-batch sweeps per worker loop"
    )
    # Graceful drain. On SIGTERM the pool stops claiming NEW jobs and lets the
    # in-flight render finish, instead of cancelling it and throwing away a paid
    # 16k-token WM call that the next claim then has to redo.
    #
    # MUST stay below the pod's terminationGracePeriodSeconds. If k8s SIGKILLs us
    # mid-drain, `release_claims` never runs, the lease just lapses, and the attempt
    # is NOT refunded — strictly worse than not draining at all. The default is
    # therefore safe under k8s's default 30s grace; raise BOTH together in gitops
    # (terminationGracePeriodSeconds + REDESIGN_DRAIN_TIMEOUT_SECONDS) to actually
    # let a minutes-long render survive a rollout.
    drain_timeout_seconds: int = Field(
        default=25,
        description="On shutdown, how long to let an in-flight render finish before cancelling",
    )

    # Direct PDF routes (GET /jobs/{id}/pdf, POST /render/pdf) lease a browser
    # straight from the request handler, outside JobWorkerPool's queue — so the
    # pool's own drain (above) never sees that lease and BrowserPool.stop() used
    # to close it out from under an in-flight page.pdf()/page.goto(). This bounds
    # how long BrowserPool.stop() waits for outstanding leases to be released
    # before closing anyway. It is deliberately short: in this deployment
    # JobWorkerPool.stop() runs first in the ASGI shutdown sequence and can
    # itself consume up to REDESIGN_DRAIN_TIMEOUT_SECONDS (540s in gitops prod)
    # of the pod's 600s terminationGracePeriodSeconds, leaving roughly 60s of
    # slack before a SIGKILL — this default must stay comfortably inside that
    # remainder, not the full grace period.
    browser_drain_timeout_seconds: int = Field(
        default=30,
        description="On shutdown, how long to wait for outstanding browser leases before closing",
    )

    # Screenshot rendering (mirrored onto ScreenshotSettings at the section level)
    screenshot_enabled: bool = Field(default=True, description="Enable screenshot rendering")
    screenshot_viewports: list[str] = Field(
        default=["desktop", "tablet", "mobile"], description="Viewport sizes to render"
    )
    screenshot_timeout_ms: int = Field(default=30000, description="Page load timeout in ms")

    @field_validator("screenshot_viewports")
    @classmethod
    def _known_viewports(cls, v: list[str]) -> list[str]:
        """A typo in REDESIGN_SCREENSHOT_VIEWPORTS silently rendered zero screenshots."""
        from src.services.screenshot_renderer import validate_viewport_names

        return validate_viewport_names(v) or []

    # Templates
    template_version: str = Field(default="v1", description="Template version to use")
    template_directory: str | None = Field(
        default=None, description="Override template directory path"
    )

    model_config = SettingsConfigDict(
        env_prefix="REDESIGN_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class CatalogSettings(BaseSettings):
    """Computed-assembly web-design settings (component catalog + vendored assets).

    Assembly mode composes a page from a fixed catalog of vetted open-source
    section blocks (Tailwind + DaisyUI + Flowbite/HyperUI) instead of asking the
    LLM to hand-write CSS. Vendored runtime assets (compiled CSS, htmx, Alpine,
    Lucide) are **inlined** into the generated artifact by default so the result
    is fully self-contained — it renders identically under Playwright, in a
    browser live-preview iframe, and as a download, with no asset-reachability or
    external-CSP concerns. The only external references a self-contained artifact
    carries are sandboxed ``embeds`` (the three.js seam).
    """

    # Directory holding vendored, pinned OSS assets (compiled app.css + JS libs).
    # Defaults to ``src/static/vendor`` inside the package.
    vendor_dir: str | None = Field(
        default=None,
        description="Override path to the vendored asset directory (default: src/static/vendor).",
    )
    inline_assets: bool = Field(
        default=True,
        description=(
            "Inline vendored CSS/JS into the generated HTML (fully self-contained). "
            "When False, assets are referenced from PUBLIC_BASE_URL/static/vendor."
        ),
    )
    default_component_set: str = Field(
        default="daisyui",
        description="Default component set / library variant used by assembly mode.",
    )
    default_theme: str = Field(
        default="corporate",
        description="Default DaisyUI theme name applied before brand-token overrides.",
    )
    # Extra CDN origins the generated HTML may reference. Default empty = fully
    # self-contained (only inlined vendored assets + sandboxed embeds). Populated
    # only when a deployment explicitly opts a specific external CDN in.
    cdn_allowlist: list[str] = Field(
        default_factory=list,
        description="Additional external CDN origins permitted in assembled HTML.",
    )
    # Hosts an assembled page may load <img> from, besides the images the service
    # inlines itself (the generated hero, as an exact data: URI). MinIO is NOT
    # trusted: http, tailnet-only, expiring. Default empty: an LLM-written image
    # URL on any other host is dropped and the block renders its designed CSS fallback, so a
    # page never depends on an unverified hotlink (live run 01a156a6: an Unsplash
    # hero blocked by net::ERR_BLOCKED_BY_ORB shipped as a missing hero).
    image_host_allowlist: list[str] = Field(
        default_factory=list,
        description="Extra image hosts (host or host:port) assembled pages may hotlink.",
    )
    max_blocks: int = Field(
        default=12, description="Max section blocks the assembler will compose per page."
    )
    # Split the assembly page-plan into many small LLM calls (1 structure call to
    # pick block types + one small fill call per block) instead of a single large
    # one-shot draw. Sized for a small, slow vLLM that a big multi-block one-shot
    # overwhelms. OFF by default: the one-shot path stays byte-identical until this
    # is flipped in GitOps. See w2v-vllm-single-small-device-strategy.
    split_assembly: bool = Field(
        default=False,
        description="Plan assembly blocks via per-block LLM calls instead of one one-shot draw.",
    )

    # Automated verification lane (assembly-mode completion gate).
    verify_enabled: bool = Field(
        default=True, description="Run the verification gate on assembled designs."
    )
    verify_max_refine: int = Field(
        default=2, description="Max refine iterations before failing loud with findings."
    )
    verify_a11y_max_serious: int = Field(
        default=0, description="Max serious/critical axe violations allowed to pass."
    )

    model_config = SettingsConfigDict(
        env_prefix="CATALOG_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""

    app_name: str = Field(default="HTML Render Service", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Environment")

    # Shared infrastructure
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    wm: WorkerManagementSettings = Field(default_factory=WorkerManagementSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    webhook: WebhookSettings = Field(default_factory=WebhookSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)

    # Domain rendering settings (read directly by lifted code as settings.X.*)
    playwright: PlaywrightSettings = Field(default_factory=PlaywrightSettings)
    screenshot: ScreenshotSettings = Field(default_factory=ScreenshotSettings)
    dalle: DalleSettings = Field(default_factory=DalleSettings)
    content_generation: ContentGenerationSettings = Field(default_factory=ContentGenerationSettings)
    flux: FluxSettings = Field(default_factory=FluxSettings)

    # Web-design domain settings (formerly `redesign`)
    redesign: DesignSettings = Field(default_factory=DesignSettings)

    # Computed-assembly settings (component catalog + vendored assets)
    catalog: CatalogSettings = Field(default_factory=CatalogSettings)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    def get_database_url(self) -> str:
        """Return the configured PostgreSQL database URL."""
        return self.database.url

    def get_redis_url(self) -> str:
        """Return the configured Redis connection URL."""
        return self.redis.url

    def is_production(self) -> bool:
        """Return True if the environment is production."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Return True if the environment is development."""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the cached settings instance (useful for testing)."""
    get_settings.cache_clear()
