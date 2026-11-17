"""In-process durable job-queue worker pool (Phase 1).

Replaces the request-time ``asyncio.create_task`` model. ``create_redesign`` /
``refine`` now enqueue a ``pending`` row carrying a ``job_options`` bundle, and a
pool of worker loops claims rows via ``FOR UPDATE SKIP LOCKED``
(:meth:`JobManager.claim_next_job`), runs the job under a lease renewed by a
heartbeat, and lets the existing background processors mark it terminal. A job
whose pod dies mid-render leaves an expired lease that any worker reclaims on its
next poll — so jobs survive restarts instead of dying with the in-process task.

The ``job_options`` bundle is the durable substitute for the per-request closure
the old model captured: everything needed to reconstruct the dispatch is stored
on the row, so a different pod (or the same pod after a restart) can resume it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import time
from typing import TYPE_CHECKING, Any, NamedTuple

from w2v_common.tenant import TenantSource

from src.config import get_settings
from src.services.job_manager import ClaimedJob, JobManager, claim_owner
from src.tenant import require_tenant

if TYPE_CHECKING:
    from fastapi import FastAPI

    from src.api.schemas import LLMConfigOverride
    from src.services.design_builder.types import RenderOptions

logger = logging.getLogger(__name__)


class WorkerPoolStatus(NamedTuple):
    """A non-mutating snapshot of the worker pool, for /health.

    ``alive`` is the count of worker tasks that have NOT finished. It is the only
    honest signal here: ``_worker_loop`` swallows per-iteration errors, so a task
    that has *ended* ended for a reason the loop could not absorb — cancellation,
    or a raise from outside its ``try``. Every ended task is one fewer claimer on
    the ``FOR UPDATE SKIP LOCKED`` queue, and ``alive == 0`` means this pod has
    stopped draining the queue entirely while still answering every HTTP probe.
    """

    started: bool
    workers: int
    alive: int


KIND_REDESIGN = "redesign"
# How long a dispatch cancelled at the lifetime ceiling gets to unwind before the worker
# stops waiting on it (a task that swallows the cancellation is then abandoned).
CANCEL_GRACE_SECONDS = 30.0
KIND_REFINEMENT = "refinement"


# ---------------------------------------------------------------------------
# job_options (de)serialization
# ---------------------------------------------------------------------------


def _render_options_to_dict(ro: RenderOptions | None) -> dict | None:
    if ro is None:
        return None
    return {
        "capture_original": ro.capture_original,
        "generate_comparison": ro.generate_comparison,
        "viewports": list(ro.viewports) if ro.viewports is not None else None,
        "full_page": ro.full_page,
        "section_screenshots": ro.section_screenshots,
    }


def _render_options_from_dict(d: dict | None) -> RenderOptions | None:
    if not d:
        return None
    from src.services.design_builder.types import RenderOptions

    vp = d.get("viewports")
    return RenderOptions(
        capture_original=d.get("capture_original", True),
        generate_comparison=d.get("generate_comparison", True),
        viewports=tuple(vp) if vp is not None else None,
        full_page=d.get("full_page", True),
        section_screenshots=d.get("section_screenshots", False),
    )


def _llm_config_from_dict(d: dict | None) -> LLMConfigOverride | None:
    if not d:
        return None
    from src.api.schemas import LLMConfigOverride

    return LLMConfigOverride(**d)


def _capture_identity() -> dict:
    """Snapshot the caller's audit identity from the request-scoped ContextVars.

    Called at enqueue time (inside the request handler, where AuthMiddleware has
    populated the identity context). The old request-time ``asyncio.create_task``
    inherited these ContextVars automatically; the detached worker loop does not,
    so we persist them on the job and re-establish them before dispatch. (tenant_id
    is already a job column; license/subject are audit headers stamped on outbound
    LLM calls — WM auth itself uses the service-account DOWNSTREAM_LICENSE_TOKEN.)
    """
    from w2v_common.auth import (
        get_correlation_id_from_context,
        get_license_id_from_context,
        get_project_id_from_context,
        get_stage_from_context,
        get_subject_id_from_context,
        get_tenant_id_from_context,
        get_tenant_source_from_context,
        get_w2v_job_id_from_context,
        get_w2v_project_id_from_context,
    )
    from w2v_common.auth.identity_context import resolve_actor_id

    # ``get_tenant_id_from_context()`` returns ``uuid.UUID | None`` since
    # w2v_common 0.67.0. This bundle is persisted into ``job_options`` (a jsonb
    # column), and a bare UUID is not JSON-serialisable — so it crosses as text
    # here and is re-parsed by ``require_tenant`` in ``_run_claimed``. ``None``
    # (no request identity: a CronJob, a startup task) stays absent rather than
    # becoming a placeholder; the job row's own ``tenant_id`` is the primary
    # source and this snapshot is only its audit fallback.
    context_tenant = get_tenant_id_from_context()

    # TENANT PROVENANCE IS PART OF THE IDENTITY, not decoration, and this is the
    # ONLY place it can be read. Every LLM call this service makes goes through
    # WMClient, which bakes ``X-Tenant-Id: Bearer <DOWNSTREAM_LICENSE_TOKEN>``,
    # and worker_management is a metered destination — so each one is a
    # DELEGATED hop. w2v_common 0.67.0's ``_apply_outbound_tenant`` refuses to
    # delegate (``UnvouchedTenantError``) unless ``tenant_source`` is in
    # VOUCHED_TENANT_SOURCES. The detached worker loop inherits no ContextVars,
    # so if the source is not snapshotted HERE (in request scope, where
    # AuthMiddleware set it) the worker re-establishes a tenant with provenance
    # ``None`` and EVERY background render dies at the WM call — or, on a lane
    # that tolerated it, ships without X-Tenant-Id and mis-attributes the cost.
    # This is not a type error and would have shipped green.
    context_source = get_tenant_source_from_context()

    return {
        "tenant_id": str(context_tenant) if context_tenant is not None else "",
        # Stored as the enum's VALUE: job_options is jsonb, and a TenantSource
        # is not JSON-serialisable. Re-hydrated in ``_run_claimed``.
        "tenant_source": context_source.value if context_source is not None else "",
        "license_id": get_license_id_from_context(),
        "subject_id": get_subject_id_from_context(),
        # Cost-correlation (run_id + DAG stage) stamped by the orchestrator's
        # X-Correlation-Id/X-Stage. The detached worker re-establishes these so
        # the WM LLM call it makes carries them through to the billing cost_event.
        "correlation_id": get_correlation_id_from_context(),
        "stage": get_stage_from_context(),
        # ATTRIBUTION (w2v-common 0.88). The job runs later, outside the request,
        # where w2v_common would treat it as SERVICE-originated and stamp this
        # service's own SERVICE_ACTOR_ID. It is not: a caller asked for it. So the
        # caller's actor (canonical UUID, "" when the caller sent none) and the
        # project/job it worked for are persisted here and re-bound at dispatch
        # by ``_bind_job_attribution``.
        "actor_id": resolve_actor_id()[0] or "",
        "project_id": get_project_id_from_context(),
        "w2v_project_id": get_w2v_project_id_from_context(),
        "w2v_job_id": get_w2v_job_id_from_context(),
    }


def _bind_job_attribution(ident: dict) -> None:
    """Re-bind the enqueue-time actor / project / job for a detached dispatch.

    Call AFTER ``set_identity_context`` (which clears the actor and project).

    A queued job is CALLER-originated work, so the context is marked inbound:
    a job carrying a stored actor bills that actor; a job WITHOUT one (a legacy
    row enqueued before the snapshot existed, or a caller that sent no actor)
    stays actor-less and is counted by w2v_common as ``inbound_without_actor``.
    It must NOT fall back to this service's SERVICE_ACTOR_ID — that would hide
    the caller's gap from billing coverage.

    Values go through w2v_common's public ``bind_attribution_from_headers``
    (trusted: they were captured from this service's own validated request
    context), so a malformed persisted value is ignored, never coerced.
    """
    from w2v_common.auth import (
        ACTOR_ID_HEADER,
        PROJECT_ID_HEADER,
        W2V_JOB_ID_HEADER,
        W2V_PROJECT_ID_HEADER,
        bind_attribution_from_headers,
    )
    from w2v_common.auth.identity_context import mark_inbound_request

    mark_inbound_request()
    headers = {
        ACTOR_ID_HEADER: ident.get("actor_id") or "",
        PROJECT_ID_HEADER: ident.get("project_id") or "",
        W2V_PROJECT_ID_HEADER: ident.get("w2v_project_id") or "",
        W2V_JOB_ID_HEADER: ident.get("w2v_job_id") or "",
    }
    bind_attribution_from_headers({k: str(v) for k, v in headers.items() if v}, trusted=True)
    if not headers[ACTOR_ID_HEADER]:
        logger.info(
            "Queued job has no stored caller actor; dispatching actor-less "
            "(counted as inbound_without_actor, no SERVICE_ACTOR_ID fallback)"
        )


def build_redesign_options(
    *,
    run_accessibility_check: bool,
    auto_fix_accessibility: bool,
    accessibility_threshold: float,
    accessibility_first_auto_theme: bool,
    generate_screenshots: bool,
    generate_concept_images: bool,
    render_options: RenderOptions | None,
    llm_config: LLMConfigOverride | None,
) -> dict:
    """Build the persisted options bundle for a ``create_redesign`` job."""
    return {
        "kind": KIND_REDESIGN,
        "run_accessibility_check": run_accessibility_check,
        "auto_fix_accessibility": auto_fix_accessibility,
        "accessibility_threshold": accessibility_threshold,
        "accessibility_first_auto_theme": accessibility_first_auto_theme,
        "generate_screenshots": generate_screenshots,
        "generate_concept_images": generate_concept_images,
        "render_options": _render_options_to_dict(render_options),
        "llm_config": llm_config.model_dump() if llm_config else None,
        "identity": _capture_identity(),
    }


def build_refinement_options(
    *,
    generate_screenshots: bool,
    generate_concept_images: bool,
    focus_areas: list[str] | None,
    llm_config: LLMConfigOverride | None,
) -> dict:
    """Build the persisted options bundle for a ``refine`` job."""
    return {
        "kind": KIND_REFINEMENT,
        "generate_screenshots": generate_screenshots,
        "generate_concept_images": generate_concept_images,
        "focus_areas": list(focus_areas) if focus_areas else None,
        "llm_config": llm_config.model_dump() if llm_config else None,
        "identity": _capture_identity(),
    }


# ---------------------------------------------------------------------------
# dispatch + lease heartbeat
# ---------------------------------------------------------------------------


async def _dispatch(app: Any, claimed: ClaimedJob) -> None:
    """Reconstruct and run the job from its persisted options bundle.

    Reuses the existing background processors, which own the
    completed/failed terminal-state writes (including per-job mark_failed on
    exception), so the worker just routes by ``kind``.
    """
    from src.api.routes._helpers import (
        _process_job_background,
        _process_refinement_background,
    )

    service = app.state.redesign_service
    session_factory = app.state.session_factory
    opts = claimed.options
    kind = opts.get("kind", KIND_REDESIGN)
    llm_config = _llm_config_from_dict(opts.get("llm_config"))

    if kind == KIND_REFINEMENT:
        await _process_refinement_background(
            service,
            session_factory,
            claimed.job_id,
            claimed.tenant_id,
            generate_screenshots=opts.get("generate_screenshots", True),
            generate_concept_images=opts.get("generate_concept_images", False),
            focus_areas=opts.get("focus_areas"),
            llm_config=llm_config,
        )
        return

    await _process_job_background(
        service,
        session_factory,
        claimed.job_id,
        claimed.tenant_id,
        run_accessibility_check=opts.get("run_accessibility_check", False),
        auto_fix_accessibility=opts.get("auto_fix_accessibility", False),
        accessibility_threshold=opts.get("accessibility_threshold", 70.0),
        accessibility_first_auto_theme=opts.get("accessibility_first_auto_theme", False),
        generate_screenshots=opts.get("generate_screenshots", True),
        generate_concept_images=opts.get("generate_concept_images", False),
        render_options=_render_options_from_dict(opts.get("render_options")),
        llm_config=llm_config,
    )


async def _heartbeat(
    session_factory: Any,
    job_id: str,
    lease_seconds: int,
    stop: asyncio.Event,
    max_lifetime_seconds: int = 0,
) -> None:
    """Renew the job's lease every ``lease_seconds/3`` until ``stop`` is set.

    STOPS RENEWING once this run has been going longer than
    ``max_lifetime_seconds``, and says so loudly.

    Why that bound has to live HERE and not only in ``claim_next_job``: the
    lifetime ceiling is evaluated on a RE-claim, i.e. only for a job whose lease
    was allowed to lapse. A job that hangs inside ``_dispatch`` — an await that
    never returns, with no timeout anywhere up the chain — kept its lease renewed
    forever by this loop, so it was never re-claimable, so the ceiling was never
    reached. With ``worker_pool_size`` defaulting to 1 that wedges the pod's
    entire queue permanently, and it does it SILENTLY: no exception, no ERROR
    line, ``/health`` green (liveness was deliberately rejected, see health.py),
    and no reaper/sweeper exists to notice. The heartbeat was asserting liveness
    it could not observe.

    The fix is deliberately NOT a cancellation: we do not kill the task and we do
    not throw away a paid 16k-token WM call that might still land. We just stop
    lying. The lease then lapses on its own, the next claim hits the EXISTING
    ceiling in ``claim_next_job``, and that writes the real failure reason.

    Elapsed is measured monotonically from the start of THIS run, not from the
    row's ``started_at``, so it is a lower bound on the job's total running life
    across attempts: if this run alone exceeds the cap, the total certainly does.
    Erring short would be wrong (a premature lapse lets a sibling double-render);
    erring long is safe, because the authoritative check re-measures from
    ``started_at`` before it fails anything.
    """
    interval = max(5, lease_seconds // 3)
    run_started = time.monotonic()
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return  # stop set → job finished
        except TimeoutError:
            pass
        elapsed = time.monotonic() - run_started
        if 0 < max_lifetime_seconds < elapsed:
            logger.error(
                "Job %s has been running %.0fs (> max lifetime %ds) and is still not "
                "finished — ceasing lease renewal so the lease lapses and the job "
                "becomes re-claimable; claim_next_job will fail it with the real reason. "
                "The in-flight task is NOT cancelled.",
                job_id,
                elapsed,
                max_lifetime_seconds,
            )
            return
        try:
            async with session_factory() as db, db.begin():
                await JobManager.renew_lease(db, job_id, lease_seconds)
        except Exception:  # best-effort; a missed beat just shortens the lease
            logger.warning("Lease renew failed for job %s", job_id, exc_info=True)


def _restore_tenant_source(value: object) -> TenantSource | None:
    """Re-hydrate the enqueue-time ``TenantSource`` from the persisted job options.

    Returns ``None`` — never a guessed default — when the job predates the
    snapshot or carries an unrecognised value. ``None`` is the honest answer and
    it fails CLOSED: ``_apply_outbound_tenant`` refuses to delegate an unvouched
    tenant, so an old job raises ``UnvouchedTenantError`` at its first WM call
    instead of silently spending an arbitrary tenant's credentials on provenance
    this process cannot actually vouch for. Substituting ``LICENSE_TOKEN`` here
    would be forging the vouch.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return TenantSource(value)
    except ValueError:
        # Not a MissingTenantError concern: TenantSource is a plain StrEnum, so
        # this ValueError is genuinely "unknown member", not a swallowed tenant
        # refusal. Degrade to unvouched rather than raise — the delegation gate
        # downstream is the thing that must refuse, and it will.
        logger.warning("Unrecognised persisted tenant_source %r; treating as unvouched", value)
        return None


async def _run_claimed(
    app: Any,
    claimed: ClaimedJob,
    lease_seconds: int,
    worker_id: str,
    max_lifetime_seconds: int = 0,
) -> None:
    """Run one claimed job under a renewed lease, bounded by the lifetime ceiling.

    THE BOUND IS ON THE WORKER'S WAIT, NOT ON THE WORK. ``_dispatch`` has no
    timeout anywhere up its chain, and this function used to ``await`` it inline.
    ``_worker_loop`` only calls ``claim_next_job`` again once this returns, and
    ``worker_pool_size`` defaults to 1 on a deployment pinned to a single
    replica — so one await that never returns stopped the pod draining its queue
    for the life of the process, permanently and silently.

    #112 (see ``_heartbeat``) fixed half of that: past ``max_lifetime_seconds``
    the heartbeat stops renewing, the lease lapses, and the existing ceiling in
    ``claim_next_job`` writes the real failure reason. But that recovery needs
    SOME worker to reach ``claim_next_job``, and the only worker is the wedged
    one. This is the other half: at the same ceiling we stop waiting.

    WHY THE SAME CONSTANT, deliberately, rather than a new number. Abandoning
    EARLIER than the heartbeat's ceiling would be a double-render bug: the
    ``finally`` stops the heartbeat, the lease lapses within ``lease_seconds``,
    and a re-claim would find ``running_for < max_lifetime_seconds`` and START
    THE JOB AGAIN alongside the still-running orphan — replaying a paid WM call.
    Abandoning at the ceiling makes that impossible: the row cannot be re-claimed
    before its lease lapses, by which time ``running_for`` (measured from
    ``started_at``, which is at or before this run's start) necessarily exceeds
    the ceiling, so ``claim_next_job`` FAILS the row instead of re-running it.
    Abandoning later is merely wasteful. So the ceiling is exactly
    ``max_job_lifetime_seconds``, and ``0`` keeps the old unbounded behaviour.

    AT THE CEILING WE CANCEL (``_cancel_expired_dispatch``). The old "abandon, never
    cancel" kept a WM call alive that could no longer matter — the row is failed on
    re-claim and the orphan's writes are fenced — while it went on spending LLM calls
    and holding a browser lease. The dispatch runs in its own context copy, so a task
    that swallows the cancellation still keeps its identity + claim-owner fence and is
    abandoned after ``CANCEL_GRACE_SECONDS``: the queue keeps draining either way.
    """
    from w2v_common.auth import (
        clear_identity_context,
        set_correlation_context,
        set_identity_context,
    )

    # Re-establish the caller's identity for this job (the detached worker loop
    # doesn't inherit the enqueue request's ContextVars). tenant comes from the
    # job row; license/subject from the persisted snapshot, for audit headers on
    # outbound LLM/S2S calls.
    ident = claimed.options.get("identity") or {}
    # FAIL LOUD on a job whose tenant is absent, a legacy slug, or the nil UUID.
    # w2v_common 0.67.0 deleted every fallback tenant, so the old
    # ``... or ""`` tail is no longer a harmless default: an empty string now
    # raises inside ``set_identity_context`` anyway, from a stack that names
    # w2v_common rather than this job. Raising HERE, before any render work or
    # any outbound LLM spend, means the failure is attributed to the job row
    # that actually carries the bad tenant — and the caller's ``_run_claimed``
    # seam marks it failed rather than rendering it unattributed.
    set_identity_context(
        tenant_id=require_tenant(
            claimed.tenant_id or ident.get("tenant_id"),
            source=f"claimed job {claimed.job_id} tenant_id",
        ),
        license_id=ident.get("license_id") or "",
        subject_id=ident.get("subject_id") or "",
        tenant_source=_restore_tenant_source(ident.get("tenant_source")),
    )
    # Caller's actor + project/job, so cost events and outbound calls carry the
    # ORIGINAL caller's attribution rather than this service's own actor.
    _bind_job_attribution(ident)
    # Re-establish cost-correlation so this job's WM LLM call propagates
    # X-Correlation-Id/X-Stage → WM stamps them on the emitted cost_event,
    # tying the spend back to the orchestrator run + stage. clear_identity_context
    # (finally, below) resets these too.
    set_correlation_context(
        ident.get("correlation_id") or "",
        ident.get("stage") or "",
    )
    stop = asyncio.Event()
    hb = asyncio.create_task(
        _heartbeat(
            app.state.session_factory,
            claimed.job_id,
            lease_seconds,
            stop,
            max_lifetime_seconds,
        )
    )
    try:
        # Bind the claim owner for every failure seam reached under this dispatch:
        # `mark_failed` / `mark_retryable` fence on it so a worker that has lost
        # the claim (lease lapsed, sibling re-claimed after a requeue) cannot
        # overwrite a job another worker is actively rendering.
        #
        # The dispatch runs as its own TASK, created INSIDE the `claim_owner`
        # block and after the identity ContextVars are set, so it inherits a
        # COPY of both (``asyncio.create_task`` copies the current context).
        # That copy is what makes abandonment safe: the `finally` below resets
        # the owner token and clears identity in THIS context only, leaving an
        # abandoned dispatch still fenced under its own worker id and still
        # carrying the tenant provenance it was started with, while the worker
        # loop goes back to claiming with a clean context.
        with claim_owner(worker_id):
            task = asyncio.create_task(_dispatch(app, claimed))
        try:
            # ``asyncio.wait`` (not ``wait_for``) on purpose, twice over:
            #   * it does NOT cancel the task when the timeout expires, so the
            #     ceiling seam decides (``_cancel_expired_dispatch``);
            #   * ``wait_for`` raises ``TimeoutError`` for BOTH "my ceiling fired"
            #     and "the dispatch itself raised TimeoutError", and conflating
            #     those would abandon a task that had in fact already finished,
            #     mislabelling a plain render timeout as a wedge.
            done, _ = await asyncio.wait(
                {task},
                timeout=max_lifetime_seconds if max_lifetime_seconds > 0 else None,
            )
            if done:
                await task  # re-raise the dispatch's exception, as before
            else:
                await _cancel_expired_dispatch(
                    task, claimed.job_id, max_lifetime_seconds, worker_id
                )
        except asyncio.CancelledError:
            # Shutdown drain past its deadline (or an outright kill). Here we DO
            # cancel: ``asyncio.wait`` does not touch the task, so without this
            # the render would run on into a closing event loop after
            # release_claims() has already refunded the attempt — and the
            # bounded drain in ``JobWorkerPool.stop`` would stop being bounded.
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
            raise
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            await hb
        clear_identity_context()


async def _reap_stale_batches(app: FastAPI) -> None:
    """Run the batch arm of the lease sweep. Best-effort: never breaks the loop.

    Lives on the worker loop rather than in a CronJob or a second daemon because
    it IS the same sweep: the worker loop is already the thing that reclaims work
    orphaned by a dead pod, and ``batch_jobs`` is simply the table that mechanism
    forgot. A separate reaper would be a second answer to "who notices abandoned
    work", and the two would drift.
    """
    from src.services.batch_manager import BatchManager

    cfg = get_settings().redesign
    if cfg.batch_no_progress_seconds <= 0:
        return
    async with app.state.session_factory() as db, db.begin():
        await BatchManager.reap_stale_batches(db, cfg.batch_no_progress_seconds)


async def _cancel_expired_dispatch(
    task: asyncio.Task[None], job_id: str, max_lifetime_seconds: int, worker_id: str
) -> None:
    """CANCEL a dispatch that outlived the job lifetime, then stop waiting on it.

    This used to ABANDON without cancelling, to keep a paid WM call that "might still
    land". It cannot usefully land: by the ceiling the lease has lapsed and the row is
    failed on re-claim, so the orphan's writes are fenced off — while the orphan kept
    spending LLM calls, a browser lease and memory for a job nobody will ever read.
    Cancelling at the ceiling is exactly as safe from double-rendering as abandoning
    was (the row still cannot be re-claimed before ``running_for`` exceeds the
    ceiling). The render's own generation deadline
    (``redesign.generation_deadline_seconds``) normally ends the work long before this.

    A task that swallows the cancellation is abandoned after
    ``CANCEL_GRACE_SECONDS``, as before — the queue must keep draining.
    """
    logger.error(
        "Job %s is still running after %ds (max job lifetime) — CANCELLING the dispatch. "
        "The heartbeat has already stopped renewing, so the lease lapses and "
        "claim_next_job fails the row with the real reason. worker=%s",
        job_id,
        max_lifetime_seconds,
        worker_id,
    )
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=CANCEL_GRACE_SECONDS)
    if done:
        if not task.cancelled() and task.exception() is not None:
            logger.error("Cancelled dispatch for job %s raised", job_id, exc_info=task.exception())
        return
    task.add_done_callback(_log_abandoned_dispatch_result)
    logger.error(
        "Job %s ignored cancellation for %.0fs — ABANDONING the await so this worker can "
        "claim other jobs. worker=%s",
        job_id,
        CANCEL_GRACE_SECONDS,
        worker_id,
    )


def _log_abandoned_dispatch_result(task: asyncio.Task[None]) -> None:
    """Retrieve an abandoned dispatch's outcome so it is logged, not swallowed.

    Nothing awaits an abandoned task, so without this its eventual exception
    surfaces only as asyncio's "Task exception was never retrieved" at GC time —
    detached from the job id. Registered only on the abandon path; the normal
    path awaits the task and propagates as before.

    The "it is a no-op" line below USED TO BE FALSE and is worth the correction:
    the claim fence stopped the abandoned dispatch's UPDATE, but its caller
    discarded the fence's return value and ran ``record_usage`` and the
    ``job.completed``/``job.failed`` webhook regardless. The row was untouched
    while the tenant was billed and notified by a worker that had already lost
    the job. The callers now gate both on the fenced write landing, which is what
    makes the sentence true.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Abandoned dispatch finally failed", exc_info=exc)
    else:
        logger.warning(
            "Abandoned dispatch finally COMPLETED after the lifetime ceiling — its "
            "terminal write is fenced on the claim owner AND on the row's terminal "
            "status, and its caller now skips record_usage and the webhook when that "
            "fence rejects, so a row already re-claimed and failed is left alone"
        )


async def _idle(stop: asyncio.Event, seconds: float) -> None:
    """Sleep, but wake immediately if shutdown is signalled."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=seconds)


async def _worker_loop(app: Any, worker_id: str, index: int, stop: asyncio.Event) -> None:
    """Claim → run, until shutdown is signalled.

    ``stop`` is checked only BETWEEN jobs. That is the whole point of the drain: a
    SIGTERM stops us picking up NEW work, but an in-flight render runs to completion
    rather than being cancelled — cancelling it would throw away a paid 16k-token WM
    call that the next claim then has to redo from scratch.
    """
    cfg = get_settings().redesign
    session_factory = app.state.session_factory
    logger.info("Job worker %s[%d] started", worker_id, index)
    # Only worker 0 sweeps. The sweep is whole-table and idempotent, so N workers
    # running it would be correct but N times the scan for nothing; the resolver's
    # SKIP LOCKED makes the cross-POD overlap (replicas:2 + HPA) harmless too.
    # -inf: sweep on the very first iteration, so a pod that comes up after the
    # crash that stranded the batches reaps them immediately rather than in 5 min.
    next_reap = float("-inf") if index == 0 else float("inf")
    while not stop.is_set():
        try:
            if time.monotonic() >= next_reap:
                next_reap = time.monotonic() + max(1.0, cfg.batch_reap_interval_s)
                try:
                    await _reap_stale_batches(app)
                except Exception:
                    # Best-effort: a failed sweep must never stop the loop from
                    # draining the render queue, which is its primary job.
                    logger.warning("Stale-batch sweep failed", exc_info=True)
            async with session_factory() as db:
                claimed = await JobManager.claim_next_job(
                    db,
                    worker_id=worker_id,
                    lease_seconds=cfg.lease_seconds,
                    max_attempts=cfg.max_attempts,
                    max_lifetime_seconds=cfg.max_job_lifetime_seconds,
                )
            if claimed is None:
                await _idle(stop, cfg.queue_poll_interval_s)
                continue
            await _run_claimed(
                app,
                claimed,
                cfg.lease_seconds,
                worker_id,
                cfg.max_job_lifetime_seconds,
            )
        except asyncio.CancelledError:
            # The drain deadline passed while this job was still running (or we were
            # killed outright). The pool's release_claims() refunds the attempt.
            logger.warning("Job worker %s[%d] cancelled mid-job", worker_id, index)
            raise
        except Exception:
            logger.exception("Job worker %s[%d] loop error", worker_id, index)
            await _idle(stop, cfg.queue_poll_interval_s)
    logger.info("Job worker %s[%d] drained cleanly", worker_id, index)


class JobWorkerPool:
    """Owns the per-pod worker tasks; started/stopped from the ASGI lifespan."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self._tasks: list[asyncio.Task[None]] = []
        self._stop = asyncio.Event()
        # claimed_by == pod id for every worker, so release_claims() can expire
        # all of this pod's in-flight leases at shutdown with one predicate.
        self._pod_id = socket.gethostname()[:64]

    def status(self) -> WorkerPoolStatus:
        """Snapshot the worker tasks for /health. Never blocks, never mutates."""
        return WorkerPoolStatus(
            started=bool(self._tasks),
            workers=len(self._tasks),
            alive=sum(1 for task in self._tasks if not task.done()),
        )

    async def start(self) -> None:
        size = max(1, get_settings().redesign.worker_pool_size)
        for i in range(size):
            self._tasks.append(
                asyncio.create_task(_worker_loop(self._app, self._pod_id, i, self._stop))
            )
        logger.info("Job worker pool started: %d worker(s) (pod=%s)", size, self._pod_id)

    async def stop(self) -> None:
        """Drain on SIGTERM: stop claiming, let in-flight renders finish, then exit.

        The old behaviour cancelled every worker immediately, killing an in-flight
        render mid-flight. The job survived (an expired lease got reclaimed), but its
        *work* did not: the next claim replayed the whole render, including the paid
        16k-token WM call. On a ``replicas:2 + HPA`` deployment whose pods cycle every
        few minutes, a minutes-long redesign could be interrupted over and over,
        paying for the same generation each time and finishing none of them.

        So: signal the loops to stop claiming, then give whatever is still running up
        to ``drain_timeout_seconds`` to finish. Only work that outlives the deadline is
        cancelled — and its lease is released, refunding the attempt, so it is retried
        rather than penalised.

        ``drain_timeout_seconds`` MUST stay below terminationGracePeriodSeconds; a
        SIGKILL mid-drain skips release_claims entirely and the attempt is not refunded.
        """
        if not self._tasks:
            return
        drain = max(0, get_settings().redesign.drain_timeout_seconds)
        self._stop.set()  # stop claiming NEW work; in-flight jobs keep running

        _, pending = await asyncio.wait(self._tasks, timeout=drain)
        if pending:
            logger.warning(
                "Drain deadline (%ds) passed with %d job(s) still running — cancelling; "
                "their leases are released and attempts refunded, so they will be retried",
                drain,
                len(pending),
            )
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        else:
            logger.info("Job worker pool drained cleanly — no work was thrown away")
        self._tasks.clear()

        # Safety net. After a clean drain there is nothing left claimed by this pod, so
        # this is a no-op; after a cancelled drain it expires the lease (and refunds the
        # attempt) so a sibling reclaims immediately instead of waiting out the TTL.
        try:
            async with self._app.state.session_factory() as db, db.begin():
                await JobManager.release_claims(db, self._pod_id)
        except Exception:  # best-effort; leases expire on their own anyway
            logger.warning("Lease release at shutdown failed", exc_info=True)
        logger.info("Job worker pool stopped")
