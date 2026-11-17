"""A completed render puts a ``w2v-html-render`` event on the cost bus — and a fenced one does not.

The defect this guards is an ABSENCE: a web-design/HTML render emitted no
billing cost event at all. A 30-day sweep of ``billing_cost_events`` carried
keys for mlx-backend, inference-router, kag, w2v-tts, w2v-video-analyzer,
mlx-ingestion-backend, ingestion, w2v-video-compiler, w2v-image, kag-graph,
karma and hirag — and no render key of any kind. Screenshots, the Playwright
render and the concept images billed nobody.

**Every assertion below reads the MARSHALLED STREAM PAYLOAD** — the actual
fields of the actual ``XADD`` on a real ``fakeredis`` — never a mock's call
kwargs. That distinction is the whole point of the file: this repo's
characteristic failure is inert wiring, a value computed, passed and read by
nobody, sitting behind a green test that asserted the call happened. A mock
kwarg assertion passes whether or not ``w2v_common`` marshals the event, whether
or not the stream name resolves, and whether or not the XADD is ever awaited. A
``XRANGE`` on the stream passes only if a consumer could really read it.

Red-proof: deleting the ``_emit_render_cost_event`` call in
``finalize_phases._phase_finalize_completed`` takes ``test_completed_render_emits_a_cost_event``
to ``XLEN == 0`` and fails it. Recorded in the PR body.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
from redis import exceptions as redis_exceptions
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.metering import drain_pending
from w2v_common.metering.cost_bus import cost_events_total

from src.database.models import RedesignJob
from src.services.design_builder import RedesignService
from src.services.design_builder.types import (
    _A11yResult,
    _GenerationResult,
    _ScreenshotsResult,
    _TokenAccumulator,
)
from src.services.metering import SERVICE_KEY

STREAM = "billing:costs"
TENANT = "7b3f4e1a-2c6d-4a51-9f10-8d2e5c7b4a63"
JOB_ID = "job-s96-render"


class _RecordingWebhook:
    """Stands in for ``WebhookDispatcher`` and REMEMBERS what the tenant was told.

    Deliberately not a ``MagicMock``: the assertion reads ``.events``, which is
    the tenant-visible fact ("your render finished"), not the internal fact
    ("dispatch was called once").
    """

    def __init__(self) -> None:
        self.events: list[str] = []

    async def dispatch(self, db, **kwargs) -> None:
        self.events.append(kwargs["event_type"])


def _emit_attempts() -> float:
    """Total cost-bus emit ATTEMPTS recorded for this service, ok and failed alike.

    ``w2v_cost_events_total`` is incremented by ``w2v_common``'s
    ``_record_outcome`` on both terminal outcomes, so it counts attempts that
    REACHED the emitter — which is exactly what "did the metering-unconfigured
    path stay out of the emitter entirely?" needs. A failed-but-attempted emit
    moves this number; a correctly no-op'd one does not.

    ``_created`` samples carry a unix timestamp as their value, so they are
    filtered out rather than summed into the total.
    """
    return sum(
        sample.value
        for metric in cost_events_total.collect()
        for sample in metric.samples
        if sample.name.endswith("_total") and sample.labels.get("service_key") == SERVICE_KEY
    )


@dataclass
class _FakeJobManager:
    """Stands in for the real ``JobManager`` at the package lookup site.

    Only the fence verdict matters here — ``mark_completed``'s boolean is the
    anti-double-bill barrier under test. The DB row itself is another test's
    subject (``test_terminal_status_fence``), so this file does not carry a
    postgres dependency just to re-assert it.
    """

    completed: bool

    async def mark_completed(self, *args, **kwargs) -> bool:
        return self.completed

    async def record_usage(self, *args, **kwargs) -> None:
        return None


def _service(metering_redis, webhook=None, metering_stream=None) -> RedesignService:
    """A REAL ``RedesignService`` — the production mixin composition, not a stub.

    Storage/generators are inert doubles because the finalize phase under test
    touches none of them beyond ``self._html_gen._model`` (read by
    ``estimate_cost``). The emitter, the fence gating it, ``_dispatch_webhook``
    and the whole ``w2v_common`` marshalling path are all real.
    """
    html_gen = MagicMock()
    html_gen._model = "test-model"
    return RedesignService(
        storage=MagicMock(),
        html_generator=html_gen,
        concept_generator=MagicMock(),
        webhook_dispatcher=webhook,
        metering_redis=metering_redis,
        metering_stream=metering_stream,
    )


async def _run_finalize(
    monkeypatch,
    *,
    completed: bool,
    redis_conn,
    tenant_id: str = TENANT,
    metering_stream: str | None = None,
) -> _RecordingWebhook:
    """Await the REAL ``_phase_finalize_completed`` end to end.

    Returns the recording webhook so a caller can assert on what the tenant was
    actually told.
    """
    import src.services.design_builder as pkg

    monkeypatch.setattr(pkg, "JobManager", _FakeJobManager(completed=completed))
    # Pin the stream so the assertion reads the same key the emitter writes,
    # regardless of a METERING_STREAM left in the ambient environment.
    monkeypatch.setenv("METERING_STREAM", STREAM)

    webhook = _RecordingWebhook()
    service = _service(redis_conn, webhook, metering_stream=metering_stream)
    job = SimpleNamespace(
        job_id=JOB_ID,
        status="processing",
        webhook_url="https://tenant.example.com/hook",
        webhook_secret=None,
        batch_id=None,
    )
    await service._phase_finalize_completed(
        # redis-py/fakeredis and the ORM session are both duck-typed here; ty
        # needs the cast, the runtime does not care.
        cast(AsyncSession, AsyncMock()),
        cast(RedesignJob, job),
        JOB_ID,
        tenant_id,
        gen_result=_GenerationResult(
            redesigned_html="<html></html>",
            redesigned_html_path="out/index.html",
            section_html_paths=None,
            component_html_path=None,
            input_tokens=1200,
            output_tokens=800,
        ),
        redesigned_html_path="out/index.html",
        a11y=_A11yResult(score=91.0, report={}, redesigned_html_path=None),
        screenshots=_ScreenshotsResult(
            original_paths=None,
            screenshot_paths={"full": "shot.png"},
            comparison_paths=None,
            diff_paths=None,
            screenshots_count=3,
        ),
        concept_image_assets=None,
        concept_count=2,
        html_validation_report=None,
        tokens=_TokenAccumulator(input=1200, output=800),
    )
    # Fire-and-forget: the XADD is a spawned task. Drain it, or the assertion
    # races the emit and this test would pass for the wrong reason.
    await drain_pending(timeout=5.0)
    return webhook


@pytest.mark.asyncio
async def test_completed_render_emits_a_cost_event(monkeypatch) -> None:
    """Exactly one readable ``w2v-html-render`` event, attributed to the job's tenant."""
    redis_conn = fakeredis.aioredis.FakeRedis(decode_responses=True)
    await _run_finalize(monkeypatch, completed=True, redis_conn=redis_conn)

    # ``xrange`` is typed as returning None by the stubs redis-py ships for the
    # async client; the cast is a typing fact, not a behavioural one.
    entries = cast("list[tuple[str, dict[str, str]]]", await redis_conn.xrange(STREAM))
    assert len(entries) == 1, (
        f"expected exactly one render cost event on {STREAM}, got {len(entries)} — "
        "a render that bills nobody is the defect this test exists for"
    )
    _msg_id, fields = entries[0]

    # Identity: the slug billing groups render spend under. Before this change
    # NO key of this shape existed anywhere in billing_cost_events.
    #
    # The pragma is for detect-secrets, whose "Secret Keyword" heuristic fires on
    # the *name* `service_key` next to a string literal. This is a routing slug
    # that is published on every cost event — the opposite of a credential.
    # Suppressed inline rather than by regenerating .secrets.baseline, so the
    # reason travels with the line and the baseline stays a record of real
    # findings. Same treatment as `SERVICE_KEY` in src/services/metering.py.
    assert fields["service_key"] == "w2v-html-render"  # pragma: allowlist secret
    assert fields["cost_source"] == "w2v-html-render"
    # WHO PAYS — the job's tenant, never a substituted default.
    assert fields["tenant_id"] == TENANT
    # Billable quantity: pages rendered, marshalled as a string by w2v_common.
    assert fields["input_tokens"] == "1"
    # The LLM slice is billed by inference-router; billing it again here would
    # double-charge the same spend.
    assert fields["output_tokens"] == "0"
    # Producer-minted idempotency key. Without it billing's inbox skips its
    # claim and bills a retried XADD unconditionally.
    assert fields["event_key"]
    assert fields["provider"] == "html-render"
    assert fields["model"] == "playwright"

    usage_extra = json.loads(fields["usage_extra"])
    assert usage_extra["unit"] == "pages"
    assert usage_extra["screenshots"] == 3
    assert usage_extra["concept_images"] == 2
    assert usage_extra["llm_tokens"] == 2000

    # Metadata the allowlist in w2v_common actually passes through.
    assert fields["job_id"] == JOB_ID
    assert fields["stage"] == "redesign"

    await redis_conn.aclose()


@pytest.mark.asyncio
async def test_fenced_off_completion_bills_nothing(monkeypatch) -> None:
    """A lost claim / cancelled job is the anti-double-bill barrier — not ``event_key``.

    ``mark_completed`` returning False means this worker does not own the
    verdict. It must bill neither the usage row nor the cost bus: ``event_key``
    dedupes a retried XADD of ONE logical emit, and cannot help across two
    workers that each emit their own event for the same job.
    """
    redis_conn = fakeredis.aioredis.FakeRedis(decode_responses=True)
    await _run_finalize(monkeypatch, completed=False, redis_conn=redis_conn)

    assert await redis_conn.xlen(STREAM) == 0, (
        "a fenced-off completion emitted a cost event — the tenant is billed twice for one render"
    )
    await redis_conn.aclose()


@pytest.mark.asyncio
async def test_metering_unconfigured_renders_normally(monkeypatch) -> None:
    """No ``METERING_REDIS_URL`` means the emitter is never entered, and the render still lands.

    ``init_metering_redis("")`` returns None, so ``metering_redis=None`` is the
    real production shape wherever metering is not configured (local dev, any
    deployment without the cost-bus DSN) — not a hypothetical.

    Two observable facts, because "no crash" is not a result:

    1. The cost-bus emit-attempt counter does not move. That is stronger than
       "no event on the stream": it proves the call returned at the
       ``_metering_redis is None`` guard rather than reaching ``w2v_common``
       and FAILING there. A failed emit would still increment this
       (``outcome="failed"``), so a silently-broken emitter cannot pass here.
    2. The tenant is still told their render completed. A billing dependency
       that is absent must not cost the tenant the render — the whole point of
       keeping emission off the render's success path.
    """
    import src.services.design_builder as pkg

    monkeypatch.setattr(pkg, "JobManager", _FakeJobManager(completed=True))
    before = _emit_attempts()

    webhook = await _run_finalize(monkeypatch, completed=True, redis_conn=None)

    assert _emit_attempts() == before, (
        "an unconfigured metering lane still reached the w2v_common emitter — "
        "the None guard is not holding, and every render would log a failed emit"
    )
    assert webhook.events == ["job.completed"], (
        "the render was not delivered to the tenant when metering was "
        f"unconfigured; got {webhook.events}"
    )


@pytest.mark.asyncio
async def test_cost_event_carries_every_required_emitter_guide_field(monkeypatch) -> None:
    """The marshalled payload carries the FULL required field set, none of them empty-by-accident.

    ``billing_service`` parses each field positionally-by-name off the stream
    entry. A field this producer never writes does not fail anything here — it
    lands as a NULL column three hops away, in a service this repo cannot see,
    and the row still looks like a successful bill. So the contract is asserted
    as a SET (`required <= actual`), not field-by-field: a future w2v_common
    that stops marshalling one of them fails here rather than silently NULLing a
    billing column.

    ``license_id``/``subject_id``/``app_id``/``provider_request_id`` are
    required-to-be-PRESENT but legitimately EMPTY east-west: no in-mesh caller
    sends the license-authz headers to the render worker, and the job is
    finalized on a background worker where the request's identity ContextVars
    are not bound. Presence is asserted; non-emptiness deliberately is not.
    """
    redis_conn = fakeredis.aioredis.FakeRedis(decode_responses=True)
    await _run_finalize(monkeypatch, completed=True, redis_conn=redis_conn)

    entries = cast("list[tuple[str, dict[str, str]]]", await redis_conn.xrange(STREAM))
    assert len(entries) == 1
    _msg_id, fields = entries[0]

    required = {
        "tenant_id",
        "license_id",
        "subject_id",
        "app_id",
        "service_key",
        "cost_source",
        "provider",
        "model",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "timestamp",
        # Optional per the guide, but this emitter always ships them: the
        # idempotency key (billing's inbox skips its dedupe claim without it)
        # and the advisory price.
        "event_key",
        "cost_usd",
        "usage_extra",
    }
    missing = required - set(fields)
    assert not missing, (
        f"cost event is missing required field(s) {sorted(missing)} — billing stores "
        "each of these as its own column and a missing one lands as NULL without "
        "failing anything visible from here"
    )

    # The fields that MUST carry a real value, because a blank one silently
    # unbills or mis-bills the render rather than merely losing detail.
    assert fields["tenant_id"] == TENANT
    assert fields["service_key"] == "w2v-html-render"  # pragma: allowlist secret
    assert fields["cost_source"] == "w2v-html-render"
    # Must match the existing `provider='html-render'` / per_request $0.02
    # pricing row, or these events land unpriced.
    assert fields["provider"] == "html-render"
    assert fields["model"] == "playwright"
    assert fields["event_key"]
    assert fields["timestamp"]
    # Numeric fields are marshalled as strings at the Redis layer; assert they
    # are parseable, not merely present.
    assert int(fields["input_tokens"]) == 1
    assert int(fields["output_tokens"]) == 0
    assert int(fields["latency_ms"]) >= 0
    assert float(fields["cost_usd"]) >= 0.0

    await redis_conn.aclose()


@pytest.mark.asyncio
async def test_emit_targets_the_explicitly_configured_stream(monkeypatch) -> None:
    """The stream is the one THIS SERVICE resolved — never w2v_common's internal default.

    The canonical cost-bus name has already moved twice (``metering:costs`` ->
    ``improvement-loop:costs`` -> ``billing:costs``). A caller that omits
    ``stream=`` inherits whatever the LINKED w2v_common happens to consider
    canonical, so an unrelated dependency bump can repoint every render's
    billing at a stream with no consumer — and the XADD still succeeds, so
    nothing anywhere goes red. That is the exact failure mode the emitter guide
    warns about.

    Red-proof: the ambient ``METERING_STREAM`` is pinned to the canonical name
    while the service is constructed with a DIFFERENT one. An implementation
    that drops ``stream=self._metering_stream`` falls back to
    ``resolve_stream()``, writes to the canonical stream, and fails both
    assertions below.
    """
    redis_conn = fakeredis.aioredis.FakeRedis(decode_responses=True)
    configured = "billing:costs-lane-under-test"
    await _run_finalize(
        monkeypatch,
        completed=True,
        redis_conn=redis_conn,
        metering_stream=configured,
    )

    assert await redis_conn.xlen(configured) == 1, (
        "the cost event did not land on the stream this service resolved — the "
        "emit is inheriting w2v_common's internal default"
    )
    assert await redis_conn.xlen(STREAM) == 0, (
        f"the cost event landed on the library default {STREAM} instead of the "
        "explicitly configured stream"
    )
    await redis_conn.aclose()


class _BrokenRedis(fakeredis.aioredis.FakeRedis):
    """A cost-bus Redis whose every ``XADD`` fails the way a real outage does.

    ``ConnectionError`` is in w2v_common's RETRYABLE set, so this exercises the
    full bounded-retry path and its terminal warning — not a single throw that
    short-circuits before the interesting code.
    """

    async def xadd(self, *args, **kwargs):
        raise redis_exceptions.ConnectionError("cost-bus redis is down")


@pytest.mark.asyncio
async def test_redis_fault_does_not_fail_the_render_and_logs_loudly(monkeypatch, caplog) -> None:
    """A dead cost bus costs the tenant NOTHING — but it is never silent.

    Two facts, because "did not crash" is not a result:

    1. The tenant still gets their finished render (``job.completed`` dispatched).
       Emission is fire-and-forget and off the success path precisely so a
       billing outage cannot fail a design that is already rendered and persisted.
    2. A WARNING naming the failure is emitted. A bare ``except: pass`` would
       satisfy (1) and leave the fleet unable to tell "nothing was rendered"
       from "everything was rendered and none of it was billed" — the same
       invisible-absence defect this whole file exists for.
    """
    redis_conn = _BrokenRedis(decode_responses=True)
    caplog.set_level(logging.WARNING)
    # Keep the bounded retry from making this test sleep for the real backoffs.
    monkeypatch.setenv("METERING_EMIT_RETRY_BACKOFFS", "0")

    webhook = await _run_finalize(monkeypatch, completed=True, redis_conn=redis_conn)

    assert webhook.events == ["job.completed"], (
        "a cost-bus outage cost the tenant their render — emission is not off "
        f"the success path; got {webhook.events}"
    )
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("cost_event_emit_failed" in r.getMessage() for r in warnings), (
        "the cost-bus failure was swallowed without a WARNING — an unbilled "
        "render that logs nothing is indistinguishable from a billed one. "
        f"Records seen: {[r.getMessage()[:80] for r in warnings]}"
    )
    await redis_conn.aclose()


@pytest.mark.asyncio
async def test_unattributable_tenant_does_not_fail_the_render_and_logs_loudly(
    monkeypatch, caplog
) -> None:
    """A missing tenant is a LOUD billing defect — never a nil-UUID row, never a failed render.

    ⚠️ Contract note, because the obvious expectation is now wrong: w2v_common
    REMOVED the nil-UUID sentinel on 2026-08-26 (``cost_bus.py``, the
    ``UNATTRIBUTED_TENANT_ID`` removal note). Substituting
    ``00000000-0000-0000-0000-000000000000`` was the worst available shape — a
    valid UUID that passes every type check and every dirty-row audit, and is
    registered in ``licenses.tenants`` as ``default-anon``, so a mis-attributed
    charge looked completely legitimate. Product-owner ruling: a ``tenant_id``
    may NEVER hold a sentinel. ``schedule_cost_event`` now raises
    ``MissingTenantError`` synchronously, in the caller's frame.

    So the assertion is NOT "the nil UUID was emitted". It is: nothing is
    emitted, the error does not propagate into the render, and it is logged at
    ERROR naming the job as UNBILLED.
    """
    redis_conn = fakeredis.aioredis.FakeRedis(decode_responses=True)
    caplog.set_level(logging.WARNING)

    webhook = await _run_finalize(monkeypatch, completed=True, redis_conn=redis_conn, tenant_id="")

    assert await redis_conn.xlen(STREAM) == 0, (
        "an unattributable render was billed to SOMEBODY — a cost event with no "
        "real tenant must not reach the bus at all"
    )
    assert webhook.events == ["job.completed"], (
        "a billing-attribution defect cost the tenant a render that was already "
        f"finished and persisted; got {webhook.events}"
    )
    assert any(
        "UNBILLED" in r.getMessage() and r.levelno >= logging.ERROR for r in caplog.records
    ), (
        "an unattributable render was swallowed quietly — this is a real revenue "
        f"defect and must be loud. Records seen: {[r.getMessage()[:80] for r in caplog.records]}"
    )
    await redis_conn.aclose()


# --- G5: full attribution, stated by the emitter --------------------------------

LICENCE = "0195f48b-1111-7000-8000-000000000001"
SUBJECT = "0195f48b-2222-7000-8000-000000000002"


@pytest.fixture
def _bound_licence():
    from w2v_common.auth import clear_identity_context, set_identity_context

    set_identity_context(TENANT, LICENCE, SUBJECT)
    yield
    clear_identity_context()


async def _single_event(redis_conn) -> dict[str, str]:
    entries = cast("list[tuple[str, dict[str, str]]]", await redis_conn.xrange(STREAM))
    assert len(entries) == 1
    return entries[0][1]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_bound_licence")
async def test_render_event_states_app_funding_and_licence(monkeypatch) -> None:
    """app/funding are STATED on the wire (billing never has to derive them)."""
    redis_conn = fakeredis.aioredis.FakeRedis(decode_responses=True)
    await _run_finalize(monkeypatch, completed=True, redis_conn=redis_conn)
    fields = await _single_event(redis_conn)

    assert fields["app_id"] == "w2v"
    assert fields["funding"] == "local_gpu"
    assert fields["license_id"] == LICENCE
    assert fields["subject_id"] == SUBJECT
    await redis_conn.aclose()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_bound_licence")
async def test_render_event_key_is_deterministic_per_job_and_stage(monkeypatch) -> None:
    """A re-emitted completion carries the SAME key (billing dedupes it); a uuid7 would not."""
    from w2v_common.metering import cost_event_key

    keys = []
    for _ in range(2):
        redis_conn = fakeredis.aioredis.FakeRedis(decode_responses=True)
        await _run_finalize(monkeypatch, completed=True, redis_conn=redis_conn)
        keys.append((await _single_event(redis_conn))["event_key"])
        await redis_conn.aclose()

    assert keys[0] == keys[1] == cost_event_key(SERVICE_KEY, JOB_ID, "redesign")
    assert cost_event_key(SERVICE_KEY, JOB_ID, "refinement") != keys[0]
    assert cost_event_key(SERVICE_KEY, "another-job", "redesign") != keys[0]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_bound_licence")
async def test_strict_mode_emits_a_fully_attributed_render(monkeypatch) -> None:
    """Under METERING_REQUIRE_ATTRIBUTION the emitter's own fields pass the strict gate."""
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    redis_conn = fakeredis.aioredis.FakeRedis(decode_responses=True)
    await _run_finalize(monkeypatch, completed=True, redis_conn=redis_conn)
    fields = await _single_event(redis_conn)
    assert fields["app_id"] == "w2v"
    assert fields["funding"] == "local_gpu"
    await redis_conn.aclose()


@pytest.mark.asyncio
async def test_strict_mode_refuses_a_render_event_without_a_licence(monkeypatch) -> None:
    """Strict: a missing licence is a LOUD failure, never a partial event on the bus."""
    from w2v_common.metering import MissingAttributionError

    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    redis_conn = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with pytest.raises(MissingAttributionError, match="license_id"):
        await _run_finalize(monkeypatch, completed=True, redis_conn=redis_conn)
    assert await redis_conn.xlen(STREAM) == 0
    await redis_conn.aclose()
