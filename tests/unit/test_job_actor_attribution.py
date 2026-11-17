"""A queued render job carries its CALLER's actor, not this service's own.

w2v-common 0.88 uses ``SERVICE_ACTOR_ID`` only for SERVICE-originated work. A
queued job runs later, in the detached worker loop, outside any request — so
without a persisted actor w2v_common sees it as service-originated and stamps
this service's own sub on its cost events and outbound calls. That is wrong: a
caller asked for the render.

These tests pin both directions:

* enqueue with an actor -> the dispatch carries THAT actor (and the project/job),
  even with a different ``SERVICE_ACTOR_ID`` configured;
* a legacy job (no stored actor) stays actor-less and is COUNTED — it never
  falls back to ``SERVICE_ACTOR_ID``, because it was caller-originated.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from w2v_common.auth import (
    ACTOR_ID_HEADER,
    W2V_JOB_ID_HEADER,
    W2V_PROJECT_ID_HEADER,
    bind_attribution_from_headers,
    clear_identity_context,
    get_identity_headers,
    get_project_id_from_context,
    get_w2v_job_id_from_context,
    get_w2v_project_id_from_context,
    set_identity_context,
)
from w2v_common.auth.identity_context import (
    ACTOR_MISSING_INBOUND,
    actor_missing_total,
    mark_inbound_request,
    resolve_actor_id,
)
from w2v_common.metering.cost_bus import _apply_ambient_metadata
from w2v_common.tenant import TenantSource

from src.services import job_worker
from src.services.job_manager import ClaimedJob
from tests.tenants import TENANT_1

CALLER_ACTOR = "a1a1a1a1-1111-4111-8111-111111111111"
OWN_SERVICE_ACTOR = "5e5e5e5e-5555-4555-8555-555555555555"
FR_PROJECT = "b2b2b2b2-2222-4222-8222-222222222222"
W2V_PROJECT = "c3c3c3c3-3333-4333-8333-333333333333"
W2V_JOB = "d4d4d4d4-4444-4444-8444-444444444444"


@pytest.fixture(autouse=True)
def _clean_identity(monkeypatch):
    monkeypatch.setenv("SERVICE_ACTOR_ID", OWN_SERVICE_ACTOR)
    clear_identity_context()
    yield
    clear_identity_context()


def _enqueue_snapshot() -> dict:
    """``_capture_identity`` inside a request that named an actor + project + job."""
    set_identity_context(
        TENANT_1, "lic", "sub", tenant_source=TenantSource.LICENSE_TOKEN, actor_id=CALLER_ACTOR
    )
    mark_inbound_request()
    bind_attribution_from_headers(
        {
            "X-Project-Id": FR_PROJECT,
            W2V_PROJECT_ID_HEADER: W2V_PROJECT,
            W2V_JOB_ID_HEADER: W2V_JOB,
        },
        trusted=True,
    )
    ident = job_worker._capture_identity()
    clear_identity_context()
    return ident


async def _dispatch_and_observe(monkeypatch, options: dict) -> dict:
    """Run the real ``_run_claimed`` with ``_dispatch`` replaced by an observer."""
    observed: dict = {}

    async def _dispatch(app, claimed) -> None:
        observed["actor"] = resolve_actor_id()
        observed["headers"] = get_identity_headers()
        observed["project"] = get_project_id_from_context()
        observed["w2v_project"] = get_w2v_project_id_from_context()
        observed["w2v_job"] = get_w2v_job_id_from_context()
        fields: dict[str, str] = {}
        _apply_ambient_metadata(fields, None, "w2v-html-render")
        observed["cost_fields"] = fields

    monkeypatch.setattr(job_worker, "_dispatch", _dispatch)
    app = SimpleNamespace(state=SimpleNamespace(session_factory=None))
    await job_worker._run_claimed(
        app,
        ClaimedJob(job_id="job-1", tenant_id=TENANT_1, options=options),
        lease_seconds=30,
        worker_id="pod",
    )
    return observed


def test_capture_persists_actor_project_and_job() -> None:
    ident = _enqueue_snapshot()
    assert ident["actor_id"] == CALLER_ACTOR
    assert ident["project_id"] == FR_PROJECT
    assert ident["w2v_project_id"] == W2V_PROJECT
    assert ident["w2v_job_id"] == W2V_JOB


async def test_enqueued_actor_is_carried_by_the_dispatch(monkeypatch) -> None:
    ident = _enqueue_snapshot()
    observed = await _dispatch_and_observe(monkeypatch, {"identity": ident})

    assert observed["actor"] == (CALLER_ACTOR, None)
    assert observed["headers"][ACTOR_ID_HEADER] == CALLER_ACTOR
    assert observed["cost_fields"]["actor_id"] == CALLER_ACTOR
    assert observed["cost_fields"]["job_id"] == W2V_JOB
    assert observed["project"] == FR_PROJECT
    assert observed["w2v_project"] == W2V_PROJECT
    assert observed["w2v_job"] == W2V_JOB
    # Never this service's own sub for caller-originated work.
    assert OWN_SERVICE_ACTOR not in observed["headers"].values()


async def test_legacy_job_without_actor_stays_actorless_and_is_counted(monkeypatch) -> None:
    legacy_identity = {
        "tenant_id": TENANT_1,
        "tenant_source": TenantSource.LICENSE_TOKEN.value,
        "license_id": "lic",
        "subject_id": "sub",
    }
    counter = actor_missing_total.labels(service="w2v-html-render", reason=ACTOR_MISSING_INBOUND)
    before = counter._value.get()

    observed = await _dispatch_and_observe(monkeypatch, {"identity": legacy_identity})

    assert observed["actor"] == (None, ACTOR_MISSING_INBOUND)
    assert ACTOR_ID_HEADER not in observed["headers"]
    assert "actor_id" not in observed["cost_fields"]
    assert OWN_SERVICE_ACTOR not in observed["headers"].values()
    assert counter._value.get() == before + 1


async def test_dispatch_does_not_leak_actor_into_the_worker_loop(monkeypatch) -> None:
    await _dispatch_and_observe(monkeypatch, {"identity": _enqueue_snapshot()})
    # Back in the worker loop: identity cleared, no inbound mark, no actor.
    assert resolve_actor_id() == (OWN_SERVICE_ACTOR, None)
