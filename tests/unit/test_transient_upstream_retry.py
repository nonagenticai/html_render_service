"""An upstream READ TIMEOUT is not a verdict, and `(no message)` is not a record.

The live failure this pins, from the nightly orchestrator E2E of 2026-08-24
(`web-design-v1`, the only red scenario of eleven)::

    FAILURE: run 01a031e6-… reached terminal `failed` — expected ['completed'].
    Error: 'RuntimeError: design job 01a031e6-… failed: ReadTimeout: (no message)'

Two defects in one line, and the orchestrator is innocent of both — its
``retry_on=()`` for a server-marked failure is deliberate, so whatever this
service writes on the row is what the customer gets.

1. **Classification.** ``httpx.ReadTimeout`` from the ``worker_management`` LLM
   call reached the pipeline's failure seam and was written as
   ``status='failed'`` — terminal, unreclaimable, on an error that says only
   "the LLM did not answer in time". It now goes back on the durable claim
   queue, bounded by the ``attempts``/``max_attempts`` poison gate that already
   existed. The control half of every test below is a genuinely PERMANENT
   failure (a 4xx, a schema violation) which must still die terminal: widening
   the retry to those would hide real bugs behind three paid re-renders.

2. **The empty message.** ``str(httpx.ReadTimeout())`` is ``''`` — httpx maps
   the httpcore error and the detail comes through blank — so the old
   ``str(exc) or f"{type(exc).__name__}: (no message)"`` produced a record with
   no diagnostic content. Hence ``describe_failure``.

Hermetic: no DB, no network, no browser.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import src.services.design_builder as design_builder
from src.api.routes._helpers import _record_background_failure
from src.services.design_builder.finalize_phases import RedesignFinalizeMixin
from src.services.design_builder.helpers import RedesignHelpersMixin
from src.services.job_manager import JobManager, claim_owner
from src.services.upstream_failure import describe_failure, is_transient_upstream

# The upstream endpoint every one of these failures came from. Built once so the
# request-bearing exceptions below all name the same target.
_WM_REQUEST = httpx.Request("POST", "https://worker-management.w2v.svc:8100/workers/w/message")

TRANSIENT = [
    pytest.param(httpx.ReadTimeout("", request=_WM_REQUEST), id="read-timeout-empty"),
    pytest.param(httpx.ReadTimeout("read timed out"), id="read-timeout-message"),
    pytest.param(httpx.ConnectTimeout("connect timed out"), id="connect-timeout"),
    pytest.param(httpx.PoolTimeout("pool exhausted"), id="pool-timeout"),
]

# The control. Each is a PERMANENT failure that must keep its terminal verdict.
PERMANENT = [
    pytest.param(
        httpx.HTTPStatusError(
            "422", request=_WM_REQUEST, response=httpx.Response(422, request=_WM_REQUEST)
        ),
        id="wm-422-schema-violation",
    ),
    pytest.param(
        httpx.HTTPStatusError(
            "400", request=_WM_REQUEST, response=httpx.Response(400, request=_WM_REQUEST)
        ),
        id="wm-400",
    ),
    pytest.param(ValueError("output_schema violation: 'sections' is not a list"), id="schema"),
    pytest.param(RuntimeError("design failed verification: [contrast] 2.1:1"), id="verification"),
    # Named explicitly because it is the near-miss: this service raises the
    # BUILTIN TimeoutError from its own concept-image waits and asyncio guards.
    # Those are local budgets, not an upstream failing to answer, and sweeping
    # them in would retry work that already ran.
    pytest.param(TimeoutError("concept image batch timed out"), id="builtin-timeout"),
    pytest.param(httpx.ConnectError("wm unreachable"), id="connect-error"),
]


def _db() -> MagicMock:
    return MagicMock(commit=AsyncMock(), rollback=AsyncMock())


class _Finalizer(RedesignFinalizeMixin, RedesignHelpersMixin):
    """The two mixins the redesign failure path spans, exactly as in prod."""


def _finalizer() -> tuple[_Finalizer, AsyncMock]:
    obj = _Finalizer.__new__(_Finalizer)
    dispatch = AsyncMock()
    obj._dispatch_webhook = dispatch  # type: ignore[method-assign]
    return obj, dispatch


def _job() -> MagicMock:
    job = MagicMock()
    job.job_id = "job-1"
    job.webhook_url = None
    job.webhook_secret = None
    job.batch_id = None
    return job


def _job_manager(*, requeued: bool) -> MagicMock:
    """A JobManager double whose ``mark_retryable`` reports a real bool.

    ``requeued`` models the row-eligibility guard: True = a queue-owned row was
    handed back to ``pending``; False = it was not eligible (a batch child, or a
    row that left ``processing``), so the caller must fall through to a terminal
    failure. An ``AsyncMock`` returning the default ``MagicMock`` would be
    *truthy* and make that fall-through untestable, which is the whole reason
    this returns an explicit bool.
    """
    return MagicMock(
        mark_failed=AsyncMock(),
        mark_retryable=AsyncMock(return_value=requeued),
        record_usage=AsyncMock(),
    )


# ---------------------------------------------------------------------------
# 1. describe_failure — the record must never be empty, and never leak a secret
# ---------------------------------------------------------------------------


class TestTheRecordSaysSomething:
    """``(no message)`` is undiagnosable. Every record names its type."""

    def test_a_bare_read_timeout_stringifies_to_nothing(self) -> None:
        """The premise of defect 2, pinned as executable fact."""
        assert str(httpx.ReadTimeout("")) == ""

    def test_empty_exception_still_yields_a_typed_non_empty_record(self) -> None:
        described = describe_failure(httpx.ReadTimeout(""))

        assert described != ""
        assert "(no message)" not in described
        assert "ReadTimeout" in described

    def test_the_upstream_endpoint_is_named_when_known(self) -> None:
        """Which upstream timed out is the first question an operator asks."""
        described = describe_failure(httpx.ReadTimeout("", request=_WM_REQUEST))

        assert "worker-management.w2v.svc:8100" in described
        assert "/workers/w/message" in described

    def test_a_credential_bearing_url_is_never_persisted(self) -> None:
        """The record is echoed to the caller — a token in the URL must not ride along."""
        request = httpx.Request(
            "POST",
            # pragma: allowlist secret — synthetic credentials, the point of the test
            "https://user:hunter2@wm.test/workers/w/message?token=s3cr3t&X-Amz-Signature=abc",  # pragma: allowlist secret
        )
        described = describe_failure(httpx.ReadTimeout("", request=request))

        assert "hunter2" not in described
        assert "s3cr3t" not in described
        assert "X-Amz-Signature" not in described
        assert "wm.test/workers/w/message" in described

    def test_a_message_bearing_exception_keeps_its_message(self) -> None:
        assert describe_failure(ValueError("bad shape")) == "ValueError: bad shape"


class TestClassification:
    """Timeouts only. Everything else keeps its terminal verdict."""

    @pytest.mark.parametrize("exc", TRANSIENT)
    def test_upstream_timeouts_are_transient(self, exc: BaseException) -> None:
        assert is_transient_upstream(exc) is True

    @pytest.mark.parametrize("exc", PERMANENT)
    def test_permanent_failures_are_not_widened(self, exc: BaseException) -> None:
        assert is_transient_upstream(exc) is False


# ---------------------------------------------------------------------------
# 2. The redesign pipeline's failure seam — the one the live job went through
# ---------------------------------------------------------------------------


class TestRedesignFailureSeam:
    """``_phase_finalize_failed`` is where the live ``failed`` row was written."""

    @pytest.mark.parametrize("exc", TRANSIENT)
    async def test_a_timeout_is_requeued_not_failed(
        self, exc: BaseException, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        jm = _job_manager(requeued=True)
        monkeypatch.setattr(design_builder, "JobManager", jm)
        obj, dispatch = _finalizer()
        db = _db()

        await obj._phase_finalize_failed(db, _job(), "job-1", "tenant-1", exc)

        jm.mark_retryable.assert_awaited_once()
        jm.mark_failed.assert_not_awaited()
        # Not a failure yet: no error-usage charge, no `job.failed` webhook to a
        # tenant whose job is about to run again.
        jm.record_usage.assert_not_awaited()
        dispatch.assert_not_awaited()
        db.commit.assert_awaited_once()

    @pytest.mark.parametrize("exc", PERMANENT)
    async def test_a_permanent_failure_still_ends_terminal(
        self, exc: BaseException, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: widening far enough to catch these would hide real bugs."""
        jm = _job_manager(requeued=True)
        monkeypatch.setattr(design_builder, "JobManager", jm)
        obj, dispatch = _finalizer()
        db = _db()

        await obj._phase_finalize_failed(db, _job(), "job-1", "tenant-1", exc)

        jm.mark_failed.assert_awaited_once()
        jm.mark_retryable.assert_not_awaited()
        assert jm.record_usage.await_args is not None
        assert jm.record_usage.await_args.kwargs["is_error"] is True
        assert dispatch.await_args_list[-1].args[2] == "job.failed"

    async def test_the_persisted_timeout_message_is_not_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What the row would have read: ``ReadTimeout: (no message)``."""
        jm = _job_manager(requeued=True)
        monkeypatch.setattr(design_builder, "JobManager", jm)
        obj, _ = _finalizer()

        await obj._phase_finalize_failed(
            _db(), _job(), "job-1", "tenant-1", httpx.ReadTimeout("", request=_WM_REQUEST)
        )

        message = jm.mark_retryable.await_args.args[2]
        assert message
        assert "(no message)" not in message
        assert "ReadTimeout" in message

    async def test_an_ineligible_row_falls_back_to_a_terminal_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A row no worker will ever re-claim must NOT be parked in ``pending``.

        ``mark_retryable`` returns False for a batch child (``job_options`` NULL)
        or a row that left ``processing``. Without this fall-through the job
        would sit ``pending`` forever with its batch counters hanging — a worse
        outcome than the terminal failure being fixed.
        """
        jm = _job_manager(requeued=False)
        monkeypatch.setattr(design_builder, "JobManager", jm)
        obj, dispatch = _finalizer()

        await obj._phase_finalize_failed(
            _db(), _job(), "job-1", "tenant-1", httpx.ReadTimeout("", request=_WM_REQUEST)
        )

        jm.mark_retryable.assert_awaited_once()
        jm.mark_failed.assert_awaited_once()
        assert dispatch.await_args_list[-1].args[2] == "job.failed"


# ---------------------------------------------------------------------------
# 3. The OUTER seam. Without it the inner requeue is immediately overwritten.
# ---------------------------------------------------------------------------


class TestBackgroundTaskSeamAgrees:
    """``_process_job_background``'s ``except`` sees the SAME exception re-raised.

    ``_phase_finalize_failed`` records and re-raises; the background wrapper
    catches it and records again on a fresh session. A plain ``mark_failed``
    there stamps ``failed`` over the requeue and restores the exact bug — so
    the outer seam must reach the same verdict.
    """

    async def test_timeout_requeues_at_the_outer_seam_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marks = MagicMock(mark_failed=AsyncMock(), mark_retryable=AsyncMock(return_value=True))
        monkeypatch.setattr("src.api.routes._helpers.JobManager", marks)

        await _record_background_failure(
            MagicMock(), "job-1", httpx.ReadTimeout("", request=_WM_REQUEST)
        )

        marks.mark_retryable.assert_awaited_once()
        marks.mark_failed.assert_not_awaited()

    async def test_permanent_failure_is_terminal_at_the_outer_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marks = MagicMock(mark_failed=AsyncMock(), mark_retryable=AsyncMock(return_value=True))
        monkeypatch.setattr("src.api.routes._helpers.JobManager", marks)

        await _record_background_failure(
            MagicMock(),
            "job-1",
            httpx.HTTPStatusError(
                "422", request=_WM_REQUEST, response=httpx.Response(422, request=_WM_REQUEST)
            ),
        )

        marks.mark_failed.assert_awaited_once()
        marks.mark_retryable.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. mark_retryable's own guards — the bound, and the rows it must refuse
# ---------------------------------------------------------------------------


class TestMarkRetryableStatement:
    """The requeue is a guarded UPDATE, and its guards are the safety property."""

    async def _capture(self, rowcount: int) -> tuple[bool, str]:
        db = MagicMock(execute=AsyncMock(return_value=MagicMock(rowcount=rowcount)))
        ok = await JobManager.mark_retryable(
            db, "job-1", "ReadTimeout: ReadTimeout('')", worker_id="podA"
        )
        statement = str(db.execute.await_args.args[0])
        return ok, statement

    async def test_it_requeues_to_pending_and_releases_the_lease(self) -> None:
        ok, statement = await self._capture(1)

        assert ok is True
        assert "UPDATE redesign_jobs" in statement
        for column in ("status", "error_message", "claimed_by", "lease_expires_at"):
            assert column in statement
        # completed_at must stay NULL: the job is not finished.
        assert "completed_at" not in statement

    async def test_it_refuses_a_row_no_worker_owns_or_that_left_processing(self) -> None:
        """``job_options IS NOT NULL`` is queue ownership; the status set is the fence."""
        _, statement = await self._capture(1)

        assert "job_options IS NOT NULL" in statement
        assert "status IN" in statement

    async def test_no_matching_row_reports_false(self) -> None:
        """The signal the caller needs to fall back to a terminal failure."""
        ok, _ = await self._capture(0)

        assert ok is False

    def test_the_retry_is_bounded_by_the_existing_poison_gate(self) -> None:
        """No new retry engine: the claim resolver already fails poison jobs.

        ``claim_next_render_job`` charges ``attempts += 1`` per claim and fails
        any row at ``attempts >= p_max_attempts``. That is what stops a
        permanently-timing-out job from being requeued forever, so it is pinned
        here — if it were ever removed, ``mark_retryable`` would become an
        unbounded loop replaying a paid 16k-token render.
        """
        from src.database.rls_resolvers import CLAIM_FN_SQL

        assert "attempts = COALESCE(attempts, 0) + 1" in CLAIM_FN_SQL
        assert "COALESCE(v_attempts, 0) >= p_max_attempts" in CLAIM_FN_SQL
        assert "poison job" in CLAIM_FN_SQL


# ---------------------------------------------------------------------------
# 5. The ownership fence has to REACH the seams, not just exist in the SQL
# ---------------------------------------------------------------------------


class TestTheSeamsFenceOnTheClaimOwner:
    """Both verdict writes carry the worker id holding the claim.

    The behaviour of the fence itself is proven against a real database in
    ``tests/unit/test_job_queue.py``; what is proven here is the wiring — that
    the worker id actually arrives at the two seams a failing job unwinds
    through. A fence the seams never pass an owner to is a fence that is always
    open.
    """

    async def test_the_outer_seam_fences_both_writes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        marks = MagicMock(
            mark_failed=AsyncMock(return_value=True),
            mark_retryable=AsyncMock(return_value=False),
        )
        monkeypatch.setattr("src.api.routes._helpers.JobManager", marks)

        with claim_owner("podA"):
            await _record_background_failure(
                MagicMock(), "job-1", httpx.ReadTimeout("", request=_WM_REQUEST)
            )

        assert marks.mark_retryable.await_args.kwargs["worker_id"] == "podA"
        assert marks.mark_failed.await_args.kwargs["worker_id"] == "podA"

    async def test_a_fenced_off_requeue_still_falls_back_to_the_terminal_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lost race must not end in silence: the caller still records a verdict."""
        marks = MagicMock(
            mark_failed=AsyncMock(return_value=False),
            mark_retryable=AsyncMock(return_value=False),
        )
        monkeypatch.setattr("src.api.routes._helpers.JobManager", marks)

        with claim_owner("podA"):
            await _record_background_failure(
                MagicMock(), "job-1", httpx.ReadTimeout("", request=_WM_REQUEST)
            )

        marks.mark_retryable.assert_awaited_once()
        marks.mark_failed.assert_awaited_once()

    async def test_the_redesign_seam_fences_on_the_owner_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        jm = _job_manager(requeued=True)
        monkeypatch.setattr(design_builder, "JobManager", jm)
        obj, _ = _finalizer()

        with claim_owner("podB"):
            await obj._phase_finalize_failed(
                _db(), _job(), "job-1", "tenant-1", httpx.ReadTimeout("", request=_WM_REQUEST)
            )

        assert jm.mark_retryable.await_args.kwargs["worker_id"] == "podB"

    async def test_no_claim_owner_outside_a_worker_dispatch(self) -> None:
        """The default is None — the seams cannot invent an owner they don't have."""
        from src.services.job_manager import get_claim_owner

        assert get_claim_owner() is None
        with claim_owner("podA"):
            assert get_claim_owner() == "podA"
        assert get_claim_owner() is None
