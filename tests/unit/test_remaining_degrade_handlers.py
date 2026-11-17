"""The last five handlers of the "degrade that could not fire" class.

PRs #67 and #68 closed the loud ones. These are the tail, and they carry three
*different* semantics — which is why they are fixed three different ways rather
than by one blanket widening:

* **Failure-recording seams** (``process_refinement_job``, ``_process_variant``)
  — a job's only chance to record why it died. Widened to ``except Exception``,
  because losing the recorded reason is strictly worse than over-catching, and
  ``CancelledError`` is a ``BaseException`` so a shutdown drain still passes
  through untouched.
* **Fail-soft with a known raiser** (``_dispatch_webhook``) — the client is
  httpx, so ``httpx.HTTPError`` is the one missing base.
* **Fail-soft where the raiser had to be traced** (critique revalidation,
  comparison generation) — these turned out to raise
  ``bs4.exceptions.ParserRejectedMarkup`` / ``StopParsing`` and PIL's
  ``DecompressionBombError`` respectively. Neither is an ``OSError``; both were
  escaping.

The through-line, as in the earlier PRs: the enclosed library raises something
that derives from ``Exception`` and not ``OSError``, so a tuple of builtins can
never match it. The existing suite did not drive any of these five seams — the
fixes were mutated back out and the suite stayed green. So each test below
injects the REAL library exception at the exact seam the function calls, and
pairs the degrade case with a positive control, so none can pass by the function
having quietly become a no-op.

Hermetic: no browser, no Redis, no network, no real PIL work.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import redis.exceptions
from bs4.exceptions import ParserRejectedMarkup, StopParsing
from PIL.Image import DecompressionBombError
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from sqlalchemy.exc import OperationalError

import src.services.design_builder as design_builder
import src.services.design_builder.lifecycle as lifecycle_mod
import src.services.design_builder.variants as variants_mod
import src.services.html_validator as html_validator_mod
from src.services.design_builder.finalize_phases import RedesignFinalizeMixin
from src.services.design_builder.helpers import RedesignHelpersMixin
from src.services.design_builder.lifecycle import RedesignLifecycleMixin
from src.services.design_builder.types import RenderOptions
from src.services.design_builder.variants import RedesignVariantsMixin

# Types that derive from Exception but NOT OSError — i.e. everything the old
# tuples were blind to. Reused across both record-the-failure seams, each one
# paired with the sink in those functions that really raises it.
OFF_TUPLE = [
    pytest.param(PlaywrightError("browser closed"), id="playwright-error"),
    pytest.param(PlaywrightTimeoutError("Timeout 30000ms exceeded"), id="playwright-timeout"),
    pytest.param(httpx.ConnectError("wm unreachable"), id="httpx-connect"),
    pytest.param(
        httpx.HTTPStatusError(
            "502",
            request=httpx.Request("POST", "https://wm.invalid/"),
            response=httpx.Response(502),
        ),
        id="httpx-status",
    ),
    pytest.param(redis.exceptions.ConnectionError("redis down"), id="redis-connection"),
    pytest.param(
        OperationalError("SELECT 1", {}, Exception("server closed the connection")),
        id="sqlalchemy-operational",
    ),
    pytest.param(KeyError("job_options"), id="keyerror"),
]

# Types the OLD tuple already caught. Kept so a regression that narrows a
# handler the *other* way — covering only the newly-named types — is caught too.
IN_TUPLE = [
    pytest.param(OSError("disk gone"), id="oserror"),
    pytest.param(ValueError("bad shape"), id="valueerror"),
    pytest.param(RuntimeError("boom"), id="runtimeerror"),
    pytest.param(TimeoutError("builtin timeout"), id="builtin-timeout"),
]

# Every httpx transport/status failure the webhook sink can produce.
HTTPX_FAILURES = [
    pytest.param(httpx.ConnectError("connection refused"), id="ConnectError"),
    pytest.param(httpx.ConnectTimeout("connect timed out"), id="ConnectTimeout"),
    pytest.param(httpx.ReadTimeout("read timed out"), id="ReadTimeout"),
    pytest.param(
        httpx.HTTPStatusError(
            "500",
            request=httpx.Request("POST", "https://hooks.acme.test/w"),
            response=httpx.Response(500),
        ),
        id="HTTPStatusError",
    ),
]


@dataclass
class _LLMResult:
    """Generation result double with real integer token counts."""

    text: str = "<html><body>after</body></html>"
    input_tokens: int = 1200
    output_tokens: int = 800


def _db() -> MagicMock:
    """Return a session double with awaitable ``commit``/``rollback``."""
    return MagicMock(commit=AsyncMock(), rollback=AsyncMock())


# ---------------------------------------------------------------------------
# 0. Executable documentation for why every one of these tuples was dead
# ---------------------------------------------------------------------------


class TestTheHierarchy:
    """The flat hierarchies and name collisions that hid this defect class."""

    @pytest.mark.parametrize(
        "exc_type",
        [
            PlaywrightError,
            PlaywrightTimeoutError,
            httpx.HTTPError,
            redis.exceptions.RedisError,
            ParserRejectedMarkup,
            StopParsing,
            DecompressionBombError,
        ],
    )
    def test_none_of_the_real_raisers_is_an_oserror(self, exc_type: type[BaseException]) -> None:
        """A builtins-only tuple cannot match any of these. Do not delete."""
        assert issubclass(exc_type, Exception)
        assert not issubclass(exc_type, OSError)
        assert not issubclass(exc_type, (ValueError, RuntimeError))

    def test_bs4_parser_rejection_is_the_untrusted_markup_case(self) -> None:
        """``FeatureNotFound`` was already covered; the parser-gave-up case was not.

        bs4's ``HTMLParserTreeBuilder.feed`` converts the ``AssertionError`` that
        ``html.parser`` raises on fatal markup into ``ParserRejectedMarkup``, so
        the exception the validator really hits on LLM-authored HTML is exactly
        the one the old tuple could not see.
        """
        from bs4.exceptions import FeatureNotFound

        assert issubclass(FeatureNotFound, ValueError)
        assert not issubclass(ParserRejectedMarkup, ValueError)

    def test_pillow_decode_errors_are_oserror_but_the_bomb_is_not(self) -> None:
        """Why the comparison handler needed exactly ONE addition, not a widening.

        Pillow's decode failures (``UnidentifiedImageError``, truncated file)
        subclass ``OSError`` and were always caught. ``DecompressionBombError``
        is the sole outlier — and it is the one a tall full-page capture pair
        can actually trip.
        """
        from PIL import UnidentifiedImageError

        assert issubclass(UnidentifiedImageError, OSError)
        assert not issubclass(DecompressionBombError, OSError)

    def test_cancellation_is_a_baseexception(self) -> None:
        """Why ``except Exception`` is a net and not a blanket."""
        assert not issubclass(asyncio.CancelledError, Exception)
        assert issubclass(asyncio.CancelledError, BaseException)


# ---------------------------------------------------------------------------
# Site 1. lifecycle.process_refinement_job — the refinement job's ONLY
#         record-the-failure seam. Records and RE-RAISES.
# ---------------------------------------------------------------------------


def _refinement_job() -> MagicMock:
    """Return a refinement job row double (built-in style → no StyleManager hit)."""
    job = MagicMock()
    job.job_id = "refine-1"
    job.parent_job_id = "parent-1"
    job.url = "https://acme.test/pricing"
    job.style = "modern"
    job.page_type = "pricing"
    job.brand_info = None
    job.site_analysis = None
    job.feedback_text = "tighten the hero copy"
    return job


def _parent_job() -> MagicMock:
    parent = MagicMock()
    parent.job_id = "parent-1"
    parent.redesigned_html_path = "parent-1/redesign.html"
    return parent


class _Refiner(RedesignLifecycleMixin, RedesignHelpersMixin):
    """The refinement lane composed as ``RedesignService`` composes it.

    The lifecycle mixin calls helpers-mixin members (``_emit_render_cost_event``
    on the billing path), and the mixins do not inherit one another — they meet
    only at ``RedesignService``. Instantiating ``RedesignLifecycleMixin`` alone
    therefore models a class that does not exist in production, and every such
    cross-mixin call raises ``AttributeError`` here while working in the real
    service. Mirrors ``_FailureFinalizer`` below.
    """


class TestRefinementJobRecordsItsFailure:
    """``process_refinement_job`` must always record why it died."""

    def _lifecycle(self, refine: AsyncMock) -> tuple[RedesignLifecycleMixin, AsyncMock]:
        """Build the mixin with ``refine_full_page`` as the failing/succeeding seam.

        The refine call is the real sink: it is a ``worker_management`` HTTP
        round-trip (httpx), sandwiched between storage reads and a screenshot
        render (Playwright) — none of which the old tuple could see.
        """
        obj = _Refiner.__new__(_Refiner)
        obj._storage = MagicMock(
            retrieve_text=AsyncMock(return_value="<html><body>before</body></html>"),
            get_job_dir=MagicMock(return_value=Path("/tmp/refine-1")),
            store_html=AsyncMock(return_value="refine-1/redesign.html"),
        )
        obj._html_gen = MagicMock(refine_full_page=refine, _model="test-model")
        dispatch = AsyncMock()
        obj._dispatch_webhook = dispatch
        return obj, dispatch

    def _patch_env(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        settings = MagicMock()
        # Screenshots off: this test drives the failure seam, not the renderer.
        settings.screenshot.enabled = False
        monkeypatch.setattr(lifecycle_mod, "get_settings", lambda: settings)
        monkeypatch.setattr(lifecycle_mod, "estimate_cost", lambda *_a, **_kw: 0.0)

        jm = MagicMock()
        jm.get_job = AsyncMock(side_effect=[_refinement_job(), _parent_job()])
        jm.mark_processing = AsyncMock()
        jm.mark_completed = AsyncMock()
        jm.mark_failed = AsyncMock()
        jm.record_usage = AsyncMock()
        monkeypatch.setattr(design_builder, "JobManager", jm)
        return jm

    @pytest.mark.parametrize("failure", OFF_TUPLE + IN_TUPLE)
    async def test_failure_is_recorded_and_still_propagates(
        self, failure: BaseException, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whatever the refine call raises, the job gets a recorded reason.

        Under the old narrow tuple the off-tuple half of this parametrisation
        skipped the handler entirely: ``mark_failed`` never ran, the row kept its
        ``processing`` state with no error, usage was never counted and the
        tenant's ``job.failed`` webhook never fired.
        """
        jm = self._patch_env(monkeypatch)
        obj, dispatch = self._lifecycle(AsyncMock(side_effect=failure))
        db = _db()

        with pytest.raises(type(failure)):
            await obj.process_refinement_job(db, "refine-1", "tenant-1")

        jm.mark_failed.assert_awaited_once()
        assert jm.mark_failed.await_args.args[1] == "refine-1"
        jm.record_usage.assert_awaited_once()
        assert jm.record_usage.await_args.kwargs["is_error"] is True
        assert dispatch.await_args_list[-1].args[2] == "job.failed"
        db.commit.assert_awaited()
        jm.mark_completed.assert_not_awaited()

    async def test_positive_control_success_marks_completed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The happy path still completes — the handler is not a blanket sink."""
        jm = self._patch_env(monkeypatch)
        obj, dispatch = self._lifecycle(AsyncMock(return_value=_LLMResult()))
        db = _db()

        await obj.process_refinement_job(db, "refine-1", "tenant-1")

        jm.mark_completed.assert_awaited_once()
        jm.mark_failed.assert_not_awaited()
        assert jm.record_usage.await_args.kwargs["pages_redesigned"] == 1
        assert dispatch.await_args_list[-1].args[2] == "job.completed"

    async def test_cancellation_propagates_and_is_not_recorded_as_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shutdown drain must not be stamped onto the row as a job failure.

        ``CancelledError`` is a ``BaseException``, so the widened clause does not
        see it: the task is cancelled and the row is left for startup reclaim.
        Work that was cut short is not the same as work that failed.
        """
        jm = self._patch_env(monkeypatch)
        obj, _dispatch = self._lifecycle(AsyncMock(side_effect=asyncio.CancelledError()))
        db = _db()

        with pytest.raises(asyncio.CancelledError):
            await obj.process_refinement_job(db, "refine-1", "tenant-1")

        jm.mark_failed.assert_not_awaited()
        jm.mark_completed.assert_not_awaited()


# ---------------------------------------------------------------------------
# Site 2. variants._process_variant — the variant's own failure recorder.
#         ABSORBS (never re-raises), so the DB write is the only trace.
# ---------------------------------------------------------------------------


class TestVariantFailureIsRecorded:
    """``_process_variant`` — a dead variant must be rolled back and marked."""

    def _variants(self, generate: AsyncMock) -> RedesignVariantsMixin:
        obj = RedesignVariantsMixin.__new__(RedesignVariantsMixin)
        obj._storage = MagicMock(
            store_html=AsyncMock(return_value="job-1/variant_modern.html"),
            get_job_dir=MagicMock(return_value=Path("/tmp/job-1")),
            persist_paths=AsyncMock(side_effect=lambda paths: paths),
        )
        obj._html_gen = MagicMock(generate_page_chunked=generate)
        obj._check_cancelled = AsyncMock(return_value=False)
        obj._dispatch_webhook = AsyncMock()
        return obj

    def _patch_env(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        monkeypatch.setattr(variants_mod, "estimate_cost", lambda *_a, **_kw: 0.0)
        jm = MagicMock()
        variant = MagicMock()
        variant.variant_id = "variant-1"
        jm.create_variant = AsyncMock(return_value=variant)
        jm.mark_variant_completed = AsyncMock()
        jm.mark_variant_failed = AsyncMock()
        monkeypatch.setattr(design_builder, "JobManager", jm)
        return jm

    def _job(self) -> MagicMock:
        job = MagicMock()
        job.job_id = "job-1"
        job.url = "https://acme.test/"
        job.page_type = "landing"
        job.critique_findings = None
        job.brand_info = None
        job.site_analysis = None
        job.tenant_id = "tenant-1"
        return job

    async def _run(self, obj: RedesignVariantsMixin, db: MagicMock) -> None:
        settings = MagicMock()
        # Screenshots off: the generation call is the seam under test.
        settings.screenshot.enabled = False
        await obj._process_variant(
            db,
            self._job(),
            "<html><body>before</body></html>",
            "modern",
            MagicMock(),
            settings,
        )

    @pytest.mark.parametrize("failure", OFF_TUPLE + IN_TUPLE)
    async def test_failure_is_absorbed_and_marked(
        self, failure: BaseException, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The variant is rolled back and marked failed, and nothing escapes.

        This handler absorbs — the caller only ever learns of the failure by
        reading the row — so under the old narrow tuple an off-tuple exception
        left the variant stuck ``pending`` AND left the half-open transaction
        from the failed generation unrolled.
        """
        jm = self._patch_env(monkeypatch)
        obj = self._variants(AsyncMock(side_effect=failure))
        db = _db()

        await self._run(obj, db)  # must not raise

        db.rollback.assert_awaited_once()
        jm.mark_variant_failed.assert_awaited_once()
        assert jm.mark_variant_failed.await_args.args[1] == "variant-1"
        jm.mark_variant_completed.assert_not_awaited()

    async def test_positive_control_success_marks_completed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A healthy variant still completes and dispatches its webhook."""
        jm = self._patch_env(monkeypatch)
        obj = self._variants(AsyncMock(return_value=_LLMResult()))
        db = _db()

        await self._run(obj, db)

        jm.mark_variant_completed.assert_awaited_once()
        jm.mark_variant_failed.assert_not_awaited()
        db.rollback.assert_not_awaited()
        hook = cast(AsyncMock, obj._dispatch_webhook)
        webhook_call = hook.await_args
        assert webhook_call is not None
        assert webhook_call.args[2] == "variant.completed"

    async def test_cancellation_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Widening to ``except Exception`` must not absorb a shutdown drain."""
        jm = self._patch_env(monkeypatch)
        obj = self._variants(AsyncMock(side_effect=asyncio.CancelledError()))
        db = _db()

        with pytest.raises(asyncio.CancelledError):
            await self._run(obj, db)

        jm.mark_variant_failed.assert_not_awaited()


# ---------------------------------------------------------------------------
# Site 3. helpers._dispatch_webhook — httpx is the transport.
# ---------------------------------------------------------------------------


def _webhook_job() -> MagicMock:
    job = MagicMock()
    job.job_id = "job-webhook"
    job.status = "failed"
    job.webhook_url = "https://hooks.acme.test/w"
    job.webhook_secret = None
    job.batch_id = None
    return job


class TestWebhookDispatchDegrades:
    """``_dispatch_webhook`` is best-effort; an httpx fault must not escape."""

    def _helpers(self, dispatch: AsyncMock) -> RedesignHelpersMixin:
        obj = RedesignHelpersMixin.__new__(RedesignHelpersMixin)
        obj._webhook = MagicMock(dispatch=dispatch)
        return obj

    @pytest.mark.parametrize("failure", HTTPX_FAILURES)
    async def test_transport_failure_is_logged_and_absorbed(
        self, failure: BaseException, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A dead tenant endpoint costs the notification, not the caller."""
        obj = self._helpers(AsyncMock(side_effect=failure))
        with caplog.at_level(logging.WARNING, logger="src.services.design_builder.helpers"):
            await obj._dispatch_webhook(MagicMock(), _webhook_job(), "job.failed", error="boom")
        assert any(
            "Webhook dispatch failed for job job-webhook" in record.message
            for record in caplog.records
        )

    @pytest.mark.parametrize("failure", IN_TUPLE)
    async def test_previously_caught_types_still_absorbed(self, failure: BaseException) -> None:
        """Adding ``httpx.HTTPError`` must not have dropped the builtins."""
        obj = self._helpers(AsyncMock(side_effect=failure))
        await obj._dispatch_webhook(MagicMock(), _webhook_job(), "job.failed")

    async def test_positive_control_dispatch_is_actually_called(self) -> None:
        """The happy path really builds and sends the event, with its payload."""
        dispatch = AsyncMock(return_value=True)
        obj = self._helpers(dispatch)

        await obj._dispatch_webhook(MagicMock(), _webhook_job(), "job.completed")

        dispatch.assert_awaited_once()
        assert dispatch.await_args is not None
        kwargs = dispatch.await_args.kwargs
        assert kwargs["event_type"] == "job.completed"
        assert kwargs["webhook_url"] == "https://hooks.acme.test/w"
        assert kwargs["payload"]["event"] == "job.completed"

    async def test_no_webhook_configured_is_a_silent_noop(self) -> None:
        """Guard the early-return contract (no dispatcher configured → nothing)."""
        obj = RedesignHelpersMixin.__new__(RedesignHelpersMixin)
        obj._webhook = None
        await obj._dispatch_webhook(MagicMock(), _webhook_job(), "job.completed")


class _FailureFinalizer(RedesignFinalizeMixin, RedesignHelpersMixin):
    """Composite of exactly the two mixins the job-failure path spans."""


class TestFailurePathSurvivesAWebhookTransportError:
    """The consequence that makes site 3 load bearing, end-to-end.

    ``_phase_finalize_failed`` dispatches ``job.failed`` *before* committing the
    failure state. With ``httpx.HTTPError`` outside the net, a tenant whose
    webhook endpoint was merely down lost the recorded reason for their job — a
    degrade handler taking out the thing it was protecting.
    """

    async def test_commit_still_happens_when_the_webhook_transport_dies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Degrade AND flow continues: the failure state is still persisted."""
        jm = MagicMock(mark_failed=AsyncMock(), record_usage=AsyncMock())
        monkeypatch.setattr(design_builder, "JobManager", jm)

        obj = _FailureFinalizer.__new__(_FailureFinalizer)
        obj._webhook = MagicMock(dispatch=AsyncMock(side_effect=httpx.ConnectError("down")))
        db = _db()

        await obj._phase_finalize_failed(
            db, _webhook_job(), "job-webhook", "tenant-1", RuntimeError("original cause")
        )

        jm.mark_failed.assert_awaited_once()
        # The persisted string now carries the exception TYPE as well as its
        # message — see tests/unit/test_transient_upstream_retry.py for why
        # (a bare `str(exc)` was empty for the failure that actually shipped).
        assert jm.mark_failed.await_args.args[2] == "RuntimeError: original cause"
        db.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# Site 4. helpers._run_critique_adapter_revalidation — bs4 is the sink.
# ---------------------------------------------------------------------------

_CLEAN_HTML = (
    '<html lang="en"><head><title>Acme</title>'
    '<meta name="description" content="Acme pricing"></head>'
    '<body><nav><a href="/">Home</a></nav><main><h1>Pricing</h1>'
    '<img src="/hero.png" alt="Hero" loading="lazy"></main>'
    "<footer>Acme</footer></body></html>"
)


class TestCritiqueRevalidationDegrades:
    """``_run_critique_adapter_revalidation`` — enrichment metadata, fail-soft."""

    def _helpers(self) -> RedesignHelpersMixin:
        return RedesignHelpersMixin.__new__(RedesignHelpersMixin)

    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param(ParserRejectedMarkup("html.parser gave up"), id="ParserRejectedMarkup"),
            pytest.param(StopParsing("tree builder cannot continue"), id="StopParsing"),
        ],
    )
    def test_parser_rejection_degrades_to_the_documented_dict(
        self, failure: BaseException, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Untrusted LLM markup the parser refuses must not fail the job."""
        monkeypatch.setattr(
            html_validator_mod.HtmlValidator, "validate", MagicMock(side_effect=failure)
        )
        result = self._helpers()._run_critique_adapter_revalidation("<html>@#$%</html>")

        assert result["adapters_ran"] == []
        assert result["passed"] is False
        assert result["findings_count"] == 0
        assert result["severity_counts"] == {}
        assert result["findings"] == []
        assert str(failure) in result["error"]

    @pytest.mark.parametrize("failure", IN_TUPLE)
    def test_previously_caught_types_still_degrade(
        self, failure: BaseException, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Widening must not drop what the old tuple already handled."""
        monkeypatch.setattr(
            html_validator_mod.HtmlValidator, "validate", MagicMock(side_effect=failure)
        )
        result = self._helpers()._run_critique_adapter_revalidation("<html></html>")
        assert result["passed"] is False
        assert result["adapters_ran"] == []

    def test_positive_control_real_validator_produces_a_real_report(self) -> None:
        """The un-patched validator runs and reports the adapter as having run."""
        result = self._helpers()._run_critique_adapter_revalidation(_CLEAN_HTML)

        assert result["adapters_ran"] == ["deterministic_html_validator"]
        assert "error" not in result
        assert isinstance(result["findings_count"], int)

    def test_positive_control_blocking_findings_flip_passed_false(self) -> None:
        """A critical a11y finding must still fail the gate — not always-True."""
        result = self._helpers()._run_critique_adapter_revalidation(
            '<html lang="en"><body><main><h1>x</h1><img src="/a.png"></main></body></html>'
        )

        assert result["passed"] is False
        assert result["findings_count"] >= 1
        assert result["severity_counts"].get("critical", 0) >= 1

    def test_major_a11y_finding_is_blocking(self) -> None:
        """A `major` a11y violation must FAIL the gate.

        The validator emits only `critical`/`major`/`minor` — it never produces
        `high`. The gate previously checked ``("critical", "high")``, so a page
        whose only a11y defect was `major` (e.g. an empty anchor) reported
        ``passed=True`` and the tenant-facing a11y report lied. This HTML has a
        single empty link (major) and no critical a11y issue: it must block.
        """
        result = self._helpers()._run_critique_adapter_revalidation(
            '<html lang="en"><body><nav><a href="/x"></a></nav>'
            "<main><h1>x</h1></main><footer>f</footer></body></html>"
        )

        assert result["severity_counts"].get("major", 0) >= 1
        assert result["severity_counts"].get("critical", 0) == 0
        assert result["passed"] is False

    def test_critical_beyond_display_cutoff_still_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `critical` ranked past the 25-row display cutoff must still block.

        The severity aggregation / blocking decision must run over EVERY a11y
        finding, not the truncated ``[:25]`` display slice. Here 25 `minor`
        findings precede a single `critical`: the old truncated aggregation
        never saw the critical and passed the job; the gate must fail it while
        the display rows stay capped at 25.
        """
        from src.services.html_validator import (
            CATEGORY_A11Y,
            SEVERITY_CRITICAL,
            SEVERITY_MINOR,
            ValidationFinding,
            ValidationReport,
        )

        findings = [
            ValidationFinding(
                category=CATEGORY_A11Y,
                severity=SEVERITY_MINOR,
                description=f"minor #{i}",
                recommendation="fix it",
            )
            for i in range(25)
        ]
        findings.append(
            ValidationFinding(
                category=CATEGORY_A11Y,
                severity=SEVERITY_CRITICAL,
                description="critical past the cutoff",
                recommendation="add alt text",
            )
        )
        monkeypatch.setattr(
            html_validator_mod.HtmlValidator,
            "validate",
            MagicMock(return_value=ValidationReport(findings=findings)),
        )

        result = self._helpers()._run_critique_adapter_revalidation("<html></html>")

        assert result["severity_counts"].get("critical", 0) == 1
        assert result["severity_counts"].get("minor", 0) == 25
        assert result["findings_count"] == 26
        assert len(result["findings"]) == 25  # display rows stay truncated
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# Site 5. finalize_phases._phase_screenshots — the generate_comparisons arm.
# ---------------------------------------------------------------------------


def _screenshot_settings() -> MagicMock:
    settings = MagicMock()
    settings.screenshot.enabled = True
    settings.screenshot.viewports = ["desktop"]
    settings.screenshot.timeout_ms = 30000
    return settings


class TestComparisonGenerationDegrades:
    """The ``generate_comparisons`` arm — a lost diff must not lose the render."""

    def _finalize(self) -> RedesignFinalizeMixin:
        obj = RedesignFinalizeMixin.__new__(RedesignFinalizeMixin)
        obj._storage = MagicMock(
            get_job_dir=MagicMock(return_value=Path("/tmp/job-cmp")),
            persist_paths=AsyncMock(side_effect=lambda paths: paths),
            retrieve_text=AsyncMock(return_value="<html><body>after</body></html>"),
        )
        return obj

    def _patch_captures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            design_builder,
            "capture_url_screenshots",
            AsyncMock(return_value={"desktop": "/tmp/orig/desktop.png"}),
        )
        monkeypatch.setattr(
            design_builder,
            "render_from_html_string",
            AsyncMock(return_value={"desktop": "/tmp/new/desktop.png"}),
        )

    async def _call(self) -> Any:
        job = MagicMock()
        job.job_id = "job-cmp"
        job.url = "https://acme.test/"
        return await self._finalize()._phase_screenshots(
            job,
            "job-cmp",
            "job-cmp/redesign.html",
            _screenshot_settings(),
            True,
            RenderOptions(viewports=("desktop",)),
        )

    async def test_decompression_bomb_degrades_to_no_comparison(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An oversized capture pair costs the diff, not the finished redesign.

        ``DecompressionBombError`` derives from ``Exception``, so under the old
        tuple it escaped this "enrichment" handler and hard-failed a job that had
        already been generated, rendered and paid for.
        """
        self._patch_captures(monkeypatch)
        monkeypatch.setattr(
            design_builder,
            "generate_comparisons",
            AsyncMock(side_effect=DecompressionBombError("Image size exceeds limit")),
        )

        result = await self._call()

        assert result.comparison_paths is None
        assert result.diff_paths is None
        # Positive control in the same assertion block: the phase completed and
        # the real render survived, so the degrade did not abandon the phase.
        assert result.original_paths == {"desktop": "/tmp/orig/desktop.png"}
        assert result.screenshot_paths == {"desktop": "/tmp/new/desktop.png"}
        assert result.screenshots_count == 1

    @pytest.mark.parametrize("failure", IN_TUPLE)
    async def test_previously_caught_types_still_degrade(
        self, failure: BaseException, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pillow's decode failures are ``OSError`` subclasses — still covered."""
        self._patch_captures(monkeypatch)
        monkeypatch.setattr(design_builder, "generate_comparisons", AsyncMock(side_effect=failure))

        result = await self._call()

        assert result.comparison_paths is None
        assert result.screenshot_paths == {"desktop": "/tmp/new/desktop.png"}

    async def test_positive_control_comparisons_are_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful comparison really populates both maps."""
        self._patch_captures(monkeypatch)
        monkeypatch.setattr(
            design_builder,
            "generate_comparisons",
            AsyncMock(
                return_value=(
                    {"desktop": "/tmp/cmp/comparison_desktop.png"},
                    {"desktop": "/tmp/cmp/diff_desktop.png"},
                )
            ),
        )

        result = await self._call()

        assert result.comparison_paths == {"desktop": "/tmp/cmp/comparison_desktop.png"}
        assert result.diff_paths == {"desktop": "/tmp/cmp/diff_desktop.png"}
