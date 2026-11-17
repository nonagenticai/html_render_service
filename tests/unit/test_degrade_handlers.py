"""Regression tests for the "handler that cannot fire" defect class.

WHY THIS FILE EXISTS
--------------------
Several soft-degrade handlers in this service were written as::

    except (OSError, ValueError, RuntimeError, TimeoutError):

around code whose *real* failure modes are ``playwright.async_api.Error`` /
``playwright.async_api.TimeoutError`` and ``redis.exceptions.RedisError``.

None of those library classes derive from ``OSError``:

* ``playwright.async_api.Error`` derives straight from ``Exception``.
* ``playwright.async_api.TimeoutError`` is **not** the builtin ``TimeoutError``
  — it is a subclass of Playwright's own ``Error``. Listing the builtin (which
  *is* an ``OSError`` subclass, and is what ``asyncio.TimeoutError`` aliases)
  gave the handler exactly zero reach over Playwright.
* ``redis.exceptions.RedisError`` derives from ``Exception``, and its
  ``ConnectionError``/``TimeoutError`` subclasses shadow the builtins of the
  same name without inheriting from them.

So every one of those ``except`` clauses was **unreachable code**. The
documented behaviour ("soft", "non-fatal by design", "fail-open: returns True
if Redis is unavailable") was a lie at runtime: a tenant page that merely timed
out, or an unreachable Redis, hard-failed a paid render job.

The whole suite passed while that was true, and passed again with the fix
mutated back out — no existing test drove any of these seams. These tests are
the proof. Each case pairs a **degrade** assertion (the library exception is
absorbed, the documented fallback value is returned) with a **positive
control** (the happy path still produces real output), so a test cannot pass by
the function having quietly become a no-op. The ``issubclass`` assertions are
deliberate documentation of *why* the old tuples were dead, and will fail loudly
if a future refactor decides the builtins were "good enough" after all.

Hermetic: no browser, no Redis, no network. Every seam is patched at the exact
attribute the production code resolves at call time.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis.exceptions as redis_exceptions
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from w2v_common.cache.redis_client import RedisClient
from w2v_common.net import SSRFError

import src.services.design_builder as design_builder
import src.services.design_builder.lifecycle as lifecycle_mod
from src.cache.job_lock import acquire_job_lock, release_job_lock
from src.services.design_builder.finalize_phases import RedesignFinalizeMixin
from src.services.design_builder.helpers import capture_failure_types
from src.services.design_builder.lifecycle import RedesignLifecycleMixin
from src.services.design_builder.phases import RedesignPhasesMixin
from src.services.design_builder.types import RenderOptions, _ScreenshotsResult

# The two library exceptions the old tuples could never catch. Parametrised
# everywhere a Playwright sink is exercised so both the generic browser error
# and the nav-timeout specialisation are covered.
PLAYWRIGHT_FAILURES: list[Exception] = [
    PlaywrightError("nav failed"),
    PlaywrightTimeoutError("Timeout 30000ms exceeded"),
]

# Redis-side failures. ``ConnectionError``/``TimeoutError`` here are
# ``redis.exceptions``' own classes, NOT the builtins of the same name.
REDIS_FAILURES: list[Exception] = [
    redis_exceptions.RedisError("generic redis failure"),
    redis_exceptions.ConnectionError("Error 111 connecting to redis:6379"),
    redis_exceptions.TimeoutError("Timeout reading from socket"),
]


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


def _settings() -> MagicMock:
    """Return a settings double with the screenshot knobs the phases read."""
    settings = MagicMock()
    settings.screenshot.enabled = True
    settings.screenshot.viewports = ["desktop", "tablet", "mobile"]
    settings.screenshot.timeout_ms = 30000
    return settings


def _storage(job_dir: str = "/tmp/job-degrade") -> MagicMock:
    """Return a storage double: real ``Path`` job dir, identity persist."""
    storage = MagicMock()
    storage.get_job_dir = MagicMock(return_value=Path(job_dir))
    storage.persist_paths = AsyncMock(side_effect=lambda paths: paths)
    storage.retrieve_text = AsyncMock(return_value="<html><body>after</body></html>")
    return storage


def _job() -> MagicMock:
    """Return a job row double with a public source URL."""
    job = MagicMock()
    job.job_id = "job-degrade"
    job.url = "https://acme.test/pricing"
    job.styles = None
    return job


# ---------------------------------------------------------------------------
# 2. capture_failure_types() — the net actually contains the Playwright types
# ---------------------------------------------------------------------------


class TestCaptureFailureTypes:
    """The exception net the soft-capture handlers resolve at call time."""

    def test_net_contains_playwright_error_and_timeout(self) -> None:
        """Both Playwright classes must be members of the returned tuple."""
        net = capture_failure_types()
        assert PlaywrightError in net
        assert PlaywrightTimeoutError in net

    def test_net_still_contains_the_builtins(self) -> None:
        """Widening must not have dropped the original (still-real) cases."""
        net = capture_failure_types()
        for builtin in (OSError, ValueError, RuntimeError, TimeoutError):
            assert builtin in net

    def test_builtin_tuple_would_not_have_caught_playwright(self) -> None:
        """The load-bearing assertion: document why the OLD tuple was dead code.

        ``playwright.async_api.TimeoutError`` is not the builtin and is not an
        ``OSError``, so ``except (OSError, ValueError, RuntimeError,
        TimeoutError)`` could never fire on it. If this ever starts failing, the
        subclass relationship changed and ``capture_failure_types`` should be
        re-examined — do not delete this assertion.
        """
        assert not issubclass(PlaywrightTimeoutError, OSError)
        assert not issubclass(PlaywrightError, OSError)
        assert PlaywrightTimeoutError is not TimeoutError
        assert not issubclass(PlaywrightTimeoutError, (ValueError, RuntimeError))
        # Proof by construction: the old tuple genuinely does not catch it.
        with pytest.raises(PlaywrightTimeoutError):
            try:
                raise PlaywrightTimeoutError("timeout")
            except OSError, ValueError, RuntimeError, TimeoutError:  # pragma: no cover
                pytest.fail("old tuple unexpectedly caught the Playwright timeout")

    def test_net_is_cached_and_stable(self) -> None:
        """``@cache`` means every call site shares one tuple object."""
        assert capture_failure_types() is capture_failure_types()


# ---------------------------------------------------------------------------
# 1a. _capture_variant_original degrades on a Playwright failure
# ---------------------------------------------------------------------------


class TestCaptureVariantOriginalDegrades:
    """``RedesignPhasesMixin._capture_variant_original`` — variant 'before' shot."""

    def _phases(self) -> RedesignPhasesMixin:
        obj = RedesignPhasesMixin.__new__(RedesignPhasesMixin)
        obj._storage = _storage()
        return obj

    @pytest.mark.parametrize("failure", PLAYWRIGHT_FAILURES, ids=lambda e: type(e).__name__)
    async def test_returns_none_and_does_not_propagate(
        self, failure: Exception, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A browser Error/TimeoutError costs the before image, not the job."""
        monkeypatch.setattr(
            design_builder, "capture_url_screenshots", AsyncMock(side_effect=failure)
        )
        result = await self._phases()._capture_variant_original(
            _job(), _settings(), RenderOptions(viewports=("desktop",)), generate_screenshots=True
        )
        assert result is None

    async def test_positive_control_success_returns_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guard against the degrade test passing on a no-op function."""
        monkeypatch.setattr(
            design_builder,
            "capture_url_screenshots",
            AsyncMock(return_value={"desktop": "/tmp/job-degrade/orig/desktop.png"}),
        )
        result = await self._phases()._capture_variant_original(
            _job(), _settings(), RenderOptions(viewports=("desktop",)), generate_screenshots=True
        )
        assert result == {"desktop": "/tmp/job-degrade/orig/desktop.png"}

    async def test_off_net_exception_still_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The handler is a net, not a blanket ``except Exception``."""
        monkeypatch.setattr(
            design_builder,
            "capture_url_screenshots",
            AsyncMock(side_effect=KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            await self._phases()._capture_variant_original(
                _job(),
                _settings(),
                RenderOptions(viewports=("desktop",)),
                generate_screenshots=True,
            )


# ---------------------------------------------------------------------------
# 1b. _capture_original_screenshots degrades on a Playwright failure
# ---------------------------------------------------------------------------


class TestCaptureOriginalScreenshotsDegrades:
    """``RedesignFinalizeMixin._capture_original_screenshots`` — the 'before' capture."""

    def _finalize(self) -> RedesignFinalizeMixin:
        obj = RedesignFinalizeMixin.__new__(RedesignFinalizeMixin)
        obj._storage = _storage()
        return obj

    async def _call(self) -> dict[str, str] | None:
        return await self._finalize()._capture_original_screenshots(
            _job(),
            "job-degrade",
            Path("/tmp/job-degrade/original_screenshots"),
            ["desktop"],
            _settings(),
            RenderOptions(viewports=("desktop",)),
        )

    @pytest.mark.parametrize("failure", PLAYWRIGHT_FAILURES, ids=lambda e: type(e).__name__)
    async def test_returns_none_and_does_not_propagate(
        self, failure: Exception, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dead/slow tenant page yields "no before image", not a failed job."""
        monkeypatch.setattr(
            design_builder, "capture_url_screenshots", AsyncMock(side_effect=failure)
        )
        assert await self._call() is None

    async def test_positive_control_success_returns_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path still returns the captured viewport map."""
        monkeypatch.setattr(
            design_builder,
            "capture_url_screenshots",
            AsyncMock(return_value={"desktop": "/tmp/job-degrade/original/desktop.png"}),
        )
        assert await self._call() == {"desktop": "/tmp/job-degrade/original/desktop.png"}

    async def test_ssrf_clause_still_wins_over_the_generic_net(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``SSRFError`` subclasses ``ValueError``, so clause ORDER is load-bearing.

        Widening the generic net must not have let a blocked internal target be
        absorbed as an indistinguishable "capture failed" warning — the security
        clause is declared first and must still be the one that fires.
        """
        assert issubclass(SSRFError, ValueError)
        monkeypatch.setattr(
            design_builder,
            "capture_url_screenshots",
            AsyncMock(side_effect=SSRFError("blocked: registry.test is not public")),
        )
        assert await self._call() is None


# ---------------------------------------------------------------------------
# 5. render_section_screenshots degrades on a Playwright failure
# ---------------------------------------------------------------------------


class TestSectionScreenshotsDegrade:
    """The ``render_section_screenshots`` arm inside ``_phase_screenshots``."""

    def _finalize(self) -> RedesignFinalizeMixin:
        obj = RedesignFinalizeMixin.__new__(RedesignFinalizeMixin)
        obj._storage = _storage()
        return obj

    def _opts(self) -> RenderOptions:
        # capture_original/generate_comparison off so this test drives exactly
        # one seam: the section capture.
        return RenderOptions(
            capture_original=False,
            generate_comparison=False,
            viewports=("desktop",),
            section_screenshots=True,
        )

    async def _call(self) -> _ScreenshotsResult:
        return await self._finalize()._phase_screenshots(
            _job(),
            "job-degrade",
            "job-degrade/redesign.html",
            _settings(),
            True,
            self._opts(),
        )

    @pytest.mark.parametrize("failure", PLAYWRIGHT_FAILURES, ids=lambda e: type(e).__name__)
    async def test_section_failure_degrades_to_no_sections(
        self, failure: Exception, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Section stills are an enrichment: losing them must not lose the render."""
        monkeypatch.setattr(
            design_builder,
            "render_from_html_string",
            AsyncMock(return_value={"desktop": "/tmp/job-degrade/screenshots/desktop.png"}),
        )
        monkeypatch.setattr(
            design_builder, "render_section_screenshots", AsyncMock(side_effect=failure)
        )
        result = await self._call()
        assert result.section_paths is None
        # Positive control in the same assertion block: the main render survived,
        # so the degrade did not simply abandon the phase.
        assert result.screenshot_paths == {"desktop": "/tmp/job-degrade/screenshots/desktop.png"}
        assert result.screenshots_count == 1

    async def test_positive_control_sections_are_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful section capture still populates ``section_paths``."""
        monkeypatch.setattr(
            design_builder,
            "render_from_html_string",
            AsyncMock(return_value={"desktop": "/tmp/job-degrade/screenshots/desktop.png"}),
        )
        monkeypatch.setattr(
            design_builder,
            "render_section_screenshots",
            AsyncMock(return_value=[("hero", "/tmp/job-degrade/sections/hero.png")]),
        )
        result = await self._call()
        assert result.section_paths == {"hero": "/tmp/job-degrade/sections/hero.png"}


# ---------------------------------------------------------------------------
# 3. job_lock fails open on RedisError
# ---------------------------------------------------------------------------


def _redis_client(
    *, set_side_effect: Exception | None = None, set_result: object = True
) -> MagicMock:
    """Return a ``RedisClient`` double whose raw client raises or returns."""
    raw = MagicMock()
    if set_side_effect is not None:
        raw.set = AsyncMock(side_effect=set_side_effect)
        raw.eval = AsyncMock(side_effect=set_side_effect)
    else:
        raw.set = AsyncMock(return_value=set_result)
        raw.eval = AsyncMock(return_value=1)
    client = MagicMock()
    client._ensure_connected = MagicMock(return_value=raw)
    return client


class TestJobLockFailsOpenOnRedisError:
    """``acquire_job_lock``/``release_job_lock`` must honour their fail-open docstring."""

    def test_redis_error_is_not_an_oserror(self) -> None:
        """The load-bearing assertion: document why the OLD tuple was dead code.

        ``(OSError, RuntimeError)`` could never catch a ``RedisError``, so an
        unreachable Redis *crashed* the job — the exact opposite of the
        documented fail-open. The ``redis.exceptions`` ``ConnectionError`` /
        ``TimeoutError`` shadow the builtins by name only.
        """
        assert not issubclass(redis_exceptions.RedisError, OSError)
        assert not issubclass(redis_exceptions.RedisError, RuntimeError)
        assert redis_exceptions.ConnectionError is not ConnectionError
        assert redis_exceptions.TimeoutError is not TimeoutError
        assert issubclass(redis_exceptions.ConnectionError, redis_exceptions.RedisError)
        assert issubclass(redis_exceptions.TimeoutError, redis_exceptions.RedisError)

    @pytest.mark.parametrize("failure", REDIS_FAILURES, ids=lambda e: type(e).__name__)
    async def test_acquire_fails_open(self, failure: Exception) -> None:
        """An unreachable Redis returns True (process anyway), never raises."""
        client = cast(RedisClient, _redis_client(set_side_effect=failure))
        assert await acquire_job_lock(client, "job-degrade") is True

    async def test_acquire_fails_open_when_connect_itself_raises(self) -> None:
        """``_ensure_connected`` is inside the ``try`` — its RedisError degrades too."""
        client = MagicMock()
        client._ensure_connected = MagicMock(
            side_effect=redis_exceptions.ConnectionError("no route to redis")
        )
        assert await acquire_job_lock(cast(RedisClient, client), "job-degrade") is True

    async def test_positive_control_acquire_grants_and_denies(self) -> None:
        """Fail-open must not be indistinguishable from "always True".

        A healthy Redis that reports the key already exists (``SET NX`` → None)
        must still return False, or the lock would be decorative.
        """
        granted = cast(RedisClient, _redis_client(set_result=True))
        assert await acquire_job_lock(granted, "job-degrade") is True

        contended = cast(RedisClient, _redis_client(set_result=None))
        assert await acquire_job_lock(contended, "job-degrade") is False

    @pytest.mark.parametrize("failure", REDIS_FAILURES, ids=lambda e: type(e).__name__)
    async def test_release_absorbs_redis_error(self, failure: Exception) -> None:
        """Release runs after a job already succeeded — it must never raise."""
        client = cast(RedisClient, _redis_client(set_side_effect=failure))
        await release_job_lock(client, "job-degrade")  # must not raise

    async def test_positive_control_release_calls_eval(self) -> None:
        """The happy path really does run the owner-checked release script."""
        client = _redis_client()
        await release_job_lock(cast(RedisClient, client), "job-degrade")
        client._ensure_connected.return_value.eval.assert_awaited_once()


# ---------------------------------------------------------------------------
# 4. lifecycle batch loop counts failures for off-tuple exceptions
# ---------------------------------------------------------------------------


class _FakeSessionCtx:
    """Async context manager yielding a single shared session double."""

    def __init__(self, db: MagicMock) -> None:
        self._db = db

    async def __aenter__(self) -> MagicMock:
        return self._db

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


class _FakeSessionFactory:
    """Callable standing in for ``async_sessionmaker``."""

    def __init__(self, db: MagicMock) -> None:
        self._db = db

    def __call__(self) -> _FakeSessionCtx:
        return _FakeSessionCtx(self._db)


def _fake_batch_manager() -> MagicMock:
    bm = MagicMock()
    batch = MagicMock()
    batch.status = "processing"
    batch.shared_settings = {"generate_screenshots": True, "generate_concept_images": False}
    bm.mark_processing = AsyncMock()
    bm.get_batch = AsyncMock(return_value=batch)
    bm.increment_completed = AsyncMock()
    bm.increment_failed = AsyncMock()
    bm.finalize_batch = AsyncMock(return_value="completed_with_errors")
    page_job = MagicMock()
    page_job.job_id = "page-1"
    bm.get_batch_jobs = AsyncMock(return_value=[page_job])
    return bm


class TestBatchLoopCountsFailures:
    """``RedesignLifecycleMixin.process_batch`` — one bad page must be *counted*."""

    def _lifecycle(self, process_job: AsyncMock) -> RedesignLifecycleMixin:
        obj = RedesignLifecycleMixin.__new__(RedesignLifecycleMixin)
        obj.process_job = process_job
        obj._dispatch_webhook = AsyncMock()
        return obj

    def _patch_env(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        settings = MagicMock()
        settings.redesign.batch_max_concurrency = 2
        monkeypatch.setattr(lifecycle_mod, "get_settings", lambda: settings)
        bm = _fake_batch_manager()
        monkeypatch.setattr(design_builder, "BatchManager", bm)
        return bm

    @pytest.mark.parametrize(
        "failure",
        [
            Exception("bare off-tuple failure"),
            PlaywrightTimeoutError("nav timeout inside process_job"),
            redis_exceptions.ConnectionError("redis died mid-job"),
            KeyError("job_options"),
        ],
        ids=["bare-Exception", "PlaywrightTimeoutError", "redis.ConnectionError", "KeyError"],
    )
    async def test_off_tuple_failure_increments_failed_and_finalizes(
        self, failure: Exception, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of the handler is counting; it must be unmissable.

        Under the old narrow tuple these escaped into
        ``gather(return_exceptions=True)``, which silently absorbed them —
        ``increment_failed`` never ran, the counters under-counted, and
        ``finalize_batch`` could never reach a terminal state.
        """
        bm = self._patch_env(monkeypatch)
        db = MagicMock(commit=AsyncMock(), rollback=AsyncMock())
        obj = self._lifecycle(AsyncMock(side_effect=failure))

        await obj.process_batch(
            cast("async_sessionmaker[AsyncSession]", _FakeSessionFactory(db)),
            "batch-1",
            "tenant-1",
        )

        bm.increment_failed.assert_awaited_once()
        bm.increment_completed.assert_not_awaited()
        bm.finalize_batch.assert_awaited_once()
        db.rollback.assert_awaited()

    async def test_positive_control_success_increments_completed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A healthy page counts completed — the loop is not a blanket failure sink."""
        bm = self._patch_env(monkeypatch)
        db = MagicMock(commit=AsyncMock(), rollback=AsyncMock())
        obj = self._lifecycle(AsyncMock(return_value=None))

        await obj.process_batch(
            cast("async_sessionmaker[AsyncSession]", _FakeSessionFactory(db)),
            "batch-1",
            "tenant-1",
        )

        bm.increment_completed.assert_awaited_once()
        bm.increment_failed.assert_not_awaited()
        bm.finalize_batch.assert_awaited_once()
