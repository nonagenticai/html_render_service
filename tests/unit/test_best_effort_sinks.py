"""Best-effort sinks must never fail the caller they serve.

Three handlers that exist purely to do something *optional* — write an audit
row, send a notification, render a comparison image — were narrow enough that
the optional work could take down the mandatory work around it.

The through-line is the same as the rest of this defect class: the enclosed call
raises something that derives from ``Exception`` but not ``OSError``, so a tuple
of builtins never matched it. What makes these three worth separating out is the
*inversion* they caused — in each case the failing thing was the least important
thing in the call stack:

* ``webhook_dispatcher._log_delivery`` only writes an audit row, but
  ``db.flush()`` raises ``sqlalchemy.exc.SQLAlchemyError``. A routine DB hiccup
  while **logging** a delivery escaped ``_log_delivery``, escaped ``dispatch``,
  and took down whatever was dispatching.
* ``finalize_phases._phase_finalize_failed`` dispatches the ``job.failed``
  webhook *inside the failure recorder*, before its own ``db.commit()``. An
  escaping notification error skipped that commit and lost the recorded reason —
  trading the thing that matters for the thing that does not.
* ``comparison_generator`` renders per-viewport comparison and diff images. PIL's
  ``Image.DecompressionBombError`` derives from ``Exception``, not ``OSError``
  (unlike ``UnidentifiedImageError`` and the truncated-file errors, which *are*
  ``OSError`` — that partial correctness is why the tuple looked right). One
  hostile screenshot took out every remaining viewport.

Each test pairs the failure case with a positive control, so none can pass by
the function having quietly become a no-op.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sqlalchemy.exc
from PIL import Image


class TestAuditLoggingNeverFailsTheDispatch:
    """``_log_delivery`` writes a row; nothing it does may reach the caller."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(
                sqlalchemy.exc.OperationalError("SELECT 1", {}, Exception("gone")),
                id="sqlalchemy-operational",
            ),
            pytest.param(sqlalchemy.exc.SQLAlchemyError("flush failed"), id="sqlalchemy-base"),
            pytest.param(RuntimeError("already covered"), id="runtime-still-covered"),
        ],
    )
    async def test_a_db_fault_while_logging_does_not_escape(self, exc: BaseException) -> None:
        """The audit write is optional; the dispatch it describes is not."""
        from src.services.webhook_dispatcher import WebhookDispatcher

        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock(side_effect=exc)

        dispatcher = WebhookDispatcher.__new__(WebhookDispatcher)
        await dispatcher._log_delivery(
            db,
            job_id="job-1",
            attempt=1,
            event_type="job.completed",
            payload={"event": "job.completed"},
            response_status=200,
            response_text="ok",
            status="delivered",
        )

    @pytest.mark.asyncio
    async def test_positive_control_row_is_actually_written(self) -> None:
        """With a healthy session the row really is added and flushed."""
        from src.services.webhook_dispatcher import WebhookDispatcher

        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        dispatcher = WebhookDispatcher.__new__(WebhookDispatcher)
        await dispatcher._log_delivery(
            db,
            job_id="job-1",
            attempt=1,
            event_type="job.completed",
            payload={"event": "job.completed"},
            response_status=200,
            response_text="ok",
            status="delivered",
        )

        db.add.assert_called_once()
        db.flush.assert_awaited_once()


class TestComparisonGeneratorSkipsOnlyTheBadViewport:
    """One hostile image must cost one viewport, not all of them."""

    @pytest.mark.asyncio
    async def test_decompression_bomb_does_not_abort_the_batch(self, tmp_path: Path) -> None:
        """PIL's bomb guard is an ``Exception``, not an ``OSError``.

        Before the fix it escaped the per-viewport handler and killed every
        remaining viewport's comparison *and* diff.
        """
        from src.services import comparison_generator as mod

        calls: list[str] = []

        def _side_by_side(orig: str, new: str, out: str) -> str:
            calls.append(out)
            if "desktop" in out:
                raise Image.DecompressionBombError("pixel bomb")
            return out

        def _pixel_diff(orig: str, new: str, out: str) -> str:
            calls.append(out)
            return out

        with (
            patch.object(mod, "generate_side_by_side", _side_by_side),
            patch.object(mod, "generate_pixel_diff", _pixel_diff),
        ):
            comparisons, diffs = await mod.generate_comparisons(
                {"desktop": "a.png", "mobile": "b.png"},
                {"desktop": "c.png", "mobile": "d.png"},
                tmp_path,
            )

        # The bombed viewport is skipped; the healthy one still renders.
        assert "desktop" not in comparisons
        assert "mobile" in comparisons
        # And the bomb did not prevent the desktop DIFF from being attempted —
        # the two try blocks are independent.
        assert "desktop" in diffs

    @pytest.mark.asyncio
    async def test_positive_control_all_viewports_render(self, tmp_path: Path) -> None:
        """Nothing raising means nothing is skipped."""
        from src.services import comparison_generator as mod

        with (
            patch.object(mod, "generate_side_by_side", lambda o, n, out: out),
            patch.object(mod, "generate_pixel_diff", lambda o, n, out: out),
        ):
            comparisons, diffs = await mod.generate_comparisons(
                {"desktop": "a.png", "mobile": "b.png"},
                {"desktop": "c.png", "mobile": "d.png"},
                tmp_path,
            )

        assert set(comparisons) == {"desktop", "mobile"}
        assert set(diffs) == {"desktop", "mobile"}


class TestTheHierarchyThatMadeTheseDead:
    """Executable documentation — this is what the builtins-only tuples missed."""

    def test_sqlalchemy_error_is_not_an_oserror(self) -> None:
        """``db.flush()`` is the only thing ``_log_delivery`` can fail on."""
        assert issubclass(sqlalchemy.exc.SQLAlchemyError, Exception)
        assert not issubclass(sqlalchemy.exc.SQLAlchemyError, OSError)

    def test_pil_bomb_is_not_an_oserror_but_its_siblings_are(self) -> None:
        """The partial correctness is the trap.

        ``UnidentifiedImageError`` really *is* an ``OSError``, so the old tuple
        handled the common bad-image case and looked right — while the hostile
        case walked straight past it.
        """
        assert not issubclass(Image.DecompressionBombError, OSError)
        assert issubclass(Image.UnidentifiedImageError, OSError)
