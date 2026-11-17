"""Tests for bounded variant fan-out and RenderOptions propagation.

``_process_variants`` used to ``asyncio.gather`` over every style at once —
each variant runs a multi-minute LLM call plus a full screenshot set, so an
unbounded fan-out hammered the redesign-html worker and multiplied render
memory. It is now capped at ``settings.redesign.variant_concurrency`` and
forwards the parent job's ``RenderOptions`` down to each variant render.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from src.config import get_settings
from src.services.design_builder.types import RenderOptions
from src.services.design_builder.variants import RedesignVariantsMixin


class _FakeSessionCtx:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *_):
        return False


def _session_factory():
    return _FakeSessionCtx()


def _mixin() -> RedesignVariantsMixin:
    return RedesignVariantsMixin.__new__(RedesignVariantsMixin)


def _job(n_styles: int) -> MagicMock:
    job = MagicMock()
    job.styles = [f"style{i}" for i in range(n_styles)]
    job.job_id = "job-x"
    return job


async def test_variant_fan_out_capped_at_configured_concurrency(monkeypatch) -> None:
    settings = get_settings()
    cap = settings.redesign.variant_concurrency
    assert cap >= 1

    live = 0
    peak = 0

    async def fake_process_variant(*_args, **_kwargs) -> None:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1

    obj = _mixin()
    monkeypatch.setattr(obj, "_process_variant", fake_process_variant)

    await obj._process_variants(
        session_factory=cast(Any, _session_factory),
        db=MagicMock(),
        job=_job(cap + 4),
        html="<html></html>",
        resolved_ctx=MagicMock(),
        settings=settings,
        generate_screenshots=True,
        render_options=None,
    )

    assert peak <= cap
    assert peak >= 1  # work actually ran


async def test_render_options_forwarded_to_each_variant(monkeypatch) -> None:
    captured: list[RenderOptions | None] = []

    async def fake_process_variant(*args, **_kwargs) -> None:
        # signature: (db, job, html, style, ctx, settings, gen_ss, render_options)
        captured.append(args[7])

    obj = _mixin()
    monkeypatch.setattr(obj, "_process_variant", fake_process_variant)

    opts = RenderOptions(full_page=False, viewports=("desktop",))
    await obj._process_variants(
        session_factory=cast(Any, _session_factory),
        db=MagicMock(),
        job=_job(3),
        html="<html></html>",
        resolved_ctx=MagicMock(),
        settings=get_settings(),
        generate_screenshots=True,
        render_options=opts,
    )

    assert len(captured) == 3
    assert all(c is opts for c in captured)


# ---------------------------------------------------------------------------
# Failure propagation out of the absorb
# ---------------------------------------------------------------------------
#
# A variant failure is absorbed so one bad style never takes out its siblings.
# The parent job's verdict nonetheless depends on WHAT killed the children (all
# transient => the parent is transient), so the exception has to leave the
# absorb by return value. It is also what the variant row records: a bare
# `httpx.ReadTimeout` stringifies to '', which used to persist an empty reason.


async def test_a_failed_variant_returns_its_exception_and_records_a_typed_reason(
    monkeypatch,
) -> None:
    import httpx

    import src.services.design_builder as design_builder

    boom = httpx.ReadTimeout("")
    jm = MagicMock(
        create_variant=AsyncMock(return_value=MagicMock(variant_id="v1")),
        mark_variant_failed=AsyncMock(),
    )
    monkeypatch.setattr(design_builder, "JobManager", jm)

    obj = _mixin()
    obj._check_cancelled = AsyncMock(return_value=False)  # type: ignore[method-assign]
    obj._html_gen = MagicMock(generate_page_chunked=AsyncMock(side_effect=boom))
    db = MagicMock(commit=AsyncMock(), rollback=AsyncMock())

    returned = await obj._process_variant(
        db, _job(1), "<html></html>", "modern", MagicMock(), get_settings(), False, None
    )

    assert returned is boom
    recorded = jm.mark_variant_failed.await_args.args[2]
    assert recorded, "an empty reason is not a record"
    assert "ReadTimeout" in recorded


async def test_the_fan_out_collects_every_child_failure(monkeypatch) -> None:
    """What the parent classifies on. A dropped exception silently downgrades an
    all-transient job to terminal (the unanimity check is length-sensitive)."""
    failures = [ValueError("a"), ValueError("b"), None, ValueError("d")]

    async def fake_process_variant(*_args, **_kwargs):
        return failures.pop(0)

    obj = _mixin()
    monkeypatch.setattr(obj, "_process_variant", fake_process_variant)

    collected = await obj._process_variants(
        session_factory=cast(Any, _session_factory),
        db=MagicMock(),
        job=_job(4),
        html="<html></html>",
        resolved_ctx=MagicMock(),
        settings=get_settings(),
        generate_screenshots=True,
        render_options=None,
    )

    assert [str(e) for e in collected] == ["a", "b", "d"]
