"""Variant finalize must FAIL the job when every variant failed, rather than
marking it `completed` with no screenshots (a silent empty `visuals` list that
ships a videoless redesign downstream)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import src.services.design_builder as design_builder
from src.services.design_builder.phases import RedesignPhasesMixin
from src.services.design_builder.types import RenderOptions


def _phases(failures: list[BaseException] | None = None) -> RedesignPhasesMixin:
    obj = RedesignPhasesMixin.__new__(RedesignPhasesMixin)
    # The fan-out returns the exception that ended each FAILED variant; the
    # parent classifies its own verdict from them.
    obj._process_variants = AsyncMock(return_value=failures or [])  # type: ignore[method-assign]
    obj._dispatch_webhook = AsyncMock()  # type: ignore[method-assign]
    # The variant path now captures a 'before' screenshot; the real capture needs
    # a browser + storage, so stub it here (its own behaviour is tested below).
    obj._capture_variant_original = AsyncMock(  # type: ignore[method-assign]
        return_value={"desktop": "https://m/orig.png"}
    )
    return obj


def _job() -> MagicMock:
    job = MagicMock()
    job.styles = ["bold editorial", "swiss grid"]
    job.job_id = "job-x"
    job.url = "https://acme.test"
    job.scope = "redesign"
    job.style = "modern"
    return job


def _variant(style: str, status: str, err: str | None = None) -> MagicMock:
    v = MagicMock()
    v.style = style
    v.status = status
    v.error_message = err
    v.redesigned_html_path = "p.html" if status == "completed" else None
    v.screenshot_paths = {"desktop": "https://m/x.png"} if status == "completed" else None
    return v


def _fake_jm(variants: list[MagicMock]) -> MagicMock:
    jm = MagicMock()
    jm.get_variants = AsyncMock(return_value=variants)
    jm.mark_failed = AsyncMock()
    jm.mark_completed = AsyncMock()
    jm.record_usage = AsyncMock()
    jm.mark_retryable = AsyncMock(return_value=True)
    return jm


@pytest.mark.asyncio
async def test_finalize_fails_when_all_variants_failed(monkeypatch) -> None:
    jm = _fake_jm(
        [_variant("bold editorial", "failed", "boom"), _variant("swiss grid", "failed", "boom2")]
    )
    monkeypatch.setattr(design_builder, "JobManager", jm)
    obj = _phases()
    db = MagicMock(commit=AsyncMock())

    await obj._phase_process_variants_and_finalize(
        db,
        _job(),
        "job-x",
        "tenant",
        "<html></html>",
        MagicMock(),
        MagicMock(),
        session_factory=None,
        generate_screenshots=True,
        render_options=None,
    )

    jm.mark_failed.assert_awaited_once()
    assert "all 2 redesign variants failed" in jm.mark_failed.await_args.args[2]
    jm.mark_completed.assert_not_awaited()
    cast(Any, obj._dispatch_webhook).assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_completes_when_one_variant_succeeds(monkeypatch) -> None:
    jm = _fake_jm(
        [_variant("bold editorial", "completed"), _variant("swiss grid", "failed", "boom")]
    )
    monkeypatch.setattr(design_builder, "JobManager", jm)
    obj = _phases()
    db = MagicMock(commit=AsyncMock())

    await obj._phase_process_variants_and_finalize(
        db,
        _job(),
        "job-x",
        "tenant",
        "<html></html>",
        MagicMock(),
        MagicMock(),
        session_factory=None,
        generate_screenshots=True,
        render_options=None,
    )

    jm.mark_completed.assert_awaited_once()
    jm.mark_failed.assert_not_awaited()
    # The deterministic 'before' capture is threaded onto the completed job, so a
    # single flaked variant still yields original + 1 variant = 2 distinct visuals.
    assert jm.mark_completed.await_args.kwargs["original_screenshot_paths"] == {
        "desktop": "https://m/orig.png"
    }


@pytest.mark.asyncio
async def test_capture_variant_original_honors_render_viewports(monkeypatch) -> None:
    """The 'before' capture screenshots the live URL (no LLM) and honors the
    render_options viewports — desktop-only when the orchestrator asks for it,
    not the 3-viewport screenshot default."""
    obj = RedesignPhasesMixin.__new__(RedesignPhasesMixin)
    storage = MagicMock()
    storage.get_job_dir = MagicMock(return_value=Path("/tmp/job-x"))
    storage.persist_paths = AsyncMock(side_effect=lambda p: p)
    obj._storage = storage  # type: ignore[attr-defined]

    cap = AsyncMock(return_value={"desktop": "https://m/orig.png"})
    monkeypatch.setattr(design_builder, "capture_url_screenshots", cap)

    settings = MagicMock()
    settings.screenshot.enabled = True
    settings.screenshot.viewports = ["desktop", "tablet", "mobile"]
    settings.screenshot.timeout_ms = 30000

    out = await obj._capture_variant_original(
        _job(),
        settings,
        RenderOptions(viewports=("desktop",)),
        generate_screenshots=True,
    )

    assert out == {"desktop": "https://m/orig.png"}
    # viewports arg (positional 3rd) is the desktop-only render option, not the default
    cap.assert_awaited_once()
    assert cap.await_args is not None
    assert cap.await_args.args[2] == ["desktop"]


@pytest.mark.asyncio
async def test_capture_variant_original_skips_when_capture_disabled() -> None:
    obj = RedesignPhasesMixin.__new__(RedesignPhasesMixin)
    settings = MagicMock()
    settings.screenshot.enabled = True

    out = await obj._capture_variant_original(
        _job(),
        settings,
        RenderOptions(capture_original=False),
        generate_screenshots=True,
    )
    assert out is None


# ---------------------------------------------------------------------------
# The multi-variant path is the SAME defect as the single-style one
# ---------------------------------------------------------------------------
#
# `_phase_finalize_failed` learned that an upstream timeout is not a verdict.
# This path never went through it: when every variant died it wrote its own
# terminal `failed`, so for `styles > 1` a fleet-wide WM timeout still produced
# an unreclaimable failure the orchestrator (retry_on=()) propagated at once.

_WM = httpx.Request("POST", "https://worker-management.w2v.svc:8100/workers/w/message")


def _failed_two() -> list[MagicMock]:
    return [
        _variant("bold editorial", "failed", "ReadTimeout"),
        _variant("swiss grid", "failed", "ReadTimeout"),
    ]


@pytest.mark.asyncio
async def test_all_variants_timing_out_requeues_the_parent(monkeypatch) -> None:
    jm = _fake_jm(_failed_two())
    monkeypatch.setattr(design_builder, "JobManager", jm)
    obj = _phases([httpx.ReadTimeout("", request=_WM), httpx.ConnectTimeout("slow")])
    db = MagicMock(commit=AsyncMock())

    await obj._phase_process_variants_and_finalize(
        db,
        _job(),
        "job-x",
        "tenant",
        "<html></html>",
        MagicMock(),
        MagicMock(),
        session_factory=None,
        generate_screenshots=True,
        render_options=None,
    )

    jm.mark_retryable.assert_awaited_once()
    jm.mark_failed.assert_not_awaited()
    # The job has not failed — it is going to run again.
    cast(Any, obj._dispatch_webhook).assert_not_awaited()


@pytest.mark.asyncio
async def test_one_permanent_child_keeps_the_parent_terminal(monkeypatch) -> None:
    """The control. A single permanent child failure means re-running pays twice
    for the same answer, so unanimity is the bar."""
    jm = _fake_jm(_failed_two())
    monkeypatch.setattr(design_builder, "JobManager", jm)
    obj = _phases(
        [
            httpx.ReadTimeout("", request=_WM),
            ValueError("output_schema violation: 'sections' is not a list"),
        ]
    )
    db = MagicMock(commit=AsyncMock())

    await obj._phase_process_variants_and_finalize(
        db,
        _job(),
        "job-x",
        "tenant",
        "<html></html>",
        MagicMock(),
        MagicMock(),
        session_factory=None,
        generate_screenshots=True,
        render_options=None,
    )

    jm.mark_failed.assert_awaited_once()
    jm.mark_retryable.assert_not_awaited()
    cast(Any, obj._dispatch_webhook).assert_awaited_once()


@pytest.mark.asyncio
async def test_a_style_that_reported_no_failure_is_not_evidence_of_a_transient_one(
    monkeypatch,
) -> None:
    """Two styles failed, but only one produced an exception — the other may have
    been cancelled or never written. `all()` over that shorter list is vacuously
    unanimous, which is why the length check is load-bearing."""
    jm = _fake_jm(_failed_two())
    monkeypatch.setattr(design_builder, "JobManager", jm)
    obj = _phases([httpx.ReadTimeout("", request=_WM)])
    db = MagicMock(commit=AsyncMock())

    await obj._phase_process_variants_and_finalize(
        db,
        _job(),
        "job-x",
        "tenant",
        "<html></html>",
        MagicMock(),
        MagicMock(),
        session_factory=None,
        generate_screenshots=True,
        render_options=None,
    )

    jm.mark_failed.assert_awaited_once()
    jm.mark_retryable.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_fenced_off_parent_requeue_falls_back_to_terminal(monkeypatch) -> None:
    """Same lost-race fall-through as the single-style seam: if the requeue is
    refused (another worker owns the claim now, or the row left the queue), the
    parent must still record a verdict rather than sit in `pending` forever."""
    jm = _fake_jm(_failed_two())
    jm.mark_retryable = AsyncMock(return_value=False)
    monkeypatch.setattr(design_builder, "JobManager", jm)
    obj = _phases([httpx.ReadTimeout("", request=_WM), httpx.ReadTimeout("", request=_WM)])
    db = MagicMock(commit=AsyncMock())

    await obj._phase_process_variants_and_finalize(
        db,
        _job(),
        "job-x",
        "tenant",
        "<html></html>",
        MagicMock(),
        MagicMock(),
        session_factory=None,
        generate_screenshots=True,
        render_options=None,
    )

    jm.mark_retryable.assert_awaited_once()
    jm.mark_failed.assert_awaited_once()
