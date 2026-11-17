"""The configured WM output-token budget must reach every ``wm.message()`` call.

Regression guard for the terminal 422 ``Model token limit (N) exceeded before
any response was generated``: worker_management defaults to a 6000-token budget
when the caller omits ``max_tokens``, and the reasoning model behind it charges
its thinking against that budget. ``WorkerManagementSettings.max_tokens`` used to
be documented as "Unused: WM's MessageRequest forbids max_tokens" — that was
false, and the value was in fact never passed. These tests pin that it is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from src.config import get_settings

if TYPE_CHECKING:
    from w2v_common.clients import WMClient


class _StubWMResponse:
    def __init__(self, output: object) -> None:
        self.output = output
        self.usage_metrics = {"prompt_tokens": 1, "completion_tokens": 2}


class _RecordingWM:
    def __init__(self, output: object) -> None:
        self._output = output
        self.calls: list[dict] = []

    async def message(self, **kwargs: object) -> _StubWMResponse:
        self.calls.append(kwargs)
        return _StubWMResponse(self._output)


def test_default_budget_is_above_the_recorded_failure_values() -> None:
    """4000, 6000, and 8000 are all recorded production truncation/422 values for a
    full-page web-design generation (~16k tokens, per timeout_seconds' own docs);
    the default must match that documented need."""
    assert get_settings().wm.max_tokens == 16000


@pytest.mark.asyncio
async def test_assembly_plan_call_carries_the_configured_budget() -> None:
    from src.services.html_generator import HtmlGenerator

    wm = _RecordingWM({"blocks": [{"type": "hero", "content": {"headline": "H"}}]})
    gen = HtmlGenerator(wm_client=cast("WMClient", wm))
    await gen.generate_assembly_plan(description="A landing page")

    assert wm.calls, "no WM call recorded"
    for call in wm.calls:
        assert call["max_tokens"] == get_settings().wm.max_tokens


@pytest.mark.asyncio
async def test_html_generation_call_carries_the_configured_budget() -> None:
    from src.services.html_generator import HtmlGenerator

    wm = _RecordingWM("<html><body><h1>hi</h1></body></html>")
    gen = HtmlGenerator(wm_client=cast("WMClient", wm))
    await gen._call_claude("make me a page")

    assert wm.calls, "no WM call recorded"
    assert wm.calls[0]["max_tokens"] == get_settings().wm.max_tokens
