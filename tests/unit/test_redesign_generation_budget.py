"""Per-job generation state: concurrent renders stay apart, and a render is BOUNDED.

Two defects an audit of #127 found in the per-section redesign path:

* the section jobs and image manifest of a render lived on the process-wide
  ``HtmlGenerator`` (``asgi.py`` builds ONE), so concurrent renders — a batch's pages,
  a job's variants, two queue workers — overwrote each other's: page A's rendered
  repair regenerated from page B's prompt;
* nothing bounded a render: up to 9 LLM calls per section over an uncapped section
  list, while the orchestrator's ``website_redesign`` stage gives ONE attempt 7200s.

Everything here is hermetic: a scripted worker_management, a scripted rendered probe.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, cast

import pytest

from src.config import get_settings
from src.services.batch_progress import batch_progress
from src.services.html_gen.rendered_checks import ContrastFailure, RenderedReport
from src.services.html_gen.section_run import GenerationBudget, SectionRun
from src.services.html_generator import HtmlGenerator

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

_DEGENERATE = "<section class='s'><style>.s{" + "--x-reverse" * 400
_CLEAN = "<section class='s'><style>.s{{color:var(--color-ink)}}</style><h2>{mark}</h2></section>"
_MARK_RE = re.compile(r"SITE-([A-Z])")


class _Resp:
    def __init__(self, output: str, completion_tokens: int = 500) -> None:
        self.output = f"```html\n{output}\n```"
        self.usage_metrics = {"prompt_tokens": 100, "completion_tokens": completion_tokens}


# ── 1. concurrent renders on ONE generator must not cross-contaminate ────────


class _SiteWM:
    """Answers each prompt for ITS site; the first draft is BAD (fails contrast), a
    repair regeneration is GOOD. Yields to the loop so concurrent renders interleave."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    async def message(self, **kwargs: object) -> _Resp:
        prompt = str(kwargs["message"])
        site = _MARK_RE.findall(prompt)
        assert len(set(site)) == 1, f"a prompt mixes sites: {sorted(set(site))}"
        repair = "RENDERED page measured" in prompt
        self.calls.append((site[0], repair))
        for _ in range(3):
            await asyncio.sleep(0)
        mark = f"{'GOOD' if repair else 'BAD'}-{site[0]}"
        return _Resp(_CLEAN.format(mark=mark))


async def _site_probe(page: str) -> RenderedReport:
    """Every BAD-<site> heading on the page fails contrast (no path: not colour-fixable)."""
    await asyncio.sleep(0)
    failures = [
        ContrastFailure(
            section=m.group(1),
            text=f"BAD-{m.group(2)}",
            ratio=1.2,
            required=4.5,
            color="rgb(0, 0, 0)",
            background="rgb(0, 0, 0)",
            selector="h2",
        )
        for m in re.finditer(
            r'data-section="([^"]+)"[^<]*>\s*<section[^>]*>.*?BAD-([A-Z])', page, re.S
        )
    ]
    return RenderedReport(contrast=failures)


def _source(site: str) -> str:
    return (
        f"<html><body><header><h1>SITE-{site} header</h1></header>"
        f"<section><h2>SITE-{site} intro</h2><p>SITE-{site} text</p></section>"
        f"<footer><p>SITE-{site} footer</p></footer></body></html>"
    )


async def test_concurrent_renders_on_one_generator_do_not_cross_contaminate() -> None:
    wm = _SiteWM()
    gen = HtmlGenerator(wm_client=cast("WMClient", wm), render_probe=_site_probe)

    results = await asyncio.gather(
        *(
            gen.generate_page_chunked(url=f"https://{s.lower()}.example/", html=_source(s))
            for s in "ABC"
        )
    )

    for site, result in zip("ABC", results, strict=True):
        others = {"A", "B", "C"} - {site}
        assert f"GOOD-{site}" in result.text, f"site {site} was not repaired from its own prompt"
        assert f"BAD-{site}" not in result.text
        assert not any(f"-{o}<" in result.text for o in others), f"site {site} shows another site"
        # tokens are this render's own: 3 drafts + 3 repairs, 100 prompt tokens each
        assert (result.input_tokens, result.output_tokens) == (600, 3000)
        assert result.quality_warnings == []
    assert sorted(wm.calls) == sorted(
        (s, r) for s in "ABC" for r in (False, True) for _ in range(3)
    )


# ── 2. the budget ────────────────────────────────────────────────────────────


class _ScriptWM:
    def __init__(self, replies: list[str], *, on_call=None) -> None:
        self._replies = replies
        self.prompts: list[str] = []
        self._on_call = on_call

    async def message(self, **kwargs: object) -> _Resp:
        self.prompts.append(str(kwargs["message"]))
        if self._on_call is not None:
            await self._on_call()
        return _Resp(self._replies[min(len(self.prompts), len(self._replies)) - 1])


def _sections(n: int) -> list[dict]:
    return [
        {"section_type": "section", "key": f"s{i}", "html": f"<section>s{i}</section>"}
        for i in range(n)
    ]


def _gen(wm: object) -> HtmlGenerator:
    return HtmlGenerator(wm_client=cast("WMClient", wm))


async def test_the_retry_cap_never_refuses_a_first_draft_and_ships_the_rest() -> None:
    clean = _CLEAN.format(mark="ok")
    # s0: two degenerate draws; s1 and s2: clean.
    replies = iter([_DEGENERATE, _DEGENERATE, clean, clean])

    class _WM(_ScriptWM):
        async def message(self, **kwargs: object) -> _Resp:
            self.prompts.append(str(kwargs["message"]))
            return _Resp(next(replies), 5970)

    wm = _WM([])
    run = SectionRun(budget=GenerationBudget(max_extra_calls=1))

    results = await _gen(wm).generate_sections(
        url="https://x.example", html="", sections=_sections(3), run=run
    )

    assert len(wm.prompts) == 4, "draft + ONE retry for s0, then a first draft each for s1, s2"
    assert list(results) == ["s1", "s2"]
    assert any("s0" in d and "DROPPED" in d for d in run.degradations), run.degradations
    assert any("retry/repair budget spent (1/1" in d for d in run.degradations)


async def test_past_the_deadline_no_call_starts_and_remaining_sections_are_dropped() -> None:
    now = [0.0]

    async def _slow() -> None:
        now[0] += 1000.0  # every call "takes" 1000s of job time

    wm = _ScriptWM([_CLEAN.format(mark="ok")], on_call=_slow)
    run = SectionRun(budget=GenerationBudget(deadline=2500.0, clock=lambda: now[0]))

    results = await _gen(wm).generate_sections(
        url="https://x.example", html="", sections=_sections(6), run=run
    )

    assert len(wm.prompts) == 3, "calls start at t=0, 1000, 2000; none at 3000 > deadline"
    assert list(results) == ["s0", "s1", "s2"]
    assert sum("DROPPED" in d for d in run.degradations) == 3
    assert any("generation deadline reached" in d for d in run.degradations)


async def test_the_deadline_cancels_a_call_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.services.html_gen.generator as generator_module
    import src.services.html_gen.section_run as section_run

    monkeypatch.setattr(section_run, "MIN_CALL_SECONDS", 0.1)
    monkeypatch.setattr(generator_module, "MIN_CALL_SECONDS", 0.1)

    async def _hang() -> None:
        await asyncio.sleep(30)

    wm = _ScriptWM([_CLEAN.format(mark="ok")], on_call=_hang)
    loop = asyncio.get_running_loop()
    run = SectionRun(budget=GenerationBudget(deadline=loop.time() + 0.5, clock=loop.time))

    t0 = loop.time()
    with pytest.raises(RuntimeError, match="no section could be generated"):
        await _gen(wm).generate_sections(
            url="https://x.example", html="", sections=_sections(2), run=run
        )

    assert loop.time() - t0 < 5, "the in-flight call was not cut at the deadline"
    assert any("during an LLM call" in d for d in run.degradations), run.degradations


async def test_every_retry_emits_a_progress_heartbeat() -> None:
    beats = []

    async def _beat() -> None:
        beats.append(1)

    clean = _CLEAN.format(mark="ok")
    replies = iter([_DEGENERATE, _DEGENERATE, clean])

    class _WM(_ScriptWM):
        async def message(self, **kwargs: object) -> _Resp:
            self.prompts.append(str(kwargs["message"]))
            return _Resp(next(replies), 5970)

    wm = _WM([])
    with batch_progress(_beat):
        await _gen(wm).generate_sections(
            url="https://x.example", html="", sections=_sections(1), run=SectionRun()
        )

    assert len(wm.prompts) == 3
    assert len(beats) == 3, "one heartbeat per retry (2) + one per finished section (1)"


async def test_page_chunked_reports_degradations_as_quality_warnings() -> None:
    wm = _ScriptWM([_DEGENERATE, _CLEAN.format(mark="ok")])
    budget = GenerationBudget(max_extra_calls=0)

    result = await _gen(wm).generate_page_chunked(
        url="https://x.example/", html=_source("A"), budget=budget
    )

    assert result.quality_warnings, "the dropped section is on the record"
    assert any("DROPPED" in w for w in result.quality_warnings)


def test_the_default_deadline_leaves_the_orchestrator_15_percent_headroom() -> None:
    """Derived from w2v_orchestrator website_review `website_redesign` (origin/main):
    one attempt = 7200s = start 60 + claim queue 1810 + wait (1+7)x(590+10)+159 + result
    120. The job (~60s before generation, ~500s after) must fit BOTH the wait ladder
    and the whole attempt with >= 15% headroom."""
    attempt_s, start_s, queue_s, result_s = 7200, 60, 1810, 120
    wait_ladder_s = (1 + 7) * (590 + 10) + 159
    worst_job_s = 60 + get_settings().redesign.generation_deadline_seconds + 500
    assert worst_job_s <= wait_ladder_s * 0.85
    assert start_s + queue_s + worst_job_s + result_s <= attempt_s * 0.85


# ── 3. the job records the compromises instead of failing ───────────────────


def test_quality_warnings_are_attached_to_the_job_record() -> None:
    from src.services.design_builder.service import RedesignService

    attach = RedesignService._attach_quality_warnings
    assert attach("j", {"valid": True}, []) == {"valid": True}
    assert attach("j", None, []) is None
    assert attach("j", {"valid": True}, ["s0 DROPPED"]) == {
        "valid": True,
        "quality_warnings": ["s0 DROPPED"],
    }


async def test_variants_share_the_job_budget_and_record_warnings_by_style(monkeypatch) -> None:
    from unittest.mock import AsyncMock, MagicMock

    import src.services.design_builder as design_builder
    from src.services.design_builder.variants import RedesignVariantsMixin
    from src.services.html_gen.types import LLMResult

    seen: list[object] = []

    async def _chunked(**kwargs: object) -> LLMResult:
        seen.append(kwargs["budget"])
        return LLMResult(text="<html></html>", quality_warnings=[f"warn-{kwargs['style']}"])

    jm = MagicMock(
        create_variant=AsyncMock(return_value=MagicMock(variant_id="v")),
        mark_variant_completed=AsyncMock(),
    )
    monkeypatch.setattr(design_builder, "JobManager", jm)
    obj = RedesignVariantsMixin.__new__(RedesignVariantsMixin)
    obj._check_cancelled = AsyncMock(return_value=False)  # type: ignore[method-assign]
    obj._dispatch_webhook = AsyncMock()  # type: ignore[method-assign]
    obj._storage = MagicMock(store_html=AsyncMock(return_value="p"))
    obj._html_gen = MagicMock(generate_page_chunked=_chunked)
    job = MagicMock(styles=["modern", "bold"], job_id="job-x")
    budget = GenerationBudget()
    warnings: dict[str, list[str]] = {}

    failures = await obj._process_variants(
        session_factory=None,
        db=MagicMock(commit=AsyncMock(), rollback=AsyncMock()),
        job=job,
        html="",
        resolved_ctx=MagicMock(),
        settings=get_settings(),
        generate_screenshots=False,
        budget=budget,
        quality_warnings=warnings,
    )

    assert failures == []
    assert len(seen) == 2 and all(b is budget for b in seen), "one budget for the whole job"
    assert warnings == {"modern": ["warn-modern"], "bold": ["warn-bold"]}


async def test_items_omitted_by_the_split_cap_are_on_the_job_record() -> None:
    wm = _ScriptWM([_CLEAN.format(mark="ok")])
    sections = _sections(1)
    sections[0]["omitted_items"] = 95
    run = SectionRun()

    await _gen(wm).generate_sections(url="https://x.example", html="", sections=sections, run=run)

    assert any("95 source item(s)" in d for d in run.degradations), run.degradations


# ── 3. the no-detectable-sections fallback runs under the SAME budget ────────

# A page WITH markup but no semantic landmarks: ``detect_sections`` finds nothing, so
# ``generate_page_chunked`` falls back to ONE ``generate_full_page`` call.
_NO_SECTIONS = "<html><body><div>SITE-Z plain text, no landmarks</div></body></html>"
_FULL_PAGE = "<!DOCTYPE html><html><body><main><h1>ok</h1></main></body></html>"


async def test_full_page_fallback_past_the_deadline_does_not_call_the_llm() -> None:
    from src.services.context_resolver import detect_sections

    assert detect_sections(_NO_SECTIONS) == [], "fixture must take the full-page fallback"
    now = [5000.0]
    wm = _ScriptWM([_FULL_PAGE])
    budget = GenerationBudget(deadline=5010.0, clock=lambda: now[0])  # 10s < MIN_CALL_SECONDS
    with pytest.raises(RuntimeError, match="generate_full_page: no page could be generated"):
        await _gen(wm).generate_page_chunked(
            url="https://z.example/", html=_NO_SECTIONS, budget=budget
        )
    assert wm.prompts == [], "a call started past the job deadline"
    assert budget.calls == 0

    # The degradation is on the run the caller passes in.
    run = SectionRun(budget=GenerationBudget(deadline=5010.0, clock=lambda: now[0]))
    with pytest.raises(RuntimeError, match="generation deadline reached"):
        await _gen(wm).generate_full_page(url="https://z.example/", html=_NO_SECTIONS, run=run)
    assert wm.prompts == []
    assert any(
        "generate_full_page" in d and "generation deadline reached" in d for d in run.degradations
    ), run.degradations


async def test_full_page_fallback_call_is_bounded_by_the_time_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.services.html_gen.generator as generator_module

    real_timeout = asyncio.timeout
    timeouts: list[float | None] = []

    def _spy(delay: float | None) -> asyncio.Timeout:
        timeouts.append(delay)
        return real_timeout(delay)

    monkeypatch.setattr(generator_module.asyncio, "timeout", _spy)
    now = [1000.0]

    async def _slow() -> None:
        now[0] += 60.0

    wm = _ScriptWM([_FULL_PAGE], on_call=_slow)
    beats: list[int] = []

    async def _beat() -> None:
        beats.append(1)

    budget = GenerationBudget(deadline=1400.0, clock=lambda: now[0], max_extra_calls=0)
    with batch_progress(_beat):
        result = await _gen(wm).generate_page_chunked(
            url="https://z.example/", html=_NO_SECTIONS, budget=budget
        )

    assert len(wm.prompts) == 1
    assert "<h1>ok</h1>" in result.text
    assert timeouts and timeouts[0] is not None
    assert timeouts[0] <= 400.0, f"the call may run past the deadline: timeout={timeouts[0]}"
    assert budget.calls == 1, "the fallback call counts toward the job's calls"
    assert beats, "the fallback call emits a progress heartbeat"
