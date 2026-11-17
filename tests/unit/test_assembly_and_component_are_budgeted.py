"""The assembly-plan and component dispatch paths run under the JOB's budget.

THE DEFECT these lock. ``GenerationBudget.spend`` is the only enforcement seam, and
it was reached only through ``HtmlGenerator._budgeted_call`` — i.e. the section and
full-page paths. ``design_builder/phases.py`` dispatched the two OTHER branches
without a budget at all (``_generate_assembly``, ``_generate_component``), and the
callees' signatures did not even accept one. Those calls therefore ran under
worker_management's own timeout only: 600s x 3 retries ~= 1803s with NO deadline
check, and ``_phase_verify_and_refine`` re-ran the assembly draw up to
``verify_max_refine`` more times without ever charging the extra-call cap. A job
could bill uncapped, long past the deadline its caller was waiting on.

Every test here asserts ENFORCEMENT (spend observed / the call refused or cut), not
that a parameter was passed. Hermetic: a scripted worker_management, no DB, no net.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest

from src.services.html_gen.section_run import BudgetExhausted, GenerationBudget
from src.services.html_generator import HtmlGenerator

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

_PLAN = {"blocks": [{"type": "hero", "content": {"headline": "H"}}, {"type": "footer"}]}
_COMPONENT = "```html\n<section id='hero'><h2>Hello</h2></section>\n```"


class _Resp:
    def __init__(self, output: object) -> None:
        self.output = output
        self.usage_metrics = {"prompt_tokens": 11, "completion_tokens": 22}


class _WM:
    """Scripted worker_management. ``hang_s`` simulates a long in-flight call."""

    def __init__(self, output: object, *, hang_s: float = 0.0) -> None:
        self._output = output
        self._hang_s = hang_s
        self.calls: list[dict] = []

    async def message(self, **kwargs: object) -> _Resp:
        self.calls.append(kwargs)
        if self._hang_s:
            await asyncio.sleep(self._hang_s)
        return _Resp(self._output)


def _gen(wm: _WM | _SplitWM | _FailingFillWM | _MalformedFillWM) -> HtmlGenerator:
    return HtmlGenerator(wm_client=cast("WMClient", wm))


def _spent_budget() -> GenerationBudget:
    """A budget whose deadline is already reached (less than MIN_CALL_SECONDS left)."""
    return GenerationBudget(deadline=100.0, clock=lambda: 100.0)


def _component_kwargs() -> dict:
    return {
        "url": "https://x.example/",
        "component_html": "<section id='hero'><h1>Old</h1></section>",
        "component_type": "hero",
    }


@pytest.fixture
def _short_min_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a call START with a sub-second deadline, so in-flight cuts are testable."""
    import src.services.html_gen.generator as generator_module
    import src.services.html_gen.section_run as section_run

    monkeypatch.setattr(section_run, "MIN_CALL_SECONDS", 0.05)
    monkeypatch.setattr(generator_module, "MIN_CALL_SECONDS", 0.05)


# ── the assembly plan draw ───────────────────────────────────────────────────


async def test_the_assembly_plan_draw_consumes_the_job_budget() -> None:
    wm = _WM(_PLAN)
    budget = GenerationBudget()

    plan, _, _ = await _gen(wm).generate_assembly_plan(description="A landing page", budget=budget)

    assert [p["type"] for p in plan] == ["hero", "footer"]
    assert budget.calls == 1, "the assembly draw was not accounted against the job budget"
    assert budget.extra_calls == 0, "a FIRST draw is not a retry"


async def test_a_verification_refine_redraw_is_charged_as_an_extra_call() -> None:
    """``_phase_verify_and_refine`` re-runs the draw; the extra-call cap must bind."""
    wm = _WM(_PLAN)
    budget = GenerationBudget(max_extra_calls=1)
    gen = _gen(wm)

    await gen.generate_assembly_plan(description="x", budget=budget, extra=True)
    assert budget.extra_calls == 1

    with pytest.raises(BudgetExhausted, match="retry/repair budget spent"):
        await gen.generate_assembly_plan(description="x", budget=budget, extra=True)
    assert len(wm.calls) == 1, "the over-cap re-draw still reached worker_management"


async def test_the_assembly_plan_draw_is_refused_once_the_deadline_is_reached() -> None:
    wm = _WM(_PLAN)

    with pytest.raises(BudgetExhausted, match="generation deadline reached"):
        await _gen(wm).generate_assembly_plan(description="x", budget=_spent_budget())

    assert wm.calls == [], "a call started after the job deadline"


@pytest.mark.usefixtures("_short_min_call")
async def test_the_assembly_plan_draw_is_cut_when_the_deadline_passes_in_flight() -> None:
    wm = _WM(_PLAN, hang_s=30)
    loop = asyncio.get_running_loop()
    budget = GenerationBudget(deadline=loop.time() + 0.3, clock=loop.time)

    t0 = loop.time()
    with pytest.raises(BudgetExhausted, match="during an LLM call"):
        await _gen(wm).generate_assembly_plan(description="x", budget=budget)

    assert loop.time() - t0 < 5, "the in-flight assembly draw was not cut at the deadline"


# ── the component draw ───────────────────────────────────────────────────────


async def test_the_component_draw_consumes_the_job_budget() -> None:
    wm = _WM(_COMPONENT)
    budget = GenerationBudget()

    result = await _gen(wm).generate_component(**_component_kwargs(), budget=budget)

    assert "<section" in result.text
    assert budget.calls == 1, "the component draw was not accounted against the job budget"


async def test_the_component_draw_is_refused_once_the_deadline_is_reached() -> None:
    wm = _WM(_COMPONENT)

    with pytest.raises(BudgetExhausted, match="generation deadline reached"):
        await _gen(wm).generate_component(**_component_kwargs(), budget=_spent_budget())

    assert wm.calls == [], "a call started after the job deadline"


@pytest.mark.usefixtures("_short_min_call")
async def test_the_component_draw_is_cut_when_the_deadline_passes_in_flight() -> None:
    wm = _WM(_COMPONENT, hang_s=30)
    loop = asyncio.get_running_loop()
    budget = GenerationBudget(deadline=loop.time() + 0.3, clock=loop.time)

    t0 = loop.time()
    with pytest.raises(BudgetExhausted, match="during an LLM call"):
        await _gen(wm).generate_component(**_component_kwargs(), budget=budget)

    assert loop.time() - t0 < 5, "the in-flight component draw was not cut at the deadline"


# ── the DISPATCH is what was broken: phases must hand the budget over ─────────


def _phase_fixture(
    wm: _WM | _SplitWM | _FailingFillWM | _MalformedFillWM, *, mode: str, scope: str
):
    """A ``RedesignPhasesMixin`` wired to a scripted WM and an in-memory store."""
    from unittest.mock import AsyncMock, MagicMock

    from src.services.design_builder.phases import RedesignPhasesMixin

    obj = RedesignPhasesMixin.__new__(RedesignPhasesMixin)
    obj._html_gen = _gen(wm)
    obj._storage = MagicMock(store_html=AsyncMock(return_value="path"))
    obj._component_extractor = MagicMock(
        extract=MagicMock(return_value=("<section id='hero'></section>", "hero")),
        reinject=MagicMock(return_value="<html></html>"),
    )
    job = MagicMock(
        mode=mode,
        scope=scope,
        component_target={"component_type": "hero"} if scope == "component" else None,
        url="https://x.example/",
        page_type="landing",
        user_context={"description": "A landing page"},
        feedback_text=None,
        interactivity="static",
        embeds=[],
        critique_findings=None,
        site_analysis=None,
    )
    theme = MagicMock(brand_info={"name": "Acme"}, style="modern", style_description=None)
    return obj, job, theme


@pytest.mark.parametrize(
    ("mode", "scope", "output"),
    [
        ("assembly", "full_page", _PLAN),
        ("freeform", "component", _COMPONENT),
        # `hero` is its OWN dispatch branch into the same component path.
        ("freeform", "hero", _COMPONENT),
    ],
)
async def test_the_dispatch_hands_the_job_budget_to_both_paths(
    mode: str, scope: str, output: object
) -> None:
    """The literal defect: ``_phase_generate_html`` dispatched these two unbudgeted."""
    wm = _WM(output)
    obj, job, theme = _phase_fixture(wm, mode=mode, scope=scope)
    budget = GenerationBudget()

    await obj._phase_generate_html(
        job, "job-1", "<html></html>", theme, None, llm_config=None, budget=budget
    )

    assert budget.calls == 1, f"{mode}/{scope} dispatch did not spend the job budget"
    assert len(wm.calls) == 1


@pytest.mark.parametrize(
    ("mode", "scope", "output"),
    [
        ("assembly", "full_page", _PLAN),
        ("freeform", "component", _COMPONENT),
        # `hero` is its OWN dispatch branch into the same component path.
        ("freeform", "hero", _COMPONENT),
    ],
)
async def test_the_dispatch_is_refused_once_the_job_deadline_is_reached(
    mode: str, scope: str, output: object
) -> None:
    wm = _WM(output)
    obj, job, theme = _phase_fixture(wm, mode=mode, scope=scope)

    with pytest.raises(BudgetExhausted, match="generation deadline reached"):
        await obj._phase_generate_html(
            job, "job-1", "<html></html>", theme, None, llm_config=None, budget=_spent_budget()
        )

    assert wm.calls == [], f"{mode}/{scope} called the LLM past the job deadline"


# ── the 3x re-run: the refine loop must be capped by the SAME budget ──────────


class _FailedReport:
    status = "failed"

    def summary(self) -> str:
        return "1 serious a11y violation"

    def to_dict(self) -> dict:
        return {"status": "failed"}


def _always_failing_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.services.verification as verification

    async def _verify(html: str, **kwargs: object) -> _FailedReport:
        return _FailedReport()

    monkeypatch.setattr(verification, "verify_html", _verify)
    monkeypatch.setattr(verification, "parse_axe_summary", lambda report: "findings")


async def test_each_verification_refine_rerun_spends_the_job_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-runs are what make an unbudgeted draw multiply; charge them as extras."""
    from unittest.mock import MagicMock

    from src.config import get_settings

    _always_failing_verification(monkeypatch)
    wm = _WM(_PLAN)
    obj, job, theme = _phase_fixture(wm, mode="assembly", scope="full_page")
    budget = GenerationBudget()
    max_refine = get_settings().catalog.verify_max_refine

    _, report = await obj._phase_verify_and_refine(
        job,
        "job-1",
        MagicMock(redesigned_html="<html></html>"),
        theme,
        MagicMock(),
        llm_config=None,
        budget=budget,
    )

    assert report is not None and report["refine_attempts"] == max_refine
    assert budget.extra_calls == max_refine, "refine re-runs were not charged as extra calls"
    assert len(wm.calls) == max_refine


async def test_the_refine_loop_stops_when_the_budget_refuses_the_next_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    _always_failing_verification(monkeypatch)
    wm = _WM(_PLAN)
    obj, job, theme = _phase_fixture(wm, mode="assembly", scope="full_page")

    _, report = await obj._phase_verify_and_refine(
        job,
        "job-1",
        MagicMock(redesigned_html="<html></html>"),
        theme,
        MagicMock(),
        llm_config=None,
        budget=GenerationBudget(max_extra_calls=0),
    )

    assert wm.calls == [], "a refine re-run ran with no extra-call budget left"
    assert report is not None and report["status"] == "failed", "the job must still fail loud"


# ── the SPLIT path: a page that ships empty sections must say so ──────────────
#
# THE DEFECT these lock. ``CATALOG_SPLIT_ASSEMBLY`` draws one structure call plus one
# fill call per block. When the job's deadline arrives mid-split the code defaults the
# current block AND every remaining one — deliberately, a partial page beats no page —
# but it returned a ``_GenerationResult`` with NO quality warning and NO degradation.
# An 8-block page that ran out of budget at block 3 therefore reported ``succeeded``
# while handing the tenant 6 empty default sections, with a single pod log line as the
# only evidence. Nothing here changes that fallback; these make it HONEST.
#
# The branch had never executed in CI at all: ``split_assembly`` is False by default,
# so every existing budget test above exercises the one-shot draw. ``_split_assembly``
# forces it on.

_SPLIT_TYPES = ["navbar", "hero", "features", "stats", "cta", "footer"]


@pytest.fixture
def _split_assembly(monkeypatch: pytest.MonkeyPatch):
    """Force ``CATALOG_SPLIT_ASSEMBLY`` on — the branch is off by default in CI."""
    from src.config import get_settings

    monkeypatch.setenv("CATALOG_SPLIT_ASSEMBLY", "true")
    get_settings.cache_clear()
    assert get_settings().catalog.split_assembly, "the split path was not actually enabled"
    yield
    get_settings.cache_clear()


class _SplitWM:
    """Structure call, then one fill per block. Each call burns budget on a fake clock.

    ``seconds_per_call`` advances the clock the budget reads, so the deadline is reached
    at a known block rather than by real waiting.
    """

    def __init__(self, clock: dict, *, seconds_per_call: float) -> None:
        self._clock = clock
        self._seconds = seconds_per_call
        self.calls: list[dict] = []

    async def message(self, **kwargs: object) -> _Resp:
        self.calls.append(kwargs)
        self._clock["t"] += self._seconds
        if len(self.calls) == 1:
            return _Resp({"blocks": [{"type": t} for t in _SPLIT_TYPES]})
        return _Resp({"content": {"headline": "filled"}})


def _split_budget(*, seconds_per_call: float, deadline: float) -> tuple[_SplitWM, GenerationBudget]:
    clock = {"t": 0.0}
    wm = _SplitWM(clock, seconds_per_call=seconds_per_call)
    return wm, GenerationBudget(deadline=deadline, clock=lambda: clock["t"])


@pytest.mark.usefixtures("_split_assembly")
async def test_the_split_path_defaults_the_remaining_blocks_and_records_the_degradation() -> None:
    """Behaviour preserved (the rest default) AND recorded (the run degrades)."""
    from src.services.html_gen.section_run import SectionRun

    # 250s deadline, 100s/call, MIN_CALL_SECONDS=30: structure + 2 fills fit, the 3rd
    # fill is refused with 50s left.
    wm, budget = _split_budget(seconds_per_call=100.0, deadline=250.0)
    run = SectionRun(budget=budget)

    plan, _, _ = await _gen(wm).generate_assembly_plan(
        description="A landing page", budget=run.budget, run=run
    )

    assert len(wm.calls) == 3, "the split did not stop at the deadline"
    # Unchanged fallback: every block still ships, the unfilled ones on defaults.
    assert [p["type"] for p in plan] == _SPLIT_TYPES
    assert plan[0]["content"] and plan[1]["content"], "the drawn blocks lost their content"
    assert [p["content"] for p in plan[2:]] == [{}, {}, {}, {}]
    # The point: 4 empty sections shipped, and the job must not call that a clean run.
    assert run.degradations, "a page that shipped 4 empty default sections recorded NOTHING"
    (message,) = run.degradations
    assert "budget exhausted" in message, f"the degradation does not name the cause: {message}"
    assert "4 block(s) shipped with default content" in message, (
        f"the degradation does not name how many blocks were defaulted: {message}"
    )


@pytest.mark.usefixtures("_split_assembly")
async def test_a_split_that_finishes_inside_the_budget_degrades_nothing() -> None:
    """A signal that fires on every run is not a signal — a clean split must be clean."""
    from src.services.html_gen.section_run import SectionRun

    wm, budget = _split_budget(seconds_per_call=1.0, deadline=10_000.0)
    run = SectionRun(budget=budget)

    plan, _, _ = await _gen(wm).generate_assembly_plan(
        description="A landing page", budget=run.budget, run=run
    )

    assert len(wm.calls) == 1 + len(_SPLIT_TYPES), "not every block was drawn"
    assert all(p["content"] for p in plan), "a fully-drawn plan came back with empty blocks"
    assert run.degradations == [], "a clean split reported a degradation"


@pytest.mark.usefixtures("_split_assembly")
async def test_the_assembly_phase_surfaces_the_split_degradation_as_a_quality_warning() -> None:
    """Where a human sees it: ``_GenerationResult.quality_warnings``.

    ``RedesignService.process_job`` hands that list to ``_attach_quality_warnings``,
    which logs "completes DEGRADED" and persists it as
    ``html_validation_report.quality_warnings`` on the job row — the same sink the
    section path's compromises land in. Empty here means the tenant is told nothing.
    """
    wm, budget = _split_budget(seconds_per_call=100.0, deadline=250.0)
    obj, job, theme = _phase_fixture(wm, mode="assembly", scope="full_page")

    result = await obj._phase_generate_html(
        job, "job-1", "<html></html>", theme, None, llm_config=None, budget=budget
    )

    assert result.redesigned_html, "the page must still ship — the fallback is deliberate"
    assert result.quality_warnings, (
        "the job reports succeeded with 4 empty default sections and no quality warning"
    )
    assert "4 block(s) shipped with default content" in result.quality_warnings[0]


# ── the TWIN defect: a fill that fails for a NON-budget reason ────────────────
#
# Same silent degradation, one branch over. The generic ``except Exception`` around a
# block fill defaulted the block, logged, and recorded NOTHING — so a WM or transport
# error on one fill delivered an EMPTY section, ``quality_warnings`` was absent, and the
# job reported ``succeeded``. Shipping the BudgetExhausted half without this one is
# worse than shipping neither: the signal then LOOKS complete.
#
# The fallback is unchanged — defaulting that one block and continuing is correct.


class _FailingFillWM:
    """Structure call, then one fill per block; the named blocks raise instead."""

    def __init__(self, failing: set[str], *, exc: Exception | None = None) -> None:
        self._failing = failing
        self._exc = exc or RuntimeError("worker_management returned 503")
        self.calls: list[dict] = []

    async def message(self, **kwargs: object) -> _Resp:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return _Resp({"blocks": [{"type": t} for t in _SPLIT_TYPES]})
        prompt = str(kwargs.get("message", ""))
        # ``build_fill_prompt`` writes "# Fill the '<key>' block (...)".
        if any(f"Fill the '{name}' block" in prompt for name in self._failing):
            raise self._exc
        return _Resp({"content": {"headline": "filled"}})


@pytest.mark.usefixtures("_split_assembly")
async def test_a_non_budget_fill_failure_defaults_the_block_and_records_it() -> None:
    """The literal defect: an empty section shipped with no quality warning at all."""
    from src.services.html_gen.section_run import SectionRun

    wm = _FailingFillWM({"features"})
    run = SectionRun(budget=GenerationBudget())

    plan, _, _ = await _gen(wm).generate_assembly_plan(
        description="A landing page", budget=run.budget, run=run
    )

    by_type = {p["type"]: p for p in plan}
    # Unchanged fallback: every block still ships, the failed one on its defaults.
    assert [p["type"] for p in plan] == _SPLIT_TYPES
    assert by_type["features"]["content"] == {}, "the failed block did not fall back to defaults"
    assert by_type["hero"]["content"], "an unrelated block lost its drawn content"
    # The point: that hole must not read as a clean run.
    assert run.degradations, "a page that shipped an EMPTY section recorded NOTHING"
    (message,) = run.degradations
    assert "features" in message, f"the degradation does not name the block: {message}"
    assert "RuntimeError" in message, f"the degradation does not name the reason: {message}"
    # Safe to persist: the class name, never the exception's text (a WM/transport error
    # can carry the request body, a response payload, or a credentialed URL).
    assert "503" not in message, f"the raw exception text was persisted: {message}"


@pytest.mark.usefixtures("_split_assembly")
async def test_every_block_failing_records_one_degradation_per_block() -> None:
    """Volume decision, locked: ONE entry per hollow block, not one aggregate.

    The loop is bounded by the structure phase's block list (a page plan), so the worst
    case is that list's length — and each entry names WHICH section is hollow, which an
    aggregate would lose. ``SectionRun.degrade`` also dedupes, so repeats collapse.
    """
    from src.services.html_gen.section_run import SectionRun

    wm = _FailingFillWM(set(_SPLIT_TYPES))
    run = SectionRun(budget=GenerationBudget())

    plan, _, _ = await _gen(wm).generate_assembly_plan(
        description="A landing page", budget=run.budget, run=run
    )

    assert [p["content"] for p in plan] == [{}] * len(_SPLIT_TYPES)
    assert len(run.degradations) == len(_SPLIT_TYPES), (
        f"expected one degradation per hollow block, got {run.degradations}"
    )
    assert {t for t in _SPLIT_TYPES if any(t in m for m in run.degradations)} == set(_SPLIT_TYPES)


# ── the THIRD twin defect (#132): malformed-but-parseable, no exception at all ────
#
# THE DEFECT this locks. ``parse_fill`` used to return ``{}`` on ANY unusable shape,
# including a reply that parsed fine as JSON/a dict but had no usable content (a
# ``content`` field that is a string/list instead of an object, or a garbled
# top-level value). No exception meant neither the ``BudgetExhausted`` handler nor
# the generic ``except Exception`` handler above ever ran, so this path recorded
# NOTHING — worse than the other two, because the reply LOOKED like success.
#
# The fix: ``parse_fill`` now raises ``FillContentMalformedError`` for those shapes, which
# the EXISTING generic ``except Exception`` handler around the fill call already
# catches, defaults the block, and calls ``run.degrade`` — the same sink the other
# two branches use. A genuinely empty reply (no output, blank text, an explicit
# ``{}``/``{"content": {}}``) must stay silent — see the well-formed test below.


class _MalformedFillWM:
    """Structure call, then one fill per block; named blocks get a malformed-but-
    parseable reply — valid JSON, wrong shape, NOT an exception from the transport."""

    def __init__(self, malformed: dict[str, object]) -> None:
        self._malformed = malformed
        self.calls: list[dict] = []

    async def message(self, **kwargs: object) -> _Resp:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return _Resp({"blocks": [{"type": t} for t in _SPLIT_TYPES]})
        prompt = str(kwargs.get("message", ""))
        for name, output in self._malformed.items():
            if f"Fill the '{name}' block" in prompt:
                return _Resp(output)
        return _Resp({"content": {"headline": "filled"}})


@pytest.mark.usefixtures("_split_assembly")
async def test_a_malformed_fill_reply_defaults_the_block_and_records_it() -> None:
    """The literal defect: a 200-shaped reply with unusable content shipped hollow
    and silent. It must default (unchanged fallback) AND degrade (the fix)."""
    from src.services.html_gen.section_run import SectionRun

    wm = _MalformedFillWM({"features": {"content": "just a string, not an object"}})
    run = SectionRun(budget=GenerationBudget())

    plan, _, _ = await _gen(wm).generate_assembly_plan(
        description="A landing page", budget=run.budget, run=run
    )

    by_type = {p["type"]: p for p in plan}
    assert [p["type"] for p in plan] == _SPLIT_TYPES
    assert by_type["features"]["content"] == {}, "the malformed block did not default"
    assert by_type["hero"]["content"], "an unrelated block lost its drawn content"
    # The point: the hollow block must not read as a clean run.
    assert run.degradations, "a malformed-but-parseable fill reply recorded NOTHING"
    (message,) = run.degradations
    assert "features" in message, f"the degradation does not name the block: {message}"
    assert "FillContentMalformedError" in message, (
        f"the degradation does not name the specific defect: {message}"
    )


@pytest.mark.usefixtures("_split_assembly")
async def test_a_well_formed_or_legitimately_empty_fill_reply_degrades_nothing() -> None:
    """A signal that fires on everything is not a signal — neither real content nor
    an explicit, schema-compliant empty content object may trip the new guard."""
    from src.services.html_gen.section_run import SectionRun

    wm = _MalformedFillWM({"features": {"content": {}}})  # explicit, legitimate empty
    run = SectionRun(budget=GenerationBudget())

    plan, _, _ = await _gen(wm).generate_assembly_plan(
        description="A landing page", budget=run.budget, run=run
    )

    by_type = {p["type"]: p for p in plan}
    assert by_type["features"]["content"] == {}
    assert by_type["hero"]["content"] == {"headline": "filled"}
    assert run.degradations == [], (
        f"a well-formed/legitimately-empty split reported a degradation: {run.degradations}"
    )


@pytest.mark.usefixtures("_split_assembly")
async def test_the_assembly_phase_surfaces_the_malformed_fill_as_a_quality_warning() -> None:
    """Where a human sees it: ``_GenerationResult.quality_warnings`` — same sink as
    the other two branches, proven end to end through ``_phase_generate_html``."""
    wm = _MalformedFillWM({"features": {"content": "just a string, not an object"}})
    obj, job, theme = _phase_fixture(wm, mode="assembly", scope="full_page")

    result = await obj._phase_generate_html(
        job, "job-1", "<html></html>", theme, None, llm_config=None, budget=GenerationBudget()
    )

    assert result.redesigned_html, "the page must still ship — the fallback is deliberate"
    assert result.quality_warnings, (
        "the job reports succeeded with a hollow block and no quality warning"
    )
    assert "features" in result.quality_warnings[0]
    assert "FillContentMalformedError" in result.quality_warnings[0]
