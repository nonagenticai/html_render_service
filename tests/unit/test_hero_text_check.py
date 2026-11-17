"""Hero visible-text check + ONE abstract regeneration (owner decision).

Live run 01a159b1-9431: the hero showed a garbled shop sign "Cafiiti Boker" despite
the negative prompt. Now the hero is checked by a vision model; text -> regenerate
once (fresh seed, abstract close-up, no storefront/signage/interior) -> re-check;
text again -> brand-colour gradient. The check failing never blocks the page.
Cap: 1 check + at most 1 regeneration + 1 re-check per job.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from w2v_common.auth import clear_identity_context, set_identity_context
from w2v_common.clients import base as client_base
from w2v_common.tenant import TenantSource

from src.services.concept_image_generator import (
    HERO_ABSTRACT_NEGATIVE_EXTRA,
    HERO_NEGATIVE_PROMPT,
    ConceptImageGenerator,
    ConceptImageResult,
)
from src.services.design_builder import RedesignService
from src.services.hero_text_check import (
    HERO_TEXT_SCHEMA,
    HERO_TEXT_WORKER,
    HeroTextChecker,
    HeroTextVerdict,
)
from src.services.html_gen.section_run import GenerationBudget, SectionRun
from tests.tenants import TENANT_1
from tests.unit.test_hero_image_and_hours import (
    _PNG,
    _FakeGen,
    _FakeStorage,
    _Phases,
    _plan,
    _run,
    _theme,
)

# --------------------------------------------------------------------------- #
# HeroTextChecker: the vision call + fail-open parsing
# --------------------------------------------------------------------------- #


def _wm(output: object = None, *, raises: BaseException | None = None) -> Any:
    wm = AsyncMock()
    if raises is not None:
        wm.message.side_effect = raises
    else:
        wm.message.return_value = type("R", (), {"output": output})()
    return wm


async def test_check_sends_one_image_block_with_strict_schema():
    wm = _wm({"has_text": True, "text_seen": "Cafiiti Boker"})
    verdict = await HeroTextChecker(wm).check(_PNG)

    assert verdict == HeroTextVerdict(has_text=True, text_seen="Cafiiti Boker")
    wm.message.assert_awaited_once()
    kwargs = wm.message.await_args.kwargs
    assert kwargs["worker_name"] == HERO_TEXT_WORKER
    assert kwargs["output_schema"] == HERO_TEXT_SCHEMA
    blocks = kwargs["content_blocks"]
    assert [b["type"] for b in blocks] == ["text", "image"]
    assert blocks[1]["media_type"] == "image/jpeg"  # the downscaled hero, not the PNG
    assert len(blocks[1]["data"]) < len(_PNG)


async def test_check_accepts_a_json_string_verdict():
    verdict = await HeroTextChecker(_wm('{"has_text": false, "text_seen": ""}')).check(_PNG)
    assert verdict == HeroTextVerdict(has_text=False)


@pytest.mark.parametrize(
    "wm",
    [
        _wm(raises=httpx.ConnectError("ir down")),
        _wm(raises=TimeoutError()),
        _wm("not json"),
        _wm({"has_text": "yes", "text_seen": ""}),
        _wm(["has_text"]),
    ],
    ids=["transport", "timeout", "unparseable", "non-bool", "non-object"],
)
async def test_check_failure_is_no_text_with_an_error(wm):
    verdict = await HeroTextChecker(wm).check(_PNG)
    assert verdict.has_text is False
    assert verdict.error


async def test_check_refused_by_the_job_budget_is_a_check_error_not_a_raise():
    wm = _wm({"has_text": True, "text_seen": "x"})
    spent = GenerationBudget(deadline=0.0, clock=lambda: 0.0)  # no time left
    verdict = await HeroTextChecker(wm).check(_PNG, budget=spent)
    assert verdict.error == "BudgetExhausted"
    wm.message.assert_not_awaited()


async def test_check_of_undecodable_bytes_never_calls_the_model():
    wm = _wm({"has_text": False, "text_seen": ""})
    verdict = await HeroTextChecker(wm).check(b"<html/>")
    assert verdict.has_text is False and verdict.error
    wm.message.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Phase wiring: every branch
# --------------------------------------------------------------------------- #


class _ScriptedChecker:
    def __init__(self, *verdicts: HeroTextVerdict) -> None:
        self._verdicts: Iterator[HeroTextVerdict] = iter(verdicts)
        self.calls = 0

    async def check(self, data: bytes, *, budget: GenerationBudget | None = None):
        del data, budget
        self.calls += 1
        return next(self._verdicts)


def _phases(*verdicts: HeroTextVerdict) -> tuple[_Phases, _FakeGen, _ScriptedChecker]:
    gen = _FakeGen()
    phases = _Phases(gen, _FakeStorage())
    checker = _ScriptedChecker(*verdicts)
    phases.__dict__["_hero_text_checker"] = checker  # what RedesignService wires
    return phases, gen, checker


_CLEAN = HeroTextVerdict(has_text=False)
_TEXT = HeroTextVerdict(has_text=True, text_seen="Cafiiti Boker")


async def _hero(phases: _Phases) -> tuple[str | None, dict | None, list[dict], SectionRun]:
    plan, run = _plan(), _run()
    url, check = await phases._assembly_hero_image("job-1", plan, _theme(), "A cafe", run=run)
    return url, check, plan, run


async def test_clean_first_image_is_used_with_one_check():
    phases, gen, checker = _phases(_CLEAN)
    url, check, plan, run = await _hero(phases)

    assert url and url.startswith("data:image/jpeg;base64,")
    assert plan[0]["content"]["image"] == url
    assert check == {"checks": 1, "outcome": "clean"}
    assert checker.calls == 1
    assert len(gen.calls) == 1 and gen.calls[0]["abstract"] is False
    assert gen.calls[0]["seed"] is None
    assert run.degradations == []


async def test_text_then_clean_retry_regenerates_once_abstract_with_a_seed():
    phases, gen, checker = _phases(_TEXT, _CLEAN)
    url, check, plan, run = await _hero(phases)

    assert url and url.startswith("data:image/jpeg;base64,")
    assert plan[0]["content"]["image"] == url
    assert checker.calls == 2
    assert len(gen.calls) == 2
    retry = gen.calls[1]
    assert retry["abstract"] is True
    assert isinstance(retry["seed"], int)
    assert check is not None
    assert check["outcome"] == "regenerated"
    assert check["checks"] == 2
    assert check["text_seen"] == "Cafiiti Boker"
    assert check["retry_seed"] == retry["seed"]
    storage = cast("_FakeStorage", phases._storage)
    assert [f for _, _, f in storage.stored] == ["hero/original.png", "hero/retry.png"]
    assert run.degradations == []  # delivered as asked: not a degradation


async def test_text_twice_falls_back_to_the_gradient():
    phases, gen, checker = _phases(_TEXT, HeroTextVerdict(has_text=True, text_seen="OPEN"))
    url, check, plan, run = await _hero(phases)

    assert url == ""  # attempted and failed: a refine must not pay again
    assert plan[0]["content"]["image"] == ""
    assert checker.calls == 2 and len(gen.calls) == 2  # the cap: no third draw
    assert check is not None and check["outcome"] == "fell_back"
    assert check["retry_text_seen"] == "OPEN"
    assert len(run.degradations) == 1
    assert "text twice" in run.degradations[0] and "gradient" in run.degradations[0]


async def test_check_error_uses_the_image_unchecked_and_warns():
    phases, gen, checker = _phases(HeroTextVerdict(has_text=False, error="ReadTimeout"))
    url, check, _, run = await _hero(phases)

    assert url and url.startswith("data:image/jpeg;base64,")
    assert checker.calls == 1 and len(gen.calls) == 1  # no regeneration on a failed check
    assert check is not None and check["outcome"] == "check_failed"
    assert run.degradations == ["hero text check failed (ReadTimeout); image used unchecked"]


async def test_recheck_error_uses_the_retry_unchecked_and_warns():
    phases, gen, _ = _phases(_TEXT, HeroTextVerdict(has_text=False, error="ConnectError"))
    url, check, _, run = await _hero(phases)

    assert url and url.startswith("data:image/jpeg;base64,")
    assert len(gen.calls) == 2
    assert check is not None and check["outcome"] == "regenerated_unchecked"
    assert len(run.degradations) == 1 and "re-check failed" in run.degradations[0]


class _SecondDrawFails(_FakeGen):
    async def generate_hero(self, **kwargs: Any) -> ConceptImageResult:
        if self.calls:
            self.calls.append(kwargs)
            raise RuntimeError("acg down")
        return await super().generate_hero(**kwargs)


async def test_failed_regeneration_falls_back_to_the_gradient():
    gen = _SecondDrawFails()
    phases = _Phases(gen, _FakeStorage())
    checker = _ScriptedChecker(_TEXT)
    phases.__dict__["_hero_text_checker"] = checker
    url, check, _, run = await _hero(phases)

    assert url == ""
    assert checker.calls == 1
    assert check is not None and check["outcome"] == "fell_back"
    assert len(run.degradations) == 1 and "regeneration failed" in run.degradations[0]


async def test_refine_reuse_carries_the_check_and_its_warning_without_new_calls():
    phases, gen, checker = _phases()
    prior: dict[str, object] = {
        "checks": 2,
        "outcome": "fell_back",
        "warning": "hero image showed text twice",
    }
    plan, run = _plan(), _run()
    url, check = await phases._assembly_hero_image(
        "job-1", plan, _theme(), "b", run=run, reuse="", reuse_check=prior
    )
    assert (url, check) == ("", prior)
    assert run.degradations == ["hero image showed text twice"]
    assert gen.calls == [] and checker.calls == 0


def test_check_outcome_is_persisted_on_the_validation_report():
    attach = RedesignService._attach_hero_image_check
    report = {"quality_warnings": ["w"]}
    assert attach(report, None) is report
    assert attach(None, {"outcome": "clean", "checks": 1}) == {
        "hero_image_check": {"outcome": "clean", "checks": 1}
    }
    assert attach(report, {"outcome": "fell_back"}) == {
        "quality_warnings": ["w"],
        "hero_image_check": {"outcome": "fell_back"},
    }


# --------------------------------------------------------------------------- #
# The abstract retry request itself
# --------------------------------------------------------------------------- #


@pytest.fixture
def _tenant(monkeypatch):
    monkeypatch.setattr(client_base, "_RETRY_DELAY", 0.0)
    set_identity_context(TENANT_1, "lic", "sub", tenant_source=TenantSource.LICENSE_TOKEN)
    yield
    clear_identity_context()


async def _noop_hook(request: httpx.Request) -> None:
    del request


async def _submitted_item(monkeypatch, **kwargs: Any) -> dict[str, Any]:
    seen: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"batch_id": "b1"})
        return httpx.Response(200, json={"status": "completed", "results": []})

    def _client(*args: Any, **kw: Any) -> httpx.AsyncClient:
        kw["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kw)

    monkeypatch.setattr(client_base.httpx, "AsyncClient", _client)
    monkeypatch.setattr(
        "src.services.concept_image_generator._require_cell_token", _noop_hook, raising=True
    )
    gen = ConceptImageGenerator()
    gen._enabled = True
    gen._poll_interval_s = 0.0
    try:
        await gen.generate_hero(brand_name="Cafe Bakker", brief="A cafe and bakery", **kwargs)
    finally:
        await gen.aclose()
    return json.loads(seen[0].content)["items"][0]


async def test_abstract_retry_is_a_close_up_with_a_seed_and_no_place(monkeypatch, _tenant):
    item = await _submitted_item(monkeypatch, abstract=True, seed=1234)

    assert item["extra_params"] == {"seed": 1234}
    assert item["negative_prompt"] == HERO_NEGATIVE_PROMPT + HERO_ABSTRACT_NEGATIVE_EXTRA
    for word in ("storefront", "signage", "interior"):
        assert word in item["negative_prompt"]
    prompt = item["prompt"].lower()
    assert "close-up" in prompt and "hands" in prompt
    assert "premises" not in prompt
    assert "no storefront" in prompt and "no room or interior" in prompt
    assert "bakker" not in prompt  # brand name never lettered


async def test_first_draw_sends_no_seed_and_the_plain_negative(monkeypatch, _tenant):
    item = await _submitted_item(monkeypatch)
    assert "extra_params" not in item
    assert item["negative_prompt"] == HERO_NEGATIVE_PROMPT
