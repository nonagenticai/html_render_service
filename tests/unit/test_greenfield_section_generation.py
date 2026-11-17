"""A greenfield page must be generated SECTION-WISE, never as one completion.

``web_design_v1`` could not produce at all: every greenfield run died on
``generate_full_page: LLM reply is not a storable page (does not end on a closed
tag (truncated))``. ``generate_page_chunked`` chunks by ``detect_sections(html)``,
and greenfield has no ``html``, so it fell through to the single full-page draw —
which cannot fit, because worker_management clamps a self-hosted provider's output
budget to a wall-clock time budget (~12k tokens) far below a full landing page.
Raising ``max_tokens`` was tried twice (8000 → 16000) and failed both times.

These tests pin the three properties of the fix:
  1. greenfield chunks (N section calls, no full-page call), planned FROM THE BRIEF;
  2. the non-greenfield path is byte-for-byte the behaviour it had;
  3. a truncation error names the token budget, so the next one is diagnosable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from src.config import get_settings
from src.services.context_resolver import ResolvedContext
from src.services.html_gen.section_plan import MAX_GREENFIELD_SECTIONS, plan_greenfield_sections
from src.services.html_gen.types import LLMResult
from src.services.html_generator import HtmlGenerator
from src.services.prompt_template import PromptTemplateLoader

if TYPE_CHECKING:
    from w2v_common.clients import WMClient


# A valid section carries its own CSS: a CSS-less section is rejected (retried once,
# then fails) since the testimonials section shipped unstyled.
_SECTION_REPLY = (
    '<section class="s"><style>.s{padding:var(--space-md)}</style>'
    "<h2>A section</h2><p>Copy.</p></section>"
)

# A real page with landmarks — the non-greenfield input.
_SOURCE_PAGE = (
    '<!DOCTYPE html><html lang="en"><body>'
    '<header class="hero"><h1>Acme</h1></header>'
    "<main><h2>What we do</h2><p>Things.</p></main>"
    "<footer><p>© Acme</p></footer>"
    "</body></html>"
)


class _StubWMResponse:
    def __init__(self, output: object, completion_tokens: int = 120) -> None:
        self.output = output
        self.usage_metrics = {"prompt_tokens": 50, "completion_tokens": completion_tokens}


class _RecordingWM:
    """Records every WM call so the test can count them and read the prompts."""

    def __init__(self, output: str = _SECTION_REPLY, completion_tokens: int = 120) -> None:
        self._output = output
        self._completion_tokens = completion_tokens
        self.calls: list[dict] = []

    async def message(self, **kwargs: object) -> _StubWMResponse:
        self.calls.append(kwargs)
        return _StubWMResponse(self._output, self._completion_tokens)


def _generator(wm: _RecordingWM) -> HtmlGenerator:
    return HtmlGenerator(PromptTemplateLoader(), wm_client=cast("WMClient", wm))


def _brief() -> ResolvedContext:
    return ResolvedContext(
        description="A booking site for a two-person sailing charter in Croatia.",
        site_type="saas",
        target_audience=["small crews"],
        goals=["build_trust"],
        feature_priorities=[{"feature": "instant booking", "priority": "high"}],
        tone="warm",
        brand_name="Kvarner Sail",
    )


# ---------------------------------------------------------------------------
# 1. GREENFIELD takes the section-wise path, not generate_full_page
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_greenfield_generates_section_wise_and_never_calls_generate_full_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*args: object, **kwargs: object) -> LLMResult:
        raise AssertionError("greenfield fell through to the one-shot full-page draw")

    monkeypatch.setattr(HtmlGenerator, "generate_full_page", _boom)

    wm = _RecordingWM()
    result = await _generator(wm).generate_page_chunked(
        url="",
        html="",
        page_type=None,
        context=_brief(),
    )

    expected = plan_greenfield_sections(context=_brief(), page_type=None)
    assert len(wm.calls) == len(expected) >= 5, (
        f"expected one WM call per planned section, got {len(wm.calls)}"
    )
    # It composed a real document out of the per-section snippets.
    assert result.text.startswith("<!DOCTYPE html>")
    assert result.text.rstrip().endswith("</html>")
    for section in expected:
        assert f'data-section="{section["section_type"]}"' in result.text


@pytest.mark.asyncio
async def test_greenfield_section_prompts_come_from_the_brief_not_from_markup() -> None:
    """The plan and the prompts must be grounded in the brief — there is no HTML to read."""
    wm = _RecordingWM()
    gen = _generator(wm)

    await gen.generate_page_chunked(url="", html="", context=_brief())

    prompts = [str(c["message"]) for c in wm.calls]
    joined = "\n".join(prompts)
    # The free-text brief, the brand name and the named feature priority all reach the model.
    assert "two-person sailing charter" in joined
    assert "Kvarner Sail" in joined
    assert "instant booking" in joined
    # And it is an AUTHORING instruction, not the refine-an-existing-section one.
    assert any("BRAND-NEW" in p for p in prompts)
    assert not any("REFINEMENT of an existing brand" in p for p in prompts)
    # Section snippets, not N nested documents.
    assert all("Do NOT output `<!DOCTYPE html>`" in str(c["system_prompt"]) for c in wm.calls), (
        "greenfield sections must use the SECTION-scoped system prompt, not the whole-page one"
    )


def test_greenfield_plan_is_driven_by_site_type_goals_and_is_capped() -> None:
    saas = [s["section_type"] for s in plan_greenfield_sections(context=_brief())]
    assert saas[0] == "nav" and saas[-1] == "footer"
    assert "pricing" in saas, "the saas outline must be the saas one, not the default"

    # A goal-driven section must be ADDED to an outline that does not already have it.
    # portfolio has no social-proof band of its own, so build_trust has to introduce one.
    bare_portfolio = [
        s["section_type"]
        for s in plan_greenfield_sections(context=ResolvedContext(site_type="portfolio"))
    ]
    assert bare_portfolio == ["nav", "hero", "work", "about", "contact", "footer"]
    assert "testimonials" not in bare_portfolio
    trusted = [
        s["section_type"]
        for s in plan_greenfield_sections(
            context=ResolvedContext(site_type="portfolio", goals=["build_trust"])
        )
    ]
    assert "testimonials" in trusted, "goal build_trust must add a social-proof section"
    assert trusted[-1] == "footer", "an added section must not displace the closing footer"

    # portfolio (6) + every goal's section (testimonials, pricing, cta, faq, contact-present)
    # overflows the cap, so the cap has to bite.
    greedy = ResolvedContext(site_type="portfolio", goals=list(_ALL_GOALS))
    capped = [s["section_type"] for s in plan_greenfield_sections(context=greedy)]
    assert len(capped) == MAX_GREENFIELD_SECTIONS, capped
    assert capped[-1] == "footer", "trimming must keep the footer, not chop the tail"


_ALL_GOALS = (
    "build_trust",
    "generate_leads",
    "boost_sales",
    "increase_conversions",
    "improve_engagement",
)


# ---------------------------------------------------------------------------
# 2. The NON-greenfield path is unchanged
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_non_greenfield_still_chunks_by_detected_landmarks() -> None:
    from src.services.context_resolver import detect_sections

    wm = _RecordingWM()
    gen = _generator(wm)

    await gen.generate_page_chunked(
        url="https://acme.test/",
        html=_SOURCE_PAGE,
        context=_brief(),
    )

    assert len(wm.calls) == len(detect_sections(_SOURCE_PAGE))
    prompts = [str(c["message"]) for c in wm.calls]
    # Still the REFINEMENT prompt, still fed the original markup, never the greenfield one.
    assert all("REFINEMENT of an existing brand" in p for p in prompts)
    assert any("Acme" in p for p in prompts)
    assert not any("BRAND-NEW" in p for p in prompts)
    assert all("REFINES an existing" in str(c["system_prompt"]) for c in wm.calls), (
        "a real page must keep the refine-existing system prompt"
    )


@pytest.mark.asyncio
async def test_a_page_with_markup_but_no_landmarks_still_falls_back_to_full_page() -> None:
    """The pre-existing fallback must survive: markup, but nothing detectable."""
    wm = _RecordingWM(output='<!DOCTYPE html><html lang="en"><body><p>x</p></body></html>')
    gen = _generator(wm)

    await gen.generate_page_chunked(
        url="https://acme.test/",
        html="<div><p>no landmarks at all</p></div>",
        context=_brief(),
    )

    assert len(wm.calls) == 1, "no-landmark page must take the single full-page draw"
    message = str(wm.calls[0]["message"])
    assert "Refine the following web page" in message, "must be the FULL-PAGE prompt"
    assert 'Design the "' not in message, "must not be a per-section prompt"


# ---------------------------------------------------------------------------
# 3. The truncation error names the token budget
# ---------------------------------------------------------------------------
_TRUNCATED = '<!DOCTYPE html><html lang="en"><body><main><h3>Cut off her'


@pytest.mark.asyncio
async def test_truncation_at_the_cap_says_so_in_the_error() -> None:
    cap = get_settings().wm.max_tokens
    wm = _RecordingWM(output=_TRUNCATED, completion_tokens=cap)
    gen = _generator(wm)

    with pytest.raises(RuntimeError) as exc:
        await gen.generate_full_page(url="https://acme.test/", html=_SOURCE_PAGE)

    msg = str(exc.value)
    assert "truncated at the" in msg and "-token ceiling" in msg, msg
    assert f"completion_tokens={cap}" in msg and f"max_tokens={cap}" in msg, msg


@pytest.mark.asyncio
async def test_truncation_below_the_cap_is_reported_as_below_the_cap() -> None:
    """The whole point: 'hit the ceiling' and 'stopped early' must read differently."""
    cap = get_settings().wm.max_tokens
    wm = _RecordingWM(output=_TRUNCATED, completion_tokens=120)
    gen = _generator(wm)

    with pytest.raises(RuntimeError) as exc:
        await gen.generate_full_page(url="https://acme.test/", html=_SOURCE_PAGE)

    msg = str(exc.value)
    assert "stopped BELOW the requested budget" in msg, msg
    assert "truncated at the" not in msg, msg
    assert f"completion_tokens=120 of max_tokens={cap}" in msg, msg


def test_token_budget_note_is_silent_without_usage() -> None:
    """No usage numbers ⇒ no invented diagnosis."""
    from src.services.html_gen.html_utils import _token_budget_note

    assert _token_budget_note(None, 16000) == ""
    assert _token_budget_note(0, 16000) == ""
    assert _token_budget_note(500, None) == ""
