"""One bad section draw must not kill a redesign; a section whose CSS misses its markup must not ship.

Found by the 2026-09-14 live acceptance runs for https://www.cliffordtraining.nl/
(after #126):

* run 1 failed outright — ONE section reply fell into a sampling loop
  (``--ct-reverse-reverse-reverse-…``), was cut at 5970/16000 tokens, and the
  truncation check raised straight out of the job with no retry;
* run 2 shipped "Zes routes" unstyled — its CSS targets ``.diensten`` while its markup
  is ``<section class="section" id="diensten">``, so the band lost padding, background,
  colour and font, and ``--card-hover``/``--card-hover-line``/``--accent-ink`` (declared
  on that dead rule) were undefined for the cards.

Replies are in ``fixtures/cliffordtraining_run2_section_replies.json`` (provenance in its
``_doc``). No live LLM is called.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from src.services.html_gen.brand_css import BRAND_TOKEN_NAMES
from src.services.html_gen.generator import SECTION_ATTEMPTS
from src.services.html_gen.html_utils import degenerate_repetition, section_reply_defect
from src.services.html_gen.section_css import analyze_section_css
from src.services.html_gen.section_run import SectionRun
from src.services.html_generator import HtmlGenerator

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

_FIXTURES = Path(__file__).parent / "fixtures"
_RUN2: dict[str, str] = json.loads(
    (_FIXTURES / "cliffordtraining_run2_section_replies.json").read_text(encoding="utf-8")
)
_RUN1_REPLIES: dict[str, list[str]] = {
    k: v
    for k, v in json.loads(
        (_FIXTURES / "cliffordtraining_section_replies.json").read_text(encoding="utf-8")
    ).items()
    if not k.startswith("_")
}
_URL = "https://www.cliffordtraining.nl/"


class _Resp:
    def __init__(self, output: str, completion_tokens: int) -> None:
        self.output = f"```html\n{output}\n```"
        self.usage_metrics = {"prompt_tokens": 100, "completion_tokens": completion_tokens}


class _ScriptedWM:
    """Answers each call with the next scripted ``(reply, completion_tokens)``."""

    def __init__(self, script: list[tuple[str, int]]) -> None:
        self._script = script
        self.prompts: list[str] = []
        self.max_tokens: list[object] = []

    async def message(self, **kwargs: object) -> _Resp:
        self.prompts.append(str(kwargs["message"]))
        self.max_tokens.append(kwargs.get("max_tokens"))
        reply, tokens = self._script[min(len(self.prompts), len(self._script)) - 1]
        return _Resp(reply, tokens)


def _gen(script: list[tuple[str, int]]) -> tuple[HtmlGenerator, _ScriptedWM]:
    wm = _ScriptedWM(script)
    return HtmlGenerator(wm_client=cast("WMClient", wm)), wm


async def _one(
    gen: HtmlGenerator,
    *,
    used_images: dict[str, str] | None = None,
    source_image_counts: dict[str, int] | None = None,
    run: SectionRun | None = None,
) -> str:
    html = await gen._generate_one_section(
        "PROMPT",
        where="generate_sections[section-2 (section: 'Hoe een traject begint.')]",
        url=_URL,
        manifest=[],
        llm_config=None,
        greenfield=False,
        run=run if run is not None else SectionRun(),
        used_images=used_images,
        source_image_counts=source_image_counts,
    )
    assert html is not None
    return html


# ── defect 1: truncated / degenerate section output is retried ──────────────


def test_the_recorded_loop_is_detected_as_degenerate() -> None:
    reason = section_reply_defect(_RUN2["degenerate_section_2"], completion_tokens=5970)
    assert reason is not None and "repetition loop" in reason
    assert "reverse" in reason


def test_a_loop_that_later_closes_its_tags_is_still_degenerate() -> None:
    closed = _RUN2["degenerate_section_2"] + ": 1;\n}\n</style>\n<h2>Hoe</h2>\n</section>"
    reason = section_reply_defect(closed, completion_tokens=5970)
    assert reason is not None and "repetition loop" in reason


@pytest.mark.parametrize(
    "text",
    [
        "<section><style>/* ================================================ */</style>"
        "<h2>x</h2></section>",
        '<section><svg><path d="M0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '
        '0 0 0 0 0 0 0 0 0 0 0 0 0 0"/></svg></section>',
        "<section><p>★★★★★</p></section>",
        # repeated ELEMENTS are design, not a loop (each carries its own tags)
        "<div class='dots'>" + '<span class="dot"></span>' * 12 + "</div>",
        "<div class='grid'>" + "<div class='cell'></div>" * 16 + "</div>",
        "<table><tbody>" + "<tr><td>Ja</td><td>Nee</td></tr>" * 12 + "</tbody></table>",
        # a short run of repeated words is text, not a loop
        "<p>" + "ha " * 30 + "</p>",
    ],
)
def test_legitimate_repetition_is_not_degenerate(text: str) -> None:
    assert degenerate_repetition(text) is None
    assert section_reply_defect(text) is None


def test_every_recorded_healthy_reply_passes_the_reply_check() -> None:
    """False-positive guard over every real reply we have (both acceptance runs)."""
    healthy = [r for replies in _RUN1_REPLIES.values() for r in replies]
    healthy += [_RUN2["section_2"], _RUN2["diensten_fixed"], _RUN2["cta"]]
    for reply in healthy:
        assert section_reply_defect(reply, completion_tokens=3000, max_tokens=16000) is None


@pytest.mark.parametrize(
    ("text", "tokens", "expected"),
    [
        ("<section><h2>x</h2><p>cut", 900, "does not end on a closed tag"),
        ("<section><h2>x</h2></section>", 16000, "output-token ceiling"),
        ("<section><style>.a{color:red}</style><div><p>x</p></div>", 900, "<section> is never"),
        ("<section><style>.a{color:red}<div>x</div></section>", 900, "<style> element"),
    ],
)
def test_truncation_shapes_are_named(text: str, tokens: int, expected: str) -> None:
    reason = section_reply_defect(text, completion_tokens=tokens, max_tokens=16000)
    assert reason is not None and expected in reason


async def test_a_degenerate_draw_is_retried_with_the_reason_and_the_retry_wins() -> None:
    gen, wm = _gen([(_RUN2["degenerate_section_2"], 5970), (_RUN2["section_2"], 2400)])

    html = await _one(gen)

    assert len(wm.prompts) == 2
    assert "REJECTED" in wm.prompts[1] and "repetition loop" in wm.prompts[1]
    assert "cut off or looped" in wm.prompts[1]
    assert "ct-start" in html and "reverse-reverse" not in html


async def test_an_empty_reply_is_retried() -> None:
    gen, wm = _gen([("", 0), (_RUN2["section_2"], 2400)])
    assert "ct-start" in await _one(gen)
    assert len(wm.prompts) == 2


async def test_a_reply_at_the_output_ceiling_is_retried() -> None:
    gen, wm = _gen([(_RUN2["section_2"], 16000), (_RUN2["section_2"], 2400)])
    await _one(gen)
    assert len(wm.prompts) == 2 and "ceiling" in wm.prompts[1]


async def test_a_section_that_never_recovers_fails_loudly_naming_the_section() -> None:
    gen, wm = _gen([(_RUN2["degenerate_section_2"], 5970)])

    with pytest.raises(RuntimeError) as exc:
        await _one(gen)

    assert len(wm.prompts) == SECTION_ATTEMPTS == 3
    msg = str(exc.value)
    assert "section-2" in msg and "Hoe een traject begint" in msg
    assert f"after {SECTION_ATTEMPTS} attempts" in msg and "repetition loop" in msg


# ── defect 1b: the provider's REAL output ceiling ───────────────────────────


def test_the_recorded_section_3_cut_is_named_as_the_output_ceiling() -> None:
    """Job 01a0a064: 11225 tokens, mid ``--header-`` declaration, under a requested 16000.

    Against the 16000 it asked for, that read as "stopped BELOW the requested budget";
    against the ~12069 the provider really allows it is the ceiling.
    """
    reply = _RUN2["truncated_section_3"]
    assert reply.endswith("--header-")
    old = section_reply_defect(reply, completion_tokens=11225, max_tokens=16000)
    assert old is not None and "stopped BELOW" in old
    new = section_reply_defect(reply, completion_tokens=11225, max_tokens=12069)
    assert new is not None and "provider's output ceiling" in new


async def test_section_calls_request_the_real_ceiling_and_retry_a_cut_reply_shorter() -> None:
    gen, wm = _gen([(_RUN2["truncated_section_3"], 11225), (_RUN2["section_2"], 2400)])
    assert gen._max_tokens == 16000

    html = await _one(gen)

    assert wm.max_tokens == [12069, 12069]
    assert len(wm.prompts) == 2
    assert "provider's output ceiling" in wm.prompts[1] and "SHORTER" in wm.prompts[1]
    assert "ct-start" in html


def test_every_section_prompt_states_the_budget_and_the_compact_css_rule() -> None:
    from src.services.html_gen.prompt_builder import css_contract_block

    block = css_contract_block("section-3")
    assert "12069 tokens" in block
    assert "--header-cta-bg: var(--color-accent)` is forbidden" in block
    assert "WCAG AA" in block


# ── defect 1c: a source section too large for one reply is split ────────────


def _wall(n: int) -> str:
    cards = "".join(
        f"<article class='rec'><p>{'Dave is een uitstekende coach. ' * 12}#{i}</p>"
        f"<cite>Klant {i}</cite></article>"
        for i in range(n)
    )
    return (
        "<main><section class='testimonials'><h2>75 LinkedIn-aanbevelingen</h2>"
        f"<p class='intro'>Uit 65 organisaties.</p><div class='grid'>{cards}</div>"
        "<a class='all' href='/aanbevelingen'>Lees alle 75</a></section>"
        "<footer><p>Clifford</p></footer></main>"
    )


def test_a_large_source_section_is_split_at_item_boundaries_keeping_every_item() -> None:
    from src.services.context_resolver import MAX_SECTION_PARTS, detect_sections

    sections = detect_sections(_wall(64))
    parts = [s for s in sections if s.get("part_of") == "testimonials"]

    assert [s["key"] for s in parts] == [
        "testimonials",
        "testimonials-part-2",
        "testimonials-part-3",
    ]
    assert len(parts) == 3 < MAX_SECTION_PARTS
    shown = [n for s in parts for n in range(64) if f"#{n}</p>" in s["html"]]
    assert shown == list(range(64)), "every item exactly once, in order"
    assert not any(s["truncated"] for s in parts)
    assert "<h2>" in parts[0]["html"] and all("<h2>" not in s["html"] for s in parts[1:])
    assert "Lees alle 75" in parts[-1]["html"] and "Lees alle 75" not in parts[0]["html"]
    assert all(s["parts"] == 3 for s in parts) and [s["part"] for s in parts] == [1, 2, 3]


def test_items_beyond_the_part_cap_are_counted_and_told_never_dropped_silently() -> None:
    """200 recommendations: 95 of them used to vanish from a part with ``truncated=False``."""
    from src.services.context_resolver import MAX_SECTION_PARTS, detect_sections
    from src.services.html_gen.prompt_builder import build_section_prompt

    sections = detect_sections(_wall(200))
    parts = [s for s in sections if s.get("part_of") == "testimonials"]

    assert len(parts) == MAX_SECTION_PARTS
    shown = {n for s in parts for n in range(200) if f"#{n}</p>" in s["html"]}
    omitted = parts[-1].get("omitted_items", 0)
    assert omitted > 0 and parts[-1]["truncated"] is True
    assert len(shown) + omitted == 200, "every item is either shown or counted"
    prompt = build_section_prompt(
        None,
        url=_URL,
        html="",
        section=parts[-1],
        page_type=None,
        style="modern",
        findings=None,
        brand=None,
        site_analysis=None,
    )
    assert f"{omitted} further item(s)" in prompt


def test_a_section_that_fits_is_not_split() -> None:
    from src.services.context_resolver import detect_sections

    assert [s.get("parts") for s in detect_sections(_wall(4))] == [None, None]


def test_a_split_part_tells_the_model_what_it_is() -> None:
    from src.services.html_gen.prompt_builder import section_part_note

    note = section_part_note({"part": 2, "parts": 3})
    assert "PART 2 of 3" in note and "NO heading" in note
    assert section_part_note({}) == ""


# ── defect 2: section CSS must apply to the section's own markup ────────────


def test_the_recorded_zes_routes_section_is_flagged() -> None:
    report = analyze_section_css(_RUN2["diensten_dead_root"], BRAND_TOKEN_NAMES)
    assert report.blocking_dead_selectors == [".diensten", ".diensten::before"]
    assert set(report.unresolved_vars) == {"--card-hover", "--card-hover-line", "--accent-ink"}
    text = " ".join(report.defects())
    assert ".diensten" in text and "--card-hover" in text


def test_the_one_token_fix_clears_it() -> None:
    report = analyze_section_css(_RUN2["diensten_fixed"], BRAND_TOKEN_NAMES)
    assert report.defects() == []


def test_the_old_declared_anywhere_check_passed_it() -> None:
    """Why a new check was needed: ``css_defects`` counts a declaration on a dead rule."""
    from src.services.html_gen.css_tokens import css_defects

    assert css_defects(_RUN2["diensten_dead_root"], BRAND_TOKEN_NAMES) == []


def test_harmless_dead_selectors_do_not_block() -> None:
    """``.cta-note`` matches nothing in the recorded CTA — an unused rule, not a broken band."""
    report = analyze_section_css(_RUN2["cta"], BRAND_TOKEN_NAMES)
    assert report.dead_selectors == [".cta-note"]
    assert report.defects() == []


def test_every_recorded_run1_reply_that_shipped_clean_is_not_blocked() -> None:
    """False-positive guard: only the known-defective first attempts may be flagged."""
    flagged = {
        key
        for key, replies in _RUN1_REPLIES.items()
        if analyze_section_css(replies[-1], BRAND_TOKEN_NAMES).defects()
    }
    assert flagged == set()


@pytest.mark.parametrize(
    ("fragment", "blocks"),
    [
        # a state variant toggled by JS is unused in a static concept, not dead
        (
            "<nav class='site-nav'><ul class='links'><li>a</li></ul></nav>"
            "<style>.site-nav{color:red}.site-nav.mobile-open .links{display:block}</style>",
            False,
        ),
        # every selector dead: the CSS was written for other markup
        ("<section class='a'><p class='b'>x</p></section><style>.x{color:red}.y p{}</style>", True),
        # a var declared on an unrelated live sibling does NOT reach its user
        (
            "<section><div class='one'></div><div class='two'></div></section>"
            "<style>.one{--tone:red}.two{color:var(--tone)}</style>",
            True,
        ),
        # declared on a live ancestor: inherited
        (
            "<section class='wrap'><div class='two'></div></section>"
            "<style>.wrap{--tone:red}.two{color:var(--tone)}</style>",
            False,
        ),
        # page token, :root-ish declaration, and a fallback all resolve
        (
            "<section class='w'><p class='t'>x</p></section>"
            "<style>:scope{--k:1px}.t{color:var(--color-ink);margin:var(--k);"
            "padding:var(--nope, 2px)}</style>",
            False,
        ),
    ],
)
def test_section_css_resolution_shapes(fragment: str, blocks: bool) -> None:
    assert bool(analyze_section_css(fragment, BRAND_TOKEN_NAMES).defects()) is blocks


@pytest.mark.parametrize(
    "fragment",
    [
        # page-anchored selectors: html/body exist in the composed page, not the fragment
        "<section class='x'><h2>a</h2></section>"
        "<style>html body .x{padding:1rem}html body .x h2{color:red}</style>",
        # the composed section wrapper, as a prefix and on its own
        "<section class='x'><h2>a</h2></section>"
        "<style>[data-section='hero']{--k:1px}[data-section=\"hero\"] .x h2{margin:var(--k)}"
        "</style>",
        "<section class='x'><h2>a</h2></section><style>body > .x{color:red}:root .x h2{}</style>",
    ],
)
def test_page_or_wrapper_anchored_selectors_are_not_dead(fragment: str) -> None:
    report = analyze_section_css(fragment, BRAND_TOKEN_NAMES)
    assert report.dead_selectors == [] and report.defects() == [], report


@pytest.mark.parametrize(
    ("fragment", "blocking"),
    [
        # THE rule: a dead single-class root whose BEM children are live
        (
            "<section class='section'><div class='diensten__card'>a</div></section>"
            "<style>.diensten{padding:4rem}.diensten__card{color:red}</style>",
            [".diensten"],
        ),
        # modifier children count too
        (
            "<section><div class='band--dark'>a</div></section>"
            "<style>.band{padding:4rem}.band--dark{color:red}</style>",
            [".band"],
        ),
        # a dead single class with NO live children of its own is an unused rule
        (
            "<section class='cta'><p class='cta__lead'>a</p></section>"
            "<style>.cta{padding:1rem}.cta__lead{color:red}.cta-note{color:blue}</style>",
            [],
        ),
        # a dead COMPOUND selector is never a block root
        (
            "<section class='a'><div class='a__b'>x</div></section>"
            "<style>.a{}.a__b{}.zz .a__b{color:red}</style>",
            [],
        ),
    ],
)
def test_the_block_root_rule(fragment: str, blocking: list[str]) -> None:
    assert analyze_section_css(fragment, BRAND_TOKEN_NAMES).blocking_dead_selectors == blocking


async def test_dead_root_css_is_retried_naming_the_selector_and_the_fix_ships() -> None:
    gen, wm = _gen([(_RUN2["diensten_dead_root"], 3500), (_RUN2["diensten_fixed"], 3500)])

    html = await _one(gen)

    assert len(wm.prompts) == 2
    assert ".diensten" in wm.prompts[1] and "--card-hover" in wm.prompts[1]
    assert 'class="section diensten"' in html


# ── image dedupe across sections ────────────────────────────────────────────


_KOFFIE = "https://www.cliffordtraining.nl/assets/img/04-dave-koffie-client.jpg"


async def test_a_photo_an_earlier_section_shows_is_retried_then_tolerated() -> None:
    """Run 2 re-showed the coffee photo of "Hoe we starten" in the closing CTA."""
    assert _KOFFIE in _RUN2["section_2"] and _KOFFIE in _RUN2["cta"]
    no_photo = _RUN2["cta"].replace(_KOFFIE, "https://www.cliffordtraining.nl/assets/img/x.jpg")
    gen, wm = _gen([(_RUN2["cta"], 3000), (no_photo, 3000)])

    html = await _one(gen, used_images={_KOFFIE: "section-2"}, source_image_counts={_KOFFIE: 1})

    assert len(wm.prompts) == 2 and _KOFFIE in wm.prompts[1]
    assert _KOFFIE not in html

    # A model that keeps re-showing it costs retries, never the page.
    gen, wm = _gen([(_RUN2["cta"], 3000)])
    html = await _one(gen, used_images={_KOFFIE: "section-2"}, source_image_counts={_KOFFIE: 1})
    assert len(wm.prompts) == SECTION_ATTEMPTS and _KOFFIE in html


async def test_a_photo_the_source_itself_repeats_may_repeat() -> None:
    gen, wm = _gen([(_RUN2["cta"], 3000)])
    await _one(gen, used_images={_KOFFIE: "section-2"}, source_image_counts={_KOFFIE: 2})
    assert len(wm.prompts) == 1


async def test_generate_sections_tells_later_sections_which_photos_are_taken() -> None:
    gen, wm = _gen([(_RUN2["section_2"], 2400), (_RUN2["diensten_fixed"], 3000)])
    source = (
        f'<main><section><img src="{_KOFFIE}"></section><section id="diensten"></section></main>'
    )

    await gen.generate_sections(
        url=_URL,
        html=source,
        sections=[
            {"section_type": "section", "key": "section-2", "html": "<section>a</section>"},
            {"section_type": "section", "key": "diensten", "html": "<section>b</section>"},
        ],
    )

    assert "already shown by earlier sections" not in wm.prompts[0]
    assert "already shown by earlier sections" in wm.prompts[1] and _KOFFIE in wm.prompts[1]


# ── defect 1c: LLM_BACKEND=vllm reports the cut-off as a 422, not as text ───


class _LengthCutoffThenReply(_ScriptedWM):
    """First call: w2v_common ``vllm_direct``'s 422 for ``finish_reason == "length"``."""

    def __init__(self, reply: str, *, status: int = 422, detail: str | None = None) -> None:
        super().__init__([(reply, 2400)])
        self._status = status
        self._detail = detail or (
            "Model token limit (12069) reached before the reply finished (finish_reason=length)"
        )

    async def message(self, **kwargs: object) -> _Resp:
        import httpx

        if not self.prompts:
            self.prompts.append(str(kwargs["message"]))
            request = httpx.Request("POST", "http://vllm.test:8001/v1/chat/completions")
            response = httpx.Response(self._status, json={"detail": self._detail}, request=request)
            raise httpx.HTTPStatusError(
                f"vllm returned {self._status}: {self._detail}", request=request, response=response
            )
        return await super().message(**kwargs)


async def test_a_finish_reason_length_422_is_retried_as_a_truncated_draw() -> None:
    """2026-10-18: LLM calls moved to w2v_common's vLLM backend, which raises a 422 for
    ``finish_reason=length`` where worker_management returned the cut-off text. The
    attempt loop only caught ``RuntimeError``, so one over-long draw failed the JOB."""
    wm = _LengthCutoffThenReply(_RUN2["section_2"])
    gen = HtmlGenerator(wm_client=cast("WMClient", wm))

    html = await _one(gen)

    assert "ct-start" in html
    assert len(wm.prompts) == 2
    assert "finish_reason=length" in wm.prompts[1]
    assert "cut off or looped" in wm.prompts[1]  # the "write a SHORTER reply" guidance


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (422, "Exceeded maximum output retries (3) for worker 'redesign-html'"),
        (500, "finish_reason=length"),
    ],
)
async def test_other_upstream_errors_still_propagate(status: int, detail: str) -> None:
    import httpx

    wm = _LengthCutoffThenReply(_RUN2["section_2"], status=status, detail=detail)
    gen = HtmlGenerator(wm_client=cast("WMClient", wm))

    with pytest.raises(httpx.HTTPStatusError):
        await _one(gen)
    assert len(wm.prompts) == 1
