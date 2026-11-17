"""Unit tests for computed-assembly web design: catalog, theme, assembler, planner.

Deterministic (no DB, no network) — these lock the security + composition contracts
the assembly path depends on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from src.api.schemas import EmbedSpec, RedesignRequest
from src.services.assembly_planner import build_assembly_prompt, parse_plan
from src.services.catalog import assemble_page, block_keys, catalog_brief, get_block
from src.services.catalog.theme import build_theme_css, readable_on, resolve_font

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_catalog_has_core_blocks():
    keys = set(block_keys())
    assert {"navbar", "hero", "features", "faq", "cta", "footer"} <= keys


def test_catalog_brief_lists_every_block():
    brief = catalog_brief()
    for key in block_keys():
        assert key in brief


def test_get_block_unknown_returns_none():
    assert get_block("does-not-exist") is None
    assert get_block("hero") is not None


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #


def test_readable_on_contrast():
    assert readable_on("#ffffff") == "#1a1a1a"
    assert readable_on("#000000") == "#ffffff"


def test_build_theme_css_applies_brand_primary():
    css = build_theme_css({"primary_color": "#e11d48"})
    assert "--color-primary:#e11d48" in css
    assert "--color-primary-content:#ffffff" in css  # dark rose -> white text


def test_build_theme_css_defaults_without_brand():
    css = build_theme_css(None)
    assert css.startswith(":root{")
    assert "--color-primary:" in css


def test_build_theme_css_reads_analysis_palette_shape():
    css = build_theme_css({"color_palette": {"primary": "#123456"}})
    assert "--color-primary:#123456" in css


def test_build_theme_css_rejects_non_hex():
    # A bogus colour must not leak into CSS; falls back to the default.
    css = build_theme_css({"primary_color": "red; } body{display:none"})
    assert "display:none" not in css
    assert "--color-primary:#2563eb" in css  # default


def test_resolve_font_from_google_fonts_list():
    family, google = resolve_font({"fonts": [{"family": "Inter", "is_google_font": True}]})
    assert family == "Inter"
    assert google == "Inter"


# --------------------------------------------------------------------------- #
# Assembler — composition + security
# --------------------------------------------------------------------------- #


def _basic_plan():
    return [
        {"type": "footer", "content": {"brand": "Acme"}},
        {"type": "navbar", "content": {"brand": "Acme", "cta_label": "Join"}},
        {"type": "hero", "content": {"headline": "Ship faster"}},
        {"type": "features", "content": {"items": [{"title": "Fast", "body": "Very"}]}},
    ]


def test_assemble_produces_full_document():
    res = assemble_page(_basic_plan(), brand={"name": "Acme"}, title="Acme")
    assert res.html.startswith("<!DOCTYPE html>")
    assert "</html>" in res.html
    assert "Ship faster" in res.html


def test_assemble_hoists_singletons():
    res = assemble_page(_basic_plan())
    # navbar first, footer last regardless of plan order
    assert res.blocks_used[0] == "navbar"
    assert res.blocks_used[-1] == "footer"


def test_assemble_drops_unknown_blocks():
    res = assemble_page([{"type": "bogus", "content": {}}, {"type": "hero", "content": {}}])
    assert "bogus" not in res.blocks_used
    assert "hero" in res.blocks_used


def test_assemble_escapes_html_in_content():
    res = assemble_page([{"type": "hero", "content": {"headline": "<script>x</script>"}}])
    assert "<script>x</script>" not in res.html
    assert "&lt;script&gt;x&lt;/script&gt;" in res.html


def test_assemble_neutralises_javascript_hrefs():
    res = assemble_page(
        [{"type": "navbar", "content": {"cta_href": "javascript:alert(1)", "cta_label": "x"}}]
    )
    assert "javascript:alert(1)" not in res.html


def test_assemble_applies_brand_theme():
    res = assemble_page(_basic_plan(), brand={"primary_color": "#e11d48"})
    assert "--color-primary:#e11d48" in res.html


def test_assemble_inlines_css_always_and_js_only_when_interactive():
    static = assemble_page([{"type": "hero", "content": {}}], interactive=False)
    assert "btn" in static.html  # css inlined
    assert "var htmx" not in static.html  # no runtime for static
    inter = assemble_page([{"type": "hero", "content": {}}], interactive=True)
    assert "var htmx" in inter.html


def test_assemble_injects_sandboxed_embed():
    res = assemble_page(
        [{"type": "hero", "content": {"headline": "hi"}}],
        embeds=[{"type": "threejs", "artifact_url": "https://x.example/s.html", "mount": "hero"}],
    )
    assert res.embeds_injected == 1
    assert 'sandbox="allow-scripts"' in res.html
    assert "s.example" not in res.html  # sanity: exact url present, not truncated
    assert "https://x.example/s.html" in res.html


def test_assemble_embed_url_is_escaped():
    res = assemble_page(
        [{"type": "hero", "content": {}}],
        embeds=[{"artifact_url": 'https://x/"><img src=x onerror=alert(1)>'}],
    )
    assert "onerror=alert(1)>" not in res.html  # quote-escaped, cannot break out


def test_assemble_respects_max_blocks(monkeypatch):
    from src.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("CATALOG_MAX_BLOCKS", "2")
    get_settings.cache_clear()
    res = assemble_page([{"type": "hero"}, {"type": "features"}, {"type": "faq"}])
    assert len(res.blocks_used) <= 2
    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


def test_parse_plan_from_structured_dict():
    plan = parse_plan({"blocks": [{"type": "hero", "content": {"headline": "H"}}]})
    assert plan == [{"type": "hero", "content": {"headline": "H"}}]


def test_parse_plan_filters_unknown_blocks():
    plan = parse_plan({"blocks": [{"type": "hero"}, {"type": "nope"}]})
    assert [p["type"] for p in plan] == ["hero"]


def test_parse_plan_from_fenced_json_string():
    raw = '```json\n{"blocks": [{"type": "cta", "content": {}}]}\n```'
    assert [p["type"] for p in parse_plan(raw)] == ["cta"]


def test_parse_plan_from_bare_list():
    assert [p["type"] for p in parse_plan([{"type": "footer"}])] == ["footer"]


def test_parse_plan_unusable_returns_empty():
    assert parse_plan(None) == []
    assert parse_plan("not json at all") == []


def test_build_assembly_prompt_includes_brief_and_catalog():
    prompt = build_assembly_prompt(
        description="A SaaS landing page",
        brand={"name": "Acme"},
        page_type="landing",
        audience="developers",
        interactivity="interactive",
    )
    assert "A SaaS landing page" in prompt
    assert "Acme" in prompt
    assert "hero" in prompt  # catalog listed


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_request_defaults_are_freeform_static():
    req = RedesignRequest(scope="landing")
    assert req.mode == "freeform"
    assert req.interactivity == "static"
    assert req.embeds == []


def test_assembly_mode_does_not_require_url():
    # freeform full_page without url is rejected; assembly is exempt.
    with pytest.raises(ValueError):
        RedesignRequest(scope="full_page")
    ok = RedesignRequest(scope="full_page", mode="assembly")
    assert ok.mode == "assembly"


def test_embed_spec_requires_http_url():
    with pytest.raises(ValueError):
        EmbedSpec(artifact_url="javascript:alert(1)")
    spec = EmbedSpec(artifact_url="https://cdn.example/scene.html")
    assert spec.sandbox is True


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        RedesignRequest(scope="landing", mode="bogus")


# --------------------------------------------------------------------------- #
# generate_assembly_plan (mocked worker_management)
# --------------------------------------------------------------------------- #


class _StubWMResponse:
    def __init__(self, output, usage=None):
        self.output = output
        self.usage_metrics = usage or {"prompt_tokens": 11, "completion_tokens": 22}


class _StubWM:
    def __init__(self, output):
        self._output = output
        self.calls: list[dict] = []

    async def message(self, **kwargs):
        self.calls.append(kwargs)
        return _StubWMResponse(self._output)


@pytest.mark.asyncio
async def test_generate_assembly_plan_happy_path():
    from src.services.html_generator import HtmlGenerator

    wm = _StubWM({"blocks": [{"type": "hero", "content": {"headline": "H"}}, {"type": "footer"}]})
    gen = HtmlGenerator(wm_client=cast("WMClient", wm))
    plan, in_tok, out_tok = await gen.generate_assembly_plan(description="A landing page")
    assert [p["type"] for p in plan] == ["hero", "footer"]
    assert (in_tok, out_tok) == (11, 22)
    # structured output schema was passed to WM
    assert wm.calls[0]["output_schema"]["required"] == ["blocks"]
    assert wm.calls[0]["worker_name"] == "web-design"


@pytest.mark.asyncio
async def test_generate_assembly_plan_fails_loud_on_empty():
    from src.services.html_generator import HtmlGenerator

    gen = HtmlGenerator(wm_client=cast("WMClient", _StubWM({"blocks": []})))
    with pytest.raises(RuntimeError, match="no usable page plan"):
        await gen.generate_assembly_plan(description="x")


@pytest.mark.asyncio
async def test_generate_assembly_plan_flag_off_is_single_oneshot_call():
    """Flag OFF: exactly ONE web-design call with the one-shot ASSEMBLY_PLAN_SCHEMA."""
    from src.services.html_generator import HtmlGenerator

    wm = _StubWM({"blocks": [{"type": "hero", "content": {"headline": "H"}}, {"type": "footer"}]})
    gen = HtmlGenerator(wm_client=cast("WMClient", wm))
    plan, _, _ = await gen.generate_assembly_plan(description="A landing page")
    assert [p["type"] for p in plan] == ["hero", "footer"]
    assert len(wm.calls) == 1
    # The one-shot schema fills content per block (structure-phase schema does not).
    item_props = wm.calls[0]["output_schema"]["properties"]["blocks"]["items"]["properties"]
    assert "content" in item_props


# --------------------------------------------------------------------------- #
# Split planner (CATALOG_SPLIT_ASSEMBLY) — structure + per-block fill
# --------------------------------------------------------------------------- #


def test_parse_structure_keeps_valid_keys_in_order():
    from src.services.assembly_planner import parse_structure

    out = parse_structure({"blocks": [{"type": "navbar"}, {"type": "hero"}, {"type": "footer"}]})
    assert out == ["navbar", "hero", "footer"]


def test_parse_structure_drops_unknown_and_dedups_singletons():
    from src.services.assembly_planner import parse_structure

    # two navbars (singleton) + a bogus key → one navbar, bogus dropped
    out = parse_structure({"blocks": [{"type": "navbar"}, {"type": "nope"}, {"type": "navbar"}]})
    assert out == ["navbar"]


def test_parse_structure_from_bare_list_of_strings():
    from src.services.assembly_planner import parse_structure

    assert parse_structure(["hero", "cta"]) == ["hero", "cta"]


def test_parse_structure_unusable_returns_empty():
    from src.services.assembly_planner import parse_structure

    assert parse_structure(None) == []
    assert parse_structure("not json") == []


def test_build_structure_prompt_names_types_but_not_content_object():
    from src.services.assembly_planner import build_structure_prompt

    prompt = build_structure_prompt(
        description="A SaaS landing page",
        brand={"name": "Acme"},
        page_type="landing",
        audience="developers",
        interactivity="static",
    )
    assert "A SaaS landing page" in prompt
    assert "hero" in prompt  # catalog types listed
    assert "TYPES only" in prompt


def test_parse_fill_extracts_content_object():
    from src.services.assembly_planner import parse_fill

    assert parse_fill({"content": {"headline": "H"}}) == {"headline": "H"}
    assert parse_fill({"headline": "H"}) == {"headline": "H"}  # bare dict tolerated
    assert parse_fill(None) == {}


def test_parse_fill_stays_silent_on_legitimately_empty_replies():
    """A signal that fires on everything is not a signal (#132) — no output at all,
    blank text, or an explicit empty content object must NOT raise."""
    from src.services.assembly_planner import parse_fill

    assert parse_fill(None) == {}
    assert parse_fill("") == {}
    assert parse_fill("   ") == {}
    assert parse_fill({}) == {}
    assert parse_fill({"content": {}}) == {}


def test_parse_fill_raises_on_malformed_but_parseable_replies():
    """#132: valid JSON/a dict that cannot be read as content must not silently {}."""
    from src.services.assembly_planner import FillContentMalformedError, parse_fill

    with pytest.raises(FillContentMalformedError):
        parse_fill({"content": "just a string, not an object"})
    with pytest.raises(FillContentMalformedError):
        parse_fill({"content": [1, 2, 3]})
    with pytest.raises(FillContentMalformedError):
        parse_fill([{"type": "hero"}])  # top-level list, not a dict
    with pytest.raises(FillContentMalformedError):
        parse_fill("not json at all, just prose")


def test_build_fill_prompt_lists_only_this_blocks_slots():
    from src.services.assembly_planner import build_fill_prompt

    manifest = get_block("hero")
    assert manifest is not None
    prompt = build_fill_prompt(
        manifest=manifest,
        description="A landing page",
        brand=None,
        page_type="landing",
        audience="general",
        interactivity="static",
    )
    assert "'hero' block" in prompt
    assert "headline" in prompt  # a hero slot


class _SequencedWM:
    """Returns queued outputs in order; optionally raises when a message contains a token."""

    def __init__(self, outputs, usage=None, fail_on=None):
        self._outputs = list(outputs)
        self._usage = usage or {"prompt_tokens": 5, "completion_tokens": 7}
        self.calls: list[dict] = []
        self._fail_on = set(fail_on or ())

    async def message(self, **kwargs):
        self.calls.append(kwargs)
        message = str(kwargs.get("message", ""))
        if any(token in message for token in self._fail_on):
            raise RuntimeError("simulated WM failure")
        output = self._outputs.pop(0) if self._outputs else {}
        return _StubWMResponse(output, self._usage)


def _enable_split(monkeypatch):
    from src.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("CATALOG_SPLIT_ASSEMBLY", "true")
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_split_builds_plan_from_structure_plus_fills(monkeypatch):
    from src.config import get_settings
    from src.services.html_generator import HtmlGenerator

    _enable_split(monkeypatch)
    try:
        wm = _SequencedWM(
            [
                {"blocks": [{"type": "navbar"}, {"type": "hero"}, {"type": "footer"}]},  # structure
                {"content": {"brand": "Acme"}},  # navbar fill
                {"content": {"headline": "Ship faster"}},  # hero fill
                {"content": {"brand": "Acme"}},  # footer fill
            ]
        )
        gen = HtmlGenerator(wm_client=cast("WMClient", wm))
        plan, in_tok, out_tok = await gen.generate_assembly_plan(description="A landing page")
        assert [p["type"] for p in plan] == ["navbar", "hero", "footer"]
        assert plan[1]["content"] == {"headline": "Ship faster"}
        assert len(wm.calls) == 4  # 1 structure + 3 fills
        assert (in_tok, out_tok) == (20, 28)  # summed across all 4 calls
        # The first call used the structure schema (types only — no content property).
        struct_item_props = wm.calls[0]["output_schema"]["properties"]["blocks"]["items"][
            "properties"
        ]
        assert "content" not in struct_item_props
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_split_tolerates_a_block_fill_failure(monkeypatch):
    from src.config import get_settings
    from src.services.html_generator import HtmlGenerator

    _enable_split(monkeypatch)
    try:
        wm = _SequencedWM(
            [
                {"blocks": [{"type": "navbar"}, {"type": "hero"}, {"type": "footer"}]},
                {"content": {"brand": "Acme"}},  # navbar ok
                {"content": {"headline": "Ship faster"}},  # (hero raises before this is used)
            ],
            fail_on={"'hero' block"},  # the hero fill call raises
        )
        gen = HtmlGenerator(wm_client=cast("WMClient", wm))
        plan, _, _ = await gen.generate_assembly_plan(description="A landing page")
        # hero still present (defaults filled by the assembler), page not aborted
        assert [p["type"] for p in plan] == ["navbar", "hero", "footer"]
        assert plan[1]["content"] == {}  # hero degraded to empty → assembler fills defaults
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_split_empty_structure_falls_back_to_oneshot(monkeypatch):
    from src.config import get_settings
    from src.services.html_generator import HtmlGenerator

    _enable_split(monkeypatch)
    try:
        wm = _SequencedWM(
            [
                {"blocks": []},  # structure yields nothing
                {"blocks": [{"type": "hero", "content": {"headline": "H"}}, {"type": "footer"}]},
            ]
        )
        gen = HtmlGenerator(wm_client=cast("WMClient", wm))
        plan, in_tok, out_tok = await gen.generate_assembly_plan(description="A landing page")
        assert [p["type"] for p in plan] == ["hero", "footer"]
        # one structure call + one one-shot fallback call
        assert len(wm.calls) == 2
        # cost includes BOTH the wasted structure call and the one-shot fallback
        assert (in_tok, out_tok) == (10, 14)
        assert (
            "content" in wm.calls[1]["output_schema"]["properties"]["blocks"]["items"]["properties"]
        )
    finally:
        get_settings.cache_clear()
