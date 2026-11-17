"""A new hero can be designed without pretending an existing component matched."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.component_extractor import ComponentExtractor, ComponentNotFoundError
from src.services.context_resolver import ResolvedContext
from src.services.design_builder.phases import RedesignPhasesMixin
from src.services.html_gen.prompt_builder import build_component_prompt


@pytest.mark.parametrize("scope", ["hero", "component"])
async def test_missing_target_can_only_generate_a_new_hero(scope):
    service = RedesignPhasesMixin()
    service._component_extractor = ComponentExtractor()
    service._storage = MagicMock(store_html=AsyncMock(return_value="saved.html"))
    service._html_gen = MagicMock(
        generate_component=AsyncMock(
            return_value=SimpleNamespace(
                text="<section class='hero'><h1>Report a bottleneck</h1></section>",
                input_tokens=1,
                output_tokens=2,
            )
        )
    )
    job = MagicMock(
        scope=scope,
        component_target={"component_type": "hero"},
        url="https://example.com/",
        page_type="app",
        critique_findings=[],
        site_analysis=None,
    )
    # A REAL ResolvedContext, as _ThemeDecision always carries one. The previous
    # fixture passed a str here, which is exactly the shape that hid the production
    # crash ('str' object has no attribute 'target_audience').
    resolved = ResolvedContext(description="Product description", target_audience=["teams"])
    theme = MagicMock(context=resolved, style="modern", brand_info={}, style_description=None)
    source = "<main><p>Report bottlenecks and vote on solutions.</p></main>"
    if scope == "component":
        with pytest.raises(ComponentNotFoundError):
            await service._generate_component(
                job, "job", source, theme, MagicMock(), llm_config=None
            )
        service._html_gen.generate_component.assert_not_awaited()
    else:
        out = await service._generate_component(
            job, "job", source, theme, MagicMock(), llm_config=None
        )
        assert "<h1>Report a bottleneck</h1>" in out.redesigned_html
        kwargs = service._html_gen.generate_component.call_args.kwargs
        # The resolved context is passed through untouched, never replaced by a string.
        assert kwargs["context"] is resolved
        assert kwargs["component_html"].startswith("<section class='hero'>")
        assert "Report bottlenecks and vote on solutions." in kwargs["component_html"]


async def test_missing_hero_prompt_builds_with_the_real_prompt_builder():
    """Regression for design job 01a17e88: the real prompt builder must accept what
    ``_generate_component`` hands it on the missing-hero path (no mocked generator)."""
    service = RedesignPhasesMixin()
    service._component_extractor = ComponentExtractor()
    service._storage = MagicMock(store_html=AsyncMock(return_value="saved.html"))
    prompts: list[str] = []

    async def fake_generate_component(**kwargs):
        prompts.append(
            build_component_prompt(
                url=kwargs["url"],
                component_html=kwargs["component_html"],
                component_type=kwargs["component_type"],
                page_type=kwargs["page_type"],
                style=kwargs["style"],
                findings=kwargs["findings"],
                brand=kwargs["brand"],
                site_analysis=kwargs["site_analysis"],
                style_description=kwargs["style_description"],
                context=kwargs["context"],
            )
        )
        return SimpleNamespace(
            text="<section class='hero'>x</section>", input_tokens=1, output_tokens=1
        )

    service._html_gen = MagicMock(generate_component=fake_generate_component)
    job = MagicMock(
        scope="hero",
        component_target={"component_type": "hero"},
        url="https://example.com/",
        page_type="app",
        critique_findings=[],
        site_analysis=None,
    )
    theme = MagicMock(
        context=ResolvedContext(primary_color="#0F766E", secondary_color="#134E4A"),
        style="modern",
        brand_info={},
        style_description=None,
    )
    source = "<main><p>Track <b>clinic</b> bottlenecks &amp; vote.</p></main>"
    await service._generate_component(job, "job", source, theme, MagicMock(), llm_config=None)

    assert len(prompts) == 1
    assert "primary=#0F766E" in prompts[0]
    assert "Track clinic bottlenecks &amp; vote." in prompts[0]
