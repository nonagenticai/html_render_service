"""Hero prompt + testimonials integrity (live web_design_v2 run 01a15968, 2026-10-20).

Brief: "Bakkerij De Molen", a bakery open Tue-Sat 7:00-15:00.

* The generated hero was a picture of a whole WEBSITE (nav bar, buttons, garbled
  "Prestengares"): the prompt said "hero image for the website of X" and quoted the
  headline. It must describe a photographic scene of the business's subject, exclude
  text/logos/UI/device frames, and carry the web/UI words only in the negative prompt.
* Testimonials ("Mark T.", "Elena R.") were invented. Like opening hours, a quote is
  kept only when the brief contains it; with none, the block is removed.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest
from bs4 import BeautifulSoup
from w2v_common.auth import clear_identity_context, set_identity_context
from w2v_common.clients import base as client_base
from w2v_common.tenant import TenantSource

from src.services.catalog import assemble_page
from src.services.catalog.integrity import IntegrityReport, enforce_testimonials
from src.services.catalog.registry import get_block
from src.services.concept_image_generator import (
    HERO_IMAGE_SIZE,
    HERO_NEGATIVE_PROMPT,
    ConceptImageGenerator,
    hero_subject,
)
from tests.tenants import TENANT_1

_LIVE_BRIEF = (
    "Create a modern landing page website for Bakkerij De Molen, a family bakery in "
    "Utrecht baking sourdough bread, croissants and apple pie. Open Tuesday to Saturday "
    "7:00-15:00. The site needs a clean UI with a hero section."
)

# Never in the POSITIVE prompt: these are what made the model draw a web page.
_WEB_WORDS = ("website", "web site", "landing page", "homepage", "web page", "mockup", "screenshot")


def _prompt(brief: str = _LIVE_BRIEF, brand: str | None = "Bakkerij De Molen") -> str:
    return ConceptImageGenerator._build_hero_prompt(
        brand_name=brand, brief=brief, primary_color="#B45309"
    )


# --------------------------------------------------------------------------- #
# Hero prompt
# --------------------------------------------------------------------------- #


def test_hero_prompt_describes_the_subject_as_a_photograph():
    prompt = _prompt().lower()
    assert "photograph" in prompt
    for subject in ("bakery", "sourdough", "croissants", "apple pie"):
        assert subject in prompt
    assert "16:9" in prompt


def test_hero_prompt_excludes_text_logos_ui_and_device_frames():
    prompt = _prompt().lower()
    for exclusion in (
        "no text",
        "no letters",
        "no words",
        "no logos",
        "no signage",
        "no interface elements",
        "no device frames",
        "no screens",
    ):
        assert exclusion in prompt


def test_hero_prompt_never_contains_web_or_ui_words():
    prompt = _prompt()
    lowered = prompt.lower()
    for word in _WEB_WORDS:
        assert word not in lowered, word
    assert not re.search(r"\bui\b", lowered)
    assert not re.search(r"\bsite\b", lowered)
    assert not re.search(r"\bpage\b", lowered)
    # The brand NAME is a lettering instruction to an image model — neutralised.
    assert "de molen" not in lowered
    assert "the business" in lowered


def test_hero_subject_strips_web_terms_but_keeps_the_business():
    subject = hero_subject("Bakkerij De Molen", _LIVE_BRIEF).lower()
    assert "family bakery in utrecht" in subject
    assert "website" not in subject and "landing" not in subject


def test_hero_prompt_without_brief_still_excludes_text():
    prompt = _prompt(brief="", brand=None).lower()
    assert "photograph" in prompt and "no text" in prompt


def test_negative_prompt_carries_the_web_exclusions():
    neg = HERO_NEGATIVE_PROMPT.lower()
    for word in (
        "text",
        "letters",
        "logos",
        "website",
        "landing page",
        "ui",
        "screenshot",
        "mockup",
        "device frame",
        "navigation bar",
        "buttons",
    ):
        assert word in neg


@pytest.fixture
def _tenant(monkeypatch):
    monkeypatch.setattr(client_base, "_RETRY_DELAY", 0.0)
    set_identity_context(TENANT_1, "lic", "sub", tenant_source=TenantSource.LICENSE_TOKEN)
    yield
    clear_identity_context()


@pytest.mark.asyncio
async def test_generate_hero_request_is_hero_usage_wide_with_negative_prompt(monkeypatch, _tenant):
    seen: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"batch_id": "b1"})
        return httpx.Response(200, json={"status": "completed", "results": []})

    def _client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(client_base.httpx, "AsyncClient", _client)
    # The cell-token guard is out of scope here (covered in test_hero_image_and_hours).
    monkeypatch.setattr(
        "src.services.concept_image_generator._require_cell_token", _noop_hook, raising=True
    )
    gen = ConceptImageGenerator()
    gen._enabled = True
    gen._poll_interval_s = 0.0
    gen._size = "1024x1024"  # the concept-mockup size must NOT leak into the hero
    try:
        await gen.generate_hero(brand_name="Bakkerij De Molen", brief=_LIVE_BRIEF)
    finally:
        await gen.aclose()

    item = json.loads(seen[0].content)["items"][0]
    assert item["target_usage"] == "hero_image"
    assert item["size"] == HERO_IMAGE_SIZE
    w, h = (int(v) for v in item["size"].split("x"))
    assert 1.7 <= w / h <= 1.8  # wide, ~16:9
    assert item["negative_prompt"] == HERO_NEGATIVE_PROMPT
    # usage_context is appended to the prompt by ACG's OpenAI provider: no text bait.
    ctx = json.dumps(item["usage_context"]).lower()
    assert "molen" not in ctx and "headline" not in ctx
    assert "website" not in item["prompt"].lower()


async def _noop_hook(request: httpx.Request) -> None:
    del request


# --------------------------------------------------------------------------- #
# Testimonials
# --------------------------------------------------------------------------- #

_INVENTED = [
    {
        "quote": "Best croissants I have ever had, every single morning.",
        "author": "Mark T.",
        "role": "Regular",
    },
    {
        "quote": "The sourdough is simply unbeatable in Utrecht.",
        "author": "Elena R.",
        "role": "Local foodie",
    },
]


def _testimonials_plan(items: list[dict]) -> list[dict]:
    return [
        {
            "type": "navbar",
            "content": {
                "brand": "Bakkerij De Molen",
                "links": [
                    {"label": "Reviews", "href": "#testimonials"},
                    {"label": "Hours", "href": "#hours"},
                ],
            },
        },
        {"type": "hero", "content": {"headline": "Bread baked at dawn"}},
        {"type": "testimonials", "content": {"headline": "Loved by locals", "items": items}},
        {"type": "footer", "content": {"brand": "Bakkerij De Molen"}},
    ]


def test_brief_without_testimonials_removes_the_block():
    res = assemble_page(_testimonials_plan(list(_INVENTED)), brief=_LIVE_BRIEF)
    soup = BeautifulSoup(res.html, "html.parser")
    assert soup.select_one("section#testimonials") is None
    assert "Loved by locals" not in res.html
    for name in ("Mark T.", "Elena R."):
        assert name not in res.html
    # No link may point at the section that no longer exists.
    assert all(a.get("href") != "#testimonials" for a in soup.select("a"))
    assert res.integrity_report is not None
    assert res.integrity_report.testimonials_dropped == 2
    assert res.integrity_report.testimonial_blocks_removed == 1


def test_empty_testimonials_block_is_removed_not_rendered_as_placeholder():
    manifest = get_block("testimonials")
    assert manifest is not None
    resolved: list[tuple[Any, dict]] = [(manifest, {"headline": "Reviews", "items": []})]
    report = IntegrityReport()
    enforce_testimonials(resolved, _LIVE_BRIEF, report)
    assert resolved == []
    assert report.testimonial_blocks_removed == 1


_BRIEF_WITH_QUOTE = (
    _LIVE_BRIEF
    + " A customer review we love: \u201cThe  sourdough here is worth the 7 o\u2019clock queue!\u201d"
    " \u2014 Anna V., Utrecht."
)


def test_brief_with_one_quote_keeps_only_that_quote():
    items = [
        # Same quote, straight quotes + collapsed whitespace + no trailing "!".
        {
            "quote": '"The sourdough here is worth the 7 o\'clock queue"',
            "author": "Anna V.",
            "role": "Utrecht",
        },
        *_INVENTED,
    ]
    res = assemble_page(_testimonials_plan(items), brief=_BRIEF_WITH_QUOTE)
    section = BeautifulSoup(res.html, "html.parser").select_one("section#testimonials")
    assert section is not None
    quotes = [q.get_text(" ", strip=True) for q in section.select("blockquote")]
    assert len(quotes) == 1
    assert "worth the 7 o'clock queue" in quotes[0]
    assert "Anna V." in section.get_text()
    for name in ("Mark T.", "Elena R."):
        assert name not in res.html
    assert res.integrity_report is not None
    assert res.integrity_report.testimonials_dropped == 2
    assert res.integrity_report.testimonial_blocks_removed == 0


def test_real_quote_with_invented_author_keeps_quote_blanks_author():
    manifest = get_block("testimonials")
    assert manifest is not None
    content: dict[str, Any] = {
        "items": [
            {
                "quote": "The sourdough here is worth the 7 o'clock queue!",
                "author": "Mark T.",
                "role": "Food critic",
            }
        ]
    }
    resolved: list[tuple[Any, dict]] = [(manifest, content)]
    enforce_testimonials(resolved, _BRIEF_WITH_QUOTE, IntegrityReport())
    assert len(resolved) == 1
    item = content["items"][0]
    assert item["author"] == "" and item["role"] == ""


def test_short_or_paraphrased_quote_is_not_grounded():
    manifest = get_block("testimonials")
    assert manifest is not None
    content: dict[str, Any] = {
        "items": [
            {"quote": "Sourdough bread", "author": "Anna V."},  # brief copy, too short
            {"quote": "The sourdough is worth the queue!", "author": "Anna V."},  # paraphrase
        ]
    }
    resolved: list[tuple[Any, dict]] = [(manifest, content)]
    report = IntegrityReport()
    enforce_testimonials(resolved, _BRIEF_WITH_QUOTE, report)
    assert resolved == []
    assert report.testimonials_dropped == 2


def test_prompts_forbid_invented_testimonials():
    from src.services.assembly_planner import (
        ASSEMBLY_FILL_SYSTEM_PROMPT,
        ASSEMBLY_STRUCTURE_SYSTEM_PROMPT,
        ASSEMBLY_SYSTEM_PROMPT,
    )

    for prompt in (ASSEMBLY_FILL_SYSTEM_PROMPT, ASSEMBLY_SYSTEM_PROMPT):
        assert "Testimonials are FACTS" in prompt
        assert "Never invent a quote or a customer name" in prompt
        assert "pricing, testimonials" not in prompt  # no longer asked to fill plausibly
    assert "ONLY when the brief itself quotes customers" in ASSEMBLY_STRUCTURE_SYSTEM_PROMPT
    manifest = get_block("testimonials")
    assert manifest is not None
    assert "never invent" in manifest.slots["items"].description
