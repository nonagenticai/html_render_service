"""Assembled-page integrity: one brand, no dead links, no unverified image hotlinks.

Regression for live web_design_v2 run 01a156a6 (html_render job 01a156a6-8b6b,
2026-10-20, "Hearth & Crumb" bakery brief): an Unsplash hero blocked by
``net::ERR_BLOCKED_BY_ORB``, five ``href="#"`` links, header "Hearth & Crumb" vs
footer "The Daily Crumb", and ``<title>landing</title>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from bs4 import BeautifulSoup

from src.services.catalog import assemble_page
from src.services.catalog.integrity import (
    assign_section_ids,
    extract_contact,
    trusted_image_hosts,
)
from src.services.verification.verifier import _brand_problems, _dead_links

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

_UNSPLASH = "https://images.unsplash.com/photo-1509440159596-0249088772ff?w=1200"


def _hearth_plan() -> list[dict]:
    """The shape the live split planner produced: independent fills, invented names."""
    return [
        {
            "type": "navbar",
            "content": {
                "brand": "Hearth & Crumb",
                "links": [
                    {"label": "Our Menu", "href": "#"},
                    {"label": "Pricing", "href": "/pricing"},
                    {"label": "Careers", "href": "#"},
                ],
                "cta_label": "Order Now",
                "cta_href": "#",
            },
        },
        {
            "type": "hero",
            "content": {
                "headline": "Bread baked at dawn",
                "primary_label": "Order Now",
                "primary_href": "#order",
                "secondary_label": "See the menu",
                "secondary_href": "",
                "image": _UNSPLASH,
            },
        },
        {
            "type": "features",
            "content": {"headline": "Our menu", "items": [{"title": "Sourdough", "body": "Slow"}]},
        },
        {
            "type": "pricing",
            "content": {
                "headline": "Bread club",
                "items": [{"name": "Weekly", "price": "$20", "cta_href": "#"}],
            },
        },
        {"type": "cta", "content": {"headline": "Visit us", "primary_label": "Get directions"}},
        {
            "type": "footer",
            "content": {
                "brand": "The Daily Crumb",
                "tagline": "The Daily Crumb bakes for the neighbourhood.",
                "copyright": "© 2026 The Daily Crumb",
                "links": [{"label": "Contact", "href": "#"}],
            },
        },
    ]


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


# --------------------------------------------------------------------------- #
# Brand + title
# --------------------------------------------------------------------------- #


def test_one_brand_everywhere_when_blocks_disagree():
    res = assemble_page(_hearth_plan())
    soup = _soup(res.html)
    brands = [el.get_text(strip=True) for el in soup.select("[data-brand]")]
    assert brands == ["Hearth & Crumb", "Hearth & Crumb"]
    assert "Daily Crumb" not in res.html  # tagline + copyright rewritten too
    assert "© 2026 Hearth &amp; Crumb" in res.html
    assert res.integrity_report is not None
    assert res.integrity_report.brand_aliases == ["The Daily Crumb"]


def test_caller_brand_name_wins_over_plan_brand():
    res = assemble_page(_hearth_plan(), brand={"name": "Crumb Co"})
    brands = {el.get_text(strip=True) for el in _soup(res.html).select("[data-brand]")}
    assert brands == {"Crumb Co"}
    assert "Hearth" not in res.html


def test_title_is_brand_and_headline_never_a_page_type_slug():
    res = assemble_page(_hearth_plan())
    title = _soup(res.html).title
    assert title is not None
    assert title.get_text() == "Hearth & Crumb — Bread baked at dawn"


def test_title_without_brand_falls_back_to_headline():
    res = assemble_page([{"type": "hero", "content": {"headline": "Fresh loaves"}}])
    title = _soup(res.html).title
    assert title is not None
    assert title.get_text() == "Fresh loaves"


def test_explicit_title_is_kept():
    res = assemble_page(_hearth_plan(), title="Custom")
    title = _soup(res.html).title
    assert title is not None
    assert title.get_text() == "Custom"


def test_placeholder_brand_default_is_not_a_brand():
    # A footer fill that failed keeps the manifest default "Brand"; it must be
    # overwritten with the real brand, not counted as a second one.
    plan = [
        {"type": "navbar", "content": {"brand": "Acme"}},
        {"type": "footer", "content": {}},
    ]
    res = assemble_page(plan)
    brands = [el.get_text(strip=True) for el in _soup(res.html).select("[data-brand]")]
    assert brands == ["Acme", "Acme"]
    assert res.integrity_report is not None
    assert res.integrity_report.brand_aliases == []


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #


def test_no_dead_links_in_assembled_page():
    res = assemble_page(_hearth_plan())
    assert _dead_links(res.html) == []
    assert 'href="#"' not in res.html


def test_links_resolve_to_matching_sections():
    soup = _soup(assemble_page(_hearth_plan()).html)
    nav = {a.get_text(strip=True): a["href"] for a in soup.select("header nav a")}
    assert nav == {"Our Menu": "#features", "Pricing": "#pricing"}  # "Careers" removed
    hero_buttons = [a["href"] for a in soup.select("#home a")]
    # "Order Now" (#order, dangling) → the CTA band; "See the menu" → features.
    assert hero_buttons == ["#get-started", "#features"]
    brand_link = soup.select_one("header a[data-brand]")
    assert brand_link is not None
    assert brand_link["href"] == "#top"


def test_contact_link_becomes_mailto_from_brief():
    res = assemble_page(
        _hearth_plan(), brief="A bakery. Orders: hello@hearthcrumb.test, call +1 555 010 0199."
    )
    soup = _soup(res.html)
    footer_links = [a["href"] for a in soup.select("footer nav a")]
    assert footer_links == ["mailto:hello@hearthcrumb.test"]


def test_external_absolute_links_are_kept():
    plan = [{"type": "navbar", "content": {"links": [{"label": "Shop", "href": "https://x.test"}]}}]
    soup = _soup(assemble_page(plan).html)
    assert [a["href"] for a in soup.select("header nav a")] == ["https://x.test"]


def test_secondary_button_with_no_target_is_hidden_not_dead():
    plan = [
        {
            "type": "hero",
            "content": {"headline": "H", "secondary_label": "Learn more", "secondary_href": "#"},
        }
    ]
    res = assemble_page(plan)
    assert "Learn more" not in res.html
    # A hero-only page has nowhere for its primary button to go: it is NOT faked
    # into a self-link — the verifier reports it as the one remaining dead link.
    assert _dead_links(res.html) == [{"href": "#", "text": "Get started"}]


def test_javascript_href_never_survives_resolution():
    plan = [
        {"type": "navbar", "content": {"cta_href": "javascript:alert(1)", "cta_label": "x"}},
        {"type": "cta", "content": {}},
    ]
    res = assemble_page(plan)
    assert "javascript:" not in res.html


def test_section_ids_are_unique():
    assert assign_section_ids(["navbar", "features", "features", "footer"]) == [
        "top",
        "features",
        "features-2",
        "contact",
    ]


def test_extract_contact():
    assert extract_contact("mail a.b@c.io or ring +44 20 7946 0958") == {
        "email": "a.b@c.io",
        "phone": "+442079460958",
    }
    assert extract_contact("Founded in 2026, 3 shops") == {}


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def test_unverified_hero_hotlink_is_replaced_by_designed_fallback():
    res = assemble_page(_hearth_plan())
    soup = _soup(res.html)
    assert soup.find("img") is None
    assert "unsplash" not in res.html
    art = soup.select_one("#home .hero-art")
    assert art is not None
    assert art.get("aria-hidden") == "true"
    assert art.get_text(strip=True) == ""  # decorative gradient — no brand-initial letter
    assert res.integrity_report is not None
    assert res.integrity_report.images_dropped == 1


def test_allowlisted_image_host_is_kept(monkeypatch):
    from src.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("CATALOG_IMAGE_HOST_ALLOWLIST", '["assets.example.test"]')
    try:
        url = "https://assets.example.test/hero.webp"
        res = assemble_page([{"type": "hero", "content": {"headline": "H", "image": url}}])
        img = _soup(res.html).find("img")
        assert img is not None
        assert img["src"] == url
    finally:
        get_settings.cache_clear()


def test_logo_hotlink_falls_back_to_name():
    plan = [
        {
            "type": "logos",
            "content": {"items": [{"name": "Mill Co", "image": "https://cdn.test/logo.png"}]},
        }
    ]
    res = assemble_page(plan)
    soup = _soup(res.html)
    assert soup.find("img") is None
    assert "Mill Co" in res.html


def test_trusted_image_hosts_accepts_host_port_and_urls():
    hosts = trusted_image_hosts(["assets.test:9000", "https://cdn.test:8443/x"])
    assert {"assets.test", "assets.test:9000", "cdn.test", "cdn.test:8443"} <= hosts
    assert "minio.test" not in hosts  # MinIO is never an implicitly trusted image host


# --------------------------------------------------------------------------- #
# Verifier checks
# --------------------------------------------------------------------------- #


def test_dead_links_lists_every_kind_of_dead_target():
    html = (
        '<section id="ok"></section>'
        '<a href="#">a</a><a href="">b</a><a>c</a><a href="#missing">d</a>'
        '<a href="/menu">e</a><a href="#ok">live</a><a href="https://x.test">live</a>'
        '<a href="mailto:a@b.test">live</a><a href="tel:+1555">live</a>'
    )
    dead = _dead_links(html)
    assert [d["text"] for d in dead] == ["a", "b", "c", "d", "e"]


def test_brand_problems_flags_generic_title_and_two_brands():
    html = (
        "<html><head><title>landing</title></head><body>"
        "<a data-brand>Hearth &amp; Crumb</a><p data-brand>The Daily Crumb</p></body></html>"
    )
    problems = _brand_problems(html)
    assert any("generic <title>" in p for p in problems)
    assert any("2 different brand names" in p for p in problems)


def test_brand_problems_flags_title_without_brand():
    html = "<title>Bread</title><a data-brand>Acme</a>"
    assert _brand_problems(html) == ["<title> 'Bread' does not name the brand 'Acme'"]


def test_assembled_hearth_page_has_no_brand_problems():
    assert _brand_problems(assemble_page(_hearth_plan()).html) == []


# --------------------------------------------------------------------------- #
# Split planner carries the first brand into later fill prompts
# --------------------------------------------------------------------------- #


class _StubResp:
    def __init__(self, output):
        self.output = output
        self.usage_metrics = {"prompt_tokens": 1, "completion_tokens": 1}


class _SeqWM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls: list[dict] = []

    async def message(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResp(self._outputs.pop(0) if self._outputs else {})


@pytest.mark.asyncio
async def test_split_fills_reuse_the_first_brand(monkeypatch):
    from src.config import get_settings
    from src.services.html_generator import HtmlGenerator

    get_settings.cache_clear()
    monkeypatch.setenv("CATALOG_SPLIT_ASSEMBLY", "true")
    try:
        wm = _SeqWM(
            [
                {"blocks": [{"type": "navbar"}, {"type": "hero"}, {"type": "footer"}]},
                {"content": {"brand": "Hearth & Crumb"}},
                {"content": {"headline": "Bread"}},
                {"content": {"brand": "Hearth & Crumb"}},
            ]
        )
        gen = HtmlGenerator(wm_client=cast("WMClient", wm))
        await gen.generate_assembly_plan(description="A bakery landing page")
        prompts = [str(c["message"]) for c in wm.calls]
        assert "Brand name:" not in prompts[1]  # navbar fill: nothing given yet
        assert "Brand name: Hearth & Crumb" in prompts[2]  # hero fill
        assert "Brand name: Hearth & Crumb" in prompts[3]  # footer fill
    finally:
        get_settings.cache_clear()


def test_fill_prompt_tells_the_model_not_to_hotlink_or_use_bare_hash():
    from src.services.assembly_planner import ASSEMBLY_FILL_SYSTEM_PROMPT, build_fill_prompt
    from src.services.catalog import get_block

    manifest = get_block("hero")
    assert manifest is not None
    prompt = build_fill_prompt(
        manifest=manifest,
        description="bakery",
        brand=None,
        page_type="landing",
        audience=None,
        interactivity="static",
    )
    assert "never an external image URL" in prompt
    assert "#get-started" in prompt
    assert "never '#' alone" in ASSEMBLY_FILL_SYSTEM_PROMPT


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        return bool(path) and Path(path).exists()
    except Exception:  # noqa: BLE001 — probe: any failure means "no chromium"
        return False


@pytest.mark.skipif(not _chromium_available(), reason="Playwright chromium not installed")
@pytest.mark.asyncio
async def test_verify_reports_broken_images_and_clean_integrity():
    from src.services.browser_pool import close_browser_pool
    from src.services.verification import verify_html

    html = assemble_page(_hearth_plan(), brand={"primary_color": "#7c2d12"}).html
    broken = html.replace("</body>", '<img alt="" src="data:image/png;base64,AAAA"></body>')
    try:
        clean = await verify_html(html, run_axe=False)
        assert clean.checks["dead_links"] == []
        assert clean.checks["broken_images"] == []
        assert clean.checks["brand_consistent"] is True
        assert not any(f.check in ("links", "brand", "images") for f in clean.findings)

        rep = await verify_html(broken, run_axe=False)
        assert len(rep.checks["broken_images"]) == 1
        assert any(f.check == "images" and f.severity == "advisory" for f in rep.findings)
    finally:
        await close_browser_pool()
