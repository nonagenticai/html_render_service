"""A single-call full-page redesign must carry ``data-section`` landmarks.

Round 4 (2026-10-06), 4/4 live website_review runs for example.com: the source page
has no detectable sections, so ``generate_page_chunked`` fell back to
``generate_full_page``, whose output had NO ``data-section`` (only the per-section
path adds it). The after-act screenshots came back as anonymous ``band-NN`` and the
script lost its section headings. ``tag_data_sections`` is the deterministic fix.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

from bs4 import BeautifulSoup

from src.services.html_gen.section_landmarks import tag_data_sections
from src.services.html_generator import HtmlGenerator

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

# The measured shape: full-width header (~76px) and footer (~90px), every piece of
# content in ONE centred ~832px column, no data-section anywhere.
MEASURED_FULL_PAGE_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<style>
body{margin:0;font:18px/1.5 Georgia,serif;background:#f4f1ea;color:#222}
header{height:76px;background:#1d2b3a;color:#fff;display:flex;align-items:center;padding:0 32px}
main{max-width:832px;margin:0 auto;padding:48px 0}
.card{background:#fff;padding:24px;border-radius:12px}
footer{height:90px;background:#1d2b3a;color:#fff;display:flex;align-items:center;padding:0 32px}
</style></head>
<body>
<header><a href="https://example.com/" class="logo">Example</a></header>
<main>
  <h1>Example Domain</h1>
  <p>This domain is for use in illustrative examples in documents.</p>
  <div class="card"><h2>More information</h2><p>You may use this domain freely.</p></div>
</main>
<footer><p>IANA-managed reserved domain</p></footer>
<script type="application/ld+json">{"@type": "WebPage"}</script>
</body></html>
"""


def _sections(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    return [(el.name, str(el["data-section"])) for el in soup.select("[data-section]")]


def test_measured_page_gets_named_blocks() -> None:
    assert _sections(MEASURED_FULL_PAGE_HTML) == []  # the defect: none at all
    out = tag_data_sections(MEASURED_FULL_PAGE_HTML)
    assert _sections(out) == [
        ("header", "header"),
        ("main", "example-domain"),
        ("footer", "footer"),
    ]
    assert "<!DOCTYPE html>" in out  # still a full document


def test_idempotent_and_existing_names_untouched() -> None:
    once = tag_data_sections(MEASURED_FULL_PAGE_HTML)
    assert tag_data_sections(once) == once
    pre = '<html><body><section data-section="hero"><h1>Hi</h1></section></body></html>'
    assert tag_data_sections(pre) == pre


def test_descends_wrappers_and_dedupes_heading_slugs() -> None:
    html = """<html><body><div class="page">
      <header><h1>Acme</h1></header>
      <main>
        <section><h2>Features</h2><p>a</p></section>
        <section><h2>Features</h2><p>b</p></section>
        <div class="spacer"></div>
        <div data-section="pricing"><h2>Plans</h2></div>
      </main>
      <footer>(c) Acme</footer>
    </div></body></html>"""
    assert _sections(tag_data_sections(html)) == [
        ("header", "header"),
        ("section", "features"),
        ("section", "features-2"),
        ("div", "pricing"),
        ("footer", "footer"),
    ]


def test_block_holding_a_named_section_is_not_tagged() -> None:
    # Tagging the outer <main> would make the model's own named section NESTED,
    # and the screenshot second pass only takes top-level data-sections.
    html = (
        "<html><body><main><h1>Title</h1>"
        '<section data-section="intro"><p>x</p></section></main></body></html>'
    )
    assert _sections(tag_data_sections(html)) == [("section", "intro")]


def test_never_tags_inline_or_script() -> None:
    html = "<html><body><span>hi</span><script>x</script><p>loose</p></body></html>"
    assert tag_data_sections(html) == html


def test_model_written_values_are_format_normalised_not_deduped() -> None:
    # build_inventory looks screenshots up by the RAW data-section: a model-written
    # "Hero Section" never matched the screenshot role name. Duplicates stay as-is.
    html = """<html><head><style>
    [data-section="Hero Section"]{padding:4rem}
    </style></head><body>
      <section data-section="Hero Section"><h1>Welcome</h1></section>
      <section data-section="hero"><h2>Again</h2></section>
      <section data-section="hero"><h2>Third</h2></section>
      <section data-section="!!!"><h2>Our Story</h2></section>
      <section data-section="%s"><p>x</p></section>
    </body></html>""" % ("Very " * 20)
    out = tag_data_sections(html)
    names = [n for _, n in _sections(out)]
    assert names == ["hero-section", "hero", "hero", "our-story", names[4]]
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", n) and len(n) <= 48 for n in names)
    assert '[data-section="hero-section"]' in out  # the page's own CSS still applies
    assert tag_data_sections(out) == out  # idempotent after normalising


class _Reply:
    def __init__(self, output: str) -> None:
        self.output = output
        self.usage_metrics = {"prompt_tokens": 1, "completion_tokens": 2}


class _StubWM:
    def __init__(self, output: str) -> None:
        self._output = output

    async def message(self, **kwargs: object) -> _Reply:
        return _Reply(self._output)


async def test_generate_full_page_output_carries_data_sections() -> None:
    """The WIRE: the single-call fallback's returned HTML is tagged, not just the helper."""
    gen = HtmlGenerator(
        wm_client=cast("WMClient", _StubWM(f"```html\n{MEASURED_FULL_PAGE_HTML}\n```"))
    )
    result = await gen.generate_full_page(
        url="https://example.com/", html="<html><body><p>Example Domain</p></body></html>"
    )
    assert _sections(result.text) == [
        ("header", "header"),
        ("main", "example-domain"),
        ("footer", "footer"),
    ]


def _style(out: str) -> str:
    style = BeautifulSoup(out, "html.parser").find("style")
    assert style is not None
    return style.get_text()


def test_duplicate_model_value_keeps_its_style_on_every_carrier() -> None:
    # Audit repro: dedup-suffixing the 2nd "hero" and renaming the rule to hero-2
    # stripped the first hero of its style.
    html = """<html><head><style>[data-section="hero"]{background:red}</style></head>
    <body><section data-section="hero"><h1>A</h1></section>
    <section data-section="hero"><h2>B</h2></section></body></html>"""
    out = tag_data_sections(html)
    assert [n for _, n in _sections(out)] == ["hero", "hero"]
    assert _style(out) == '[data-section="hero"]{background:red}'
    assert out == html


def test_format_rename_never_chains_into_an_existing_value() -> None:
    # Audit repro: "Hero" -> "hero" then "hero" -> "hero-2" sent BOTH rules to hero-2.
    css = '[data-section="Hero"]{color:red}[data-section="hero"]{color:blue}'
    html = f"""<html><head><style>{css}</style></head><body>
    <section data-section="Hero"><h1>A</h1></section>
    <section data-section="hero"><h2>B</h2></section></body></html>"""
    out = tag_data_sections(html)
    assert [n for _, n in _sections(out)] == ["Hero", "hero"]
    assert _style(out) == css
    assert tag_data_sections(out) == out


def test_css_rename_touches_only_exact_attribute_selectors() -> None:
    css = (
        '[data-section="About Us"]{padding:1rem}'
        "[data-section='About Us' i]{margin:0}"
        '[data-section^="About Us"]{color:red}'
        '[data-section*="About Us"]{color:blue}'
        '.x::before{content:"[data-section=\\"About Us\\"]"}'
        '.y::after{content:"[data-section=hero]"}'
        '/* [data-section="About Us"] */'
    )
    html = f"""<html><head><style>{css}</style></head><body>
    <section data-section="About Us"><h2>About</h2></section></body></html>"""
    out = tag_data_sections(html)
    assert [n for _, n in _sections(out)] == ["about-us"]
    expected = css.replace('[data-section="About Us"]{', '[data-section="about-us"]{', 1).replace(
        "[data-section='About Us' i]", '[data-section="about-us" i]'
    )
    assert _style(out) == expected
    assert '.y::after{content:"[data-section=hero]"}' in _style(out)
    assert tag_data_sections(out) == out


def test_value_referenced_from_script_is_not_renamed() -> None:
    html = """<html><head><script>document.querySelector('[data-section="Hero"]')</script>
    </head><body><section data-section="Hero"><h1>A</h1></section></body></html>"""
    assert tag_data_sections(html) == html
