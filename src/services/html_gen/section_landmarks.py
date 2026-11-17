"""Give a single-call full-page redesign the ``data-section`` landmarks it lacks.

Only the per-section path wraps its output in ``<section data-section=...>``. The
``generate_full_page`` fallback (a source page with no detectable sections, e.g.
example.com) returned whatever band structure the model chose: a full-width
header, a centred content column, a footer — and no ``data-section`` anywhere.
``render_section_screenshots`` then found no named section and cut anonymous
``band-NN`` stills, so the website_review after-act script had no section heading
to ground a narration line on (4/4 live runs, round 4, 2026-10-06).

This is the deterministic half of the fix (the prompt asks for the attribute too,
but a model instruction is not a guarantee): walk the body's top-level blocks,
descend through pure wrappers, and tag every content block that lacks the
attribute with a slug of its first heading (else its tag name). A block that
already contains a named section is left alone so the named inner one stays
top-level.

Values the MODEL wrote itself get a pure FORMAT normalisation to the same slug
form (``[a-z0-9-]``, <=48 chars): the orchestrator's ``build_inventory`` looks a
screenshot up by its raw ``data-section``, so ``"Hero Section"`` would otherwise
never match the role name. The rename never retargets the page's own styles:

* the full old->new mapping is built first; a value is renamed only when every
  element carrying it maps to the same new value and that value collides with no
  other existing (or already-claimed) value — ``Hero`` next to a ``hero`` stays
  ``Hero``. Model-written duplicates are NOT suffixed (two ``hero`` blocks both
  stay ``hero``, so ``[data-section="hero"]`` still styles both);
* a value referenced from a ``<script>`` is never renamed (a JS selector is not
  something we can rewrite safely);
* ``<style>`` text is rewritten in ONE regex pass whose alternation consumes CSS
  comments and quoted strings untouched, so only a real ``[data-section=...]``
  attribute selector with the exact ``=`` operator is renamed — never ``^=``/
  ``*=``/``~=``/``|=``/``$=``, never text inside ``content:"..."``.

Our own generated tags still de-duplicate against every value on the page.
Idempotent.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bs4 import BeautifulSoup, Tag

__all__ = ["tag_data_sections"]

# Block containers that can be a page band. Inline / raw-text / metadata elements
# (script, style, a, span, img, ...) are never tagged.
_BLOCK_TAGS = frozenset(
    {"header", "nav", "main", "section", "article", "aside", "footer", "div", "form"}
)
# Structural chrome: the heading inside a header is usually the site title, which
# would collide with the hero's heading — name these by their role instead.
_ROLE_NAMED_TAGS = frozenset({"header", "nav", "footer"})
# Ignored when deciding whether a container is a pure wrapper.
_NON_CONTENT_TAGS = frozenset({"script", "style", "noscript", "template", "link", "meta"})
_MAX_SLUG_LEN = 48
_MAX_WRAPPER_DEPTH = 4


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:_MAX_SLUG_LEN].strip("-")


def _element_children(tag: Tag) -> list[Tag]:
    from bs4 import Tag as _Tag

    return [
        c for c in tag.children if isinstance(c, _Tag) and c.name.lower() not in _NON_CONTENT_TAGS
    ]


def _has_content(tag: Tag) -> bool:
    return (
        bool(tag.get_text(strip=True)) or tag.find(["img", "picture", "svg", "video"]) is not None
    )


def _is_wrapper(tag: Tag) -> bool:
    """A container whose meaningful children are ALL blocks (>=2): descend into it.

    A container holding loose headings/paragraphs is itself the band — splitting it
    would leave that loose text untagged.
    """
    if tag.has_attr("data-section"):
        return False
    kids = _element_children(tag)
    if tag.name.lower() in _ROLE_NAMED_TAGS:
        return False
    return len(kids) >= 2 and all(k.name.lower() in _BLOCK_TAGS for k in kids)


def _blocks(container: Tag, depth: int = 0) -> list[Tag]:
    out: list[Tag] = []
    kids = _element_children(container)
    # A single wrapper (``<div class="page">`` / ``<main>``) around everything:
    # the bands are inside it.
    if depth < _MAX_WRAPPER_DEPTH and len(kids) == 1 and kids[0].name.lower() in _BLOCK_TAGS:
        only = kids[0]
        if not only.has_attr("data-section") and _is_wrapper(only):
            return _blocks(only, depth + 1)
    for kid in kids:
        if kid.name.lower() not in _BLOCK_TAGS:
            continue
        if depth < _MAX_WRAPPER_DEPTH and _is_wrapper(kid):
            out.extend(_blocks(kid, depth + 1))
        else:
            out.append(kid)
    return out


def _name_for(block: Tag) -> str:
    tag = block.name.lower()
    if tag not in _ROLE_NAMED_TAGS:
        heading = block.find(["h1", "h2", "h3"])
        if heading is not None:
            slug = _slug(heading.get_text(" ", strip=True))
            if slug:
                return slug
    role = block.get("role")
    return _slug(role) if isinstance(role, str) and _slug(role) else tag


# One pass over a stylesheet: comments and quoted strings are matched (and kept
# verbatim) by the first two branches, so a selector-looking run inside them can
# never reach the third branch. Only the exact ``=`` operator is accepted.
_CSS_SCAN_RE = re.compile(
    r"(?P<comment>/\*.*?\*/)"
    r"|(?P<string>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
    r"|(?P<sel>\[\s*(?i:data-section)\s*=\s*"
    r"(?:\"(?P<dq>[^\"\\]*)\"|'(?P<sq>[^'\\]*)'|(?P<uq>[A-Za-z0-9_-]+))"
    r"(?P<tail>\s*(?:[iIsS]\s*)?\]))",
    re.DOTALL,
)


def _raw_value(el: Tag) -> str:
    raw = el.get("data-section")
    return raw if isinstance(raw, str) else " ".join(raw or [])


def _rename_css(css: str, mapping: dict[str, str]) -> str:
    def _sub(m: re.Match[str]) -> str:
        if m.group("sel") is None:
            return m.group(0)
        value = next(v for v in (m.group("dq"), m.group("sq"), m.group("uq")) if v is not None)
        new = mapping.get(value)
        if new is None:
            return m.group(0)
        return f'[data-section="{new}"{m.group("tail")}'

    return _CSS_SCAN_RE.sub(_sub, css)


def _rename_mapping(carriers: dict[str, list[Tag]], script_text: str) -> dict[str, str]:
    """old->new for every value whose rename is a pure, collision-free normalisation."""
    mapping: dict[str, str] = {}
    claimed = set(carriers)
    for raw, els in carriers.items():
        slug = _slug(raw)
        targets = {slug} if slug else {_name_for(el) or "section" for el in els}
        if len(targets) != 1:
            continue  # carriers would split: leave the model's value alone
        (target,) = targets
        if target == raw or target in claimed or (raw and raw in script_text):
            continue
        mapping[raw] = target
        claimed.add(target)
    return mapping


def _normalise_existing(soup: BeautifulSoup, used: set[str]) -> bool:
    """Format-normalise model-written ``data-section`` values without retargeting CSS.

    Fills ``used`` with every final value so generated tags dedupe against them.
    """
    carriers: dict[str, list[Tag]] = {}
    for el in soup.select("[data-section]"):
        carriers.setdefault(_raw_value(el), []).append(el)
    script_text = "\n".join(s.get_text() for s in soup.find_all("script"))
    mapping = _rename_mapping(carriers, script_text)

    for raw, els in carriers.items():
        final = mapping.get(raw, raw)
        used.add(final)
        if final != raw:
            for el in els:
                el["data-section"] = final
    if mapping:
        for style in soup.find_all("style"):
            css = style.string
            if not css:
                continue
            new = _rename_css(css, mapping)
            if new != css:
                style.string = new
    return bool(mapping)


def _unique(base: str, used: set[str]) -> str:
    name, n = base, 2
    while name in used:
        name, n = f"{base}-{n}", n + 1
    used.add(name)
    return name


def tag_data_sections(html: str) -> str:
    """Return ``html`` with each top-level content block carrying ``data-section``.

    No-op (byte-identical) when nothing needs tagging or renaming.
    """
    from bs4 import BeautifulSoup

    if not html or not html.strip():
        return html
    soup = BeautifulSoup(html, "html.parser")
    root = soup.body or soup
    used: set[str] = set()
    changed = _normalise_existing(soup, used)
    for block in _blocks(root):
        if block.has_attr("data-section") or block.select_one("[data-section]"):
            continue
        if not _has_content(block):
            continue
        block["data-section"] = _unique(_name_for(block) or "section", used)
        changed = True
    return str(soup) if changed else html
