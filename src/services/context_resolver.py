"""Resolve and merge user context from multiple sources for redesign prompts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from bs4 import Tag

if TYPE_CHECKING:
    from bs4 import BeautifulSoup
    from bs4.element import PageElement

logger = logging.getLogger(__name__)

# How many image-manifest entries to carry into a prompt. Bounded so the
# "reuse these exact URLs" block stays small even on image-heavy pages.
IMAGE_MANIFEST_MAX = 40

# Per-section original-markup budget, in characters. The binding limit on a section
# call is the OUTPUT ceiling (worker_management clamps self-hosted output to ~12k
# tokens), not the 131k context: input only costs prefill. The model re-lays-out the
# section's TEXT, not its markup, and ~12k chars of markup carry ~4-5k chars of text
# (~1.5k tokens) — plus 1.5-3k tokens of section CSS, still far under the ceiling.
# 5000 was too small once content was the point: it cut a 19k ``main`` mid-card and
# still cuts a 9-testimonial section. Now that sections are outermost-landmark chunks
# (not ``main`` wholesale) and scripts/comments are stripped first, 12000 covers every
# real section seen; anything longer is cut at an ELEMENT boundary and flagged.
SECTION_MARKUP_BUDGET = 12000
SECTION_TEXT_PREVIEW_BUDGET = 600


@dataclass
class ResolvedContext:
    """Merged context from user input, site analysis, brand, and inference."""

    description: str | None = None
    site_type: str | None = None
    target_audience: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    focus_pages: list[str] = field(default_factory=list)
    exclude_pages: list[str] = field(default_factory=list)
    feature_priorities: list[dict] = field(default_factory=list)
    visual_preferences: dict = field(default_factory=dict)
    tone: str = "professional"
    primary_color: str = "#2563EB"
    secondary_color: str = "#1E40AF"
    brand_name: str | None = None
    font_family: str | None = None
    # Display (headings) + body fonts kept distinct so the redesign reloads the
    # brand's real type pairing (e.g. Fraunces display + IBM Plex Sans body)
    # instead of collapsing to a single family or system-ui.
    display_font: str | None = None
    body_font: str | None = None
    google_font_links: list[str] = field(default_factory=list)
    logo_url: str | None = None
    # Hex colours the SOURCE site actually declares/uses (``{"name", "value"}``), read
    # by :func:`extract_palette`. The accent guard picks a contrasting accent from
    # these instead of inventing one when the supplied secondary ~= the brand colour.
    source_palette: list[dict] = field(default_factory=list)
    # The source document's ``<html lang>`` and ``<title>`` (see extract_page_meta).
    page_lang: str | None = None
    page_title: str | None = None


# Patterns for inferring site type from HTML content
_SITE_TYPE_PATTERNS: dict[str, list[str]] = {
    "ecommerce": [
        r"add[- ]to[- ]cart",
        r"shopping[- ]cart",
        r"checkout",
        r"product[- ]price",
        r"buy[- ]now",
    ],
    "saas": [
        r"sign[- ]up",
        r"free[- ]trial",
        r"pricing[- ]plan",
        r"get[- ]started",
        r"dashboard",
    ],
    "blog": [
        r"blog[- ]post",
        r"article",
        r"published[- ]on",
        r"read[- ]more",
        r"author",
    ],
    "portfolio": [
        r"portfolio",
        r"my[- ]work",
        r"projects",
        r"case[- ]stud",
    ],
    "corporate": [
        r"about[- ]us",
        r"our[- ]team",
        r"careers",
        r"investor",
        r"annual[- ]report",
    ],
    "landing": [
        r"hero[- ]section",
        r"call[- ]to[- ]action",
        r"cta",
        r"landing",
    ],
}


_URL_PATH_PATTERNS: dict[str, list[str]] = {
    "blog": [r"/blog", r"/posts", r"/articles", r"/news"],
    "ecommerce": [r"/product", r"/shop", r"/store", r"/cart", r"/collection"],
    "saas": [r"/pricing", r"/features", r"/integrations", r"/docs"],
    "portfolio": [r"/portfolio", r"/work", r"/projects", r"/gallery"],
    "corporate": [r"/about", r"/careers", r"/investor", r"/press"],
    "landing": [r"/lp/", r"/promo/", r"/offer/"],
}

_META_TAG_PATTERNS: dict[str, list[str]] = {
    "ecommerce": [r"og:type.*product", r"shopify", r"woocommerce", r"magento"],
    "blog": [r"og:type.*article", r"wordpress", r"ghost"],
    "saas": [r"og:type.*website", r"intercom", r"hubspot"],
}


def _score_pattern_table(
    text: str, table: dict[str, list[str]], scores: dict[str, int], weight: int
) -> None:
    """Increment ``scores`` by ``weight`` for each pattern hit in ``table``."""
    for site_type, patterns in table.items():
        for p in patterns:
            if re.search(p, text):
                scores[site_type] = scores.get(site_type, 0) + weight


def _score_dom_signatures(html_lower: str, scores: dict[str, int]) -> None:
    if re.search(r'class="[^"]*product[- ]grid', html_lower):
        scores["ecommerce"] = scores.get("ecommerce", 0) + 3
    if re.search(r"<article", html_lower):
        scores["blog"] = scores.get("blog", 0) + 2
    if re.search(r'class="[^"]*pricing[- ]table', html_lower):
        scores["saas"] = scores.get("saas", 0) + 3


def infer_site_type(html: str, url: str | None = None) -> str | None:
    """Infer site type from HTML content and URL using heuristic pattern matching.

    Args:
        html: Raw HTML to analyze.
        url: Optional page URL for URL-path heuristics.

    Returns:
        Inferred site type string (ecommerce, saas, blog, etc.) or None.
    """
    if not html:
        return None

    scores: dict[str, int] = {}
    if url:
        _score_pattern_table(url.lower(), _URL_PATH_PATTERNS, scores, 2)

    html_lower = html.lower()
    _score_pattern_table(html_lower, _META_TAG_PATTERNS, scores, 2)
    # DOM structure patterns: per-pattern increment of 1 (variable weight).
    for site_type, patterns in _SITE_TYPE_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, html_lower))
        if score > 0:
            scores[site_type] = scores.get(site_type, 0) + score
    _score_dom_signatures(html_lower, scores)

    if not scores:
        return None
    return max(scores, key=lambda k: scores[k])


# Landmark elements a page is chunked by. ``main`` is a WRAPPER, never a chunk of its
# own when it holds landmarks: generating it AND its inner sections produced the hero
# twice (duplicate ``id="hero-heading"``) on the cliffordtraining.nl redesign.
_LANDMARK_TAGS = ("header", "nav", "main", "footer", "aside", "section")
_KNOWN_SECTION_TYPES = (
    "hero",
    "features",
    "pricing",
    "testimonials",
    "cta",
    "faq",
    "contact",
    "about",
    "team",
)
# Elements that are never page content: dropped before a section is serialized so the
# markup budget is spent on content, not on analytics scripts or JSON-LD.
_NON_CONTENT_TAGS = ("script", "noscript", "template")
# Loose text directly inside a wrapper landmark (outside its child landmarks) shorter
# than this is whitespace/decoration, not a section worth its own LLM call.
_MIN_LOOSE_TEXT_CHARS = 40
# Below this much remaining budget a cut child element is dropped, not descended into.
_MIN_DESCEND_BUDGET = 200


def _is_landmark(node: PageElement) -> bool:
    return isinstance(node, Tag) and node.name in _LANDMARK_TAGS


def _outermost_landmarks(root: Tag) -> list[Tag]:
    """Landmarks under ``root`` with no landmark between them and ``root``."""
    found: list[Tag] = []
    for tag in root.find_all(_LANDMARK_TAGS):
        if not isinstance(tag, Tag):
            continue
        parent = tag.parent
        while parent is not None and parent is not root and not _is_landmark(parent):
            parent = parent.parent
        if parent is None or parent is root:
            found.append(tag)
    return found


def _loose_wrapper(nodes: list[PageElement], source: str) -> Tag | None:
    """Wrap loose wrapper content in a ``<div>`` chunk, or None when it is trivial."""
    from bs4 import BeautifulSoup

    text = " ".join(
        n.get_text(" ", strip=True) if isinstance(n, Tag) else str(n).strip() for n in nodes
    ).strip()
    if len(text) < _MIN_LOOSE_TEXT_CHARS:
        return None
    wrapper = BeautifulSoup("<div></div>", "html.parser").div
    if wrapper is None:  # pragma: no cover - the literal always parses
        return None
    wrapper["data-source-wrapper"] = source
    for node in nodes:
        wrapper.append(node.extract())
    return wrapper


def _chunk_landmarks(tag: Tag) -> list[Tag]:
    """Decompose a WRAPPER landmark into its own outermost child landmarks.

    ``main`` holding any landmark, or any landmark holding two or more, is a wrapper:
    its children are the sections. Recursive, so ``main > section.page > section*``
    still chunks to the leaves that carry content. Loose content directly inside the
    wrapper (outside every child landmark) becomes its own chunk, in source order, so
    no text is dropped.
    """
    children = _outermost_landmarks(tag)
    if not ((tag.name == "main" and children) or len(children) >= 2):
        return [tag]
    out: list[Tag] = []
    loose: list[PageElement] = []

    def _flush() -> None:
        wrapper = _loose_wrapper(loose, tag.name)
        if wrapper is not None:
            out.append(wrapper)
        loose.clear()

    for node in list(tag.children):
        if not isinstance(node, Tag) or (
            not _is_landmark(node) and node.find(_LANDMARK_TAGS) is None
        ):
            loose.append(node)
            continue
        _flush()
        inner = [node] if _is_landmark(node) else _outermost_landmarks(node)
        for landmark in inner:
            out.extend(_chunk_landmarks(landmark))
    _flush()
    return out


def _detected_type(tag: Tag) -> str:
    if tag.get("data-source-wrapper"):
        return "section"
    classes = " ".join(tag.get("class") or [])
    identifier = f"{classes} {tag.get('id', '')}".lower()
    for known_type in _KNOWN_SECTION_TYPES:
        if known_type in identifier:
            return known_type
    return tag.name if tag.name in _LANDMARK_TAGS else "section"


def _base_key(tag: Tag, detected: str) -> str:
    """A generic landmark is keyed by what the SOURCE calls it (id, then first class),
    so the composed page's ``data-section`` names stay meaningful."""
    if detected not in ("section", "aside"):
        return detected
    classes = tag.get("class") or []
    hint = tag.get("id") or (classes[0] if classes else "")
    return re.sub(r"[^a-z0-9]+", "-", str(hint).lower()).strip("-") or detected


def _unique_key(base: str, used: set[str]) -> str:
    key = base or "section"
    n = 2
    while key in used:
        key = f"{base}-{n}"
        n += 1
    used.add(key)
    return key


def _attr(name: str, value: str | list[str] | None) -> str:
    from html import escape

    if isinstance(value, list):
        value = " ".join(value)
    if value is None:
        return f" {name}"
    return f' {name}="{escape(str(value), quote=True)}"'


def _fit_markup(node: Tag, budget: int) -> str:
    """Serialize ``node`` keeping whole children while they fit (see truncate_markup)."""
    from bs4 import BeautifulSoup, NavigableString

    if isinstance(node, BeautifulSoup):
        open_tag = close_tag = ""
    else:
        attrs = "".join(_attr(k, v) for k, v in node.attrs.items())
        open_tag, close_tag = f"<{node.name}{attrs}>", f"</{node.name}>"
    remaining = budget - len(open_tag) - len(close_tag)
    parts: list[str] = []
    for child in node.children:
        text = str(child)
        if len(text) <= remaining:
            parts.append(text)
            remaining -= len(text)
            continue
        if isinstance(child, Tag) and remaining > _MIN_DESCEND_BUDGET:
            parts.append(_fit_markup(child, remaining))
        elif isinstance(child, NavigableString) and remaining > 0:
            # Text is cut at a word boundary, never mid-word.
            parts.append(text[:remaining].rsplit(" ", 1)[0])
        break
    return f"{open_tag}{''.join(parts)}{close_tag}"


def truncate_markup(markup: str | Tag, budget: int) -> tuple[str, bool]:
    """Serialize markup within ``budget`` chars, cutting ONLY at element boundaries.

    A raw ``str(tag)[:budget]`` cut the cliffordtraining.nl ``main`` mid-card, so the
    model saw "Verandering vraagt om ander leiders" with no ending and card 6 not at
    all — and silently shipped the stub. Here whole child elements are kept while they
    fit; the first child that does not fit is descended into with the remaining budget,
    and everything after it is dropped. Every open element is closed. Returns
    ``(markup, truncated)`` so the prompt can SAY that the source was cut.
    """
    from bs4 import BeautifulSoup

    root = BeautifulSoup(markup, "html.parser") if isinstance(markup, str) else markup
    full = str(root)
    if len(full) <= budget:
        return full, False
    return _fit_markup(root, budget), True


def detect_sections(html: str) -> list[dict]:
    """Chunk a page into NON-OVERLAPPING outermost sections, in source order.

    Each dict carries ``key`` (unique across the page — the composed page stores
    sections by it), ``section_type``, ``heading``, ``role``, ``aria_label``, ``tag``,
    ``html`` (element-boundary-truncated original markup), ``truncated`` and
    ``text_preview``.

    Before this, every ``header/nav/main/footer/aside/section`` was returned INCLUDING
    nested ones, so ``main`` and the hero inside it were both generated (hero twice),
    and sections named by type collided and overwrote each other.
    """
    from bs4 import BeautifulSoup, Comment

    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    for junk in soup.find_all(_NON_CONTENT_TAGS):
        junk.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    body = soup.body or soup

    chunks: list[Tag] = []
    for landmark in _outermost_landmarks(body):
        chunks.extend(_chunk_landmarks(landmark))

    sections: list[dict] = []
    used: set[str] = set()
    for chunk in chunks:
        detected = _detected_type(chunk)
        heading = chunk.find(re.compile(r"^h[1-6]$"))
        split = split_oversized_section(chunk, SECTION_MARKUP_BUDGET)
        parts = split.parts
        base_key = _base_key(chunk, detected)
        first_key = ""
        for index, part in enumerate(parts, start=1):
            markup, truncated = truncate_markup(part, SECTION_MARKUP_BUDGET)
            key = _unique_key(base_key if index == 1 else f"{base_key}-part-{index}", used)
            first_key = first_key or key
            section = _section_dict(chunk, part, detected, heading, key, markup, truncated)
            if len(parts) > 1:
                section.update(part=index, parts=len(parts), part_of=first_key)
            if index == len(parts) and split.omitted_items:
                # Never silently: the prompt says the section was cut and by how much.
                section.update(truncated=True, omitted_items=split.omitted_items)
            sections.append(section)
    return sections


# A source section larger than the markup budget is split into at most this many parts
# (each its own generation call, bounded by the job's generation budget). Items that do
# not fit even then are OMITTED — counted on the last part (``omitted_items``, and
# ``truncated=True``) so the prompt tells the model; never dropped silently.
MAX_SECTION_PARTS = 6


@dataclass(frozen=True)
class SectionSplit:
    """The parts of one source section, and how many trailing items did not fit."""

    parts: list[Tag]
    omitted_items: int = 0


def _item_container(chunk: Tag) -> list[int] | None:
    """Child-index path to the element holding the section's repeated items, or None.

    The deepest descendant that carries most of the section's markup across >= 3
    element children (a card grid, a testimonial list). Only element children count.
    """
    total = len(str(chunk))
    path: list[int] = []
    node = chunk
    found: list[int] | None = None
    while True:
        kids = [c for c in node.children if isinstance(c, Tag)]
        if len(kids) >= 3 and len(str(node)) >= total * 0.6:
            found = list(path)
        big = [(i, k) for i, k in enumerate(kids) if len(str(k)) >= total * 0.6]
        if not big:
            return found
        index, node = big[0]
        path.append(index)


def _element_children(node: Tag) -> list[Tag]:
    return [c for c in node.children if isinstance(c, Tag)]


def _group_items(items: list[Tag], room: int) -> tuple[list[list[int]], int]:
    """Consecutive item indices packed into groups of at most ``room`` chars.

    Returns ``(groups, omitted)`` — ``omitted`` counts the trailing items that did not
    fit into ``MAX_SECTION_PARTS`` groups.
    """
    groups: list[list[int]] = [[]]
    size = 0
    for i, item in enumerate(items):
        n = len(str(item))
        if groups[-1] and size + n > room:
            if len(groups) == MAX_SECTION_PARTS:
                return groups, len(items) - i
            groups.append([])
            size = 0
        groups[-1].append(i)
        size += n
    return groups, 0


def _build_part(chunk: Tag, path: list[int], group: list[int], *, first: bool, last: bool) -> Tag:
    """A copy of ``chunk`` holding only ``group``'s items.

    Content BEFORE the item container (heading, intro) stays in the first part only,
    content AFTER it (a "read all" link) in the last part only.
    """
    import copy

    part = copy.copy(chunk)  # bs4 copies the whole subtree
    node = part
    for index in path:
        kids = _element_children(node)
        for j, kid in enumerate(kids):
            if (j < index and not first) or (j > index and not last):
                kid.decompose()
        node = kids[index]
    for j, item in enumerate(_element_children(node)):
        if j not in group:
            item.decompose()
    return part


def split_oversized_section(chunk: Tag, budget: int) -> SectionSplit:
    """Split a section too large for one prompt/reply into parts at ITEM boundaries.

    A whole-section reply for a large source (a 75-recommendation testimonial wall)
    cannot fit the provider's output ceiling, and truncating the source silently
    dropped the trailing items. Each part is a copy of the section keeping the path to
    its item container (so its classes still describe the design). A section with no
    item container, or small enough, is returned whole; at most ``MAX_SECTION_PARTS``,
    with any items beyond them counted in ``omitted_items``.
    """
    if len(str(chunk)) <= budget:
        return SectionSplit([chunk])
    path = _item_container(chunk)
    if path is None:
        return SectionSplit([chunk])
    container = chunk
    for index in path:
        container = _element_children(container)[index]
    overhead = len(str(chunk)) - len(str(container))
    groups, omitted = _group_items(
        _element_children(container), max(budget - overhead, budget // 2)
    )
    if len(groups) < 2:
        return SectionSplit([chunk])
    return SectionSplit(
        [
            _build_part(chunk, path, group, first=n == 0, last=n == len(groups) - 1)
            for n, group in enumerate(groups)
        ],
        omitted,
    )


def _section_dict(
    chunk: Tag,
    part: Tag,
    detected: str,
    heading: Tag | None,
    key: str,
    markup: str,
    truncated: bool,
) -> dict:
    return {
        "key": key,
        "section_type": detected,
        "heading": heading.get_text(strip=True) if heading else None,
        "role": chunk.get("role") or None,
        "aria_label": chunk.get("aria-label") or None,
        "tag": chunk.name,
        # The section's own original markup so the redesign prompt can read the
        # real content, colours and image ``src`` URLs. Cut only at element
        # boundaries; ``truncated`` tells the prompt when it was cut.
        "html": markup,
        "truncated": truncated,
        "text_preview": part.get_text(" ", strip=True)[:SECTION_TEXT_PREVIEW_BUDGET],
    }


def _classify_image_role(tag: Any, identifier: str) -> str:
    """Best-effort role for an <img>: logo / content (og handled separately)."""
    if "logo" in identifier or "brand" in identifier:
        return "logo"
    # An image directly inside the <header>/<nav> with no obvious content role
    # is almost always the logo.
    for parent in tag.parents:
        name = getattr(parent, "name", None)
        if name in ("header", "nav"):
            return "logo"
        if name in ("main", "article", "section", "footer"):
            break
    return "content"


def extract_image_manifest(html: str, base_url: str | None) -> list[dict]:
    """Extract every reusable image from the original page, ABSOLUTIZED.

    The redesign HTML is rendered/screenshotted off the source origin, so any
    relative ``src`` (e.g. ``/assets/book/cover.png``) 404s and the image goes
    blank. We resolve every URL against ``base_url`` up front and hand the model
    an explicit "reuse these exact absolute URLs" list — decoupled from the
    truncated section markup so images never fall outside the prompt window.

    Returns a deduped, capped list of ``{role, url, alt}`` dicts.
    ``data:`` URIs are kept verbatim (already self-contained).
    """
    from bs4 import BeautifulSoup

    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    manifest: list[dict] = []

    def _add(role: str, raw: Any, alt: Any) -> None:
        if not isinstance(raw, str):
            return
        raw = raw.strip()
        if not raw or raw.startswith(("javascript:", "#")):
            return
        url = raw if raw.startswith("data:") else urljoin(base_url or "", raw)
        if not url or url in seen:
            return
        seen.add(url)
        alt_text = alt.strip() if isinstance(alt, str) else ""
        manifest.append({"role": role, "url": url, "alt": alt_text or None})

    # og:image first (canonical social preview, always absolute-worthy).
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        _add("og", og.get("content"), None)

    for img in soup.find_all("img"):
        classes = " ".join(img.get("class") or [])
        identifier = f"{classes} {img.get('id', '')}".lower()
        role = _classify_image_role(img, identifier)
        # Prefer src; fall back to the first srcset candidate.
        src = img.get("src")
        srcset = img.get("srcset")
        if not src and isinstance(srcset, str):
            src = srcset.split(",")[0].strip().split(" ")[0]
        _add(role, src, img.get("alt"))
        if len(manifest) >= IMAGE_MANIFEST_MAX:
            break

    return manifest[:IMAGE_MANIFEST_MAX]


# CSS custom-property names that conventionally hold the heading vs body font.
_DISPLAY_VAR_HINTS = ("display", "heading", "serif", "title", "head")
_BODY_VAR_HINTS = ("sans", "body", "text", "base", "copy")
_FONT_VALUE_RE = re.compile(r"font-family\s*:\s*([^;}{]+)", re.IGNORECASE)
_FONT_VAR_RE = re.compile(r"(--[\w-]*font[\w-]*)\s*:\s*([^;}{]+)", re.IGNORECASE)


def _clean_font_family(value: str) -> str | None:
    """Take the first concrete family name from a font-family value list."""
    first = value.split(",")[0].strip().strip("'\"")
    # Drop CSS keywords / var() references — we want a nameable family.
    if (
        not first
        or first.startswith("var(")
        or first.lower()
        in {
            "inherit",
            "initial",
            "unset",
            "system-ui",
            "sans-serif",
            "serif",
            "monospace",
        }
    ):
        return None
    return first


def _fonts_from_css_vars(css_text: str) -> tuple[str | None, str | None]:
    """Classify ``--font-*`` custom properties into (display, body) families."""
    display: str | None = None
    body: str | None = None
    for var_name, value in _FONT_VAR_RE.findall(css_text):
        family = _clean_font_family(value)
        if not family:
            continue
        name = var_name.lower()
        if display is None and any(h in name for h in _DISPLAY_VAR_HINTS):
            display = family
        elif body is None and any(h in name for h in _BODY_VAR_HINTS):
            body = family
    return display, body


def _families_from_google_links(google_links: list[str]) -> list[str]:
    """The ``family=`` names in Google-Fonts URLs, in order."""
    families: list[str] = []
    for href in google_links:
        for fam in re.findall(r"family=([^:&]+)", href):
            families.append(fam.replace("+", " ").strip())
    return families


# ``<script type="application/json" id=...>`` block the Playwright fetch appends to the
# captured DOM (see ``design_builder.helpers.SOURCE_STYLES_JS``): what the browser
# ACTUALLY applied — body vs heading font-family and the colours in use. A page's real
# CSS usually lives in an external stylesheet the resolver never sees, so without it
# the font roles were guessed from Google-Fonts URL order (body became Fraunces).
SOURCE_STYLES_SCRIPT_ID = "hrs-source-styles"
_HEX_RE = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_COLOR_VAR_RE = re.compile(r"(--[\w-]+)\s*:\s*(#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3}))\b")
_CSS_RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_HEADING_SELECTOR_RE = re.compile(r"(?:^|[\s,>+~])h[1-3](?:$|[\s,.:#\[>+~])")
_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8})*$")


def _normalize_hex(value: str) -> str | None:
    v = value.strip()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", v):
        v = "#" + "".join(ch * 2 for ch in v[1:])
    if re.fullmatch(r"#[0-9a-fA-F]{6}", v):
        return v.upper()
    return None


def extract_source_styles(html: str) -> dict:
    """The computed-style snapshot the fetch embedded, or ``{}`` when absent/invalid."""
    import json

    from bs4 import BeautifulSoup

    if not html or SOURCE_STYLES_SCRIPT_ID not in html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    node = soup.find("script", attrs={"id": SOURCE_STYLES_SCRIPT_ID})
    if node is None:
        return {}
    try:
        data = json.loads(node.get_text())
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _stylesheet_font_links(soup: BeautifulSoup) -> list[str]:
    """Google-Fonts STYLESHEET hrefs only.

    A bare ``<link rel="preconnect" href="https://fonts.googleapis.com">`` is not a
    stylesheet; collecting it emitted ``<link rel="stylesheet"
    href="https://fonts.googleapis.com">`` into every redesign. Only ``rel`` containing
    ``stylesheet`` (or a ``preload as=style``) with a ``/css`` / ``/css2`` path counts.
    """
    from urllib.parse import urlparse

    links: list[str] = []
    for link in soup.find_all("link"):
        href = link.get("href")
        if not isinstance(href, str):
            continue
        parsed = urlparse(href)
        if parsed.hostname != "fonts.googleapis.com" or not parsed.path.startswith("/css"):
            continue
        rel = link.get("rel") or []
        rels = {r.lower() for r in (rel if isinstance(rel, list) else str(rel).split())}
        is_style = "stylesheet" in rels or (
            "preload" in rels and str(link.get("as", "")).lower() == "style"
        )
        if is_style and href not in links:
            links.append(href)
    return links


def _fonts_from_css_rules(css_text: str) -> tuple[str | None, str | None]:
    """(display, body) from which selectors a ``font-family`` is APPLIED to."""
    display: str | None = None
    body: str | None = None
    for selector, block in _CSS_RULE_RE.findall(css_text):
        fam_match = _FONT_VALUE_RE.search(block)
        if not fam_match:
            continue
        family = _clean_font_family(fam_match.group(1))
        if not family:
            continue
        sel = selector.strip().lower()
        parts = [p.strip() for p in sel.split(",")]
        if body is None and any(p in ("body", "html", ":root") for p in parts):
            body = family
        if display is None and _HEADING_SELECTOR_RE.search(sel):
            display = family
    return display, body


def extract_fonts(html: str) -> dict:
    """Extract the page's real font pairing so the redesign can reload it.

    Returns ``{"google_links": [...], "display": str|None, "body": str|None}``.
    Role evidence, strongest first: the computed ``font-family`` of ``body`` vs the
    first heading (embedded by the Playwright fetch); inline ``<style>`` rules applied
    to ``body`` vs ``h1-h3``; ``--font-*`` custom properties classified by name; and
    only as a last resort the ``family=`` order of a Google-Fonts URL.
    """
    from bs4 import BeautifulSoup

    if not html:
        return {"google_links": [], "display": None, "body": None}
    soup = BeautifulSoup(html, "html.parser")
    google_links = _stylesheet_font_links(soup)

    computed = extract_source_styles(html)
    display = _clean_font_family(str(computed.get("heading_font") or "")) or None
    body = _clean_font_family(str(computed.get("body_font") or "")) or None

    css_text = " ".join(s.get_text(" ") for s in soup.find_all("style"))
    rule_display, rule_body = _fonts_from_css_rules(css_text)
    var_display, var_body = _fonts_from_css_vars(css_text)
    display = display or rule_display or var_display
    body = body or rule_body or var_body

    # Fallback: family= order in the Google-Fonts URL.
    if (display is None or body is None) and google_links:
        families = [
            f for f in _families_from_google_links(google_links) if f not in (display, body)
        ]
        if display is None and families:
            display = families.pop(0)
        if body is None and families:
            body = families[0]

    return {"google_links": google_links, "display": display, "body": body}


def extract_palette(html: str) -> list[dict]:
    """Hex colours the source site declares or visibly uses, deduped, in evidence order.

    ``[{"name": "--accent", "value": "#B8501F"}, ...]``. Sources: the fetch's
    computed-style snapshot (root custom properties + colours of links/buttons),
    ``--*`` colour custom properties in inline ``<style>``, and ``theme-color``.
    """
    from bs4 import BeautifulSoup

    if not html:
        return []
    out: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, raw: object) -> None:
        value = _normalize_hex(str(raw)) if raw else None
        if value and value not in seen:
            seen.add(value)
            out.append({"name": name, "value": value})

    computed = extract_source_styles(html)
    for item in computed.get("vars") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            _add(str(item[0]), item[1])
    for item in computed.get("used") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            _add(str(item[0]), item[1])

    soup = BeautifulSoup(html, "html.parser")
    css_text = " ".join(s.get_text(" ") for s in soup.find_all("style"))
    for name, value in _COLOR_VAR_RE.findall(css_text):
        _add(name, value)
    theme = soup.find("meta", attrs={"name": "theme-color"})
    if theme is not None:
        _add("theme-color", theme.get("content"))
    return out


def extract_page_meta(html: str) -> dict:
    """``{"lang": ..., "title": ...}`` of the source document (either may be None)."""
    from bs4 import BeautifulSoup

    if not html:
        return {"lang": None, "title": None}
    soup = BeautifulSoup(html, "html.parser")
    lang = soup.html.get("lang") if soup.html is not None else None
    lang = str(lang).strip() if lang else None
    if lang and not _LANG_RE.match(lang):
        lang = None
    title_tag = soup.find("title")
    title = " ".join(title_tag.get_text(" ").split()) if title_tag else None
    return {"lang": lang, "title": title or None}


def resolve_context(
    *,
    user_context: dict | None = None,
    site_analysis: dict | None = None,
    brand: dict | None = None,
    html: str | None = None,
) -> ResolvedContext:
    """Merge context from multiple sources into a single ResolvedContext.

    Priority order: user_context > site_analysis > brand > inferred from HTML.

    Args:
        user_context: User-provided context dict.
        site_analysis: Site analysis results dict.
        brand: Brand identity dict.
        html: Raw HTML for inferring site type.

    Returns:
        A fully resolved ResolvedContext with merged values.
    """
    ctx = ResolvedContext()
    fonts: dict = {}

    if html:
        inferred_type = infer_site_type(html)
        if inferred_type:
            ctx.site_type = inferred_type
        # Base font pairing from the page's own markup; brand/site_analysis
        # layers below override when they carry explicit fonts.
        fonts = extract_fonts(html)
        ctx.display_font = fonts.get("display")
        ctx.body_font = fonts.get("body")
        ctx.google_font_links = fonts.get("google_links") or []
        ctx.source_palette = extract_palette(html)
        meta = extract_page_meta(html)
        ctx.page_lang = meta.get("lang")
        ctx.page_title = meta.get("title")

    _apply_brand_layer(ctx, brand)
    _apply_site_analysis_layer(ctx, site_analysis)
    _apply_user_context_layer(ctx, user_context)
    if html:
        # What the browser APPLIED outranks a brand extractor's guess at font ROLES:
        # the guess put the display face on body text (body = Fraunces).
        computed = extract_source_styles(html)
        if computed.get("heading_font") and fonts.get("display"):
            ctx.display_font = fonts["display"]
        if computed.get("body_font") and fonts.get("body"):
            ctx.body_font = fonts["body"]
    return ctx


def _apply_brand_layer(ctx: ResolvedContext, brand: dict | None) -> None:
    if not brand:
        return
    ctx.primary_color = brand.get("primary_color", ctx.primary_color)
    ctx.secondary_color = brand.get("secondary_color", ctx.secondary_color)
    ctx.brand_name = brand.get("name", ctx.brand_name)
    ctx.font_family = brand.get("font_family", ctx.font_family)
    # Distinct display/body fonts when the brand extractor supplies them;
    # font_family stays the single-field fallback for display.
    if brand.get("display_font") or brand.get("font_family"):
        ctx.display_font = brand.get("display_font") or brand.get("font_family")
    if brand.get("body_font"):
        ctx.body_font = brand["body_font"]
    ctx.logo_url = brand.get("logo_url", ctx.logo_url)


def _apply_site_analysis_layer(ctx: ResolvedContext, site_analysis: dict | None) -> None:
    if not site_analysis:
        return
    ctx.description = site_analysis.get("description", ctx.description)
    if site_analysis.get("target_audience"):
        ctx.target_audience = site_analysis["target_audience"]
    ctx.tone = site_analysis.get("tone", ctx.tone) or ctx.tone
    vis = site_analysis.get("visual_identity") or {}
    if vis.get("primary_color"):
        ctx.primary_color = vis["primary_color"]
    if vis.get("secondary_color"):
        ctx.secondary_color = vis["secondary_color"]


def _apply_user_context_layer(ctx: ResolvedContext, user_context: dict | None) -> None:
    if not user_context:
        return
    direct_fields = (
        "description",
        "site_type",
        "target_audience",
        "goals",
        "focus_pages",
        "exclude_pages",
        "feature_priorities",
        "visual_preferences",
    )
    for field_name in direct_fields:
        if user_context.get(field_name):
            setattr(ctx, field_name, user_context[field_name])


# Mapping of user goals to design actions for prompt injection
GOAL_DESIGN_ACTIONS: dict[str, str] = {
    "increase_conversions": "Optimize CTA placement, use contrasting button colors, add urgency elements",
    "improve_engagement": "Add interactive elements, improve content hierarchy, use compelling visuals",
    "reduce_bounce_rate": "Improve above-the-fold content, speed up perceived load time, clear value proposition",
    "build_trust": "Add testimonials, trust badges, clear contact info, professional imagery",
    "generate_leads": "Prominent lead capture forms, compelling offer, clear benefits list",
    "boost_sales": "Product highlights, social proof, easy checkout flow, urgency indicators",
    "improve_seo": "Semantic HTML, proper heading hierarchy, image alt text, structured data",
    "enhance_accessibility": "WCAG compliance, keyboard navigation, screen reader support, color contrast",
}
