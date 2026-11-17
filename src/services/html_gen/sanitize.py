"""Deterministic post-processing of LLM section markup before it is composed/shipped.

Every function here is LLM-independent insurance for a defect that shipped in the
cliffordtraining.nl redesign:

* inline ``on*`` handlers — the model wrote ``onerror="…class=\\"x\\"…"``; HTML has no
  backslash escape, so the lenient parser round-trip turned it into a garbage
  attribute (``logo-text-fallback\\"=""``) and leaked the rest of the handler as
  visible text. They are removed at STRING level, before any parse;
* Cloudflare artefacts — ``/cdn-cgi/`` email-decode scripts and
  ``[email protected]`` placeholders copied from the captured DOM;
* relative ``href``s that 404 once the concept is viewed off-origin;
* duplicate ``id``s across independently generated sections;
* the ``→`` glyph, which no brand webfont subset covers, so the PDF renderer fell
  back to WenQuanYi Zen Hei for it;
* section CSS leaking across sections — every section is generated independently from
  a source site that reuses class names (``.stat-num``, ``.eyebrow``, ``.btn``), so one
  section's rules silently restyled another's (hero statistics turned ink-on-navy).
  Each section's ``<style>`` is wrapped in ``@scope`` to its own band;
* ``loading="lazy"`` images, which a headless full-page screenshot / PDF print never
  scrolls into view, so they rendered as empty boxes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from .css_tokens import honor_contradicted_fallbacks
from .images import _abs_srcset, _abs_url

if TYPE_CHECKING:
    from bs4 import BeautifulSoup
    from bs4.element import PageElement

__all__ = [
    "decode_cfemail",
    "finalize_fragment",
    "scope_css",
    "strip_inline_event_handlers",
]

_TAG_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9:-]*")
_ATTR_NAME_RE = re.compile(r"[^\s=/>\"'<]+")
_RAW_TEXT_TAGS = ("script", "style")


def _scan_quoted(text: str, i: int, *, backslash_escapes: bool) -> int:
    """Index just past the closing quote of the value starting at ``text[i]``."""
    quote = text[i]
    j = i + 1
    n = len(text)
    while j < n:
        ch = text[j]
        if backslash_escapes and ch == "\\" and j + 1 < n:
            j += 2
            continue
        if ch == quote:
            return j + 1
        j += 1
    return n


def _skip_space(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _scan_attr_value(text: str, i: int, attr_name: str) -> int:
    """Index just past ``attr_name``'s optional ``= value`` starting at ``text[i]``."""
    n = len(text)
    k = _skip_space(text, i)
    if k >= n or text[k] != "=":
        return i
    k = _skip_space(text, k + 1)
    if k < n and text[k] in "\"'":
        # Only a JS handler value is scanned with backslash escapes: that is the LLM's
        # (invalid-HTML) spelling that broke the parser.
        return _scan_quoted(text, k, backslash_escapes=attr_name.startswith("on"))
    while k < n and not text[k].isspace() and text[k] != ">":
        k += 1
    return k


def _scan_tag(text: str, start: int) -> tuple[int, str, list[tuple[int, int, str]]]:
    """Scan the start tag at ``text[start] == '<'``.

    Returns ``(end, tag_name, attrs)`` where ``attrs`` are ``(span_start, span_end,
    name)`` for each attribute, spans including the leading whitespace.
    """
    m = _TAG_NAME_RE.match(text, start + 1)
    if m is None:
        return start + 1, "", []
    name = m.group(0).lower()
    i = m.end()
    n = len(text)
    attrs: list[tuple[int, int, str]] = []
    while i < n:
        ws_start = i
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        if text[i] == ">":
            return i + 1, name, attrs
        if text.startswith("/>", i):
            return i + 2, name, attrs
        am = _ATTR_NAME_RE.match(text, i)
        if am is None:
            i += 1
            continue
        attr_name = am.group(0).lower()
        i = _scan_attr_value(text, am.end(), attr_name)
        attrs.append((ws_start, i, attr_name))
    return n, name, attrs


def strip_inline_event_handlers(html: str) -> str:
    """Remove every ``on*`` attribute from every start tag, at string level.

    Runs BEFORE any HTML parse (see module docstring). Comments and the raw text of
    ``<script>``/``<style>`` are skipped verbatim.
    """
    if not html or not re.search(r"\son[a-z]+\s*=", html, re.IGNORECASE):
        return html
    out: list[str] = []
    i = 0
    n = len(html)
    while i < n:
        lt = html.find("<", i)
        if lt == -1:
            out.append(html[i:])
            break
        out.append(html[i:lt])
        if html.startswith("<!--", lt):
            end = html.find("-->", lt + 4)
            end = n if end == -1 else end + 3
            out.append(html[lt:end])
            i = end
            continue
        end, name, attrs = _scan_tag(html, lt)
        if not name:
            out.append("<")
            i = lt + 1
            continue
        cursor = lt
        for span_start, span_end, attr_name in attrs:
            if attr_name.startswith("on"):
                out.append(html[cursor:span_start])
                cursor = span_end
        out.append(html[cursor:end])
        i = end
        if name in _RAW_TEXT_TAGS:
            close = re.compile(rf"</{name}\s*>", re.IGNORECASE).search(html, i)
            stop = n if close is None else close.start()
            out.append(html[i:stop])
            i = stop
    return "".join(out)


def decode_cfemail(encoded: str) -> str | None:
    """Decode a Cloudflare email-protection hex string (first byte is the XOR key)."""
    try:
        data = bytes.fromhex(encoded.strip())
    except ValueError:
        return None
    if len(data) < 2:
        return None
    key = data[0]
    try:
        return bytes(b ^ key for b in data[1:]).decode("utf-8")
    except UnicodeDecodeError:
        return None


_CF_EMAIL_PATH = "/cdn-cgi/l/email-protection"
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ID_REF_ATTRS = ("aria-labelledby", "aria-describedby", "aria-controls", "aria-owns", "for")
_ARROWS = {
    "→": "M2 8h11M9 4l4 4-4 4",  # →
    "←": "M14 8H3M7 4 3 8l4 4",  # ←
}
_NO_ARROW_PARENTS = ("script", "style", "svg", "title", "textarea", "option", "code", "pre")


def _decode_cloudflare_emails(soup: BeautifulSoup) -> None:
    for node in soup.select("[data-cfemail]"):
        email = decode_cfemail(str(node.get("data-cfemail", "")))
        if email is None:
            continue
        parent = node.parent
        node.replace_with(email)
        if parent is not None and getattr(parent, "name", None) == "a":
            href = parent.get("href")
            if isinstance(href, str) and _CF_EMAIL_PATH in href:
                parent["href"] = f"mailto:{email}"
    for a in soup.find_all("a", href=True):
        href = str(a.get("href", ""))
        if _CF_EMAIL_PATH not in href:
            continue
        email = decode_cfemail(href.split("#", 1)[1]) if "#" in href else None
        if email:
            a["href"] = f"mailto:{email}"
        else:
            del a["href"]


def _absolutize_links(soup: BeautifulSoup, base_url: str | None) -> None:
    for tag in soup.find_all(["a", "area", "link", "form"]):
        attr = "action" if tag.name == "form" else "href"
        value = tag.get(attr)
        if not isinstance(value, str):
            continue
        value = value.strip()
        if value.lower().startswith("javascript:"):
            del tag[attr]
            continue
        if not base_url or not value or value.startswith(("#", "//")) or _SCHEME_RE.match(value):
            continue
        tag[attr] = urljoin(base_url, value)


def _absolutize_media(
    soup: BeautifulSoup, base_url: str | None, manifest: list[dict] | None
) -> None:
    if not base_url:
        return
    by_filename: dict[str, str] = {}
    for img in manifest or []:
        url = img.get("url", "")
        if url and not url.startswith("data:"):
            by_filename.setdefault(url.rsplit("/", 1)[-1].split("?")[0], url)
    for img in soup.find_all(["img", "source", "video", "audio"]):
        for attr in ("src", "poster"):
            value = img.get(attr)
            if isinstance(value, str) and value:
                img[attr] = _abs_url(value, by_filename, base_url)
        srcset = img.get("srcset")
        if isinstance(srcset, str) and srcset:
            img["srcset"] = _abs_srcset(srcset, by_filename, base_url)


def _replace_arrows(soup: BeautifulSoup) -> None:
    from bs4 import BeautifulSoup, NavigableString
    from bs4.element import Comment

    for text in list(soup.find_all(string=lambda t: bool(t) and any(a in str(t) for a in _ARROWS))):
        if isinstance(text, Comment) or any(
            getattr(p, "name", None) in _NO_ARROW_PARENTS for p in text.parents
        ):
            continue
        pieces = re.split(f"([{''.join(_ARROWS)}])", str(text))
        new_nodes: list[PageElement] = []
        for piece in pieces:
            if piece in _ARROWS:
                svg = BeautifulSoup(
                    '<svg class="hrs-arrow" aria-hidden="true" focusable="false" '
                    'viewBox="0 0 16 16" width="16" height="16"><path d="'
                    f'{_ARROWS[piece]}" fill="none" stroke="currentColor" '
                    'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
                    "html.parser",
                ).svg
                if svg is not None:
                    new_nodes.append(svg)
            elif piece:
                new_nodes.append(NavigableString(piece))
        if new_nodes:
            text.replace_with(*new_nodes)


def _dedupe_ids(soup: BeautifulSoup, seen_ids: set[str]) -> None:
    renames: dict[str, str] = {}
    for tag in soup.find_all(id=True):
        old = str(tag.get("id"))
        if old not in seen_ids:
            seen_ids.add(old)
            continue
        n = 2
        new = f"{old}-{n}"
        while new in seen_ids:
            n += 1
            new = f"{old}-{n}"
        seen_ids.add(new)
        tag["id"] = new
        renames.setdefault(old, new)
    if not renames:
        return
    for tag in soup.find_all(True):
        for attr in _ID_REF_ATTRS:
            value = tag.get(attr)
            if isinstance(value, str):
                tag[attr] = " ".join(renames.get(tok, tok) for tok in value.split())
        href = tag.get("href")
        if isinstance(href, str) and href.startswith("#") and href[1:] in renames:
            tag["href"] = f"#{renames[href[1:]]}"


# At-rules that are not valid inside ``@scope`` stay at the top level of the sheet.
_UNSCOPABLE_AT_RULES = (
    "@font-face",
    "@keyframes",
    "@-webkit-keyframes",
    "@import",
    "@charset",
    "@property",
    "@counter-style",
    "@page",
)
_ROOTISH_PRELUDE_RE = re.compile(
    r"^\s*(?::root|html|body)(\s*,\s*(?::root|html|body))*\s*$", re.IGNORECASE
)


def _split_top_level_css(css: str) -> list[str]:
    """Split a stylesheet into top-level statements (rules / at-rules), brace-aware."""
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    n = len(css)
    while i < n:
        ch = css[i]
        if css.startswith("/*", i):
            end = css.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch in "\"'":
            end = css.find(ch, i + 1)
            i = n if end == -1 else end + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
            if depth == 0:
                parts.append(css[start : i + 1])
                start = i + 1
        elif ch == ";" and depth == 0:
            parts.append(css[start : i + 1])
            start = i + 1
        i += 1
    if css[start:].strip():
        parts.append(css[start:])
    return [p for p in parts if p.strip()]


def scope_css(css: str, scope_selector: str) -> str:
    """Wrap ``css`` in ``@scope (scope_selector)`` so it cannot style other sections.

    ``:root``/``html``/``body`` rules are rewritten to ``:scope`` (they would match
    nothing inside a scope), and at-rules that ``@scope`` cannot contain are kept at the
    top level. Idempotent: already-scoped CSS is returned unchanged.
    """
    if not css.strip() or f"@scope ({scope_selector})" in css:
        return css
    outside: list[str] = []
    inside: list[str] = []
    for stmt in _split_top_level_css(css):
        head = stmt.lstrip()
        if head.lower().startswith(_UNSCOPABLE_AT_RULES):
            outside.append(head)
            continue
        brace = stmt.find("{")
        if brace != -1 and _ROOTISH_PRELUDE_RE.match(stmt[:brace]):
            stmt = ":scope " + stmt[brace:]
        inside.append(stmt.strip())
    body = "\n".join(inside)
    scoped = f"@scope ({scope_selector}) {{\n{body}\n}}" if inside else ""
    return "\n".join([*outside, scoped]).strip()


def _honor_fallbacks(soup: BeautifulSoup) -> None:
    for style in soup.find_all("style"):
        css = str(style.string if style.string is not None else style.get_text())
        fixed = honor_contradicted_fallbacks(css)
        if fixed != css:
            style.string = fixed
    for tag in soup.find_all(style=True):
        value = tag.get("style")
        if isinstance(value, str):
            tag["style"] = honor_contradicted_fallbacks(value)


def _scope_styles(soup: BeautifulSoup, scope_selector: str) -> None:
    for style in soup.find_all("style"):
        css = style.string if style.string is not None else style.get_text()
        style.string = scope_css(str(css), scope_selector)


def _eager_images(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["img", "iframe"]):
        if str(tag.get("loading", "")).lower() == "lazy":
            del tag["loading"]


def finalize_fragment(
    html: str,
    *,
    base_url: str | None,
    manifest: list[dict] | None = None,
    seen_ids: set[str] | None = None,
    scope_selector: str | None = None,
) -> str:
    """Sanitize one generated section fragment (idempotent).

    Order matters: handlers are stripped from the RAW string first, because the parse
    below is exactly what mangles them. ``seen_ids`` is shared across the sections of
    one composed page so a repeated ``id`` is renamed (with its in-section references).
    """
    from bs4 import BeautifulSoup

    if not html:
        return html
    soup = BeautifulSoup(strip_inline_event_handlers(html), "html.parser")
    for script in soup.find_all("script"):
        # A static concept page runs no JS (the prompt forbids it). JSON-LD is data.
        if str(script.get("type", "")).lower() != "application/ld+json":
            script.decompose()
    _decode_cloudflare_emails(soup)
    _absolutize_links(soup, base_url)
    _absolutize_media(soup, base_url, manifest)
    _replace_arrows(soup)
    _eager_images(soup)
    _honor_fallbacks(soup)
    if seen_ids is not None:
        _dedupe_ids(soup, seen_ids)
    if scope_selector:
        _scope_styles(soup, scope_selector)
    return str(soup)
