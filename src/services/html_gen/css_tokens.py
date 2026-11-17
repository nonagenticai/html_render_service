"""Validation of the CSS a generated section/page relies on.

Two defects shipped silently in the cliffordtraining.nl redesign and are caught here:

* ``var(--step-6)`` / ``var(--paper-warm)`` / ``var(--color-border)`` … with no
  fallback and no definition anywhere — the declaration is invalid at computed-value
  time, so the hero headline fell back to the inherited body size and borders and text
  colours vanished, with nothing in the logs;
* a section that came back as markup with NO CSS at all (testimonials), rendering as
  unstyled default HTML in the middle of a designed page.

A THIRD defect shipped later, and this module was actively wrong about it. Since
``sanitize.scope_css`` wraps every section's ``<style>`` in ``@scope
([data-section="…"])``, a custom property declared in section A genuinely does NOT
reach section B — but this checker treated a property declared ANYWHERE on the page as
defined, and so reported "105 defined / 99 used / zero undefined" for a page whose
``.blog-section { background: var(--paper-warm); }`` resolved to nothing and rendered
transparent. The gate could not see the exact bleed that scoping introduced.

So the analysis is now SCOPE-AWARE (:func:`css_token_findings`):

* CSS outside any ``@scope``d section — the composed page's ``:root`` token block —
  defines for the whole page;
* CSS inside a scoped section defines only within that section. Resolution there is
  delegated to :func:`section_css.analyze_section_css`, which already resolves
  ``var()`` against the section's own DOM (a declaration on a selector that matches
  nothing, or on an element that is not an ancestor of the user, does not count);
* a fragment that carries no ``@scope`` at all — every section reply BEFORE
  ``finalize_fragment`` scopes it — is analysed exactly as before, so per-section
  generation-time checking is unchanged.

Fallbacks are no longer ignored either. ``var(--x, 1rem)`` where ``--x`` is undefined
still renders, so it is not fatal, but it is the same authoring mistake and it is the
shape a token rename leaves behind — it is reported at ``"warning"`` severity while a
no-fallback usage stays ``"error"``.

What a severity MEANS depends on where the check runs, and only one place can retry:

* per SECTION, during generation (:func:`css_defects`), an error is retryable — there is
  budget left and a redraw is cheap, so it blocks that draw;
* per composed PAGE (``generator.compose_section_page``), NOTHING blocks. Every LLM call
  is already spent and there is no retry; both tiers are recorded as degradations so the
  tenant gets a page with an honest, visible defect instead of nothing at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bs4 import BeautifulSoup, Tag

__all__ = [
    "TokenFinding",
    "css_defects",
    "css_token_findings",
    "declared_custom_properties",
    "describe_tokens",
    "has_content_markup",
    "honor_contradicted_fallbacks",
    "undefined_tokens",
]

# Any ``var(--name)`` usage; group 2 is ``,`` when a fallback follows, ``)`` when not —
# that group IS the severity split, so there is no separate no-fallback pattern here.
# (There was one; it survived as dead code after this module stopped using it, and an
# unmeasured leftover is the last thing this particular module should carry.)
_VAR_ANY_RE = re.compile(r"var\(\s*(--[A-Za-z0-9_-]+)\s*(,|\))")
# The marker that a section's CSS was isolated by ``sanitize.scope_css``.
_SCOPE_AT_RE = re.compile(r"@scope\s*\(", re.IGNORECASE)
_DECLARATION_RE = re.compile(r"(--[A-Za-z0-9_-]+)\s*:")
_STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>(.*?)</style\s*>", re.IGNORECASE | re.DOTALL)
_STYLE_ATTR_RE = re.compile(r"""\sstyle\s*=\s*("[^"]*"|'[^']*')""", re.IGNORECASE)
_ELEMENT_RE = re.compile(r"<(?!style\b|/|!)[A-Za-z][A-Za-z0-9-]*[\s/>]", re.IGNORECASE)


def _css_text(html: str) -> str:
    blocks = [m.group(1) for m in _STYLE_BLOCK_RE.finditer(html)]
    attrs = [m.group(1)[1:-1] for m in _STYLE_ATTR_RE.finditer(html)]
    return "\n".join(blocks + attrs)


def has_content_markup(html: str) -> bool:
    """Does ``html`` contain any element at all outside its ``<style>`` blocks?

    The one composition failure that is NOT a CSS defect: a page with nothing in it.
    """
    return _ELEMENT_RE.search(_STYLE_BLOCK_RE.sub("", html)) is not None


def declared_custom_properties(html: str) -> set[str]:
    """Every ``--x`` declared anywhere in the markup's CSS (blocks + style attributes)."""
    return set(_DECLARATION_RE.findall(_css_text(html)))


@dataclass(frozen=True)
class TokenFinding:
    """One custom property used where it is not defined.

    ``severity`` is ``"error"`` for ``var(--x)`` (no fallback — the declaration is
    invalid at computed-value time and the property falls back to its inherited or
    initial value) and ``"warning"`` for ``var(--x, fallback)`` (it renders, but the
    token the author named does not exist). ``scope`` is the ``data-section`` name of
    the ``@scope``d section the usage sits in, or ``None`` for page-level CSS.
    """

    token: str
    severity: str
    scope: str | None = None

    def describe(self) -> str:
        where = f' in section "{self.scope}"' if self.scope else ""
        return f"{self.token}{where}"


def _scoped_sections(soup: BeautifulSoup) -> list[tuple[str | None, Tag]]:
    """The ``[data-section]`` elements whose own CSS was isolated by ``@scope``.

    A section with no ``<style>``, or one whose style survived unscoped, is NOT
    isolated: its declarations really do reach the rest of the page, so it is left in
    the page-level bucket and judged exactly as before.
    """
    out: list[tuple[str | None, Tag]] = []
    for el in soup.select("[data-section]"):
        styles = el.find_all("style")
        if styles and any(_SCOPE_AT_RE.search(s.get_text()) for s in styles):
            name = el.get("data-section")
            out.append((str(name) if name is not None else None, el))
    return out


def _findings_for(css: str, defined: set[str], scope: str | None) -> list[TokenFinding]:
    """Undefined-token findings for one bucket of CSS, tiered by fallback presence."""
    return [
        TokenFinding(token, "error" if tail == ")" else "warning", scope)
        for token, tail in _VAR_ANY_RE.findall(css)
        if token not in defined
    ]


def css_token_findings(
    html: str, defined: set[str] | frozenset[str] | tuple[str, ...]
) -> list[TokenFinding]:
    """Every custom property used where it is not defined, scope-aware and tiered.

    ``defined`` is the page's token contract (``BRAND_TOKEN_NAMES``). Page-level CSS —
    everything outside an ``@scope``d section — additionally defines whatever it
    declares, for the whole page. Inside a scoped section only that section's own
    declarations apply, resolved against the section's DOM by
    :func:`section_css.analyze_section_css` so that a declaration on a dead selector
    does not count either.

    Deduplicated on (token, severity, scope); an ``"error"`` for a token suppresses its
    ``"warning"`` in the same scope, since the fatal usage is the one to report.
    """
    from bs4 import BeautifulSoup

    from .section_css import analyze_section_css

    soup = BeautifulSoup(html, "html.parser")
    sections = _scoped_sections(soup)
    for _name, el in sections:
        el.extract()

    page_css = _css_text(str(soup))
    page_tokens = set(defined) | set(_DECLARATION_RE.findall(page_css))
    findings = _findings_for(page_css, page_tokens, None)

    for name, el in sections:
        fragment = str(el)
        section_css_text = _css_text(fragment)
        report = analyze_section_css(fragment, page_tokens)
        errors = set(report.unresolved_vars)
        findings.extend(TokenFinding(t, "error", name) for t in sorted(errors))
        # Fallback usages: not resolvable per-element by ``analyze_section_css``, so the
        # laxer set test applies — the section's own declarations all count.
        section_defined = page_tokens | set(_DECLARATION_RE.findall(section_css_text))
        findings.extend(
            f
            for f in _findings_for(section_css_text, section_defined, name)
            if f.severity == "warning"
        )

    seen: set[tuple[str, str, str | None]] = set()
    errored = {(f.token, f.scope) for f in findings if f.severity == "error"}
    out: list[TokenFinding] = []
    for f in findings:
        if f.severity == "warning" and (f.token, f.scope) in errored:
            continue
        key = (f.token, f.severity, f.scope)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def undefined_tokens(html: str, defined: set[str] | frozenset[str] | tuple[str, ...]) -> set[str]:
    """The ``"error"``-tier token names of :func:`css_token_findings`, without scopes.

    Used where only the SET matters (assertions, set arithmetic). Anything a human or
    an LLM reads should go through :func:`describe_tokens` instead, which keeps the
    section name — dropping it is how the page-wide bleed stayed invisible.
    """
    return {f.token for f in css_token_findings(html, defined) if f.severity == "error"}


def css_defects(fragment: str, defined: set[str] | frozenset[str] | tuple[str, ...]) -> list[str]:
    """Human-readable reasons a generated section fragment must not be shipped."""
    defects: list[str] = []
    has_markup = has_content_markup(fragment)
    if has_markup and not _css_text(fragment).strip():
        defects.append(
            "the section has markup but NO CSS — include one <style> element with the "
            "section's hand-written, class-scoped CSS"
        )
    findings = css_token_findings(fragment, defined)
    missing = [f for f in findings if f.severity == "error"]
    if missing:
        defects.append(
            "these custom properties are used without a fallback but are not defined: "
            + describe_tokens(missing)
            + " — use only the page tokens listed, declare the property inside your own "
            "section CSS, or give var() a fallback"
        )
    return defects


def describe_tokens(findings: list[TokenFinding]) -> str:
    """``--paper-warm in section "blog", --x`` — the token AND where it was used.

    The section name is the whole point of the scope-aware analysis: a bare
    ``--paper-warm`` on a ten-band page tells a reader nothing about which band went
    transparent. Every message a human or an LLM reads goes through here.
    """
    return ", ".join(sorted({f.describe() for f in findings}))


# Page tokens whose lightness is FIXED by the token contract (brand/accent vary per site,
# so they are not judged). A section that writes ``var(--color-text, #ffffff)`` on a navy
# band was authored against a page where ``--color-text`` did not exist and rendered
# its white fallback; now that the token is defined as dark ink, the same CSS would
# render ink-on-navy. The fallback is the author's evident intent, so it wins.
_DARK_TOKENS = frozenset({"--color-ink", "--color-text", "--color-text-muted", "--color-muted"})
_LIGHT_TOKENS = frozenset({"--color-surface", "--color-card", "--color-surface-alt"})
_VAR_WITH_FALLBACK_RE = re.compile(
    r"var\(\s*(--[A-Za-z0-9_-]+)\s*,\s*(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}|white|black)\s*\)"
)


def _luminance(color: str) -> float:
    from src.services.accessibility_theme import _hex_to_rgb, _normalize_hex, _relative_luminance

    named = {"white": "#FFFFFF", "black": "#000000"}
    value = named.get(color.lower()) or _normalize_hex(color) or "#000000"
    return _relative_luminance(_hex_to_rgb(value))


def honor_contradicted_fallbacks(css: str) -> str:
    """Replace ``var(--fixed-token, fallback)`` by ``fallback`` when they contradict.

    Contradiction = a DARK token with a light fallback (luminance > 0.4) or a LIGHT
    token with a dark fallback (luminance < 0.2). Agreeing fallbacks are left alone.
    """

    def _sub(m: re.Match[str]) -> str:
        name, fallback = m.group(1), m.group(2)
        lum = _luminance(fallback)
        if (name in _DARK_TOKENS and lum > 0.4) or (name in _LIGHT_TOKENS and lum < 0.2):
            return fallback
        return m.group(0)

    return _VAR_WITH_FALLBACK_RE.sub(_sub, css)
