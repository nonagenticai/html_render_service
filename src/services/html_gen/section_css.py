"""Does a generated section's CSS actually apply to the section's markup?

The cliffordtraining.nl "Zes routes" section shipped with every root rule dead: its
CSS was written against ``.diensten`` while the markup it emitted was
``<section class="section" id="diensten">``. Nothing matched, so the band lost its
padding, background, text colour and body font — and the section-local tokens that
rule declared (``--card-hover``, ``--card-hover-line``, ``--accent-ink``) were never
defined for the cards that used them. :func:`css_tokens.undefined_tokens` passed
it, because it treats a property declared ANYWHERE as defined.

This module resolves both properties against the fragment's own DOM:

* **dead selectors** — a rule whose selector matches no element of the section;
* **unresolved custom properties** — ``var(--x)`` (no fallback) used by an element
  on which ``--x`` is not inherited from a LIVE declaration (a matching rule on the
  element or an ancestor, an inline ``style``) nor from the page's ``:root`` tokens.

Why section-local tokens stay allowed (not ":root tokens only"): in the recorded
run 97 of 134 custom properties were declared per section, almost all of them
component tokens derived from page tokens (``--cta-muted: color-mix(... var(--color-
on-brand) ...)``). That is a legitimate, idiomatic pattern; banning it would reject
nearly every section and turn one retryable defect into a retry storm. The defect was
never locality — it was a declaration on a selector that matched nothing, which this
check catches exactly.

Selector matching is deliberately lenient (errs towards "matches"): pseudo-classes
and pseudo-elements are stripped (``:hover``/``::before`` rules style an element that
exists), and a selector soupsieve cannot evaluate counts as matched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .sanitize import _split_top_level_css

if TYPE_CHECKING:
    from bs4 import BeautifulSoup, Tag

__all__ = ["CssRule", "SectionCssReport", "analyze_section_css", "parse_css_rules"]

# At-rules whose block contains nested style rules.
_GROUPING_AT_RULES = ("@media", "@supports", "@container", "@layer", "@scope", "@document")
_STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>(.*?)</style\s*>", re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_DECL_RE = re.compile(r"(--[A-Za-z0-9_-]+)\s*:")
_VAR_NO_FALLBACK_RE = re.compile(r"var\(\s*(--[A-Za-z0-9_-]+)\s*\)")
# ``::before`` / ``:hover`` / ``:nth-child(2n+1)`` / ``:not(.x)`` / ``:is(a, b)``.
_PSEUDO_RE = re.compile(r"::?[A-Za-z-]+(\((?:[^()]|\([^()]*\))*\))?")
# A single-class compound: ``.diensten`` (pseudo-classes already stripped).
_BLOCK_ROOT_RE = re.compile(r"^\.([A-Za-z0-9_-]+)$")
_ROOTISH = frozenset({":root", "html", "body", "*", ":scope", ""})
# One leading page/wrapper compound plus the combinator after it (see ``_matchable``).
_WRAPPER_PREFIX_RE = re.compile(
    r"""^(?:html|body|\*|:root|:scope|\[data-section(?:\s*[~|^$*]?=\s*(?:"[^"]*"|'[^']*'|[^\]\s]+))?\s*\])"""
    r"""(?:[.#\[][^\s>+~]*)?(?:\s*[>+~]\s*|\s+)""",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CssRule:
    """One style rule: its selector list and its declaration block text."""

    selectors: tuple[str, ...]
    body: str


@dataclass
class SectionCssReport:
    """What :func:`analyze_section_css` found."""

    dead_selectors: list[str] = field(default_factory=list)
    live_selectors: list[str] = field(default_factory=list)
    unresolved_vars: dict[str, list[str]] = field(default_factory=dict)
    rule_count: int = 0

    @property
    def blocking_dead_selectors(self) -> list[str]:
        """The dead selectors that break the section, not merely unused leftovers.

        Measured on the recorded run: 4 of 10 sections carry SOME dead selector, and 3 of
        those are harmless (a logo-text fallback for an image that loaded, an unused
        ``.cta-note``); the wave-1 recording has sections whose reply also carried another
        section's CSS (dead once #126 scoped each section, and invisible). Retrying on
        any dead selector would re-roll healthy sections and put the whole page at risk
        for nothing — a ">25% dead" ratio flagged 3 of those healthy recorded sections.
        What visibly broke "Zes routes" has two shapes, and only those block:

        * a dead BLOCK ROOT — ``.diensten`` matches nothing while its BEM children
          ``.diensten__inner``/``.diensten__card`` do: the band's own padding,
          background, colour and font are gone;
        * total mismatch — the section has CSS rules and NOT ONE selector matches.
        """
        if not self.dead_selectors:
            return []
        if not self.live_selectors:
            return list(self.dead_selectors)
        live_classes = {
            c for sel in self.live_selectors for c in re.findall(r"\.([A-Za-z0-9_-]+)", sel)
        }
        roots: list[str] = []
        for sel in self.dead_selectors:
            m = _BLOCK_ROOT_RE.match(_matchable(sel))
            if m and any(
                c.startswith((m.group(1) + "__", m.group(1) + "--")) for c in live_classes
            ):
                roots.append(sel)
        return roots

    def defects(self) -> list[str]:
        """Human-readable BLOCKING reasons, phrased for the retry prompt."""
        out: list[str] = []
        dead = self.blocking_dead_selectors
        if dead:
            shown = ", ".join(dead[:12])
            more = f" (+{len(dead) - 12} more)" if len(dead) > 12 else ""
            out.append(
                f"these CSS selectors match NO element in your markup, so their rules never "
                f"apply: {shown}{more} — make every selector target a class/element that "
                "exists in the markup you output (e.g. put the class on the section element)"
            )
        if self.unresolved_vars:
            parts = [
                f"{name} (used by {', '.join(sorted(set(users))[:3])})"
                for name, users in sorted(self.unresolved_vars.items())
            ]
            out.append(
                "these custom properties are used where they are NOT defined — they are "
                "declared only on a selector that does not match the element or any of its "
                "ancestors: "
                + "; ".join(parts)
                + " — declare them on an element that actually wraps their users, use a page "
                "token, or give var() a fallback"
            )
        return out


def _split_selector_list(prelude: str) -> tuple[str, ...]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in prelude:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur).strip())
    return tuple(p for p in parts if p)


def parse_css_rules(css: str) -> list[CssRule]:
    """Every style rule in ``css``, descending into grouping at-rules."""
    rules: list[CssRule] = []
    for stmt in _split_top_level_css(_COMMENT_RE.sub("", css)):
        brace = stmt.find("{")
        if brace == -1 or not stmt.rstrip().endswith("}"):
            continue
        prelude = stmt[:brace].strip()
        inner = stmt[brace + 1 : stmt.rstrip().rfind("}")]
        if prelude.startswith("@"):
            if prelude.lower().startswith(_GROUPING_AT_RULES):
                rules.extend(parse_css_rules(inner))
            continue
        rules.append(CssRule(selectors=_split_selector_list(prelude), body=inner))
    return rules


def _matchable(selector: str) -> str:
    """``selector`` without pseudo-classes/elements, dangling combinators and wrapper prefixes.

    A leading compound naming the PAGE (``html``, ``body``, ``:root``, ``*``) or the
    composed SECTION WRAPPER (``[data-section="hero"]``) exists in the composed page but
    never in the fragment, so ``html body .x`` / ``[data-section="hero"] .x`` are judged
    by ``.x`` — they are not dead roots.
    """
    s = _PSEUDO_RE.sub("", selector).strip()
    s = re.sub(r"^[>+~\s]+", "", s) + " "  # the space lets a bare wrapper compound strip
    while True:
        stripped = _WRAPPER_PREFIX_RE.sub("", s, count=1)
        if stripped == s:
            break
        s = stripped
    return re.sub(r"[>+~\s]+$", "", s)


def _select(soup: BeautifulSoup, selector: str) -> list[Tag] | None:
    """Elements matching ``selector``; ``None`` when it cannot be evaluated."""
    s = _matchable(selector)
    if s.lower() in _ROOTISH:
        return None
    try:
        return list(soup.select(s))
    except Exception:  # noqa: BLE001 — soupsieve raises several types for exotic selectors
        return None


def _without_state_classes(selector: str) -> str:
    """``.site-nav.mobile-open .links`` -> ``.site-nav .links`` (first class per compound)."""
    return re.sub(r"(\.[A-Za-z0-9_-]+)((?:\.[A-Za-z0-9_-]+)+)", r"\1", selector)


def _css_of(html: str) -> str:
    return "\n".join(m.group(1) for m in _STYLE_BLOCK_RE.finditer(html))


def _add_unique(bucket: list[str], value: str) -> None:
    if value not in bucket:
        bucket.append(value)


class _Resolver:
    """Which custom properties each element of the fragment inherits from live CSS."""

    def __init__(self, soup: BeautifulSoup, page_tokens: set[str]) -> None:
        self.soup = soup
        self.global_decls = set(page_tokens)
        # id(element) -> custom properties declared ON it (live rule / inline style).
        self.declared_on: dict[int, set[str]] = {}
        self.matched: list[tuple[CssRule, str, list[Tag]]] = []

    def match_rule(self, rule: CssRule, report: SectionCssReport) -> None:
        decls = set(_DECL_RE.findall(rule.body))
        for sel in rule.selectors:
            hits = _select(self.soup, sel)
            if hits is None:
                # Root-ish (``:root``/``:scope``) or unevaluable: applies section-wide.
                self.global_decls |= decls
                continue
            base = _without_state_classes(sel)
            if not hits and base != sel and _select(self.soup, base):
                # A JS-toggled state variant (``.is-open``/``.active``) of a live base:
                # unused in a static concept, not a mismatch. Not counted as dead.
                continue
            _add_unique(report.live_selectors if hits else report.dead_selectors, sel)
            self.matched.append((rule, sel, hits))
            for el in hits:
                self.declared_on.setdefault(id(el), set()).update(decls)

    def inherited(self, el: Tag) -> set[str]:
        names = set(self.declared_on.get(id(el), set()))
        for parent in el.parents:
            names |= self.declared_on.get(id(parent), set())
        return names

    def unresolved(self, used_text: str, el: Tag) -> set[str]:
        used = set(_VAR_NO_FALLBACK_RE.findall(used_text)) - self.global_decls
        return used - self.inherited(el) if used else set()


def analyze_section_css(
    fragment: str, page_tokens: set[str] | frozenset[str] | tuple[str, ...]
) -> SectionCssReport:
    """Resolve the fragment's CSS against its own markup (see module docstring)."""
    from bs4 import BeautifulSoup

    report = SectionCssReport()
    rules = parse_css_rules(_css_of(fragment))
    report.rule_count = len(rules)
    if not rules:
        return report
    soup = BeautifulSoup(fragment, "html.parser")
    for style in soup.find_all("style"):
        style.decompose()

    resolver = _Resolver(soup, set(page_tokens))
    for rule in rules:
        resolver.match_rule(rule, report)
    inline = list(soup.find_all(style=True))
    for el in inline:
        resolver.declared_on.setdefault(id(el), set()).update(
            _DECL_RE.findall(str(el.get("style")))
        )
    for rule, sel, hits in resolver.matched:
        for el in hits:
            for name in resolver.unresolved(rule.body, el):
                _add_unique(report.unresolved_vars.setdefault(name, []), sel)
    for el in inline:
        for name in resolver.unresolved(str(el.get("style")), el):
            _add_unique(report.unresolved_vars.setdefault(name, []), f"<{el.name} style>")
    return report
