"""Brand palette / font resolution and the shared ``:root`` brand-token CSS."""

from __future__ import annotations

from html import escape
from urllib.parse import urlparse

from src.services.accessibility_theme import (
    _contrast_ratio,
    _mix,
    _normalize_hex,
    ensure_distinct_accent,
)
from src.services.context_resolver import ResolvedContext, extract_fonts
from src.services.font_pairing import ensure_distinct_body_font

__all__ = [
    "BRAND_TOKEN_NAMES",
    "SIZING_TOKEN_VALUES",
    "_DEFAULT_PRIMARY",
    "_DEFAULT_SECONDARY",
    "_brand_tokens_css",
    "_font_stack",
    "_format_font_directive",
    "_google_fonts_head",
    "_resolve_fonts",
    "_resolve_palette",
]

# Hardcoded brand defaults used ONLY as a last resort. When the resolved palette still
# equals these, we treat the palette as "unknown" and tell the model to derive colours
# from the original markup instead of printing this generic blue.
_DEFAULT_PRIMARY = "#2563EB"
_DEFAULT_SECONDARY = "#1E40AF"


def _resolve_palette(
    *,
    context: ResolvedContext | None,
    brand: dict | None,
    site_analysis: dict | None,
) -> tuple[str, str, str, bool]:
    """Resolve (audience, primary_color, secondary_color, palette_known).

    ``palette_known`` is True only when a real brand palette was supplied (via the
    resolved context, an explicit brand dict, or site_analysis.visual_identity). When
    False the colours are the generic defaults and prompts should instruct the model to
    derive the palette from the original markup rather than printing a blue hex.
    """
    audience = "general"
    primary_color = _DEFAULT_PRIMARY
    secondary_color = _DEFAULT_SECONDARY
    palette_known = False

    if context:
        audience = ", ".join(context.target_audience[:5]) if context.target_audience else "general"
        primary_color = context.primary_color
        secondary_color = context.secondary_color
        palette_known = primary_color != _DEFAULT_PRIMARY or secondary_color != _DEFAULT_SECONDARY
        return audience, primary_color, secondary_color, palette_known

    if site_analysis:
        audience = ", ".join(site_analysis.get("target_audience", [])[:5]) or "general"
        vis = site_analysis.get("visual_identity") or {}
        if vis.get("primary_color"):
            primary_color = vis["primary_color"]
            palette_known = True
        if vis.get("secondary_color"):
            secondary_color = vis["secondary_color"]
            palette_known = True
    if brand:
        if brand.get("primary_color"):
            primary_color = brand["primary_color"]
            palette_known = True
        if brand.get("secondary_color"):
            secondary_color = brand["secondary_color"]
            palette_known = True
    return audience, primary_color, secondary_color, palette_known


def _resolve_fonts(
    context: ResolvedContext | None, html: str
) -> tuple[str | None, str | None, list[str]]:
    """Resolve (display_font, body_font, google_links) from context then the markup.

    Display and body are resolved independently and can land on the SAME family (the
    cliffordtraining.nl artifact shipped ``--font-body: "Fraunces"``, the serif display
    face, with Inter loaded and unused). :func:`ensure_distinct_body_font` repairs that
    from ``google_links`` — the site's own evidence of the face it loaded and never
    used — the typographic counterpart of ``ensure_distinct_accent`` on the colour side.
    """
    if context and (context.display_font or context.body_font or context.google_font_links):
        display, body = context.display_font, context.body_font
        links = context.google_font_links or []
    else:
        fonts = extract_fonts(html)
        display, body = fonts.get("display"), fonts.get("body")
        links = fonts.get("google_links") or []
    return display, ensure_distinct_body_font(display, body, links), links


def _font_stack(family: str | None, *, kind: str) -> str:
    """A CSS font stack for a brand family, with a sane fallback by kind."""
    fallback = (
        "Georgia, 'Times New Roman', serif"
        if kind == "display"
        else "system-ui, -apple-system, sans-serif"
    )
    if family:
        return f'"{family}", {fallback}'
    return fallback


def _is_font_stylesheet(href: str) -> bool:
    """A Google-Fonts CSS API URL (``/css`` or ``/css2``) — never the bare origin."""
    parsed = urlparse(href)
    return parsed.hostname == "fonts.googleapis.com" and parsed.path.startswith("/css")


def _google_fonts_head(
    google_links: list[str] | None, display_font: str | None, body_font: str | None
) -> str:
    """`<link>` tags that load the brand fonts.

    Prefer the original page's Google-Fonts stylesheet links verbatim; otherwise
    synthesize a css2 request from the display/body family names. ``preconnect`` hints
    go out as ``rel="preconnect"`` to the ORIGINS only — the bare
    ``https://fonts.googleapis.com`` origin is never emitted as a stylesheet (a context
    resolved before that fix may still carry it, so it is filtered here too).
    Empty when no real fonts exist.
    """
    preconnect = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    )
    sheets = [h for h in (google_links or []) if _is_font_stylesheet(h)]
    if sheets:
        links = "\n".join(
            f'<link rel="stylesheet" href="{escape(h, quote=True)}">' for h in dict.fromkeys(sheets)
        )
        return f"{preconnect}\n{links}"
    families = [
        f.replace(" ", "+") + ":wght@400;500;600;700"
        for f in dict.fromkeys(x for x in (display_font, body_font) if x)
    ]
    if not families:
        return ""
    query = "&".join(f"family={f}" for f in families)
    return (
        f"{preconnect}\n"
        f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?{query}&display=swap">'
    )


# The COMPLETE custom-property contract a composed page defines. Section prompts list
# these names verbatim and ``css_tokens.undefined_tokens`` rejects any other
# ``var(--x)`` without a fallback: sections used ``--step-5``/``--step-6``/``--step--1``/
# ``--space-2xl``/``--color-text``/``--color-border``… that nobody defined, so the hero
# headline collapsed to the inherited size and borders/text colours silently vanished.
_TYPE_SCALE = {
    "--step--2": "clamp(.69rem, .66rem + .12vw, .78rem)",
    "--step--1": "clamp(.83rem, .79rem + .2vw, .94rem)",
    "--step-0": "clamp(1rem, .95rem + .3vw, 1.125rem)",
    "--step-1": "clamp(1.25rem, 1.1rem + .6vw, 1.5rem)",
    "--step-2": "clamp(1.6rem, 1.3rem + 1.2vw, 2.1rem)",
    "--step-3": "clamp(2.2rem, 1.6rem + 2.4vw, 3.3rem)",
    "--step-4": "clamp(3rem, 2rem + 4vw, 5rem)",
    "--step-5": "clamp(3.6rem, 2.4rem + 5.2vw, 6.25rem)",
    "--step-6": "clamp(4.2rem, 2.8rem + 6.4vw, 7.5rem)",
}
# Conventional magnitudes: models read ``--space-xl`` as ~4rem. The previous 8rem xl
# (and 4rem lg) was spent as nav gaps and paddings and pushed a desktop nav to 2090px.
_SPACE_SCALE = {
    "--space-3xs": ".125rem",
    "--space-2xs": ".25rem",
    "--space-xs": ".5rem",
    "--space-sm": "1rem",
    "--space-md": "1.5rem",
    "--space-lg": "2.5rem",
    "--space-xl": "4rem",
    "--space-2xl": "6rem",
    "--space-3xl": "8rem",
}
_SHAPE_TOKENS = {
    "--radius-sm": ".375rem",
    "--radius-md": ".75rem",
    "--radius-lg": "1.5rem",
    "--radius-full": "999px",
    "--shadow-sm": "0 1px 2px rgb(0 0 0 / .06), 0 1px 3px rgb(0 0 0 / .08)",
    "--shadow-md": "0 4px 12px rgb(0 0 0 / .08), 0 2px 4px rgb(0 0 0 / .06)",
    "--maxw": "72rem",
    "--measure": "65ch",
}
_COLOR_TOKEN_NAMES = (
    "--color-brand",
    "--color-on-brand",
    "--color-accent",
    "--color-on-accent",
    "--color-accent-text",
    "--color-ink",
    "--color-text",
    "--color-text-muted",
    "--color-muted",
    "--color-surface",
    "--color-surface-alt",
    "--color-card",
    "--color-border",
    "--color-line",
)
# Sizing tokens whose VALUES the section prompt states, so the model can size with them.
SIZING_TOKEN_VALUES: dict[str, str] = {**_TYPE_SCALE, **_SPACE_SCALE, **_SHAPE_TOKENS}
BRAND_TOKEN_NAMES: tuple[str, ...] = (
    *_COLOR_TOKEN_NAMES,
    "--font-display",
    "--font-body",
    *_TYPE_SCALE,
    *_SPACE_SCALE,
    *_SHAPE_TOKENS,
)

_INK = "#1A1A1A"
_SURFACE = "#FFFFFF"


def _on_color(background: str) -> str:
    """White or ink text — whichever reads better on ``background``."""
    bg = _normalize_hex(background)
    if bg is None:
        return _SURFACE
    return _SURFACE if _contrast_ratio(_SURFACE, bg) >= _contrast_ratio(_INK, bg) else _INK


def _text_safe(color: str, surface: str = _SURFACE, min_ratio: float = 4.5) -> str:
    """``color`` darkened just enough to be body-size text on ``surface`` (WCAG AA)."""
    c = _normalize_hex(color)
    if c is None:
        return _INK
    for step in range(21):
        candidate = _mix(c, "#000000", step * 0.05)
        if _contrast_ratio(candidate, surface) >= min_ratio:
            return candidate
    return _INK


# ``--color-accent-text`` is the eyebrow/link colour on ANY light surface a section
# picks, not only on white. Calibrated to exactly 4.5:1 on white it measured 4.44:1 on
# the recorded warm-paper band (#F5F1EA) — two sections below AA. 5.6:1 on white keeps
# >= 4.5:1 on every surface down to relative luminance 0.8 (4.5 x 1.05 / 0.85).
_ACCENT_TEXT_MIN_RATIO_ON_WHITE = 5.6

_SURFACE_NAME_HINTS = ("paper", "surface", "background", "bg", "cream", "sand")


def _surface_alt(brand: str, source_palette: list[dict] | None) -> str:
    """An alternate light surface: the site's own paper/background tone when it
    declares one that body ink still reads on, else a faint brand tint."""
    for item in source_palette or []:
        value = _normalize_hex(str(item.get("value", "")))
        name = str(item.get("name", "")).lower()
        if (
            value
            and any(h in name for h in _SURFACE_NAME_HINTS)
            and _contrast_ratio(_INK, value) >= 12.0
        ):
            return value
    return _mix(_normalize_hex(brand) or _INK, _SURFACE, 0.94)


def _brand_tokens_css(
    primary: str,
    secondary: str,
    *,
    display_font: str | None,
    body_font: str | None,
    palette_known: bool,
    source_palette: list[dict] | None = None,
    google_links: list[str] | None = None,
) -> str:
    """The `:root` brand-token contract every redesign emits (see ``BRAND_TOKEN_NAMES``).

    Single source of truth for colours, fonts, type/space/radius scale. When the
    palette is unknown we emit neutral ink tones (NEVER the #2563EB sentinel) so the
    composed page is still visible while the model derives real colours per section.
    The accent is guaranteed distinguishable from the brand colour
    (:func:`ensure_distinct_accent`), taken from the source site's palette first, and
    ``--font-body`` is guaranteed to be a different family from ``--font-display``
    (:func:`ensure_distinct_body_font`) — a body face equal to the display face is
    dropped to the generic body stack rather than shipped.
    """
    if palette_known:
        brand = primary
        accent = ensure_distinct_accent(primary, secondary, source_palette)
        brand_note = ""
    else:
        brand = "#1a1a1a"
        accent = ensure_distinct_accent(brand, "#444444", source_palette)
        brand_note = " /* unknown — derive the real brand colour from markup */"
    colors = {
        "--color-brand": f"{brand};{brand_note}",
        "--color-on-brand": f"{_on_color(brand)};",
        "--color-accent": f"{accent};",
        "--color-on-accent": f"{_on_color(accent)};",
        "--color-accent-text": f"{_text_safe(accent, min_ratio=_ACCENT_TEXT_MIN_RATIO_ON_WHITE)};",
        "--color-ink": "#1a1a1a;",
        "--color-text": "var(--color-ink);",
        "--color-text-muted": "#555555;",
        "--color-muted": "var(--color-text-muted);",
        "--color-surface": "#ffffff;",
        "--color-surface-alt": f"{_surface_alt(brand, source_palette)};",
        "--color-card": "var(--color-surface);",
        "--color-border": "rgb(0 0 0 / .12);",
        "--color-line": "var(--color-border);",
    }
    lines = [f"  {name}: {value}" for name, value in colors.items()]
    lines.append(f"  --font-display: {_font_stack(display_font, kind='display')};")
    # Display and body are resolved independently by several callers (this one is not
    # the only path — see generator.compose_section_page), so the distinctness bar is
    # enforced HERE, at the token itself, not only where the fonts are resolved.
    lines.append(
        f"  --font-body: "
        f"{_font_stack(ensure_distinct_body_font(display_font, body_font, google_links), kind='body')};"
    )
    lines.extend(f"  {name}: {value};" for name, value in _TYPE_SCALE.items())
    lines.extend(f"  {name}: {value};" for name, value in _SPACE_SCALE.items())
    lines.extend(f"  {name}: {value};" for name, value in _SHAPE_TOKENS.items())
    root = "\n".join(lines)
    return f""":root {{
{root}
}}
* {{ box-sizing: border-box; }}
/* The `hidden` attribute must win over a section's own `display`: the UA rule is a
   plain `[hidden] {{ display: none }}`, so `.cb-panel {{ display: flex }}` on a
   `<section hidden>` chat dialog painted it — a fixed white panel over 85vh of the page,
   covering the navy hero (2026-10-18; it also made the contrast repair recolour the
   hero's white text to #555). */
[hidden]:not([hidden="until-found"]) {{ display: none !important; }}
body {{ margin: 0; font-family: var(--font-body); color: var(--color-ink);
  background: var(--color-surface); line-height: 1.6; }}
h1, h2, h3, h4 {{ font-family: var(--font-display); line-height: 1.15; }}
img {{ max-width: 100%; height: auto; }}
dd {{ margin-inline-start: 0; }}
a {{ color: inherit; }}
nav a {{ white-space: nowrap; }}
.hrs-arrow {{ display: inline-block; width: 1em; height: 1em; vertical-align: -0.125em; }}
/* Print/PDF: sections FLOW across pages — keeping a section whole moved every one that
   did not fit the remaining space to the next page and left pages 40-70% empty. Breaks
   are avoided only inside small atomic pieces: a card, a quote, a figure, a button, a
   stat. (The PDF renderer also lifts the hint from any piece too tall for a page.) */
article, figure, blockquote, a, button, [class*="card"], [class*="stat"], [class*="btn"] {{
  break-inside: avoid; page-break-inside: avoid; }}
h1, h2, h3 {{ break-after: avoid; page-break-after: avoid; }}"""


def _format_font_directive(
    display_font: str | None, body_font: str | None, google_head: str
) -> str:
    """Instruction block telling the model to reload + apply the brand fonts."""
    if not (display_font or body_font or google_head):
        return (
            "## Fonts\nDerive the type pairing from the original markup and keep it — "
            "do NOT fall back to system-ui or a generic font."
        )
    parts = ["## Fonts — keep the brand's real type pairing"]
    if google_head:
        parts.append("Load them by placing these exact tags in `<head>`:")
        parts.append(google_head)
    if display_font:
        parts.append(f'Use "{display_font}" (via --font-display) for headings (h1-h4).')
    if body_font:
        parts.append(f'Use "{body_font}" (via --font-body) for body text.')
    parts.append("Never substitute system-ui or a generic sans/serif.")
    return "\n".join(parts)
