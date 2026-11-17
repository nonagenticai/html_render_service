"""Deterministic brand-token → DaisyUI theme mapping.

Computed theming: brand colours/fonts extracted upstream (analysis_service brand
extraction, or a project brand block) map **deterministically** onto DaisyUI v5's
theme CSS variables (``--color-primary`` etc). No LLM is involved — given the same
brand tokens this always produces the same theme, which is what makes assembled
designs reproducible and testable.

The emitted CSS is a single ``:root { … }`` rule injected AFTER the inlined DaisyUI
stylesheet, so equal-specificity source order lets the brand overrides win over the
named base theme.
"""

from __future__ import annotations

import re

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

# DaisyUI colour variables we drive from brand tokens. Each maps to a
# ``(brand_key, fallback)`` pair; ``*-content`` values are computed for contrast.
_DEFAULT_PRIMARY = "#2563eb"
_DEFAULT_SECONDARY = "#7c3aed"
_DEFAULT_ACCENT = "#0891b2"


def _norm_hex(value: str | None) -> str | None:
    """Return a normalised 6-digit ``#rrggbb`` hex, or None if not a valid hex."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    if not _HEX_RE.match(v):
        return None
    v = v.lower()
    if len(v) == 4:  # #rgb -> #rrggbb
        v = "#" + "".join(ch * 2 for ch in v[1:])
    return v


def _luminance(hex_color: str) -> float:
    """Relative luminance (0..1) of a #rrggbb colour (WCAG formula)."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


# ── WCAG AA contrast enforcement ─────────────────────────────────────────────
# axe-core's ``color-contrast`` rule (the one that intermittently red-lined the
# web-design-v2 E2E) requires 4.5:1 for normal text and 3:1 for large text
# (≥24px, or ≥18.66px bold). Every foreground the assembler derives from a brand
# palette is clamped to the NORMAL-text threshold here, at theme-construction
# time, so the emitted CSS is compliant BEFORE the page is rendered — no LLM
# output and no refine pass can reintroduce the violation.
#
# The brand's own surface colours (``--color-primary`` etc.) are NEVER rewritten:
# that would change the design the brand asked for. Only the *foreground* text
# colours computed against them are moved toward the accessible end.
_AA_NORMAL = 4.5

# DaisyUI ``corporate`` (``catalog.default_theme``) light surfaces, resolved from
# the vendored app.css: --color-base-100 oklch(100% 0 0) and --color-base-200
# oklch(93% 0 0). These are the two backgrounds brand-derived *text* lands on
# (``.text-brand`` in the hero eyebrow and the stats figures).
_BASE_100 = "#ffffff"
_BASE_200 = "#e8e8e8"

# The CTA subheading's design-intent muting. Historically this was a Tailwind
# ``opacity-90`` on the element, which axe composites into the effective
# foreground — dropping e.g. white-on-#1f6feb from 4.63:1 to 4.07:1 and failing
# the gate. We keep the muted LOOK but compute an opaque colour: as muted as the
# design asks, then walked back toward full strength until it measures AA.
_MUTED_ALPHA = 0.90

# Softened "black" — visually preferred over pure #000 for large filled areas.
# Escalated to true black only when it cannot reach AA on that background.
_SOFT_DARK = "#1a1a1a"


def _rgb(hex_color: str) -> tuple[int, int, int]:
    """Return the ``(r, g, b)`` 0-255 channels of a normalised ``#rrggbb``."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return r, g, b


def _hex(r: float, g: float, b: float) -> str:
    """Clamp and format float channels back to ``#rrggbb``."""
    return "#" + "".join(f"{max(0, min(255, round(c))):02x}" for c in (r, g, b))


def contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """Return the WCAG 2.x contrast ratio (1..21) between two ``#rrggbb`` colours."""
    lighter, darker = sorted((_luminance(fg_hex), _luminance(bg_hex)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _blend(fg_hex: str, bg_hex: str, alpha: float) -> str:
    """Composite ``fg_hex`` over ``bg_hex`` at ``alpha``, in sRGB — as a browser does."""
    fg, bg = _rgb(fg_hex), _rgb(bg_hex)
    return _hex(*(alpha * fg[i] + (1.0 - alpha) * bg[i] for i in range(3)))


def readable_on(hex_color: str, *, min_ratio: float = _AA_NORMAL) -> str:
    """Return a text colour that MEASURES ``min_ratio`` against ``hex_color``.

    The soft pair (``#1a1a1a`` / ``#ffffff``) is preferred for its look and is
    returned whenever it already meets the threshold. It is escalated to whichever
    pure end (``#000000`` / ``#ffffff``) measures highest otherwise — the previous
    implementation picked purely on a luminance threshold of 0.45 and so happily
    emitted white on a mid-luminance brand primary at ~2:1.

    NOTE: no foreground can reach 4.5:1 on a background whose relative luminance
    sits near 0.20 (the best either pure end achieves there is ~4.17:1). For those
    the maximum is returned and the brand surface is left alone by design; closing
    that residual window would mean rewriting the brand's own colour.
    """
    soft = _SOFT_DARK if _luminance(hex_color) > 0.45 else "#ffffff"
    if contrast_ratio(soft, hex_color) >= min_ratio:
        return soft
    return max(("#000000", "#ffffff"), key=lambda c: contrast_ratio(c, hex_color))


def muted_on(
    hex_color: str,
    content_hex: str,
    *,
    alpha: float = _MUTED_ALPHA,
    min_ratio: float = _AA_NORMAL,
) -> str:
    """Return an opaque, de-emphasised variant of ``content_hex`` over ``hex_color``.

    Starts at the design-intent ``alpha`` and walks back toward full strength
    until the composited colour measures ``min_ratio``, so secondary copy on a
    brand-coloured surface keeps its softer look wherever the palette allows and
    silently gives it up where it does not.
    """
    step = 0.02
    a = alpha
    while a < 1.0:
        blended = _blend(content_hex, hex_color, a)
        if contrast_ratio(blended, hex_color) >= min_ratio:
            return blended
        a += step
    return content_hex


def darken_for_text(
    hex_color: str,
    *,
    backgrounds: tuple[str, ...] = (_BASE_100, _BASE_200),
    min_ratio: float = _AA_NORMAL,
) -> str:
    """Darken ``hex_color``, preserving hue, until it reads on the light surfaces.

    The raw brand primary is great for buttons/backgrounds (where we compute a
    readable content colour) but used as small text it fails WCAG contrast. This
    returns a same-hue, darkened variant whose MEASURED contrast is ``min_ratio``
    or better against every background in ``backgrounds`` — the "brand ink" used
    for accent text (``.text-brand``) so a11y passes for ANY brand colour. The
    previous implementation aimed at a fixed luminance target instead of measuring,
    which left the guarantee implicit and tied to one base theme.
    """
    r, g, b = _rgb(hex_color)
    factor = 1.0
    for _ in range(40):
        candidate = _hex(r * factor, g * factor, b * factor)
        if all(contrast_ratio(candidate, bg) >= min_ratio for bg in backgrounds):
            return candidate
        factor *= 0.9
    return "#000000"


def _brand_color(brand: dict, *keys: str) -> str | None:
    """Return the first valid hex colour found under any of ``keys`` in ``brand``.

    Understands both the flat BrandInfoInput shape (``primary_color``) and the
    richer analysis_service palette (``color_palette.primary``).
    """
    for key in keys:
        if _norm_hex(brand.get(key)):
            return _norm_hex(brand[key])
    palette = brand.get("color_palette")
    if isinstance(palette, dict):
        for key in keys:
            short = key.replace("_color", "")
            if _norm_hex(palette.get(short)):
                return _norm_hex(palette[short])
    return None


def resolve_font(brand: dict | None) -> tuple[str | None, str | None]:
    """Return ``(font_family, google_font_name)`` from brand tokens, if any."""
    if not brand:
        return None, None
    family = brand.get("font_family") or brand.get("font_family_primary")
    google = brand.get("google_font")
    if not google and isinstance(brand.get("fonts"), list):
        for f in brand["fonts"]:
            if isinstance(f, dict) and f.get("is_google_font") and f.get("family"):
                google = f["family"]
                family = family or f["family"]
                break
    return (str(family) if family else None, str(google) if google else None)


def build_theme_css(brand: dict | None) -> str:
    """Return the ``:root { … }`` CSS rule mapping brand tokens to DaisyUI vars.

    Always returns a valid rule (defaults when brand is absent), so assembly is
    never blocked by missing brand data.
    """
    brand = brand or {}
    primary = _brand_color(brand, "primary_color", "primary") or _DEFAULT_PRIMARY
    secondary = _brand_color(brand, "secondary_color", "secondary") or _DEFAULT_SECONDARY
    accent = _brand_color(brand, "accent_color", "accent") or secondary or _DEFAULT_ACCENT

    decls = [
        f"--color-primary:{primary}",
        f"--color-primary-content:{readable_on(primary)}",
        f"--color-secondary:{secondary}",
        f"--color-secondary-content:{readable_on(secondary)}",
        f"--color-accent:{accent}",
        f"--color-accent-content:{readable_on(accent)}",
        # Readable-on-light brand colour for accent TEXT (see darken_for_text).
        f"--brand-ink:{darken_for_text(primary)}",
        # De-emphasised copy on a brand-primary surface (CTA subheading). Opaque
        # and AA-clamped — see muted_on; replaces a fixed `opacity-90` that axe
        # composited into a sub-4.5:1 effective foreground.
        f"--brand-on-primary-muted:{muted_on(primary, readable_on(primary))}",
    ]

    family, _google = resolve_font(brand)
    if family:
        # Applied via our own utility on <body> (see page shell), not a DaisyUI var.
        safe_family = family.replace("\n", " ").replace(";", " ").replace('"', "'")
        decls.append(f'--brand-font:"{safe_family}", ui-sans-serif, system-ui, sans-serif')

    return ":root{" + ";".join(decls) + "}"
