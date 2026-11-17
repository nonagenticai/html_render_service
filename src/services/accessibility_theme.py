"""Accessibility-first theme planning for redesign generation."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass

from src.services.context_resolver import ResolvedContext

_SEVERITY_WEIGHT = {
    "critical": 4,
    "high": 3,
    "major": 3,
    "medium": 2,
    "low": 1,
    "minor": 1,
    "info": 0,
}

_A11Y_HINTS = (
    "accessibility",
    "a11y",
    "wcag",
    "aria",
    "keyboard",
    "alt",
    "contrast",
    "readability",
    "legibility",
)

_COLOR_HINTS = ("contrast", "color", "colour", "palette", "readability", "legibility")


@dataclass
class AccessibilityThemePlan:
    """Resolved accessibility-first theme plan."""

    style: str
    style_description: str | None
    brand_info: dict | None
    context: ResolvedContext
    signals: dict
    applied: bool

    def to_dict(self) -> dict:
        """Serialize the theme plan for result metadata."""
        return {
            "style": self.style,
            "style_description": self.style_description,
            "signals": self.signals,
            "applied": self.applied,
            "brand_info": self.brand_info,
            "colors": {
                "primary": self.context.primary_color,
                "secondary": self.context.secondary_color,
            },
        }


def _normalize_hex(color: str | None) -> str | None:
    if not isinstance(color, str):
        return None
    value = color.strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return value.upper()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", value):
        return ("#" + value[1] * 2 + value[2] * 2 + value[3] * 2).upper()
    return None


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _mix_with_black(color: str, factor: float) -> str:
    r, g, b = _hex_to_rgb(color)
    f = min(1.0, max(0.0, factor))
    return _rgb_to_hex(
        (
            int(r * (1 - f)),
            int(g * (1 - f)),
            int(b * (1 - f)),
        )
    )


def _linearize(channel: int) -> float:
    c = channel / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(color: tuple[int, int, int]) -> float:
    r, g, b = color
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def _contrast_ratio(foreground: str, background: str) -> float:
    l1 = _relative_luminance(_hex_to_rgb(foreground))
    l2 = _relative_luminance(_hex_to_rgb(background))
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _ensure_contrast_with_white(bg_color: str, min_ratio: float = 4.5) -> str:
    if _contrast_ratio("#FFFFFF", bg_color) >= min_ratio:
        return bg_color
    for step in range(1, 16):
        candidate = _mix_with_black(bg_color, step * 0.06)
        if _contrast_ratio("#FFFFFF", candidate) >= min_ratio:
            return candidate
    return "#1F2937"


# WCAG 1.4.11 non-text contrast: an accent must be distinguishable from the brand
# colour it sits next to (buttons on a brand band, rules, highlighted words). The
# cliffordtraining.nl redesign shipped ``--color-brand:#0E1B2C`` with
# ``--color-accent:#0b1523`` (1.1:1) — dark buttons on dark, and an invisible accent
# word in the hero headline.
MIN_ACCENT_BRAND_CONTRAST = 3.0


def _saturation(color: str) -> float:
    r, g, b = (c / 255.0 for c in _hex_to_rgb(color))
    hi, lo = max(r, g, b), min(r, g, b)
    if hi == lo:
        return 0.0
    light = (hi + lo) / 2
    return (hi - lo) / (1 - abs(2 * light - 1))


def _mix(color: str, other: str, factor: float) -> str:
    (r1, g1, b1), (r2, g2, b2) = _hex_to_rgb(color), _hex_to_rgb(other)
    f = min(1.0, max(0.0, factor))
    return _rgb_to_hex(
        (round(r1 + (r2 - r1) * f), round(g1 + (g2 - g1) * f), round(b1 + (b2 - b1) * f))
    )


def ensure_distinct_accent(
    brand: str,
    accent: str,
    palette: list[dict] | None = None,
    *,
    min_ratio: float = MIN_ACCENT_BRAND_CONTRAST,
) -> str:
    """Return an accent with at least ``min_ratio`` contrast against ``brand``.

    A supplied accent that already clears the bar is kept. Otherwise the accent is
    taken from the SOURCE site's own palette (``[{"name", "value"}]`` from
    ``context_resolver.extract_palette``): chromatic candidates only (greys and the
    page's paper/ink tones are not accents), ``--*accent*``-named ones first, then the
    most saturated. Only when the site offers nothing usable is one derived, by moving
    the accent away from the brand's lightness until it clears the bar.
    """
    brand_hex = _normalize_hex(brand)
    accent_hex = _normalize_hex(accent)
    if brand_hex is None:
        return accent
    if accent_hex is not None and _contrast_ratio(brand_hex, accent_hex) >= min_ratio:
        return accent

    candidates: list[tuple[int, float, float, str]] = []
    for order, item in enumerate(palette or []):
        value = _normalize_hex(str(item.get("value", "")))
        if value is None or _contrast_ratio(brand_hex, value) < min_ratio:
            continue
        sat = _saturation(value)
        if sat < 0.25:
            continue
        named_accent = "accent" in str(item.get("name", "")).lower()
        candidates.append((0 if named_accent else 1, -sat, float(order), value))
    if candidates:
        return sorted(candidates)[0][3]

    base = accent_hex or brand_hex
    brand_is_dark = _relative_luminance(_hex_to_rgb(brand_hex)) < 0.18
    target = "#FFFFFF" if brand_is_dark else "#000000"
    for step in range(1, 21):
        candidate = _mix(base, target, step * 0.05)
        if _contrast_ratio(brand_hex, candidate) >= min_ratio:
            return candidate
    return target


def _derive_signals(findings: list[dict] | None) -> dict:
    total_a11y = 0
    color_pressure = 0
    max_severity_weight = 0

    for finding in findings or []:
        severity = str(finding.get("severity", "medium")).lower()
        weight = _SEVERITY_WEIGHT.get(severity, 2)
        text = " ".join(
            str(finding.get(k, "")) for k in ("category", "title", "description", "recommendation")
        ).lower()
        category = str(finding.get("category", "")).lower()

        is_a11y = category in {"accessibility", "a11y"} or any(h in text for h in _A11Y_HINTS)
        if not is_a11y:
            continue

        total_a11y += 1
        max_severity_weight = max(max_severity_weight, weight)
        if any(h in text for h in _COLOR_HINTS):
            color_pressure += weight

    return {
        "a11y_findings": total_a11y,
        "color_pressure": color_pressure,
        "max_severity_weight": max_severity_weight,
    }


def _select_style(current_style: str, signals: dict) -> str:
    style = current_style
    color_pressure = int(signals.get("color_pressure", 0))
    max_sev = int(signals.get("max_severity_weight", 0))

    if style == "bold" and color_pressure >= 3:
        return "modern"
    if style == "elegant" and color_pressure >= 5:
        return "modern"
    if style == "modern" and color_pressure >= 8 and max_sev >= 3:
        return "minimalist"
    return style


def build_accessibility_theme_plan(
    *,
    style: str,
    style_description: str | None,
    brand_info: dict | None,
    context: ResolvedContext,
    critique_findings: list[dict] | None,
) -> AccessibilityThemePlan:
    """Build an accessibility-first theme and color plan."""
    signals = _derive_signals(critique_findings)
    selected_style = _select_style(style, signals)

    updated_ctx = deepcopy(context)
    updated_brand = deepcopy(brand_info) if brand_info else {}

    primary = _normalize_hex(updated_ctx.primary_color) or _normalize_hex(
        updated_brand.get("primary_color")
    )
    secondary = _normalize_hex(updated_ctx.secondary_color) or _normalize_hex(
        updated_brand.get("secondary_color")
    )
    primary = primary or "#2563EB"
    secondary = secondary or "#1E40AF"

    # Ensure button/brand colors can support white foreground text for core actions.
    fixed_primary = _ensure_contrast_with_white(primary, min_ratio=4.5)
    fixed_secondary = _ensure_contrast_with_white(secondary, min_ratio=4.5)

    # Darkening both colours towards white-text contrast can collapse the accent onto
    # the brand (#0E1B2C / #0b1523). Re-establish a visible accent from the site's own
    # palette AFTER the white-contrast pass; the brand colour is never rewritten here.
    fixed_secondary = ensure_distinct_accent(
        fixed_primary, fixed_secondary, getattr(updated_ctx, "source_palette", None)
    )

    updated_ctx.primary_color = fixed_primary
    updated_ctx.secondary_color = fixed_secondary

    visual_preferences = (
        deepcopy(updated_ctx.visual_preferences)
        if isinstance(updated_ctx.visual_preferences, dict)
        else {}
    )
    visual_preferences["text_color"] = "#111827"
    visual_preferences["background_color"] = "#FFFFFF"
    visual_preferences["contrast_target"] = "WCAG-AA"
    updated_ctx.visual_preferences = visual_preferences

    updated_brand["primary_color"] = fixed_primary
    updated_brand["secondary_color"] = fixed_secondary
    updated_brand["accessibility_palette"] = {
        "primary": fixed_primary,
        "secondary": fixed_secondary,
        "text": "#111827",
        "background": "#FFFFFF",
    }

    suffix = (
        "Accessibility-first theme directives: enforce WCAG AA contrast; use "
        f"primary {fixed_primary}, secondary {fixed_secondary}, body text #111827 on #FFFFFF. "
        "Avoid low-contrast muted text for key information and CTAs."
    )
    next_style_description = (
        f"{style_description} {suffix}".strip() if style_description else suffix
    )

    colors_changed = fixed_primary != primary or fixed_secondary != secondary
    style_changed = selected_style != style
    applied = style_changed or colors_changed or bool(signals.get("a11y_findings"))

    return AccessibilityThemePlan(
        style=selected_style,
        style_description=next_style_description,
        brand_info=updated_brand or None,
        context=updated_ctx,
        signals=signals,
        applied=applied,
    )
