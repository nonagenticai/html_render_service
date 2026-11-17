"""Self-contained inline-SVG icons for catalog blocks.

The ``features`` block has an ``icon`` slot the LLM fills. Models fill it with
icon-font ligature names (``bolt``, ``shield``, ``rocket_launch`` — Material
Symbols vocabulary), but no icon font is ever loaded: the assembled page is
self-contained (inlined CSS, no external stylesheet), so a ligature name rendered
as text is just a visible word on the card (live web_design_v2 run 01a1588f-2686).

Icons are therefore resolved **in code** to a small vendored set of inline SVGs
(stroke-based, ``currentColor``). Resolution never emits text: a known name,
alias or emoji maps to its glyph; anything else maps to a neutral default glyph;
an empty value renders nothing. No font, no network, no visible word.
"""

from __future__ import annotations

import re

from markupsafe import Markup

# 24x24 viewBox, stroke paths. Hand-authored, no external licence attached.
_PATHS: dict[str, str] = {
    "bolt": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "lock": '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "star": (
        '<path d="m12 2 3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 '
        '2 9.27l6.91-1.01L12 2z"/>'
    ),
    "heart": (
        '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 '
        '7.78L12 21.23l8.84-8.84a5.5 5.5 0 0 0 0-7.78z"/>'
    ),
    "check": '<circle cx="12" cy="12" r="10"/><path d="m8 12 3 3 5-6"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "users": (
        '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
        '<path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>'
    ),
    "chart": '<path d="M3 3v18h18"/><path d="m7 15 4-4 3 3 5-6"/>',
    "cloud": '<path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/>',
    "rocket": (
        '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 '
        '0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 '
        '2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/>'
    ),
    "globe": (
        '<circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 '
        '15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>'
    ),
    "settings": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "mail": '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 6L2 7"/>',
    "leaf": (
        '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 '
        '10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>'
    ),
    "cart": (
        '<circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/>'
        '<path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/>'
    ),
    "code": '<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>',
    "truck": (
        '<path d="M1 3h15v13H1zM16 8h4l3 3v5h-7z"/><circle cx="5.5" cy="18.5" r="2.5"/>'
        '<circle cx="18.5" cy="18.5" r="2.5"/>'
    ),
    "award": '<circle cx="12" cy="8" r="7"/><path d="M8.21 13.89 7 23l5-3 5 3-1.21-9.12"/>',
    "home": '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
    "money": '<path d="M12 1v22M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "coffee": (
        '<path d="M18 8h1a4 4 0 0 1 0 8h-1M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8zM6 1v3'
        'M10 1v3M14 1v3"/>'
    ),
    "phone": '<rect x="5" y="2" width="14" height="20" rx="2"/><path d="M12 18h.01"/>',
    "target": (
        '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/>'
        '<circle cx="12" cy="12" r="2"/>'
    ),
    "idea": '<path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.74V16h8v-1.26A7 7 0 0 0 12 2z"/>',
    "sparkles": '<path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z"/>',
}

DEFAULT_ICON = "sparkles"

#: The names the fill prompt offers the model (canonical keys only).
ICON_NAMES: tuple[str, ...] = tuple(sorted(_PATHS))

# Ligature / icon-library names models actually emit → canonical key.
_ALIASES: dict[str, str] = {
    "flash": "bolt",
    "flash_on": "bolt",
    "electric_bolt": "bolt",
    "zap": "bolt",
    "lightning": "bolt",
    "fast": "bolt",
    "speed": "bolt",
    "bolt_outlined": "bolt",
    "security": "shield",
    "verified_user": "shield",
    "shield_check": "shield",
    "privacy_tip": "shield",
    "gpp_good": "shield",
    "safety": "shield",
    "lock_outline": "lock",
    "locked": "lock",
    "padlock": "lock",
    "key": "lock",
    "star_rate": "star",
    "grade": "star",
    "stars": "star",
    "kid_star": "star",
    "favorite": "heart",
    "love": "heart",
    "volunteer_activism": "heart",
    "check_circle": "check",
    "task_alt": "check",
    "done": "check",
    "verified": "check",
    "checkmark": "check",
    "done_all": "check",
    "schedule": "clock",
    "access_time": "clock",
    "timer": "clock",
    "history": "clock",
    "time": "clock",
    "alarm": "clock",
    "hourglass": "clock",
    "group": "users",
    "groups": "users",
    "people": "users",
    "person": "users",
    "team": "users",
    "community": "users",
    "handshake": "users",
    "user": "users",
    "analytics": "chart",
    "insights": "chart",
    "trending_up": "chart",
    "bar_chart": "chart",
    "show_chart": "chart",
    "monitoring": "chart",
    "query_stats": "chart",
    "growth": "chart",
    "dashboard": "chart",
    "graph": "chart",
    "cloud_done": "cloud",
    "cloud_upload": "cloud",
    "cloud_sync": "cloud",
    "storage": "cloud",
    "rocket_launch": "rocket",
    "launch": "rocket",
    "public": "globe",
    "language": "globe",
    "world": "globe",
    "travel_explore": "globe",
    "earth": "globe",
    "international": "globe",
    "tune": "settings",
    "build": "settings",
    "construction": "settings",
    "gear": "settings",
    "cog": "settings",
    "integration_instructions": "settings",
    "extension": "settings",
    "support_agent": "chat",
    "forum": "chat",
    "chat_bubble": "chat",
    "support": "chat",
    "headset_mic": "chat",
    "message": "chat",
    "comment": "chat",
    "help": "chat",
    "email": "mail",
    "envelope": "mail",
    "inbox": "mail",
    "send": "mail",
    "eco": "leaf",
    "spa": "leaf",
    "nature": "leaf",
    "park": "leaf",
    "local_florist": "leaf",
    "organic": "leaf",
    "sustainability": "leaf",
    "plant": "leaf",
    "grass": "leaf",
    "shopping_cart": "cart",
    "shopping_bag": "cart",
    "store": "cart",
    "storefront": "cart",
    "shop": "cart",
    "sell": "cart",
    "terminal": "code",
    "developer_mode": "code",
    "api": "code",
    "data_object": "code",
    "manage_search": "search",
    "find": "search",
    "explore": "search",
    "visibility": "search",
    "local_shipping": "truck",
    "delivery": "truck",
    "shipping": "truck",
    "delivery_dining": "truck",
    "workspace_premium": "award",
    "emoji_events": "award",
    "military_tech": "award",
    "trophy": "award",
    "medal": "award",
    "badge": "award",
    "quality": "award",
    "house": "home",
    "cottage": "home",
    "event": "calendar",
    "calendar_month": "calendar",
    "calendar_today": "calendar",
    "date_range": "calendar",
    "event_available": "calendar",
    "booking": "calendar",
    "payments": "money",
    "attach_money": "money",
    "savings": "money",
    "paid": "money",
    "euro": "money",
    "price": "money",
    "price_check": "money",
    "credit_card": "money",
    "wallet": "money",
    "dollar": "money",
    "local_cafe": "coffee",
    "restaurant": "coffee",
    "bakery_dining": "coffee",
    "cafe": "coffee",
    "breakfast_dining": "coffee",
    "lunch_dining": "coffee",
    "local_dining": "coffee",
    "ramen_dining": "coffee",
    "food": "coffee",
    "cake": "coffee",
    "cookie": "coffee",
    "smartphone": "phone",
    "phone_iphone": "phone",
    "devices": "phone",
    "mobile": "phone",
    "call": "phone",
    "phone_android": "phone",
    "track_changes": "target",
    "my_location": "target",
    "adjust": "target",
    "goal": "target",
    "lightbulb": "idea",
    "tips_and_updates": "idea",
    "emoji_objects": "idea",
    "bulb": "idea",
    "psychology": "idea",
    "auto_awesome": "sparkles",
    "magic": "sparkles",
    "auto_fix_high": "sparkles",
    "sparkle": "sparkles",
    "new_releases": "sparkles",
}

_EMOJI: dict[str, str] = {
    "\u26a1": "bolt",
    "\U0001f512": "lock",
    "\U0001f510": "lock",
    "\U0001f6e1": "shield",
    "\u2b50": "star",
    "\U0001f31f": "star",
    "\u2764": "heart",
    "\U0001f496": "heart",
    "\u2705": "check",
    "\u2714": "check",
    "\U0001f680": "rocket",
    "\U0001f4c8": "chart",
    "\U0001f4ca": "chart",
    "\U0001f30d": "globe",
    "\U0001f30e": "globe",
    "\U0001f30f": "globe",
    "\U0001f310": "globe",
    "\u23f1": "clock",
    "\u23f0": "clock",
    "\U0001f552": "clock",
    "\U0001f4ac": "chat",
    "\U0001f331": "leaf",
    "\U0001f343": "leaf",
    "\U0001f33f": "leaf",
    "\U0001f465": "users",
    "\U0001f91d": "users",
    "\U0001f4a1": "idea",
    "\U0001f3af": "target",
    "\U0001f69a": "truck",
    "\U0001f3c6": "award",
    "\U0001f4b0": "money",
    "\U0001f4b5": "money",
    "\u2615": "coffee",
    "\U0001f950": "coffee",
    "\U0001f35e": "coffee",
    "\U0001f6d2": "cart",
    "\U0001f4c5": "calendar",
    "\U0001f3e0": "home",
    "\U0001f4f1": "phone",
    "\U0001f50d": "search",
    "\u2601": "cloud",
    "\u2699": "settings",
    "\U0001f527": "settings",
    "\u2709": "mail",
    "\U0001f4e7": "mail",
    "\u2728": "sparkles",
    "\U0001f4bb": "code",
}

# Icon-library class prefixes a model may echo (``material-symbols-outlined bolt``,
# ``fa-shield``, ``mdi-rocket``) and style suffixes on ligature names.
_PREFIX_RE = re.compile(
    r"^(?:material[-_ ]?(?:symbols|icons)(?:[-_ ](?:outlined|rounded|sharp))?[\s:/]*|"
    r"(?:fa[srlb]?|fas|far|mdi|bi|icon|lucide|heroicon|ph)[-_:])"
)
_LOOKUP: dict[str, str] = {**{key: key for key in _PATHS}, **_ALIASES}
_SUFFIX_RE = re.compile(r"_(?:outline|outlined|rounded|sharp|filled|fill|alt|o|2)$")


def resolve_icon(value: object) -> str | None:
    """Map an LLM-filled icon value to a canonical icon key.

    ``None`` means "render no icon" (empty value). Every non-empty value resolves to
    a glyph — a word we do not recognise gets :data:`DEFAULT_ICON`, never itself.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    for ch in raw.replace("\ufe0f", ""):
        if ch in _EMOJI:
            return _EMOJI[ch]
    name = raw.casefold()
    name = _PREFIX_RE.sub("", name)
    name = re.sub(r"[\s\-]+", "_", name).strip("_")
    # Exact name, then without a style suffix, then the first known token of a
    # compound name (``rocket_launch_outlined``, ``fast_delivery``).
    for candidate in (name, _SUFFIX_RE.sub("", name), *name.split("_")):
        if candidate in _LOOKUP:
            return _LOOKUP[candidate]
    # Unknown word, emoji or symbol: a neutral glyph — never the word itself, and
    # never trust the render host's system fonts to carry an arbitrary emoji.
    return DEFAULT_ICON


def icon_svg(value: object, *, size: int = 32) -> Markup:
    """Render ``value`` as an inline, decorative SVG (or nothing for an empty value).

    Registered as the ``icon`` Jinja filter; the output is code-built markup from
    the fixed table above, so marking it safe cannot inject LLM content.
    """
    key = resolve_icon(value)
    if key is None:
        return Markup("")
    size = int(size)
    # S704: every interpolated value is code-owned — ``key`` is a canonical key of
    # the fixed ``_PATHS`` table and ``size`` is coerced to int; no LLM text reaches it.
    return Markup(  # noqa: S704  # nosec B704
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" '
        f'height="{size}" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
        f'focusable="false" data-icon="{key}">{_PATHS[key]}</svg>'
    )
