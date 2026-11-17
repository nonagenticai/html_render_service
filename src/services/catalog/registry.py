"""Component catalog — the vetted set of section blocks assembly mode composes from.

"Everything that can be computed, shall be computed": rather than asking the LLM to
hand-write CSS, assembly mode composes a page from this fixed catalog of open-source
section blocks (Tailwind + DaisyUI markup). The LLM's job narrows to **selecting**
blocks and **filling content slots**; layout, theming and asset wiring are
deterministic code (see ``assembler.py`` / ``theme.py``).

Each block is described by a :class:`BlockManifest` (its slots + metadata) and backed
by an autoescaping Jinja partial in ``templates/web_design/v2/blocks/<key>.html.j2``.
The manifest is the single source of truth for what the LLM may fill and what the
assembler will render — unknown slots are dropped, missing slots fall back to the
manifest default, and every value is HTML-escaped at render time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.services.catalog.icons import ICON_NAMES

SlotType = Literal["text", "list", "url", "image", "icon"]


@dataclass(frozen=True)
class SlotSpec:
    """One fillable content slot on a block."""

    type: SlotType = "text"
    required: bool = False
    description: str = ""
    default: Any = None
    # For ``list`` slots: the shape of each item (slot_name -> SlotType).
    item_slots: dict[str, SlotType] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockManifest:
    """Metadata + slot contract for a single catalog block."""

    key: str
    title: str
    category: Literal["nav", "hero", "content", "social", "conversion", "footer"]
    description: str
    slots: dict[str, SlotSpec]
    # Singleton blocks (nav/footer) may appear at most once and are placed
    # first/last regardless of plan ordering.
    singleton: bool = False
    # Uses a client-side runtime (Alpine/htmx/JS); only emitted for
    # interactivity="interactive". Static fallbacks use <details> etc.
    interactive: bool = False
    # Exposes a named mount point (id) that ``embeds`` can target.
    embed_mount: str | None = None

    def defaults(self) -> dict[str, Any]:
        """Return the default content dict for this block."""
        return {name: spec.default for name, spec in self.slots.items()}


# Shared slot guidance. The assembler enforces both rules in code
# (``integrity.py``); the wording just stops the model wasting tokens on output
# that will be rewritten or dropped.
_LINK_HINT = (
    "an in-page anchor (#features, #pricing, #faq, #testimonials, #get-started, "
    "#contact) or a mailto:/tel: from the brief — never an invented URL"
)
_IMAGE_HINT = "leave empty — imagery comes from the design system; never an external image URL"


# Hours and addresses are FACTS about a venue, not copy: an invented "Mon-Fri
# 9am-5pm" is a false statement a customer acts on. ``integrity.enforce_hours``
# drops any row whose time is not in the brief, so the wording only saves tokens.
_HOURS_HINT = (
    "ONLY the opening hours written in the brief, copied faithfully (day, time); "
    "if the brief states no hours, return an empty list — never invent times"
)
_ADDRESS_HINT = "only an address written in the brief; empty if the brief gives none"
# Testimonials are a FACT too (a real person said it). ``integrity.enforce_testimonials``
# keeps only quotes found verbatim in the brief and removes an empty block.
_TESTIMONIALS_HINT = (
    "ONLY customer quotes written in the brief, copied word for word with the name "
    "the brief gives; if the brief quotes no one, return an empty list — never invent "
    "a quote or a customer"
)


_ICON_HINT = (
    "one name from: " + ", ".join(ICON_NAMES) + " (rendered as an inline SVG; any other "
    "value falls back to a generic glyph)"
)


def _txt(desc: str, default: Any = None, required: bool = False) -> SlotSpec:
    return SlotSpec(type="text", description=desc, default=default, required=required)


# ---------------------------------------------------------------------------
# The catalog. Ordered by typical page position; the assembler re-sorts
# singletons (nav first, footer last) but otherwise preserves plan order.
# ---------------------------------------------------------------------------

_CATALOG: dict[str, BlockManifest] = {
    "navbar": BlockManifest(
        key="navbar",
        title="Navigation bar",
        category="nav",
        singleton=True,
        description="Top navigation with brand name, links and a primary CTA.",
        slots={
            "brand": _txt("Brand / site name", default="Brand"),
            "links": SlotSpec(
                type="list",
                description=f"Nav links; each href is {_LINK_HINT}",
                default=[],
                item_slots={"label": "text", "href": "url"},
            ),
            "cta_label": _txt("Primary CTA label", default="Get started"),
            "cta_href": SlotSpec(type="url", description=f"CTA link: {_LINK_HINT}", default="#"),
        },
    ),
    "hero": BlockManifest(
        key="hero",
        title="Hero",
        category="hero",
        embed_mount="hero-embed",
        description="Above-the-fold headline, subheading, CTAs and an optional media mount.",
        slots={
            "eyebrow": _txt("Small kicker above the headline", default=""),
            "headline": _txt("Main headline", default="A headline that sells", required=True),
            "subheading": _txt("Supporting sentence", default=""),
            "primary_label": _txt("Primary CTA label", default="Get started"),
            "primary_href": SlotSpec(
                type="url", description=f"Primary CTA link: {_LINK_HINT}", default="#"
            ),
            "secondary_label": _txt("Secondary CTA label", default=""),
            "secondary_href": SlotSpec(
                type="url", description=f"Secondary CTA link: {_LINK_HINT}", default="#"
            ),
            "image": SlotSpec(
                type="image",
                description=(
                    "Hero image: leave empty — the service generates it from the brief; "
                    "never an external image URL"
                ),
                default="",
            ),
        },
    ),
    "logos": BlockManifest(
        key="logos",
        title="Logo cloud",
        category="social",
        description="A strip of partner / customer logos or names.",
        slots={
            "title": _txt("Section lead-in", default="Trusted by teams everywhere"),
            "items": SlotSpec(
                type="list",
                description=f"Logos (name; image: {_IMAGE_HINT})",
                default=[],
                item_slots={"name": "text", "image": "image"},
            ),
        },
    ),
    "features": BlockManifest(
        key="features",
        title="Feature grid",
        category="content",
        description="A grid of feature cards, each with an icon, title and body.",
        slots={
            "headline": _txt("Section headline", default="Everything you need"),
            "subheading": _txt("Section subheading", default=""),
            "items": SlotSpec(
                type="list",
                description=f"Feature cards; icon: {_ICON_HINT}",
                default=[],
                item_slots={"icon": "icon", "title": "text", "body": "text"},
            ),
        },
    ),
    "stats": BlockManifest(
        key="stats",
        title="Stats band",
        category="content",
        description="A row of headline metrics (value + label).",
        slots={
            "items": SlotSpec(
                type="list",
                description="Stats",
                default=[],
                item_slots={"value": "text", "label": "text"},
            ),
        },
    ),
    "testimonials": BlockManifest(
        key="testimonials",
        title="Testimonials",
        category="social",
        description="Customer quotes with attribution.",
        slots={
            "headline": _txt("Section headline", default="Loved by our customers"),
            "items": SlotSpec(
                type="list",
                description=_TESTIMONIALS_HINT,
                default=[],
                item_slots={"quote": "text", "author": "text", "role": "text"},
            ),
        },
    ),
    "pricing": BlockManifest(
        key="pricing",
        title="Pricing",
        category="conversion",
        description="Pricing tiers with feature lists and CTAs.",
        slots={
            "headline": _txt("Section headline", default="Simple, transparent pricing"),
            "items": SlotSpec(
                type="list",
                description="Pricing tiers",
                default=[],
                item_slots={
                    "name": "text",
                    "price": "text",
                    "period": "text",
                    "cta_label": "text",
                    "cta_href": "url",
                },
            ),
        },
    ),
    "faq": BlockManifest(
        key="faq",
        title="FAQ",
        category="content",
        interactive=True,
        description="Expandable question/answer list (native <details>; no JS required).",
        slots={
            "headline": _txt("Section headline", default="Frequently asked questions"),
            "items": SlotSpec(
                type="list",
                description="Q&A pairs",
                default=[],
                item_slots={"question": "text", "answer": "text"},
            ),
        },
    ),
    "hours": BlockManifest(
        key="hours",
        title="Opening hours",
        category="content",
        description="Opening hours (day + time) and an optional visit address.",
        slots={
            "headline": _txt("Section headline", default="Opening hours"),
            "items": SlotSpec(
                type="list",
                description=f"Opening hours rows: {_HOURS_HINT}",
                default=[],
                item_slots={"day": "text", "time": "text"},
            ),
            "address": _txt(f"Visit address: {_ADDRESS_HINT}", default=""),
        },
    ),
    "cta": BlockManifest(
        key="cta",
        title="Call to action",
        category="conversion",
        description="A closing call-to-action band.",
        slots={
            "headline": _txt("CTA headline", default="Ready to get started?"),
            "subheading": _txt("Supporting line", default=""),
            "primary_label": _txt("CTA label", default="Get started"),
            "primary_href": SlotSpec(
                type="url", description=f"CTA link: {_LINK_HINT}", default="#"
            ),
        },
    ),
    "footer": BlockManifest(
        key="footer",
        title="Footer",
        category="footer",
        singleton=True,
        description="Site footer with brand, tagline and link columns.",
        slots={
            "brand": _txt("Brand / site name", default="Brand"),
            "tagline": _txt("Short tagline", default=""),
            "links": SlotSpec(
                type="list",
                description=f"Footer links; each href is {_LINK_HINT}",
                default=[],
                item_slots={"label": "text", "href": "url"},
            ),
            "copyright": _txt("Copyright line", default=""),
        },
    ),
}


def all_blocks() -> dict[str, BlockManifest]:
    """Return the full catalog (key -> manifest)."""
    return dict(_CATALOG)


def get_block(key: str) -> BlockManifest | None:
    """Return the manifest for ``key`` or None if unknown."""
    return _CATALOG.get(key)


def block_keys() -> list[str]:
    """Return the catalog keys in canonical order."""
    return list(_CATALOG.keys())


def catalog_brief() -> str:
    """Return a compact human/LLM description of the catalog for prompt building."""
    lines: list[str] = []
    for m in _CATALOG.values():
        slot_hint = ", ".join(sorted(m.slots)) or "(no slots)"
        flags = []
        if m.singleton:
            flags.append("singleton")
        if m.interactive:
            flags.append("interactive")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        lines.append(f"- {m.key}: {m.description} slots: {slot_hint}{suffix}")
    return "\n".join(lines)
