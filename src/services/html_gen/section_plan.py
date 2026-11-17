"""Greenfield section planning — the page outline for a design-from-brief run.

``detect_sections`` reads an outline out of an existing page's landmarks. A
greenfield job has no existing page (``html`` is empty), so that returns ``[]``
and ``generate_page_chunked`` used to fall back to one ``generate_full_page``
completion. That completion cannot fit: the local vLLM is a self-hosted provider
and worker_management clamps the output budget to a *time* budget (~12k tokens),
well under a full landing page, so every greenfield run died on
``does not end on a closed tag (truncated)``.

This module supplies the missing input: an ordered outline derived from the
**brief** (``ResolvedContext`` — description, site type, goals, feature
priorities) rather than from markup that does not exist. The dicts it returns are
shaped exactly like ``detect_sections`` output so ``generate_sections`` and
``compose_section_page`` consume them unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.context_resolver import ResolvedContext

__all__ = ["MAX_GREENFIELD_SECTIONS", "plan_greenfield_sections"]

# Sequential per-section LLM calls: the outline is capped so a greenfield page
# cannot spend an unbounded number of WM round-trips (and so the composed page
# stays a landing page rather than a sprawl).
MAX_GREENFIELD_SECTIONS = 9

# Outline per site/page type. Every plan opens with the nav/header band and closes
# with the footer; the middle is the type's conventional narrative.
_PAGE_PLANS: dict[str, tuple[str, ...]] = {
    "saas": ("nav", "hero", "features", "pricing", "testimonials", "faq", "cta", "footer"),
    "ecommerce": ("nav", "hero", "products", "features", "testimonials", "cta", "footer"),
    "portfolio": ("nav", "hero", "work", "about", "contact", "footer"),
    "blog": ("nav", "hero", "posts", "about", "cta", "footer"),
    "agency": ("nav", "hero", "services", "work", "testimonials", "cta", "footer"),
    "landing": ("nav", "hero", "features", "about", "testimonials", "cta", "footer"),
}
_DEFAULT_PLAN: tuple[str, ...] = ("nav", "hero", "features", "about", "cta", "contact", "footer")

# A stated goal that the base outline does not already serve adds its section.
# Keys match ``context_resolver.GOAL_DESIGN_ACTIONS``.
_GOAL_SECTIONS: dict[str, str] = {
    "build_trust": "testimonials",
    "generate_leads": "contact",
    "boost_sales": "pricing",
    "increase_conversions": "cta",
    "improve_engagement": "faq",
}

_SECTION_PURPOSE: dict[str, str] = {
    "nav": "Site header: brand wordmark/logo, primary navigation links, one primary action.",
    "hero": (
        "Above-the-fold hero: one <h1> stating the core value proposition, a supporting "
        "subhead, and the primary call-to-action. No image behind the headline text."
    ),
    "features": "Three to six concrete capabilities, each with a short heading and a sentence.",
    "services": "The services offered, each with a short heading and a sentence of detail.",
    "products": "A grid of representative products or offers with names and short descriptions.",
    "work": "A selection of representative work/case studies with names and one-line outcomes.",
    "posts": "A list of recent article cards: title, one-line summary, and a read link.",
    "pricing": "Two or three plans with names, what each includes, and a call-to-action each.",
    "testimonials": (
        "Social-proof band written as believable, GENERIC quotes — never a named real "
        "customer, exact rating or count."
    ),
    "faq": "Four to six anticipated questions with concise answers.",
    "about": "Who is behind this and why it exists — a short narrative, not a bullet dump.",
    "cta": "A closing conversion band: one strong statement plus the primary action.",
    "contact": "A way to get in touch: a short labelled form and/or contact details.",
    "footer": "Footer: secondary navigation groups, legal line, and the brand mark.",
}


def _base_plan(context: ResolvedContext | None, page_type: str | None) -> tuple[str, ...]:
    """Pick the outline for the brief's site/page type."""
    for key in (page_type, getattr(context, "site_type", None)):
        if key:
            plan = _PAGE_PLANS.get(str(key).strip().lower())
            if plan:
                return plan
    return _DEFAULT_PLAN


def _section_brief(name: str, context: ResolvedContext | None) -> str:
    """The per-section instruction, grounded in the brief where the brief speaks."""
    parts = [_SECTION_PURPOSE.get(name, f"The page's {name} section.")]
    if context is None:
        return parts[0]
    if name in ("features", "services") and context.feature_priorities:
        named = [
            f"{fp.get('feature', '?')} ({fp.get('priority', 'medium')} priority)"
            for fp in context.feature_priorities
        ]
        parts.append("Cover these, highest priority first: " + "; ".join(named) + ".")
    if name in ("hero", "cta") and context.goals:
        parts.append("This page's stated goals: " + ", ".join(context.goals) + ".")
    if name in ("nav", "footer", "hero") and context.brand_name:
        parts.append(f"The brand is {context.brand_name}.")
    return " ".join(parts)


def plan_greenfield_sections(
    *,
    context: ResolvedContext | None,
    page_type: str | None = None,
) -> list[dict]:
    """Return the greenfield page outline as ``detect_sections``-shaped dicts.

    The outline comes from the brief (``page_type`` / ``context.site_type`` choose
    the narrative; ``context.goals`` add the sections those goals need;
    ``context.feature_priorities`` and ``context.brand_name`` ground the per-section
    instructions). ``html``/``text_preview`` are empty by construction — there is no
    source page — which is precisely why ``build_section_prompt`` must run its
    greenfield branch for these.
    """
    names = list(_base_plan(context, page_type))
    for goal in getattr(context, "goals", None) or []:
        extra = _GOAL_SECTIONS.get(goal)
        if extra and extra not in names:
            # Before the footer, which always closes the page.
            names.insert(max(len(names) - 1, 0), extra)

    if len(names) > MAX_GREENFIELD_SECTIONS:
        # Keep the closing footer when trimming the middle.
        names = [*names[: MAX_GREENFIELD_SECTIONS - 1], names[-1]]

    return [
        {
            "section_type": name,
            "heading": None,
            "role": None,
            "aria_label": None,
            "tag": "section",
            "html": "",
            "text_preview": "",
            "greenfield_brief": _section_brief(name, context),
        }
        for name in names
    ]
