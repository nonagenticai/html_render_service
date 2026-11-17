"""Prompt construction for the HTML generator.

Each builder tries a Jinja template first (when a ``PromptTemplateLoader`` is
supplied) and falls back to an inline string kept in lockstep with the template.
The builders are free functions taking ``template_loader`` explicitly; the
``HtmlGenerator`` facade exposes thin method wrappers that pass
``self._template_loader``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import get_settings
from src.services.component_extractor import COMPONENT_DESIGN_TIPS
from src.services.context_resolver import (
    GOAL_DESIGN_ACTIONS,
    SECTION_MARKUP_BUDGET,
    ResolvedContext,
    truncate_markup,
)

from .brand_css import (
    BRAND_TOKEN_NAMES,
    SIZING_TOKEN_VALUES,
    _format_font_directive,
    _google_fonts_head,
    _resolve_fonts,
    _resolve_palette,
)
from .html_utils import _truncate_html
from .images import _format_image_manifest

if TYPE_CHECKING:
    from src.services.prompt_template import PromptTemplateLoader

__all__ = [
    "REDESIGN_SYSTEM_PROMPT",
    "SECTION_MARKUP_PROMPT_BUDGET",
    "STYLE_DESCRIPTIONS",
    "_format_context_block",
    "_format_findings",
    "build_component_prompt",
    "build_full_page_prompt",
    "build_refinement_prompt",
    "build_section_prompt",
    "css_contract_block",
]

REDESIGN_SYSTEM_PROMPT = (
    "You are an editorial art director who REFINES an existing website's own visual "
    "language — you never re-theme it into a generic template. You read the site's real "
    "colours, type pairing, logo and imagery from the provided markup and keep them, "
    "improving only layout, hierarchy, spacing, rhythm and accessibility. "
    "You output a single self-contained HTML file that declares the brand's colours, "
    "fonts and scale as CSS custom properties in one `:root` block and styles everything "
    "with hand-written CSS against those tokens. "
    "HARD RULES: never use Tailwind or any other utility-class framework or CDN; never use "
    "`system-ui` (or a generic sans/serif) as the brand font when a real font is given; "
    "never substitute a generic blue/white palette; never invent content — no fake social "
    "proof, testimonials, ratings, counts ('Trusted by 10,000+ teams') or urgency the "
    "original page does not contain; preserve all real text, links and image URLs verbatim."
)

# Opinionated, brand-led direction text injected per requested style. Deliberately
# avoids SaaS-template language (cards, gradients, conversion) so the model styles
# against the brand's own character rather than a memorised utility look.
STYLE_DESCRIPTIONS: dict[str, str] = {
    "modern": (
        "Contemporary editorial layout: confident type scale, generous and intentional "
        "whitespace, a clear reading rhythm, and restrained motion. Let the brand's own "
        "colours and fonts carry the personality — no generic gradients or stock cards."
    ),
    "minimalist": (
        "Pared-back, typographic design: ample negative space, one disciplined accent "
        "from the brand palette, strong vertical rhythm. Every element earns its place."
    ),
    "bold": (
        "High-contrast, expressive layout: large display headings in the brand's own "
        "display font, decisive asymmetry, confident use of the brand's accent colour."
    ),
    "elegant": (
        "Refined, premium feel: the brand's serif/sans pairing used deliberately, muted "
        "tones from its real palette, luxurious spacing and subtle detail."
    ),
}

# Re-truncation guard applied to the per-section original markup before it goes into a
# prompt. Bounds the section path's token cost even if a caller passes a large slice
# (``site_analysis.content_sections`` is not produced by ``detect_sections``). Same
# budget as detection, and cut only at element boundaries.
SECTION_MARKUP_PROMPT_BUDGET = SECTION_MARKUP_BUDGET


# A reply is roughly 3.5 characters per token of HTML/CSS; the reasoning model also
# spends part of the SAME budget thinking, so a section is asked to use under half.
_CHARS_PER_TOKEN = 3.5
_REPLY_SHARE_OF_BUDGET = 0.45
MAX_SECTION_LOCAL_PROPERTIES = 8


def section_reply_char_target() -> int:
    """Characters a section reply should stay under to fit the provider's real ceiling."""
    ceiling = get_settings().wm.output_token_ceiling
    return int(ceiling * _REPLY_SHARE_OF_BUDGET * _CHARS_PER_TOKEN / 1000) * 1000


def section_part_note(section: dict) -> str:
    """Tell the model it is writing ONE part of a source section that was split."""
    parts = int(section.get("parts") or 1)
    if parts <= 1:
        return ""
    part = int(section.get("part") or 1)
    lead = (
        "It carries the section's heading and introduction."
        if part == 1
        else "It is composed DIRECTLY after the previous part: continue the same visual "
        "design (same background, card style and grid), and add NO heading or intro of "
        "its own."
    )
    return (
        f"NOTE: this source section was too large for one reply, so it was split into "
        f"{parts} parts at item boundaries. You are writing PART {part} of {parts}, "
        f"showing only the items below. {lead}"
    )


def css_contract_block(section_key: str | None = None) -> str:
    """The exact CSS/markup contract a section snippet must follow.

    Lists EVERY custom property the composed page defines, verbatim — the model used
    to be shown three examples and invented the rest (``--step-6``, ``--paper-warm``,
    ``--color-border``), none of which existed. ``css_tokens.css_defects`` enforces
    the same list after generation.
    """
    prefix = section_key or "section"
    tokens = ", ".join(
        f"`{t}` ({SIZING_TOKEN_VALUES[t]})" if t in SIZING_TOKEN_VALUES else f"`{t}`"
        for t in BRAND_TOKEN_NAMES
    )
    ceiling = get_settings().wm.output_token_ceiling
    return (
        "## Output budget (hard — a reply that runs past it is cut off and rejected)\n"
        f"- Your WHOLE reply, reasoning included, is cut off at about {ceiling} tokens. "
        f"Aim to finish well inside it: markup plus CSS under ~{section_reply_char_target()} "
        "characters, the `<style>` under a third of that.\n"
        "- Compact CSS: style with the page tokens DIRECTLY (`color: var(--color-accent)`). "
        f"Declare at most {MAX_SECTION_LOCAL_PROPERTIES} section-local custom properties, and "
        "never one that only renames a page token or another property "
        "(`--header-cta-bg: var(--color-accent)` is forbidden — write "
        "`var(--color-accent)` where it is used). No duplicate or near-duplicate rules, no "
        "vendor-prefixed copies, no commented-out CSS, no generated property names.\n\n"
        "## CSS contract (enforced — a reply that breaks it is rejected)\n"
        f"- The page defines EXACTLY these custom properties, and no others: {tokens}.\n"
        "- Any other `var(--x)` must either be declared inside your own section CSS or be "
        "given a fallback: `var(--x, value)`.\n"
        "- Put the section's CSS in ONE `<style>` element inside the `<section>`, scoped "
        "to a class on that section. A section with no CSS is rejected. Do NOT declare "
        "`:root`, `@font-face` or `@import`, and do NOT add `<link>` tags.\n"
        "- NO JavaScript: no `<script>`, no inline event-handler attributes (`onclick`, "
        "`onerror`, `onload`, …), no `javascript:` URLs. It is a static concept page.\n"
        f"- Every `id` must be unique on the page: prefix ids with `{prefix}-`.\n"
        "- Keep links as absolute URLs to the original site (relative paths break "
        "off-origin).\n"
        "- A statistic's number and its label share one alignment (both left or both "
        "centred), the label directly under or over its number.\n"
        "- Use `var(--color-on-brand)` / `var(--color-on-accent)` for text placed on "
        "`var(--color-brand)` / `var(--color-accent)` backgrounds.\n"
        "- Text contrast is MEASURED on the rendered page (WCAG AA: 4.5:1, 3:1 for large "
        "text). `--color-text-muted`, `--color-accent-text` and `--color-ink` are for LIGHT "
        "surfaces only; on a dark or brand/accent-coloured surface use "
        "`var(--color-on-brand)` / `var(--color-on-accent)`. Never set accent-coloured text "
        "on an accent-coloured background or gradient.\n"
        "- Every CSS selector must match an element in YOUR markup (put the section's own "
        "class on the `<section>`), and every section-local custom property must be declared "
        "on an element that wraps the elements using it."
    )


def _format_findings(findings: list[dict] | None) -> str:
    """Format critique findings into a concise summary for the prompt."""
    if not findings:
        return "No specific critique findings available."
    lines = []
    for f in findings[:40]:
        severity = f.get("severity", "medium")
        title = f.get("title", "")
        desc = f.get("description", "")
        lines.append(f"- [{severity}] {title}: {desc}")
        rec = f.get("recommendation")
        if rec:
            lines.append(f"  Recommendation: {rec}")
    return "\n".join(lines)


def _format_context_block(context: ResolvedContext | None) -> str:  # noqa: C901 — flat append chain
    """Format resolved context into a prompt section."""
    if context is None:
        return ""
    parts: list[str] = []
    # The free-text brief is the single most important grounding signal — for a
    # greenfield design it IS the subject (brand, sections, content). It was being
    # dropped here, so the model invented an unrelated business; lead with it.
    if context.description:
        parts.append(f"- What to build (follow this closely): {context.description}")
    if context.brand_name:
        parts.append(f"- Brand name: {context.brand_name}")
    if context.target_audience:
        parts.append(f"- Audience: {', '.join(context.target_audience[:5])}")
    if context.site_type:
        parts.append(f"- Site type: {context.site_type}")
    if context.goals:
        parts.append("- Goals:")
        for goal in context.goals:
            action = GOAL_DESIGN_ACTIONS.get(goal, goal)
            parts.append(f"  - {goal}: {action}")
    if context.feature_priorities:
        parts.append("- Feature priorities:")
        for fp in context.feature_priorities:
            parts.append(f"  - {fp.get('feature', '?')}: {fp.get('priority', 'medium')}")
    if context.visual_preferences:
        vp = context.visual_preferences
        if isinstance(vp, dict):
            prefs = [f"{k}={v}" for k, v in vp.items() if v]
            if prefs:
                parts.append(f"- Visual preferences: {', '.join(prefs)}")
    return "\n".join(parts)


def build_full_page_prompt(
    template_loader: PromptTemplateLoader | None,
    *,
    url: str,
    html: str,
    page_type: str | None,
    style: str,
    findings: list[dict] | None,
    brand: dict | None,
    site_analysis: dict | None,
    style_description: str | None = None,
    context: ResolvedContext | None = None,
    image_manifest: list[dict] | None = None,
    greenfield: bool = False,
) -> str:
    """Build the full-page prompt — refine-existing, or design-from-brief (greenfield)."""
    settings = get_settings()
    style_desc = style_description or STYLE_DESCRIPTIONS.get(style, STYLE_DESCRIPTIONS["modern"])
    findings_text = _format_findings(findings)
    html_excerpt = _truncate_html(html, settings.redesign.max_html_chars)

    audience, primary_color, secondary_color, palette_known = _resolve_palette(
        context=context, brand=brand, site_analysis=site_analysis
    )
    if context:
        tone = context.tone
    elif site_analysis:
        tone = site_analysis.get("tone", "professional") or "professional"
    else:
        tone = "professional"

    context_block = _format_context_block(context)
    display_font, body_font, google_links = _resolve_fonts(context, html)
    google_head = _google_fonts_head(google_links, display_font, body_font)
    font_directive = _format_font_directive(display_font, body_font, google_head)
    # Greenfield passes None, not the (necessarily empty) manifest: there is no
    # source page, so "no images could be extracted" is not a finding to report —
    # and the greenfield Requirements block DELIBERATELY instructs placehold.co
    # usage, which is correct when designing from a brief. Only a real page that
    # yielded nothing gets the empty-manifest warning. See _format_image_manifest.
    image_block = _format_image_manifest(None if greenfield else image_manifest)

    # Greenfield: no source page to refine — design from the brief. Uses a distinct
    # template (no "Original Page"/preserve-content sections) + greenfield system prompt.
    if greenfield:
        rendered = (
            template_loader.render(
                "greenfield.md.j2",
                page_type=page_type,
                audience=audience,
                primary_color=primary_color,
                secondary_color=secondary_color,
                palette_known=palette_known,
                tone=tone,
                context_block=context_block,
                style=style,
                style_desc=style_desc,
                font_directive=font_directive,
                display_font=display_font,
                body_font=body_font,
            )
            if template_loader
            else None
        )
        if rendered:
            return rendered
        # Inline fallback (kept in lockstep with greenfield.md.j2).
        palette_line = (
            f"- Palette: primary={primary_color}, secondary={secondary_color} "
            "(use as --color-brand / --color-accent)."
            if palette_known
            else "- Palette: choose an on-brief palette that fits the tone — NOT a "
            "generic blue/white SaaS default."
        )
        ctx_section = f"\n## What to build\n{context_block}\n" if context_block else ""
        return (
            f"Design a brand-new {page_type or 'landing'} page FROM SCRATCH that matches "
            "the brief below. Invent coherent, on-brief content and a production-quality "
            "layout. Output a SINGLE self-contained HTML file with one `:root` token block "
            "+ hand-written CSS. No Tailwind, no CDN.\n\n"
            f"## Brief\n- Audience: {audience}\n- Tone: {tone}\n{palette_line}\n"
            f"{font_directive}\n{ctx_section}\n"
            f"## Design Direction: {style}\n{style_desc}\n\n"
            "## Requirements\n"
            "1. Author real, on-brief copy for every section (hero, value props, features, "
            "CTA, footer).\n"
            "2. Use placeholder images (https://placehold.co/...) tinted to the palette, with "
            "descriptive alt, ONLY as discrete inline content (cards/gallery) — never behind "
            "live text.\n"
            "3. Hero & section backgrounds: a solid brand colour or subtle palette gradient, "
            "NEVER a text-baked placeholder image behind a headline/CTA.\n"
            "4. Strong hierarchy, whitespace, responsiveness, WCAG AA contrast.\n"
            '4. Include <!DOCTYPE html>, <html lang="en">, and the viewport meta.\n\n'
            "Output ONLY the HTML code in a ```html code block."
        )

    # Try Jinja2 template first
    if template_loader:
        rendered = template_loader.render(
            "full_page.md.j2",
            url=url,
            page_type=page_type,
            html_excerpt=html_excerpt,
            audience=audience,
            primary_color=primary_color,
            secondary_color=secondary_color,
            palette_known=palette_known,
            tone=tone,
            context_block=context_block,
            findings_text=findings_text,
            style=style,
            style_desc=style_desc,
            font_directive=font_directive,
            image_manifest_block=image_block,
            display_font=display_font,
            body_font=body_font,
        )
        if rendered:
            return rendered

    # Inline fallback (kept in lockstep with full_page.md.j2)
    palette_line = (
        f"- Brand colours: primary={primary_color}, secondary={secondary_color} "
        "— declare these as --color-brand / --color-accent."
        if palette_known
        else "- Brand colours: derive them from the Current HTML above and declare them "
        "as --color-brand / --color-accent — do NOT introduce a new palette or any blue."
    )
    prompt = f"""Refine the following web page. This is a REFINEMENT of an existing brand,
not a re-theme. Read the real colours, type pairing, logo and imagery from the Current
HTML below and KEEP them — the result must be instantly recognisable as the same brand.

Output a SINGLE self-contained HTML file. Declare the brand's colours, fonts and scale as
CSS custom properties in one `:root` block in a `<style>` tag, then style everything with
hand-written CSS against those tokens (e.g. `var(--color-brand)`, `var(--font-display)`,
`var(--step-3)`, `var(--space-lg)`). Do NOT use Tailwind or any utility-class framework or
CDN. Do NOT use system-ui as the brand font. Do NOT invent social proof, testimonials,
ratings, counts or urgency the page does not contain.

## Original Page
- URL: {url}
- Page type: {page_type or "unknown"}
- Current HTML:
{html_excerpt}

## Site Context
- Audience: {audience}
{palette_line}
- Tone: {tone}

{font_directive}

{image_block}
"""
    if context_block:
        prompt += f"""
## User Context
{context_block}
"""
    prompt += f"""
## Issues to Address
{findings_text}

## Design Direction: {style}
{style_desc}

## Requirements
1. Self-contained HTML: one `:root` token block + hand-written CSS. No Tailwind / no CDN.
2. Preserve the site's actual content and links (don't invent text or claims)
3. Preserve brand identity (colours, the real fonts, logo); improve execution only
4. Reuse the exact absolute image URLs listed above; placeholder only for a genuinely
   missing image
5. Improve: visual hierarchy, whitespace, typographic rhythm, responsiveness, accessibility
6. Address the critique findings above
7. Include <meta name="viewport"> for responsiveness
8. Wrap each visual band (header, hero, every content block, footer) in its own
   full-width `<section data-section="<slug>">` (slug from its heading); put the
   max-width container INSIDE it, never around several bands

Output ONLY the HTML code in a ```html code block."""
    return prompt


def _build_greenfield_section_prompt(
    template_loader: PromptTemplateLoader | None,
    *,
    section: dict,
    section_type: str,
    page_type: str | None,
    style: str,
    style_desc: str,
    context: ResolvedContext | None,
    audience: str,
    primary_color: str,
    secondary_color: str,
    palette_known: bool,
    css_contract: str = "",
) -> str:
    """Design-from-brief prompt for ONE section of a brand-new page.

    Mirrors ``greenfield.md.j2``'s framing (author real copy; placehold.co only as
    discrete inline content) but scoped to a single ``<section>`` snippet that a
    shared token block will wrap — so it never asks for ``<!DOCTYPE html>``.
    """
    section_brief = section.get("greenfield_brief") or f"The page's {section_type} section."
    context_block = _format_context_block(context)
    tone = context.tone if context else "professional"

    if template_loader:
        rendered = template_loader.render(
            "section_greenfield.md.j2",
            section_type=section_type,
            section_brief=section_brief,
            page_type=page_type,
            audience=audience,
            tone=tone,
            primary_color=primary_color,
            secondary_color=secondary_color,
            palette_known=palette_known,
            context_block=context_block,
            style=style,
            style_desc=style_desc,
            css_contract_block=css_contract,
        )
        if rendered:
            return rendered

    # Inline fallback (kept in lockstep with section_greenfield.md.j2).
    palette_line = (
        f"- Brand colours: primary={primary_color}, secondary={secondary_color} "
        "(use as var(--color-brand) / var(--color-accent))."
        if palette_known
        else "- Brand colours: use var(--color-brand) / var(--color-accent) — the page "
        "picks the palette; do NOT introduce a new one or any generic blue."
    )
    heading_line = (
        "- This is the ONLY section that may contain the page's single `<h1>`."
        if section_type == "hero"
        else "- Start from an `<h2>` — the page's `<h1>` belongs to the hero, not here."
    )
    ctx_section = f"\n## What to build\n{context_block}\n" if context_block else ""
    return (
        f'Design the "{section_type}" section of a BRAND-NEW {page_type or "landing"} page, '
        "from the brief below. There is no existing site — invent coherent, on-brief copy "
        "and a production-quality layout for THIS SECTION ONLY.\n\n"
        "Output a single HTML `<section>` SNIPPET styled with hand-written CSS that "
        "references the page's shared brand tokens (`var(--color-brand)`, "
        "`var(--font-display)`, `var(--step-2)`, `var(--space-md)`, …). Do NOT use Tailwind. "
        "Do NOT declare your own `:root`, `<style>` font-faces or `<link>` fonts, and do NOT "
        "emit `<!DOCTYPE>`/`<html>`/`<head>`/`<body>` — the page supplies them.\n\n"
        f"## This section\n- Section: {section_type}\n"
        f"- What it must do: {section_brief}\n{heading_line}\n\n"
        f"## Brief\n- Audience: {audience}\n- Tone: {tone}\n{palette_line}\n{ctx_section}\n"
        f"## Design Direction: {style}\n{style_desc}\n\n"
        f"{css_contract}\n\n"
        "## Requirements\n"
        "1. One `<section>` element using brand-token CSS (no Tailwind, no utility classes)\n"
        "2. Author real, on-brief copy — no Lorem ipsum, no TODO, no fabricated named "
        "customers, exact ratings or counts\n"
        "3. Reference the shared tokens for colours/fonts/spacing — do not redefine them\n"
        "4. Placeholder images (https://placehold.co/...) only as discrete inline content "
        "with descriptive alt; never behind a headline or CTA\n"
        "5. Mobile-first responsive; semantic HTML; WCAG AA contrast; visible focus styles\n\n"
        "Output ONLY the HTML in a ```html code block."
    )


def build_section_prompt(
    template_loader: PromptTemplateLoader | None,
    *,
    url: str,
    html: str,
    section: dict,
    page_type: str | None,
    style: str,
    findings: list[dict] | None,
    brand: dict | None,
    site_analysis: dict | None,
    style_description: str | None = None,
    context: ResolvedContext | None = None,
    image_manifest: list[dict] | None = None,
    greenfield: bool = False,
) -> str:
    """Build a section-level prompt: refine-an-existing section, or design one from the brief.

    The section is a SNIPPET composed into a page that already declares the shared
    ``:root`` brand tokens + fonts — so the snippet must reference ``var(--*)`` and
    NOT declare its own ``:root``/`<style>`/font links.

    ``greenfield=True`` swaps the whole prompt: there is no original markup to preserve
    (``section["html"]`` is empty by construction — see
    :func:`~src.services.html_gen.section_plan.plan_greenfield_sections`), so the model
    is told to AUTHOR the section from the brief instead of refining one. Reusing the
    refinement prompt here would order the model to "keep the section's real content"
    while showing it none.
    """
    style_desc = style_description or STYLE_DESCRIPTIONS.get(style, STYLE_DESCRIPTIONS["modern"])
    findings_text = _format_findings(findings)
    section_type = section.get("section_type", "unknown")
    heading = section.get("heading", "")
    text_preview = section.get("text_preview", "")

    # The original section markup (real colours / fonts / image src URLs) so the
    # model can refine the EXISTING brand rather than invent a generic theme.
    # Fall back to a whole-page excerpt when no per-section slice exists.
    section_markup, cut_here = truncate_markup(
        section.get("html") or html or "", SECTION_MARKUP_PROMPT_BUDGET
    )
    truncated = bool(section.get("truncated")) or cut_here
    truncation_note = (
        "NOTE: this markup was TRUNCATED at an element boundary to fit the prompt — "
        "trailing content is missing. Lay out every item shown COMPLETELY; do not invent "
        "the missing items and do not end on a half-finished card or sentence."
        if truncated
        else ""
    )
    omitted = int(section.get("omitted_items") or 0)
    omitted_note = (
        f"NOTE: {omitted} further item(s) of this source section did not fit the generation "
        "budget and are NOT included. Do not invent them; after the items shown, add one "
        f'short line that points to the rest (e.g. "and {omitted} more").'
        if omitted
        else ""
    )
    truncation_note = "\n".join(
        n for n in (section_part_note(section), truncation_note, omitted_note) if n
    )
    css_contract = css_contract_block(section.get("key") or section_type)

    audience, primary_color, secondary_color, palette_known = _resolve_palette(
        context=context, brand=brand, site_analysis=site_analysis
    )
    image_block = _format_image_manifest(image_manifest)

    if greenfield:
        return _build_greenfield_section_prompt(
            template_loader,
            section=section,
            section_type=section_type,
            page_type=page_type,
            style=style,
            style_desc=style_desc,
            context=context,
            audience=audience,
            primary_color=primary_color,
            secondary_color=secondary_color,
            palette_known=palette_known,
            css_contract=css_contract,
        )

    if template_loader:
        rendered = template_loader.render(
            "section.md.j2",
            url=url,
            page_type=page_type,
            section_type=section_type,
            heading=heading,
            text_preview=text_preview,
            section_markup=section_markup,
            audience=audience,
            primary_color=primary_color,
            secondary_color=secondary_color,
            palette_known=palette_known,
            findings_text=findings_text,
            style=style,
            style_desc=style_desc,
            image_manifest_block=image_block,
            truncation_note=truncation_note,
            css_contract_block=css_contract,
        )
        if rendered:
            return rendered

    palette_line = (
        f"- Brand colours: primary={primary_color}, secondary={secondary_color} "
        "(available as var(--color-brand) / var(--color-accent))."
        if palette_known
        else "- Brand colours: derive the real colours from the markup below (use "
        "var(--color-brand) / var(--color-accent)) — do NOT introduce a new palette or any blue."
    )
    return f"""Refine the "{section_type}" section of this web page.
This is a REFINEMENT of an existing brand, not a re-theme. Keep the section's real
content, colours, type and imagery — it must read as the same brand.

Output a single HTML `<section>` SNIPPET styled with hand-written CSS (in one `<style>`
inside the section) that references the page's shared brand tokens (`var(--color-brand)`,
`var(--font-display)`, `var(--step-2)`, `var(--space-md)`, etc.). Do NOT use Tailwind or any
utility framework. Do NOT declare your own `:root`, `@font-face` or `<link>` fonts — the page
supplies them. Do NOT invent content (no fake testimonials, ratings, counts or urgency).

## Original section markup
{truncation_note}
```html
{section_markup}
```

## Context
- Page: {url} ({page_type or "unknown"})
- Section: {section_type}
- Current heading: {heading}
- Current content preview: {text_preview}
- Audience: {audience}
{palette_line}
- Issues: {findings_text}

{image_block}

## Design Direction: {style}
{style_desc}

{css_contract}

## Requirements
1. Output a single `<section>` element using brand-token CSS (no Tailwind, no utility classes)
2. Lay out the ACTUAL content of this section — ALL of it, invent nothing
3. Preserve brand identity; improve execution (hierarchy, whitespace, contrast, a11y)
4. Reuse the exact absolute image URLs listed above; placeholder only for a genuinely
   missing image

Output ONLY the HTML in a ```html code block."""


def build_component_prompt(
    *,
    url: str,
    component_html: str,
    component_type: str,
    page_type: str | None,
    style: str,
    findings: list[dict] | None,
    brand: dict | None,
    site_analysis: dict | None,
    style_description: str | None = None,
    context: ResolvedContext | None = None,
) -> str:
    """Build a component-level redesign prompt."""
    style_desc = style_description or STYLE_DESCRIPTIONS.get(style, STYLE_DESCRIPTIONS["modern"])
    findings_text = _format_findings(findings)

    _, primary_color, secondary_color, palette_known = _resolve_palette(
        context=context, brand=brand, site_analysis=site_analysis
    )
    color_line = (
        f"- Brand colours: primary={primary_color}, secondary={secondary_color}"
        if palette_known
        else "- Brand colours: derive from the current HTML — do NOT introduce a new palette"
    )

    design_tips = COMPONENT_DESIGN_TIPS.get(component_type, "")
    tips_section = f"\n## Component-Specific Tips\n{design_tips}\n" if design_tips else ""

    return f"""Refine the "{component_type}" component of this web page.
Output a self-contained HTML snippet styled with hand-written CSS (no Tailwind / no utility
framework). The refined component will replace the original in the page.

## Component to Refine
Type: {component_type}
Current HTML:
```html
{_truncate_html(component_html, 20000)}
```

## Page Context
- Page: {url} ({page_type or "unknown"})
{color_line}
- Issues: {findings_text}
{tips_section}
## Design Direction: {style}
{style_desc}

## Requirements
1. Output the refined component HTML styled with hand-written CSS (no Tailwind)
2. Preserve all content and links — invent nothing
3. Keep the brand's real colours, fonts and imagery
4. Reuse the original image `src` URLs exactly as they appear
5. Ensure the component works standalone and when embedded in a page

Output ONLY the HTML in a ```html code block."""


def build_refinement_prompt(
    template_loader: PromptTemplateLoader | None,
    *,
    url: str,
    previous_html: str,
    feedback: str,
    page_type: str | None,
    style: str,
    brand: dict | None,
    site_analysis: dict | None,
    style_description: str | None = None,
    focus_areas: list[str] | None = None,
) -> str:
    """Build a refinement prompt with previous HTML and feedback."""
    settings = get_settings()
    style_desc = style_description or STYLE_DESCRIPTIONS.get(style, STYLE_DESCRIPTIONS["modern"])
    html_excerpt = _truncate_html(previous_html, settings.redesign.max_html_chars)

    primary_color = ""
    if brand and brand.get("primary_color"):
        primary_color = brand["primary_color"]

    # Try Jinja2 template
    if template_loader:
        rendered = template_loader.render(
            "refinement.md.j2",
            url=url,
            page_type=page_type,
            style=style,
            style_desc=style_desc,
            primary_color=primary_color,
            html_excerpt=html_excerpt,
            feedback=feedback,
            focus_areas=focus_areas,
        )
        if rendered:
            return rendered

    focus_section = ""
    if focus_areas:
        focus_section = (
            "\n## Focus Areas (prioritize these)\n"
            + "\n".join(f"- {area}" for area in focus_areas)
            + "\n"
        )
    color_line = f"- Primary colour: {primary_color}\n" if primary_color else ""

    return f"""Refine the following redesigned web page based on user feedback.
Output an updated self-contained HTML file. Keep the page's existing CSS approach
(brand-token CSS) — do NOT introduce Tailwind or any utility-class framework.

## Previous Redesign
- URL: {url}
- Page type: {page_type or "unknown"}
- Style: {style} - {style_desc}
{color_line}- HTML:
{html_excerpt}

## User Feedback
{feedback}
{focus_section}## Requirements
1. Apply the feedback changes to the existing redesign
2. Keep the overall design direction ({style}) intact
3. Self-contained HTML, brand-token CSS (no Tailwind / no CDN)
4. Preserve the site's actual content and image URLs
5. Only change what the feedback asks for

Output ONLY the HTML code in a ```html code block."""
