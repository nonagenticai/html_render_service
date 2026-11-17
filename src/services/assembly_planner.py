"""Assembly page-planner — the narrow LLM step of computed-assembly web design.

In assembly mode the LLM does NOT write HTML/CSS. Its only job is to return a
**page plan**: an ordered list of catalog blocks with their content slots filled.
Everything visual (layout, theming, markup, assets) is then produced
deterministically by the assembler. This module owns the plan's JSON schema, the
prompt, and the tolerant parsing of the model's response.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from src.services.catalog import BlockManifest, block_keys, catalog_brief, get_block

logger = logging.getLogger(__name__)

ASSEMBLY_SYSTEM_PROMPT = (
    "You are a senior web-design director composing a landing page from a FIXED "
    "catalog of pre-built, on-brand section blocks. You do NOT write HTML or CSS. "
    "You SELECT blocks from the catalog and FILL their content slots with concise, "
    "specific, benefit-led copy tailored to the brief. Rules:\n"
    "- Use ONLY block types from the catalog. Never invent block types or slots.\n"
    "- Start with a 'navbar' and end with a 'footer'. Include a 'hero' near the top.\n"
    "- Choose 5-9 blocks that tell a coherent story for the brief. Include an 'hours' "
    "block when the business has a place customers visit.\n"
    "- Write real copy — never 'Lorem ipsum', never invented stats/testimonials "
    "unless the brief provides them.\n"
    "- List slots (features, pricing, faq, stats) must be filled with plausible, "
    "on-topic items.\n"
    "- Testimonials are FACTS: use ONLY customer quotes written in the brief, word "
    "for word. Never invent a quote or a customer name. If the brief quotes no one, "
    "do not use a 'testimonials' block (in a fill, return an empty list).\n"
    "- Use ONE brand name, spelled identically, in every block (navbar, footer, copy).\n"
    "- Links are in-page anchors to sections on this page (#features, #pricing, #faq, "
    "#testimonials, #get-started, #contact) or mailto:/tel: from the brief; never '#' "
    "alone and never invented URLs. Leave image slots empty.\n"
    "- Opening hours, times and addresses are FACTS: use ONLY the ones written in the "
    "brief. Never invent them — not in an 'hours' block, not in an FAQ answer, not "
    "anywhere. If the brief gives no hours, leave the hours items empty and do not "
    "write an FAQ answer that states opening times.\n"
    "Return ONLY the structured plan."
)

# JSON schema handed to worker_management for structured output. Content is an
# open object because each block's slots differ — the assembler validates + coerces
# per-block against the catalog manifest (dropping unknown slots), so a loose schema
# here is safe and keeps the model unconstrained on copy.
ASSEMBLY_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "A catalog block key."},
                    "content": {"type": "object", "description": "Slot values for the block."},
                },
                "required": ["type"],
            },
        }
    },
    "required": ["blocks"],
}


def build_assembly_prompt(
    *,
    description: str,
    brand: dict | None,
    page_type: str | None,
    audience: str | None,
    interactivity: str,
    refine_findings: str | None = None,
) -> str:
    """Build the user prompt for the page-planner.

    ``refine_findings`` (the verification lane's hard findings from a prior
    attempt) are surfaced so the model can adjust the plan to fix them.
    """
    brand = brand or {}
    lines = [
        "# Brief",
        description.strip() or "A modern marketing landing page.",
        "",
        f"Page type: {page_type or 'landing'}",
        f"Audience: {audience or 'general'}",
    ]
    if refine_findings:
        lines += [
            "",
            "# Fix these issues from the previous attempt",
            refine_findings,
        ]
    if brand.get("name"):
        lines.append(f"Brand name: {brand['name']}")
    if brand.get("tagline"):
        lines.append(f"Tagline: {brand['tagline']}")
    if brand.get("brand_voice"):
        lines.append(f"Voice: {brand['brand_voice']}")
    if interactivity == "interactive":
        lines.append("Interactivity: allowed (FAQ uses expandable sections; keep copy scannable).")
    lines += [
        "",
        "# Available blocks (type: description; slots)",
        catalog_brief(),
        "",
        "Return a plan: an ordered list of blocks with filled content slots.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Split planner (CATALOG_SPLIT_ASSEMBLY) — decompose the one-shot draw into a
# small structure call (pick block TYPES) + one small fill call per block. Sized
# for a small, slow vLLM that a big multi-block one-shot overwhelms. The one-shot
# schema/prompt/parse above are untouched; these are additive.
# ---------------------------------------------------------------------------

ASSEMBLY_STRUCTURE_SYSTEM_PROMPT = (
    "You are a senior web-design director choosing the SECTION STRUCTURE of a landing "
    "page from a FIXED catalog of pre-built block types. You do NOT write copy, HTML, "
    "or CSS in this step — you ONLY pick an ordered list of block TYPES. Rules:\n"
    "- Use ONLY block types from the catalog. Never invent types.\n"
    "- Start with a 'navbar' and end with a 'footer'. Include a 'hero' near the top.\n"
    "- Choose 5-9 blocks that tell a coherent story for the brief.\n"
    "- Include an 'hours' block when the business has a place customers visit "
    "(shop, cafe, restaurant, studio, clinic).\n"
    "- Include a 'testimonials' block ONLY when the brief itself quotes customers.\n"
    "Return ONLY the ordered list of block types."
)

# Structure-phase schema: block TYPES only (no content). Kept loose/positional-safe.
ASSEMBLY_STRUCTURE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "A catalog block key."},
                },
                "required": ["type"],
            },
        }
    },
    "required": ["blocks"],
}

ASSEMBLY_FILL_SYSTEM_PROMPT = (
    "You are a senior web-design copywriter filling the content slots of ONE section "
    "block on an on-brand landing page. You do NOT write HTML or CSS — you return a "
    "JSON object mapping the block's slot names to concise, specific, benefit-led "
    "copy tailored to the brief. Rules:\n"
    "- Fill ONLY the slots listed for this block. Never invent slot names.\n"
    "- Write real copy — never 'Lorem ipsum', never invented stats/testimonials "
    "unless the brief provides them.\n"
    "- For list slots, return a JSON array of items with the item fields shown.\n"
    "- If a brand name is given, use it exactly; never invent a different one.\n"
    "- Links are in-page anchors (#features, #pricing, #faq, #testimonials, "
    "#get-started, #contact) or mailto:/tel: from the brief; never '#' alone and never "
    "invented URLs. Leave image slots empty.\n"
    "- Opening hours, times and addresses are FACTS: use ONLY the ones written in the "
    "brief. Never invent them — not in an 'hours' block, not in an FAQ answer, not "
    "anywhere. If the brief gives no hours, leave the hours items empty and do not "
    "write an FAQ answer that states opening times.\n"
    "- Testimonials are FACTS: use ONLY customer quotes written in the brief, word "
    "for word. Never invent a quote or a customer name. If the brief quotes no one, "
    "do not use a 'testimonials' block (in a fill, return an empty list).\n"
    "Return ONLY the JSON content object for this one block."
)

# Fill-phase schema: intentionally LOOSE (an open object). A typed per-block schema
# would place properties named after slot keys — some catalog slots ('items',
# 'title') are JSON-Schema keywords, which the vLLM xgrammar keyword sanitizer can
# corrupt (worker_management #121). The slot contract is carried in the prompt text
# (sanitizer-safe); the assembler's _coerce_content validates + coerces at render.
ASSEMBLY_FILL_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"content": {"type": "object", "description": "Slot values for the block."}},
    "required": ["content"],
}


def _brief_context_lines(
    *,
    description: str,
    brand: dict | None,
    page_type: str | None,
    audience: str | None,
    interactivity: str,
    refine_findings: str | None,
) -> list[str]:
    """Shared brief/brand/refine context lines for the split-phase prompts."""
    brand = brand or {}
    lines = [
        "# Brief",
        description.strip() or "A modern marketing landing page.",
        "",
        f"Page type: {page_type or 'landing'}",
        f"Audience: {audience or 'general'}",
    ]
    if brand.get("name"):
        lines.append(f"Brand name: {brand['name']}")
    if brand.get("tagline"):
        lines.append(f"Tagline: {brand['tagline']}")
    if brand.get("brand_voice"):
        lines.append(f"Voice: {brand['brand_voice']}")
    if interactivity == "interactive":
        lines.append("Interactivity: allowed (FAQ uses expandable sections; keep copy scannable).")
    if refine_findings:
        lines += ["", "# Fix these issues from the previous attempt", refine_findings]
    return lines


def build_structure_prompt(
    *,
    description: str,
    brand: dict | None,
    page_type: str | None,
    audience: str | None,
    interactivity: str,
    refine_findings: str | None = None,
) -> str:
    """User prompt for the structure phase — pick an ordered list of block types."""
    lines = _brief_context_lines(
        description=description,
        brand=brand,
        page_type=page_type,
        audience=audience,
        interactivity=interactivity,
        refine_findings=refine_findings,
    )
    lines += [
        "",
        "# Available block types (type: description; slots)",
        catalog_brief(),
        "",
        "Return an ordered list of block TYPES only (no content).",
    ]
    return "\n".join(lines)


def _slot_contract_lines(manifest: BlockManifest) -> list[str]:
    """Describe one block's fillable slots (name → type + description) for the prompt."""
    lines: list[str] = []
    for name, spec in sorted(manifest.slots.items()):
        detail = spec.description or ""
        if spec.type == "list":
            item_shape = ", ".join(f"{k}: {t}" for k, t in sorted(spec.item_slots.items()))
            lines.append(
                f"- {name} (list of items; each item has: {item_shape or 'text'}): {detail}"
            )
        else:
            lines.append(f"- {name} ({spec.type}): {detail}")
    return lines or ["- (no slots — return an empty object)"]


def build_fill_prompt(
    *,
    manifest: BlockManifest,
    description: str,
    brand: dict | None,
    page_type: str | None,
    audience: str | None,
    interactivity: str,
    refine_findings: str | None = None,
) -> str:
    """User prompt for the fill phase — fill ONE block's content slots."""
    lines = _brief_context_lines(
        description=description,
        brand=brand,
        page_type=page_type,
        audience=audience,
        interactivity=interactivity,
        refine_findings=refine_findings,
    )
    lines += [
        "",
        f"# Fill the '{manifest.key}' block ({manifest.description})",
        "Slots to fill:",
        *_slot_contract_lines(manifest),
        "",
        'Return a JSON object of the form {"content": { <slot>: <value>, ... }}.',
    ]
    return "\n".join(lines)


def parse_structure(output: object) -> list[str]:
    """Coerce a structure-phase response into an ordered list of valid catalog keys.

    Accepts the structured dict (``{"blocks":[{"type":...}]}``), a bare list of
    strings/dicts, or a JSON string. Drops unknown keys and de-duplicates singleton
    blocks (navbar/footer), preserving order. Returns ``[]`` on anything unusable.
    """
    data = _loads_lenient(output) if isinstance(output, str) else output
    blocks: object = None
    if isinstance(data, dict):
        mapping = cast("dict[str, object]", data)
        blocks = mapping.get("blocks")
        if blocks is None:
            blocks = mapping.get("plan")
    elif isinstance(data, list):
        blocks = data
    if not isinstance(blocks, list):
        logger.warning("Assembly structure response was not a block list: %r", type(output))
        return []
    valid_keys = set(block_keys())
    ordered: list[str] = []
    seen_singletons: set[str] = set()
    for entry in blocks:
        if isinstance(entry, dict):
            item = cast("dict[str, object]", entry)
            key = str(item.get("type") or item.get("block") or "").strip()
        else:
            key = str(entry or "").strip()
        if key not in valid_keys:
            continue
        manifest = get_block(key)
        if manifest is not None and manifest.singleton:
            if key in seen_singletons:
                continue
            seen_singletons.add(key)
        ordered.append(key)
    return ordered


class FillContentMalformedError(Exception):
    """A fill-phase reply parsed as JSON/a dict but had no usable ``content`` shape.

    Distinguishes "the model returned junk" from "the model legitimately left this
    block's content empty" — see :func:`parse_fill`. The caller (``generator.py``'s
    ``_assembly_plan_split``) already wraps the ``parse_fill`` call in a generic
    ``except Exception`` that records ``run.degrade`` with the exception's class name
    and defaults the block's content; this class is not caught anywhere more specific,
    so raising it here is enough to make it land in
    ``html_validation_report.quality_warnings`` without inventing a new channel.
    """


def parse_fill(output: object) -> dict:
    """Coerce a fill-phase response into a raw content dict (assembler coerces later).

    Accepts ``{"content": {...}}``, a bare content dict, or a JSON string.

    Returns ``{}`` — silently, no degradation — only for a reply that is
    *legitimately* empty: no output at all (``None``), blank text, a bare ``{}``,
    or an explicit ``{"content": {}}``. All of these are schema-compliant ways for
    the model to say "nothing for this block".

    Raises :class:`FillContentMalformedError` for a reply that IS JSON/dict-shaped (or
    text that fails to parse as JSON at all) but cannot be read as an object: a
    non-dict top-level value, a non-dict ``content`` field, or garbled text. This
    used to fall through to the same silent ``{}`` as a legitimate empty reply,
    which made a hollow block indistinguishable from an intentionally empty one
    (see issue #132) — the caller records it as a degradation instead.
    """
    if output is None:
        return {}
    if isinstance(output, str):
        if not output.strip():
            return {}
        data = _loads_lenient(output)
        if data is None:
            raise FillContentMalformedError("fill reply text did not parse as JSON")
    else:
        data = output
    if isinstance(data, dict):
        mapping = cast("dict[str, object]", data)
        if "content" not in mapping:
            return mapping  # bare content dict (legacy shape), tolerated as-is
        content = mapping["content"]
        if isinstance(content, dict):
            return content
        raise FillContentMalformedError(
            f"fill reply 'content' field was {type(content).__name__}, not an object"
        )
    raise FillContentMalformedError(
        f"fill reply top-level was {type(data).__name__}, not an object"
    )


def parse_plan(output: Any) -> list[dict]:
    """Coerce a worker_management response into a list of plan block entries.

    Accepts either the structured dict (``{"blocks": [...]}``), a bare list, or a
    JSON string (optionally fenced). Returns ``[]`` on anything unusable — the
    caller decides how to fail.
    """
    data = output
    if isinstance(data, str):
        data = _loads_lenient(data)
    if isinstance(data, dict):
        blocks = data.get("blocks", data.get("plan"))
    elif isinstance(data, list):
        blocks = data
    else:
        blocks = None
    if not isinstance(blocks, list):
        logger.warning("Assembly plan response was not a block list: %r", type(output))
        return []
    valid_keys = set(block_keys())
    cleaned: list[dict] = []
    for entry in blocks:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("type") or entry.get("block") or "").strip()
        if key not in valid_keys:
            continue
        content = entry.get("content")
        cleaned.append({"type": key, "content": content if isinstance(content, dict) else {}})
    return cleaned


def _loads_lenient(text: str) -> Any:
    """Parse JSON that may be wrapped in ``` fences or have leading prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    try:
        return json.loads(t)
    except ValueError, TypeError:
        start, end = t.find("{"), t.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(t[start : end + 1])
            except ValueError, TypeError:
                return None
        return None
