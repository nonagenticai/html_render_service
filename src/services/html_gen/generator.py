"""The ``HtmlGenerator`` facade: LLM orchestration + section composition.

String/CSS/image/prompt work is delegated to the sibling modules; this class
owns the WM transport, token accumulation, template loader, and the async
generate/refine entry points. The private ``_build_*_prompt`` methods are thin
wrappers over :mod:`prompt_builder` free functions (kept as methods because
tests call them on the instance).
"""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from html import escape as _attr_escape
from typing import TYPE_CHECKING

import httpx

from src.config import get_settings
from src.llm_client import get_wm_client
from src.llm_credentials import LLMConfigLike, model_override_from
from src.services.batch_progress import report_progress
from src.services.context_resolver import (
    ResolvedContext,
    extract_fonts,
    extract_image_manifest,
    extract_page_meta,
    extract_palette,
)

from .brand_css import BRAND_TOKEN_NAMES, _brand_tokens_css, _google_fonts_head, _resolve_palette
from .css_tokens import css_defects, css_token_findings, describe_tokens, has_content_markup
from .html_utils import (
    _balance_section_fragment,
    _extract_html_block,
    _require_html,
    section_reply_defect,
)
from .prompt_builder import (
    REDESIGN_SYSTEM_PROMPT,
    build_component_prompt,
    build_full_page_prompt,
    build_refinement_prompt,
    build_section_prompt,
    section_reply_char_target,
)
from .rendered_checks import (
    RenderedReport,
    fix_contrast_css,
    fix_logo_surfaces,
    probe_rendered_page,
)
from .sanitize import finalize_fragment
from .section_css import analyze_section_css
from .section_landmarks import tag_data_sections
from .section_plan import plan_greenfield_sections
from .section_run import (
    MIN_CALL_SECONDS,
    BudgetExhausted,
    GenerationBudget,
    SectionJob,
    SectionRun,
    budget_window,
)
from .types import LLMResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Container

    from w2v_common.clients import WMClient, WMResponse

    from src.services.prompt_template import PromptTemplateLoader

logger = logging.getLogger(__name__)

__all__ = ["HtmlGenerator"]

_TITLE_SUFFIX = "Redesign Concept"
# One call + up to two retries per section. One bad draw (a sampling loop, a reply cut
# off by worker_management's time budget) must not cost the whole redesign; a section
# that is still unusable after three draws is a real defect and fails loudly.
SECTION_ATTEMPTS = 3
_IMG_SRC_RE = re.compile(r"<img\b[^>]*?\ssrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)


def _is_logo(url: str) -> bool:
    return "logo" in url.rsplit("/", 1)[-1].lower()


def _img_srcs(html: str) -> list[str]:
    return [m.group(1).strip() for m in _IMG_SRC_RE.finditer(html or "")]


def _source_image_counts(html: str, base_url: str) -> dict[str, int]:
    """How often the SOURCE page shows each (absolutized) image."""
    from urllib.parse import urljoin

    counts: dict[str, int] = {}
    for src in _img_srcs(html):
        url = src if src.startswith("data:") else urljoin(base_url or "", src)
        counts[url] = counts.get(url, 0) + 1
    return counts


def _reused_images(
    section_html: str, used: dict[str, str], source_counts: dict[str, int]
) -> list[str]:
    """Images this section repeats from an EARLIER section although the source showed them once.

    Logos are exempt (a nav and a footer legitimately share one).
    """
    return sorted(
        {
            src
            for src in _img_srcs(section_html)
            if src in used and not _is_logo(src) and source_counts.get(src, 0) <= 1
        }
    )


# LLM regeneration rounds for sections the RENDERED page still shows failing WCAG AA
# contrast after the deterministic colour fix.
RENDER_REPAIR_ROUNDS = 2
# One rendered-page probe (Chromium layout + background capture) may take this long.
PROBE_TIMEOUT_S = 240.0


def _llm_backend(client: object) -> str:
    """Which backend the LLM client dispatches to (``wm`` / ``vllm``), for log lines."""
    return str(getattr(client, "llm_backend", None) or "wm")


def _is_length_cutoff(exc: httpx.HTTPStatusError) -> bool:
    """A 422 that says the reply hit ``max_tokens`` (``finish_reason=length``)."""
    if exc.response.status_code != 422:
        return False
    try:
        body = exc.response.text
    except Exception:  # noqa: BLE001 — an unread streamed body: judge by the message
        body = ""
    return "finish_reason=length" in f"{body} {exc}"


def _retry_guidance(defects: list[str]) -> str:
    text = " ".join(defects)
    if "truncated" in text or "degenerate" in text or "ceiling" in text:
        return (
            "\nYour previous reply was cut off or looped — it did not fit the output "
            f"budget. Write a SHORTER reply (under ~{section_reply_char_target() // 2} "
            "characters): style with page tokens directly, declare almost no custom "
            "properties, drop decorative extras, and finish the section with its closing "
            "tags."
        )
    if "contrast" in text:
        return (
            "\nOn dark, brand- or accent-coloured surfaces use var(--color-on-brand) / "
            "var(--color-on-accent) for text (a color-mix with at most 15% transparency is "
            "fine); keep --color-text-muted and --color-accent-text for light surfaces."
        )
    return ""


@dataclass(frozen=True)
class _Draw:
    """The verdict on one section reply."""

    html: str | None  # the finalized fragment when the reply is complete
    defects: list[str]
    soft_only: bool = False

    @property
    def clean(self) -> bool:
        return self.html is not None and not self.defects


def _judge_draw(
    result: LLMResult,
    *,
    max_tokens: int,
    url: str,
    manifest: list[dict],
    used_images: dict[str, str],
    source_image_counts: dict[str, int],
) -> _Draw:
    extracted = _extract_html_block(result.text)
    reply_defect = section_reply_defect(
        extracted, completion_tokens=result.output_tokens, max_tokens=max_tokens
    )
    if reply_defect is not None:
        return _Draw(None, [f"unusable reply: {reply_defect}"])
    section_html = finalize_fragment(extracted, base_url=url, manifest=manifest)
    defects = css_defects(section_html, BRAND_TOKEN_NAMES)
    defects += analyze_section_css(section_html, BRAND_TOKEN_NAMES).defects()
    if defects:
        return _Draw(section_html, defects)
    reused = _reused_images(section_html, used_images, source_image_counts)
    if reused:
        return _Draw(
            section_html,
            [
                "these photos are already shown by an earlier section — use a different "
                "image from the list, or none: " + ", ".join(reused)
            ],
            soft_only=True,
        )
    return _Draw(section_html, [])


def _out_of_budget(
    run: SectionRun, where: str, exc: BudgetExhausted, fallback: str | None, defects: list[str]
) -> str | None:
    """Record why a section stopped short of a clean draft; return what may still ship."""
    shown = f": {'; '.join(d[:200] for d in defects)}" if defects else ""
    if fallback is not None:
        run.degrade(f"{where}: {exc} — shipped a draft with unresolved defects{shown}")
    else:
        run.degrade(f"{where}: {exc} — no usable draft{shown}")
    logger.warning("%s: out of budget: %s", where, exc)
    return fallback


def _carry_brand(brand: dict | None, content: dict) -> dict | None:
    """Adopt the first real brand name a split fill wrote, if none was given."""
    if (brand or {}).get("name"):
        return brand
    found = str(content.get("brand") or "").strip()
    if not found or found.lower() == "brand":
        return brand
    return {**(brand or {}), "name": found}


def _note_hollow_block(run: SectionRun | None, block_type: str, exc: Exception) -> None:
    """Record that one assembly block shipped on its manifest defaults, and why.

    Only the exception CLASS is recorded. This lands in
    ``html_validation_report.quality_warnings`` — a stored, customer-adjacent field —
    and ``str(exc)`` on a worker_management or transport error can carry the request
    body, a response payload, or a credentialed URL. The class name is bounded and
    secret-free; the full message stays in the caller's log line.
    """
    if run is None:
        return
    run.degrade(
        f"assembly: block {block_type} fill failed "
        f"({type(exc).__name__}); shipped with default content"
    )


def _failing_sections(report: RenderedReport, sections: dict[str, str]) -> dict[str, list[str]]:
    return {k: v for k, v in report.contrast_by_section().items() if k in sections}


def _css_attr_value(value: str) -> str:
    """``value`` safe inside a double-quoted CSS attribute selector."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _unique_section_key(base: str, taken: Container[str]) -> str:
    key = base or "section"
    n = 2
    while key in taken:
        key = f"{base}-{n}"
        n += 1
    return key


def _document_lang_and_title(
    title: str | None,
    *,
    context: ResolvedContext | None,
    brand: dict | None,
    html: str | None,
) -> tuple[str, str]:
    """``(lang, <title>)`` for a composed page, from the SOURCE page first.

    ``lang="en"`` was hard-coded (a Dutch page shipped as English) and the title was
    the raw URL. An explicit ``title`` that is not a URL still wins.
    """
    meta = extract_page_meta(html) if html else {"lang": None, "title": None}
    lang = (getattr(context, "page_lang", None) if context else None) or meta.get("lang") or "en"
    explicit = (title or "").strip()
    if explicit and not explicit.lower().startswith(("http://", "https://", "www.")):
        base = explicit
    else:
        base = (
            (getattr(context, "page_title", None) if context else None)
            or meta.get("title")
            or (getattr(context, "brand_name", None) if context else None)
            or ((brand or {}).get("name"))
            or ""
        )
    return lang, f"{base} — {_TITLE_SUFFIX}" if base else _TITLE_SUFFIX


class HtmlGenerator:
    """Generate redesigned HTML+Tailwind pages using an OpenAI-compatible LLM API."""

    WORKER_NAME = "redesign-html"
    # Generation worker (greenfield/assembly). Rework uses WORKER_NAME.
    ASSEMBLY_WORKER_NAME = "web-design"

    def __init__(
        self,
        template_loader: PromptTemplateLoader | None = None,
        *,
        wm_client: WMClient | None = None,
        render_probe: Callable[[str], Awaitable[RenderedReport]] | None = None,
    ) -> None:
        settings = get_settings()
        self._max_tokens = settings.wm.max_tokens
        # What a call can really get (worker_management's silent time-budget clamp).
        self._output_ceiling = settings.wm.output_token_ceiling
        self._render_probe = render_probe or probe_rendered_page
        # NO per-job state on this instance: it is process-wide (asgi.py) and shared by
        # concurrent jobs, batch pages and variants. A render's state is a SectionRun.
        self._template_loader = template_loader
        self._generation_temperature = 0.7
        self._refinement_temperature = 0.4
        self._wm = wm_client if wm_client is not None else get_wm_client()
        # Diagnostic-only label used by ``estimate_cost(... , model=...)``
        # callers in finalize_phases.py and lifecycle.py. The post-refactor
        # (de43a15) routes through worker_management which resolves the
        # actual model dynamically per worker name, so we just expose the
        # worker label here. ``estimate_cost`` ignores its ``model`` arg
        # (hardcoded Claude Sonnet pricing) — restoring the attribute
        # primarily prevents an AttributeError that aborts the redesign
        # finalize phase and is surfaced as
        # ``StageRetriesExhaustedError: stage 'website_redesign.wait'``
        # in w2v_orchestrator. See e2e-post-promote website_review run.
        self._model = self.WORKER_NAME

    async def generate_assembly_plan(
        self,
        *,
        description: str,
        brand: dict | None = None,
        page_type: str | None = None,
        audience: str | None = None,
        interactivity: str = "static",
        refine_findings: str | None = None,
        llm_config: LLMConfigLike | None = None,
        budget: GenerationBudget | None = None,
        extra: bool = False,
        run: SectionRun | None = None,
    ) -> tuple[list[dict], int, int]:
        """Assembly mode: ask the LLM to SELECT catalog blocks + FILL content slots.

        Returns ``(plan, input_tokens, output_tokens)``. The plan is validated
        against the catalog by :func:`assembly_planner.parse_plan`; the assembler
        renders it deterministically. Raises RuntimeError on empty/unusable output
        so the job fails loud rather than completing empty.

        When ``CATALOG_SPLIT_ASSEMBLY`` is set, the single one-shot draw is replaced
        by one small structure call (pick block types) + one small fill call per
        block — sized for a small, slow vLLM. The public contract is unchanged.

        ``budget`` is the JOB's generation budget (the same object the section and
        full-page paths spend): every draw below runs under it and is refused once the
        job's deadline leaves less than ``MIN_CALL_SECONDS``. Without one a fresh
        budget from settings bounds this plan alone. ``extra=True`` marks a re-draw
        driven by a verification failure, so the refine loop is capped by
        ``max_extra_llm_calls`` like any other retry; a first draw never is.

        ``run`` is this render's :class:`SectionRun`. The split path can finish a page
        with blocks it never got to fill (the deadline arrives mid-split); that is a
        DEGRADED deliverable, not a failure, so it is recorded in ``run.degradations``
        — the same place the section path records its compromises, and the same place
        the job turns into ``html_validation_report.quality_warnings``. Without a
        ``run`` the degradation has nowhere to go but the log.
        """
        params = {
            "description": description,
            "brand": brand,
            "page_type": page_type,
            "audience": audience,
            "interactivity": interactivity,
            "refine_findings": refine_findings,
            "llm_config": llm_config,
            "budget": budget if budget is not None else GenerationBudget.from_settings(),
            "extra": extra,
        }
        if get_settings().catalog.split_assembly:
            return await self._assembly_plan_split(**params, run=run)
        return await self._assembly_plan_oneshot(**params)

    async def _assembly_plan_oneshot(
        self,
        *,
        description: str,
        brand: dict | None,
        page_type: str | None,
        audience: str | None,
        interactivity: str,
        refine_findings: str | None,
        llm_config: LLMConfigLike | None,
        budget: GenerationBudget,
        extra: bool = False,
    ) -> tuple[list[dict], int, int]:
        """One-shot assembly draw: SELECT blocks + FILL all slots in a single call."""
        import time

        from src.services.assembly_planner import (
            ASSEMBLY_PLAN_SCHEMA,
            ASSEMBLY_SYSTEM_PROMPT,
            build_assembly_prompt,
            parse_plan,
        )

        prompt = build_assembly_prompt(
            description=description,
            brand=brand,
            page_type=page_type,
            audience=audience,
            interactivity=interactivity,
            refine_findings=refine_findings,
        )
        logger.info(
            "LLM call worker=%s backend=%s (assembly plan)",
            self.ASSEMBLY_WORKER_NAME,
            _llm_backend(self._wm),
        )
        t0 = time.monotonic()
        async with self._budget_window(budget, extra=extra):
            response = await self._wm.message(
                worker_name=self.ASSEMBLY_WORKER_NAME,
                message=prompt,
                system_prompt=ASSEMBLY_SYSTEM_PROMPT,
                output_schema=ASSEMBLY_PLAN_SCHEMA,
                model_override=model_override_from(llm_config),
                max_tokens=self._max_tokens,
            )
        elapsed = time.monotonic() - t0
        usage = response.usage_metrics or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)

        plan = parse_plan(response.output)
        logger.info(
            "Assembly plan in %.2fs: %d blocks (input=%d output=%d)",
            elapsed,
            len(plan),
            input_tokens,
            output_tokens,
        )
        if not plan:
            raise RuntimeError(
                f"WM worker {self.ASSEMBLY_WORKER_NAME} returned no usable page plan "
                f"after {elapsed:.1f}s (assembly mode)"
            )
        return plan, input_tokens, output_tokens

    async def _assembly_plan_split(
        self,
        *,
        description: str,
        brand: dict | None,
        page_type: str | None,
        audience: str | None,
        interactivity: str,
        refine_findings: str | None,
        llm_config: LLMConfigLike | None,
        budget: GenerationBudget,
        extra: bool = False,
        run: SectionRun | None = None,
    ) -> tuple[list[dict], int, int]:
        """Split assembly plan: 1 structure call + 1 fill call per block.

        Each call is small (tiny context + output), so a slow vLLM completes the whole
        plan comfortably. Per-block fill failures degrade to that block's manifest
        defaults (the assembler fills them) and never fail the page. If the structure
        phase yields no blocks, falls back to the one-shot draw once. Returns the
        per-attempt ``(plan, input_tokens, output_tokens)`` summed across all calls.

        Every call runs under ``budget``. Only the FIRST call of the draw carries
        ``extra`` (a verification-driven re-draw): the per-block fills are this draw's
        first drafts — the page needs every block — so like a section's first draft
        they are deadline-bound but never refused by ``max_extra_llm_calls``.

        Stopping mid-split SHIPS the remaining blocks on their manifest defaults — the
        deliberate choice, a partial page beats no page. It is still a quality loss the
        tenant paid for, so it is recorded in ``run.degradations`` with the number of
        defaulted blocks; the job persists that as
        ``html_validation_report.quality_warnings``. A pod log line is not evidence
        anyone downstream can see.
        """
        import time

        from src.services.assembly_planner import (
            ASSEMBLY_FILL_SCHEMA,
            ASSEMBLY_FILL_SYSTEM_PROMPT,
            ASSEMBLY_STRUCTURE_SCHEMA,
            ASSEMBLY_STRUCTURE_SYSTEM_PROMPT,
            build_fill_prompt,
            build_structure_prompt,
            parse_fill,
            parse_plan,
            parse_structure,
        )
        from src.services.catalog import get_block

        model_override = model_override_from(llm_config)
        in_tokens = 0
        out_tokens = 0

        async def _call(
            message: str, system_prompt: str, schema: dict, *, is_extra: bool = False
        ) -> WMResponse:
            nonlocal in_tokens, out_tokens
            async with self._budget_window(budget, extra=is_extra):
                resp = await self._wm.message(
                    worker_name=self.ASSEMBLY_WORKER_NAME,
                    message=message,
                    system_prompt=system_prompt,
                    output_schema=schema,
                    model_override=model_override,
                    max_tokens=self._max_tokens,
                )
            usage = resp.usage_metrics or {}
            i = int(usage.get("prompt_tokens", 0) or 0)
            o = int(usage.get("completion_tokens", 0) or 0)
            in_tokens += i
            out_tokens += o
            return resp

        # Phase A — structure: pick an ordered list of catalog block types.
        t0 = time.monotonic()
        struct_resp = await _call(
            build_structure_prompt(
                description=description,
                brand=brand,
                page_type=page_type,
                audience=audience,
                interactivity=interactivity,
                refine_findings=refine_findings,
            ),
            ASSEMBLY_STRUCTURE_SYSTEM_PROMPT,
            ASSEMBLY_STRUCTURE_SCHEMA,
            is_extra=extra,
        )
        block_types = parse_structure(struct_resp.output)
        logger.info(
            "Assembly structure: %d blocks in %.2fs", len(block_types), time.monotonic() - t0
        )
        if not block_types:
            logger.warning(
                "Assembly split structure phase returned no blocks; falling back to one-shot"
            )
            plan, os_in, os_out = await self._assembly_plan_oneshot(
                description=description,
                brand=brand,
                page_type=page_type,
                audience=audience,
                interactivity=interactivity,
                refine_findings=refine_findings,
                llm_config=llm_config,
                budget=budget,
                # The structure draw already spent this draw's ``extra``; the
                # fallback substitutes for it rather than retrying on top of it.
                extra=False,
            )
            # Bill the (wasted) structure call too, so the reported cost is accurate.
            return plan, in_tokens + os_in, out_tokens + os_out

        # Phase B — fill each block's slots with one small call. A failed or empty
        # fill degrades to the block's defaults (the assembler coerces content).
        #
        # Every fill is an INDEPENDENT call, so without a brand name each block
        # invents its own (live run 01a156a6: navbar "Hearth & Crumb", footer
        # "The Daily Crumb"). The first brand a fill writes is carried into every
        # later fill prompt; the assembler then enforces it deterministically.
        fill_brand: dict | None = brand
        blocks: list[dict] = []
        for idx, block_type in enumerate(block_types):
            manifest = get_block(block_type)
            if manifest is None:
                continue
            content: dict = {}
            if manifest.slots:
                try:
                    fill_resp = await _call(
                        build_fill_prompt(
                            manifest=manifest,
                            description=description,
                            brand=fill_brand,
                            page_type=page_type,
                            audience=audience,
                            interactivity=interactivity,
                            refine_findings=refine_findings,
                        ),
                        ASSEMBLY_FILL_SYSTEM_PROMPT,
                        ASSEMBLY_FILL_SCHEMA,
                    )
                    content = parse_fill(fill_resp.output)
                    fill_brand = _carry_brand(fill_brand, content)
                except BudgetExhausted as exc:
                    # The job's deadline, not this block: no later fill can start
                    # either. Stop drawing and let the assembler default the rest.
                    # Counted from the blocks actually emitted, not from ``idx``: a
                    # block_type with no manifest is skipped above and never lands in
                    # ``blocks``, so ``idx`` overstates how much of the page got filled.
                    filled = len(blocks)
                    blocks.append({"type": block_type, "content": {}})
                    blocks.extend(
                        {"type": bt, "content": {}}
                        for bt in block_types[idx + 1 :]
                        if get_block(bt) is not None
                    )
                    defaulted = len(blocks) - filled
                    logger.warning(
                        "Assembly fill stopped at block %s: %s; %d of %d blocks use defaults",
                        block_type,
                        exc,
                        defaulted,
                        len(blocks),
                    )
                    # The page still ships — that is the deliberate fallback — but it
                    # ships with empty default sections the tenant did not order. Record
                    # it where every other budget compromise is recorded, so the job
                    # completes DEGRADED rather than silently `succeeded`.
                    if run is not None:
                        run.degrade(
                            f"assembly: budget exhausted at block {filled + 1}/{len(blocks)} "
                            f"({exc}); {defaulted} block(s) shipped with default content"
                        )
                    break
                except Exception as exc:  # noqa: BLE001 — one block must not fail the page
                    logger.warning(
                        "Assembly fill for block %s failed (%s); using defaults",
                        block_type,
                        exc,
                    )
                    content = {}
                    # Twin of the ``BudgetExhausted`` handler above: the block still
                    # ships on its manifest defaults — that fallback is deliberate and
                    # unchanged — but an empty section the tenant did not order must not
                    # read as a clean run. One entry PER BLOCK: the loop is bounded by
                    # the structure phase's block list (a page plan), each entry names
                    # WHICH section is hollow, and ``generate_sections`` already records
                    # one degradation per failed section.
                    _note_hollow_block(run, block_type, exc)
            blocks.append({"type": block_type, "content": content})

        plan = parse_plan({"blocks": blocks})
        logger.info(
            "Assembly split plan: %d blocks (input=%d output=%d)", len(plan), in_tokens, out_tokens
        )
        if not plan:
            raise RuntimeError(
                f"WM worker {self.ASSEMBLY_WORKER_NAME} returned no usable page plan "
                "(assembly split mode)"
            )
        return plan, in_tokens, out_tokens

    async def generate_full_page(
        self,
        *,
        url: str,
        html: str,
        page_type: str | None = None,
        style: str = "modern",
        findings: list[dict] | None = None,
        brand: dict | None = None,
        site_analysis: dict | None = None,
        style_description: str | None = None,
        context: ResolvedContext | None = None,
        llm_config: LLMConfigLike | None = None,
        run: SectionRun | None = None,
    ) -> LLMResult:
        """Generate a complete redesigned HTML page via Claude API.

        Returns LLMResult with extracted HTML text and token usage.

        The ONE call runs under ``run.budget`` like every section call: it is refused
        when the job's deadline leaves less than ``MIN_CALL_SECONDS``, is cut when the
        deadline passes in flight, and counts toward the job's call total. Without
        this, the no-detectable-sections fallback of a late variant started a
        WM-timeout-long call after the job deadline. A refused/cut call has no draft to
        ship, so it is recorded in ``run.degradations`` and raised. Without a ``run`` a
        budget from settings bounds this call alone.
        """
        run = run if run is not None else SectionRun(budget=GenerationBudget.from_settings())
        # Greenfield: scope=landing with no source url/html → design FROM the brief, not
        # refine an existing page. Switches both the user prompt and the system prompt.
        greenfield = not (html or "").strip()
        manifest = extract_image_manifest(html, url)
        if not greenfield and not manifest:
            # Recorded, not inferred: a real page that yields zero <img src> is the
            # trigger for the placeholder-box failure mode (CSS backgrounds /
            # data-src lazy-loading). The prompt states it too (see
            # _format_image_manifest); this makes it countable in logs so the
            # frequency is measurable rather than anecdotal.
            logger.warning(
                "image manifest empty for url=%s (%d chars of source HTML) — page images are "
                "likely CSS backgrounds or lazy-loaded; prompt will warn against placeholders",
                url,
                len(html or ""),
            )
        prompt = self._build_full_page_prompt(
            url=url,
            html=html,
            page_type=page_type,
            style=style,
            findings=findings,
            brand=brand,
            site_analysis=site_analysis,
            style_description=style_description,
            context=context,
            image_manifest=manifest,
            greenfield=greenfield,
        )
        # A multi-minute call is work in flight: keep the batch reaper's clock moving.
        await report_progress()
        try:
            result = await self._budgeted_call(
                prompt,
                run=run,
                extra=False,
                llm_config=llm_config,
                greenfield=greenfield,
                max_tokens=self._max_tokens,
                section_mode=False,
            )
        except BudgetExhausted as exc:
            run.degrade(f"generate_full_page: {exc} — no page generated")
            logger.warning("generate_full_page: out of budget: %s", exc)
            raise RuntimeError(
                "generate_full_page: no page could be generated: " + " | ".join(run.degradations)
            ) from exc
        await report_progress()
        html_out = _require_html(
            _extract_html_block(result.text),
            "generate_full_page",
            completion_tokens=result.output_tokens,
            max_tokens=self._max_tokens,
        )
        # Only the per-section path wraps its bands in ``data-section``; without the
        # attribute the after-act screenshots fall back to anonymous ``band-NN``
        # stills and the script has no heading to ground on. Deterministic, not a
        # prompt hope (see section_landmarks).
        result.text = tag_data_sections(
            finalize_fragment(html_out, base_url=url, manifest=manifest)
        )
        result.quality_warnings = list(run.degradations)
        return result

    async def generate_sections(
        self,
        *,
        url: str,
        html: str,
        sections: list[dict],
        page_type: str | None = None,
        style: str = "modern",
        findings: list[dict] | None = None,
        brand: dict | None = None,
        site_analysis: dict | None = None,
        style_description: str | None = None,
        context: ResolvedContext | None = None,
        llm_config: LLMConfigLike | None = None,
        greenfield: bool = False,
        run: SectionRun | None = None,
    ) -> dict[str, str]:
        """Generate HTML for key sections of a page.

        ``greenfield=True`` means there is no source page: each section is AUTHORED
        from the brief (design-from-brief prompt + section-scoped greenfield system
        prompt) rather than refined from markup that does not exist.

        ``run`` carries this render's state (budget, section jobs for the rendered
        repair, tokens, degradations) — pass the same object to
        :meth:`repair_rendered_page`. Without one, a run bounded by the configured
        budget is created (and its state is then unreachable to the caller).

        A section the budget leaves no usable draft for is DROPPED and recorded in
        ``run.degradations``; only a render that yields no section at all raises.
        """
        run = run if run is not None else SectionRun(budget=GenerationBudget.from_settings())
        # Whole-page image manifest (absolutized) so each per-section call reuses real
        # image URLs even when its truncated markup slice dropped the <img> tag.
        manifest = extract_image_manifest(html, url)
        source_counts = _source_image_counts(html, url)
        # image url -> the section that already shows it (for cross-section dedupe).
        used_images: dict[str, str] = {}
        results: dict[str, str] = {}
        run.manifest = manifest
        for section in sections:
            section_type = section.get("section_type", "unknown")
            # Stored by a UNIQUE key: keyed by type, two generic ``section``s (or a
            # ``main`` and its hero) overwrote each other in this dict.
            key = _unique_section_key(section.get("key") or section_type, run.jobs)
            prompt = self._build_section_prompt(
                url=url,
                html=html,
                section=section,
                page_type=page_type,
                style=style,
                findings=findings,
                brand=brand,
                site_analysis=site_analysis,
                style_description=style_description,
                context=context,
                image_manifest=manifest,
                greenfield=greenfield,
            )
            taken = sorted(
                u for u in used_images if not _is_logo(u) and source_counts.get(u, 0) <= 1
            )
            if taken:
                prompt += (
                    "\n\n## Images already shown by earlier sections — do NOT reuse them here\n"
                    + "\n".join(f"- {u} (in {used_images[u]})" for u in taken)
                )
            heading = str(section.get("heading") or "").strip()
            label = f"{key} ({section_type}{f': {heading[:60]!r}' if heading else ''})"
            job = SectionJob(
                prompt=prompt,
                where=f"generate_sections[{label}]",
                url=url,
                manifest=manifest,
                llm_config=llm_config,
                greenfield=greenfield,
                source_image_counts=source_counts,
            )
            run.jobs[key] = job
            omitted = int(section.get("omitted_items") or 0)
            if omitted:
                run.degrade(
                    f"{job.where}: {omitted} source item(s) beyond the section split cap "
                    "were not generated (the section says so)"
                )
            section_html = await self._generate_one_section(
                prompt,
                where=job.where,
                url=url,
                manifest=manifest,
                llm_config=llm_config,
                greenfield=greenfield,
                run=run,
                used_images=used_images,
                source_image_counts=source_counts,
            )
            if section_html is not None:
                results[key] = section_html
                for src in _img_srcs(section_html):
                    used_images.setdefault(src, key)
            else:
                run.degrade(f"{job.where}: section DROPPED from the page (no usable draft)")
            # THE progress signal, and it has to be here rather than one level up.
            # This loop is sequential and ``sections`` is uncapped — it is
            # ``detect_sections``' landmark list — so a section-heavy
            # page spends N x the WM timeout inside this one call. Announcing only
            # when the whole page finishes made hours of healthy work
            # indistinguishable from a dead batch orchestrator, and the stale-batch
            # reaper duly reaped it. A no-op outside a batch page render.
            await report_progress()
        if sections and not results:
            raise RuntimeError(
                "generate_sections: no section could be generated: " + " | ".join(run.degradations)
            )
        return results

    async def _budgeted_call(
        self,
        prompt: str,
        *,
        run: SectionRun,
        extra: bool,
        llm_config: LLMConfigLike | None,
        greenfield: bool,
        max_tokens: int,
        section_mode: bool = True,
    ) -> LLMResult:
        """One LLM call under the job's budget, bounded by the time it has left.

        Raises :class:`BudgetExhausted` when the call may not start, or when the job
        deadline expires while it is in flight (the call is cancelled — a reply that
        lands after the caller stopped waiting is worth nothing).
        """
        async with self._budget_window(run.budget, extra=extra):
            return await self._call_claude(
                prompt,
                temperature=self._generation_temperature,
                llm_config=llm_config,
                greenfield=greenfield,
                section_mode=section_mode,
                max_tokens=max_tokens,
                run=run,
            )

    async def _section_call(
        self,
        prompt: str,
        *,
        run: SectionRun,
        extra: bool,
        llm_config: LLMConfigLike | None,
        greenfield: bool,
        max_tokens: int,
    ) -> LLMResult | list[str]:
        """One section draw: the reply, or the defects of a reply that never arrived.

        An empty/aborted reply (``_call_claude``'s own ``RuntimeError``) and a reply cut
        off at ``max_tokens`` are RETRYABLE defects of this draw, not job failures.
        ``LLM_BACKEND=vllm`` reports the cut-off as a 422 ``finish_reason=length``
        (w2v_common ``vllm_direct``); worker_management returned the cut-off text
        instead, which :func:`section_reply_defect` catches. Same defect, same remedy —
        retry with the "shorter reply" guidance — so one over-long draw can no longer
        escape the attempt loop and fail the whole job. :class:`BudgetExhausted` and
        every other error propagate.
        """
        try:
            return await self._budgeted_call(
                prompt,
                run=run,
                extra=extra,
                llm_config=llm_config,
                greenfield=greenfield,
                max_tokens=max_tokens,
            )
        except RuntimeError as exc:
            return [str(exc)]
        except httpx.HTTPStatusError as exc:
            if not _is_length_cutoff(exc):
                raise
            return [
                "unusable reply: reply reached the output-token ceiling (truncated, "
                f"finish_reason=length at max_tokens={max_tokens})"
            ]

    def _budget_window(
        self, budget: GenerationBudget, *, extra: bool
    ) -> AbstractAsyncContextManager[None]:
        """Account ONE LLM call and bound it by the time the job has left.

        The single enforcement seam for every dispatch path — the section/full-page
        calls (via :meth:`_budgeted_call`), the assembly plan draws and the component
        draw. Without it those last two ran under worker_management's own timeout
        (600s x 3 retries ~= 1803s) with no deadline check at all, so a job could pass
        its deadline mid-call and, for assembly, do it again on every refine re-run.

        Delegates to :func:`.section_run.budget_window`, the free-function form of
        this seam used outside :class:`HtmlGenerator` (accessibility scoring,
        refinement).

        Raises :class:`BudgetExhausted` when the call may not start, or when the job
        deadline expires while it is in flight (the call is cancelled).
        """
        return budget_window(budget, extra=extra)

    async def _generate_one_section(
        self,
        prompt: str,
        *,
        where: str,
        url: str,
        manifest: list[dict],
        llm_config: LLMConfigLike | None,
        greenfield: bool,
        run: SectionRun,
        used_images: dict[str, str] | None = None,
        source_image_counts: dict[str, int] | None = None,
        first_draft: bool = True,
    ) -> str | None:
        """One section, validated; retried up to ``SECTION_ATTEMPTS - 1`` times, then fails.

        Every attempt's defects are named in the next prompt. BLOCKING defects:

        * an unusable reply — empty, truncated (no closing tags, output ceiling spent),
          or degenerate (a repetition loop) — :func:`section_reply_defect`. Production
          run 01a0a045-602a died on ONE such draw (``--ct-reverse-reverse-…``) because
          the truncation check raised straight out of the job with no retry;
        * markup without CSS, or ``var(--x)`` nothing defines (:func:`css_defects`);
        * CSS that does not apply to its own markup — a dead block-root selector, or a
          custom property declared only on a selector that matches no ancestor of its
          users (:func:`analyze_section_css`).

        A SOFT defect — re-showing a photo an earlier section already shows — is
        retried too, but never fails the page: the best soft-only draw is kept and used
        if later draws are worse.

        THE BUDGET WINS OVER THE RETRY: when ``run.budget`` refuses a call (deadline or
        retry cap), the best COMPLETE draw so far ships — its remaining defects recorded
        in ``run.degradations`` — or ``None`` is returned when there is none. Only a
        section that exhausts every attempt WITHIN budget raises: that is a real defect,
        not a budget decision. ``first_draft=False`` (a repair regeneration) makes even
        the first call an extra one. Every extra call emits a progress heartbeat.
        """
        attempt_prompt = prompt
        defects: list[str] = []
        soft_fallback: str | None = None
        # A complete reply whose only defects are CSS ones: shippable when out of budget.
        usable_fallback: str | None = None
        budget = self._section_budget()
        for attempt in range(1, SECTION_ATTEMPTS + 1):
            extra = attempt > 1 or not first_draft
            if extra:
                # A retry/repair is work in flight: keep the batch reaper's clock moving.
                await report_progress()
            try:
                result = await self._section_call(
                    attempt_prompt,
                    run=run,
                    extra=extra,
                    llm_config=llm_config,
                    greenfield=greenfield,
                    max_tokens=budget,
                )
            except BudgetExhausted as exc:
                return _out_of_budget(run, where, exc, soft_fallback or usable_fallback, defects)
            if isinstance(result, list):
                defects = result
            else:
                draw = _judge_draw(
                    result,
                    max_tokens=budget,
                    url=url,
                    manifest=manifest,
                    used_images=used_images or {},
                    source_image_counts=source_image_counts or {},
                )
                if draw.clean:
                    return draw.html
                defects = draw.defects
                if draw.soft_only:
                    soft_fallback = soft_fallback or draw.html
                elif draw.html is not None:
                    usable_fallback = draw.html
            logger.warning(
                "%s attempt %d/%d rejected: %s",
                where,
                attempt,
                SECTION_ATTEMPTS,
                "; ".join(defects),
            )
            attempt_prompt = (
                f"{prompt}\n\n## Your previous reply was REJECTED — fix these and reply again\n"
                + "\n".join(f"- {d[:600]}" for d in defects)
                + _retry_guidance(defects)
            )
        if soft_fallback is not None:
            logger.warning("%s: using a draw that re-shows an earlier section's photo", where)
            return soft_fallback
        raise RuntimeError(
            f"{where}: section still unusable after {SECTION_ATTEMPTS} attempts: "
            f"{'; '.join(d[:600] for d in defects)}"
        )

    def _section_budget(self) -> int:
        """Per-section ``max_tokens``: the request, capped at the provider's real ceiling.

        Asking for 16000 when worker_management will clamp to ~12069 buys nothing and
        hides the real limit from the truncation check.
        """
        return min(self._max_tokens, self._output_ceiling)

    async def _probe_page(self, page: str, run: SectionRun) -> RenderedReport | None:
        """The rendered-page measurement — ADVISORY: a probe failure never fails a job."""
        left = run.budget.remaining()
        if left is not None and left < MIN_CALL_SECONDS:
            run.degrade("rendered contrast/logo checks skipped: generation deadline reached")
            return None
        try:
            async with asyncio.timeout(
                PROBE_TIMEOUT_S if left is None else min(left, PROBE_TIMEOUT_S)
            ):
                return await self._render_probe(page)
        except Exception as exc:  # noqa: BLE001 — a measurement must never cost the deliverable
            reason = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("rendered page probe failed (advisory, job continues): %s", reason)
            run.degrade(f"rendered contrast/logo checks could not run ({reason})")
            return None

    async def repair_rendered_page(
        self,
        sections: dict[str, str],
        *,
        compose: Callable[[dict[str, str]], str],
        run: SectionRun,
    ) -> tuple[dict[str, str], str]:
        """Render the composed page, fix what can be computed, regenerate what cannot.

        Two defects only exist after the cascade is resolved (see
        :mod:`.rendered_checks`). Cheapest fix first:

        1. a logo on a surface it cannot be read on is swapped/tinted deterministically;
        2. text below WCAG AA gets a computed colour (nearest page token that reaches AA,
           else the original darkened/lightened) and the page is re-measured;
        3. only text no colour change fixes regenerates its section with the measured
           failures named — ``RENDER_REPAIR_ROUNDS`` rounds, each call under the job
           budget.

        What is still failing after that DEGRADES the job — recorded in
        ``run.degradations`` — rather than failing it: a design with one low-contrast
        element beats no design. A probe error is advisory the same way.
        Returns ``(sections, composed_page)``.
        """
        results = dict(sections)
        page = compose(results)
        for round_no in range(RENDER_REPAIR_ROUNDS + 1):
            await report_progress()
            report = await self._probe_page(page, run)
            if report is None:
                return results, page
            results, logo_changes = fix_logo_surfaces(results, report, run.manifest)
            if logo_changes:
                logger.info("rendered page: %s", "; ".join(logo_changes))
                page = compose(results)
            results, page, failing = await self._recolour(results, page, report, compose, run)
            if failing is None or not failing:
                return results, page
            regenerated = 0
            if round_no < RENDER_REPAIR_ROUNDS:
                regenerated = await self._regenerate_for_contrast(
                    failing, results, run, round_no=round_no
                )
            if not regenerated:
                for key, problems in failing.items():
                    run.degrade(
                        f"rendered page: section {key} text below WCAG AA after {round_no} "
                        f"regeneration round(s): {'; '.join(problems[:3])}"
                    )
                return results, page
            page = compose(results)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _recolour(
        self,
        results: dict[str, str],
        page: str,
        report: RenderedReport,
        compose: Callable[[dict[str, str]], str],
        run: SectionRun,
    ) -> tuple[dict[str, str], str, dict[str, list[str]] | None]:
        """Apply the computed colour fix and re-measure.

        Returns ``(results, page, still_failing)``; ``still_failing`` is ``None`` when
        the re-measurement could not run (the recoloured page is kept: its colours were
        computed to pass).
        """
        failing = _failing_sections(report, results)
        if not failing:
            return results, page, failing
        recoloured, changes, _ = fix_contrast_css(results, report)
        if not changes:
            return results, page, failing
        logger.info("rendered page contrast recoloured: %s", "; ".join(changes))
        candidate = compose(recoloured)
        remeasured = await self._probe_page(candidate, run)
        if remeasured is None:
            return recoloured, candidate, None
        return recoloured, candidate, _failing_sections(remeasured, recoloured)

    async def _regenerate_for_contrast(
        self,
        failing: dict[str, list[str]],
        results: dict[str, str],
        run: SectionRun,
        *,
        round_no: int,
    ) -> int:
        """Regenerate each failing section (in place in ``results``); returns how many."""
        regenerated = 0
        for key, problems in failing.items():
            job = run.jobs.get(key)
            if job is None:
                continue
            logger.warning(
                "%s: rendered text contrast below WCAG AA (round %d): %s",
                job.where,
                round_no + 1,
                "; ".join(problems),
            )
            others = {
                src: other
                for other, fragment in results.items()
                if other != key
                for src in _img_srcs(fragment)
            }
            try:
                new = await self._generate_one_section(
                    job.prompt
                    + "\n\n## The RENDERED page measured text below WCAG AA contrast in this "
                    "section — fix every one\n"
                    + "\n".join(f"- {p}" for p in problems)
                    + _retry_guidance(["contrast"]),
                    where=job.where,
                    url=job.url,
                    manifest=job.manifest,
                    llm_config=job.llm_config,
                    greenfield=job.greenfield,
                    run=run,
                    used_images=others,
                    source_image_counts=job.source_image_counts,
                    first_draft=False,
                )
            except RuntimeError as exc:
                run.degrade(f"{job.where}: contrast regeneration failed, kept the draft ({exc})")
                continue
            if new is not None:
                results[key] = new
                regenerated += 1
        return regenerated

    async def generate_page_chunked(
        self,
        *,
        url: str,
        html: str,
        page_type: str | None = None,
        style: str = "modern",
        findings: list[dict] | None = None,
        brand: dict | None = None,
        site_analysis: dict | None = None,
        style_description: str | None = None,
        context: ResolvedContext | None = None,
        llm_config: LLMConfigLike | None = None,
        budget: GenerationBudget | None = None,
    ) -> LLMResult:
        """Generate a full page as bounded per-section chunks.

        ``budget`` is the JOB's generation budget (shared by a job's variants); without
        one a fresh budget from settings bounds this render alone.

        Detects the page's semantic sections and generates each one in its own
        small, sequential LLM call (each well under the worker's token cap and the
        provider read timeout), then composes them into a single page. This avoids
        the single ~16k-token generation whose wall-time exceeds the 600s timeout
        budget under vLLM concurrency, which surfaces as ``finish_reason='abort'``.

        GREENFIELD (no source ``html``) takes the same per-section path, but its
        outline comes from the BRIEF (:func:`plan_greenfield_sections`) instead of
        from landmark detection — there is no markup to detect landmarks in.
        ``detect_sections("")`` returns ``[]``, so greenfield used to fall through to
        the single ``generate_full_page`` completion, and that completion does not
        fit: worker_management clamps a self-hosted provider's output budget to a
        wall-clock time budget well below a full landing page, so EVERY greenfield
        run failed on ``does not end on a closed tag (truncated)``. Raising
        ``max_tokens`` cannot fix that; generating less per call can.

        Falls back to :meth:`generate_full_page` only when a page WITH markup yields
        no detectable sections (e.g. no semantic landmarks).
        """
        from src.services.context_resolver import detect_sections

        greenfield = not (html or "").strip()
        if greenfield:
            sections = plan_greenfield_sections(context=context, page_type=page_type)
            logger.info(
                "generate_page_chunked: greenfield for %s — %d sections planned from the "
                "brief (%s)",
                url,
                len(sections),
                ", ".join(s["section_type"] for s in sections),
            )
        else:
            sections = detect_sections(html)

        run = SectionRun(budget=budget or GenerationBudget.from_settings())
        if not sections:
            logger.info(
                "generate_page_chunked: no sections detected for %s; "
                "falling back to single full-page generation",
                url,
            )
            # Under the SAME job budget as the section path (see generate_full_page).
            return await self.generate_full_page(
                url=url,
                html=html,
                page_type=page_type,
                style=style,
                findings=findings,
                brand=brand,
                site_analysis=site_analysis,
                style_description=style_description,
                context=context,
                llm_config=llm_config,
                run=run,
            )

        section_results = await self.generate_sections(
            url=url,
            html=html,
            sections=sections,
            page_type=page_type,
            style=style,
            findings=findings,
            brand=brand,
            site_analysis=site_analysis,
            style_description=style_description,
            context=context,
            llm_config=llm_config,
            greenfield=greenfield,
            run=run,
        )
        section_results, composite = await self.repair_rendered_page(
            section_results,
            compose=lambda results: self.compose_section_page(
                results,
                style=style,
                brand=brand,
                context=context,
                html=html,
                base_url=url,
                run=run,
            ),
            run=run,
        )
        # Tokens and degradations of THIS render only (a SectionRun is per call).
        return LLMResult(
            text=composite,
            input_tokens=run.prompt_tokens,
            output_tokens=run.completion_tokens,
            quality_warnings=list(run.degradations),
        )

    async def generate_component(
        self,
        *,
        url: str,
        component_html: str,
        component_type: str,
        page_type: str | None = None,
        style: str = "modern",
        findings: list[dict] | None = None,
        brand: dict | None = None,
        site_analysis: dict | None = None,
        style_description: str | None = None,
        context: ResolvedContext | None = None,
        llm_config: LLMConfigLike | None = None,
        budget: GenerationBudget | None = None,
    ) -> LLMResult:
        """Generate a redesigned component. Returns LLMResult.

        The ONE call runs under the job's ``budget`` exactly like the section and
        full-page calls: refused when the deadline leaves less than
        ``MIN_CALL_SECONDS``, cut when the deadline passes in flight, and counted
        toward the job's call total. Without it this call ran under
        worker_management's own timeout only, so a ``hero``/``component`` job could
        carry on for ~1803s past its own deadline. Without a ``budget`` a fresh one
        from settings bounds this call alone.
        """
        run = SectionRun(budget=budget or GenerationBudget.from_settings())
        prompt = self._build_component_prompt(
            url=url,
            component_html=component_html,
            component_type=component_type,
            page_type=page_type,
            style=style,
            findings=findings,
            brand=brand,
            site_analysis=site_analysis,
            style_description=style_description,
            context=context,
        )
        result = await self._budgeted_call(
            prompt,
            run=run,
            extra=False,
            llm_config=llm_config,
            greenfield=False,
            max_tokens=self._max_tokens,
            section_mode=False,
        )
        # Same guard as the page paths: the component text is stored as
        # ``component.html`` AND reinjected into ``redesign.html`` unconditionally
        # (design_builder/phases.py:570-576), so a truncated component ships too.
        result.text = _require_html(_extract_html_block(result.text), "generate_component")
        return result

    async def refine_full_page(
        self,
        *,
        url: str,
        previous_html: str,
        feedback: str,
        page_type: str | None = None,
        style: str = "modern",
        brand: dict | None = None,
        site_analysis: dict | None = None,
        style_description: str | None = None,
        focus_areas: list[str] | None = None,
        llm_config: LLMConfigLike | None = None,
        budget: GenerationBudget | None = None,
    ) -> LLMResult:
        """Refine a previous redesign based on feedback. Returns LLMResult.

        The call runs under ``budget`` like every other LLM dispatch in this
        module (see ``_budget_window``): refused when the job's deadline leaves
        less than ``MIN_CALL_SECONDS``, cut when the deadline passes in flight.
        Without a ``budget`` this call is unbounded — callers that create a job
        (``process_refinement_job``) always pass one; direct/ad-hoc callers may
        not.
        """
        prompt = self._build_refinement_prompt(
            url=url,
            previous_html=previous_html,
            feedback=feedback,
            page_type=page_type,
            style=style,
            brand=brand,
            site_analysis=site_analysis,
            style_description=style_description,
            focus_areas=focus_areas,
        )
        if budget is not None:
            try:
                async with self._budget_window(budget, extra=False):
                    result = await self._call_claude(
                        prompt, temperature=self._refinement_temperature, llm_config=llm_config
                    )
            except BudgetExhausted as exc:
                logger.warning("refine_full_page: out of budget: %s", exc)
                raise
        else:
            result = await self._call_claude(
                prompt, temperature=self._refinement_temperature, llm_config=llm_config
            )
        # Same guard its siblings use: an unfenced reply comes back VERBATIM from
        # ``_extract_html_block``, and the refine lifecycle stores ``result.text``
        # as ``redesign.html`` unconditionally — so LLM prose became the page.
        result.text = _require_html(_extract_html_block(result.text), "refine_full_page")
        return result

    def compose_section_page(
        self,
        sections: dict[str, str],
        title: str | None = None,
        style: str = "modern",
        *,
        brand: dict | None = None,
        context: ResolvedContext | None = None,
        html: str | None = None,
        base_url: str | None = None,
        run: SectionRun | None = None,
    ) -> str:
        """Compose individual section HTML snippets into a full page.

        Emits ONE shared brand-token block (real colours + Google-Fonts links + the
        type/space scale) so the independently-generated sections — which reference
        ``var(--*)`` and never declare their own ``:root``/fonts — cohere visually.
        Never emits Tailwind, system-ui, or the #2563EB sentinel.

        The document takes the SOURCE page's ``lang`` and a title derived from the
        source ``<title>`` / brand name (never the raw URL). Every section is
        sanitized with one shared id set, so ids are unique across the page, and the
        composed CSS is validated scope-aware: a ``var(--x)`` nothing defines FOR THAT
        SECTION is recorded in ``run.degradations`` (see :meth:`_record_token_findings`)
        — the page still ships, with the defect visible to the user.
        """
        _, primary, secondary, palette_known = _resolve_palette(
            context=context, brand=brand, site_analysis=None
        )
        display_font = getattr(context, "display_font", None) if context else None
        body_font = getattr(context, "body_font", None) if context else None
        google_links = getattr(context, "google_font_links", None) if context else None
        if not (display_font or body_font) and brand:
            display_font = brand.get("display_font") or brand.get("font_family")
            body_font = brand.get("body_font")
        # Last resort: pull the real type pairing straight from the original markup.
        if not (display_font or body_font) and html:
            fonts = extract_fonts(html)
            display_font = display_font or fonts.get("display")
            body_font = body_font or fonts.get("body")
            google_links = google_links or fonts.get("google_links")

        source_palette = (getattr(context, "source_palette", None) if context else None) or (
            extract_palette(html) if html else []
        )
        tokens = _brand_tokens_css(
            primary,
            secondary,
            display_font=display_font,
            body_font=body_font,
            palette_known=palette_known,
            source_palette=source_palette,
            google_links=google_links,
        )
        google_head = _google_fonts_head(google_links, display_font, body_font)
        lang, page_title = _document_lang_and_title(title, context=context, brand=brand, html=html)
        seen_ids: set[str] = set()

        # Wrap each logical section in a real ``<section data-section>`` landmark
        # (not just a comment marker). render_section_screenshots selects capturable
        # bands with ``section, header, footer, [data-section]`` — a top-level wrapper
        # makes every section a full-width band with a meaningful, LLM-mappable name,
        # so the after-act gets one settled still per section instead of collapsing
        # onto the ~2 nested landmarks that happened to pass the full-width filter.
        section_html = "\n\n".join(
            f'<section data-section="{_attr_escape(name, quote=True)}" '
            f'aria-label="{_attr_escape(name, quote=True)}">\n'
            + finalize_fragment(
                _balance_section_fragment(content),
                base_url=base_url,
                seen_ids=seen_ids,
                # Section CSS is authored in isolation; scope it to its own band so a
                # reused source class (.stat-num, .btn) cannot restyle another section.
                scope_selector=f'[data-section="{_css_attr_value(name)}"]',
            )
            + "\n</section>"
            for name, content in sections.items()
        )
        page = f"""<!DOCTYPE html>
<html lang="{_attr_escape(lang, quote=True)}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_attr_escape(page_title, quote=False)}</title>
    {google_head}
    <style>
{tokens}
    </style>
</head>
<body>
{section_html}
</body>
</html>"""
        # The emptiness test must read the SECTIONS, not the composed document: the
        # document always carries <html>/<body> and one <section> wrapper per entry, so
        # asking whether the PAGE has markup can never be false — an unfalsifiable
        # guard, which is the same class of defect as the token gate this change fixes.
        if not any(has_content_markup(content) for content in sections.values()):
            raise RuntimeError(
                "compose_section_page: no section contributed any content — "
                f"{len(sections)} section(s), all empty"
            )
        self._record_token_findings(page, run)
        return page

    @staticmethod
    def _record_token_findings(page: str, run: SectionRun | None) -> None:
        """Record the composed page's undefined-token findings as DEGRADATIONS.

        THE BUDGET WINS OVER THE RETRY, and composition is downstream of the budget:
        by the time we are here every LLM call for the page is already spent and there
        is no retry left. ``_generate_section`` deliberately ships the best COMPLETE
        draw with its defects recorded (``usable_fallback`` exists precisely for a draw
        whose only defects are CSS defects, ``analyze_section_css`` unresolved
        properties included, and ``_out_of_budget`` logs "shipped a draft with
        unresolved defects"). Raising here would destroy that deliverable over a defect
        the section pipeline had already decided to ship — the tenant would get nothing
        instead of a page with one transparent band.

        So a finding degrades, it does not fail. A degradation is not advisory theatre:
        ``degraded`` is computed server-side as ``degradations OR notes`` and the
        frontend renders the codes, so the user sees "Completed with warnings" and the
        token name — the same path ``duration_out_of_tolerance`` surfaces on.

        The ONLY hard failure left is a page with no usable content at all, and that is
        checked by the caller against the SECTIONS — never against the composed
        document, which always has markup.
        """
        findings = css_token_findings(page, BRAND_TOKEN_NAMES)
        for severity, message in (
            (
                "error",
                "composed CSS references undefined custom properties without a fallback "
                "(they render with no value): ",
            ),
            (
                "warning",
                "composed CSS references undefined custom properties that rendered their "
                "fallback: ",
            ),
        ):
            tier = [f for f in findings if f.severity == severity]
            if not tier:
                continue
            # describe_tokens names the SECTION, not just the token: on a ten-band page
            # "--paper-warm" alone does not say which band went transparent.
            text = "compose_section_page: " + message + describe_tokens(tier)
            if run is not None:
                run.degrade(text)
            else:
                logger.warning(text)

    async def _call_claude(
        self,
        prompt: str,
        temperature: float | None = None,
        llm_config: LLMConfigLike | None = None,
        greenfield: bool = False,
        section_mode: bool = False,
        max_tokens: int | None = None,
        run: SectionRun | None = None,
    ) -> LLMResult:
        """Call WM and return ``LLMResult`` with token usage.

        ``temperature`` is accepted for legacy compatibility but ignored —
        WM controls sampling per-worker config. Per-call ``model_override``
        comes from ``llm_config.model`` when provided. ``greenfield`` swaps the
        refine-existing system prompt for the design-from-brief one; with
        ``section_mode`` it swaps again, to the SECTION-scoped greenfield prompt —
        the whole-page greenfield prompt orders every reply to start with
        ``<!DOCTYPE html>``, which would make each section its own document and
        nest N documents inside the composed page. ``section_mode`` alone (the
        existing refine-a-section path) changes nothing.
        """
        import time

        del temperature  # WM workers own sampling params.
        system_prompt = REDESIGN_SYSTEM_PROMPT
        if self._template_loader:
            if greenfield:
                template = (
                    "system_prompt_greenfield_section.md.j2"
                    if section_mode
                    else "system_prompt_greenfield.md.j2"
                )
            else:
                template = "system_prompt.md.j2"
            rendered = self._template_loader.render(template)
            if rendered:
                system_prompt = rendered

        model_override = model_override_from(llm_config)
        logger.info("LLM call worker=%s backend=%s", self.WORKER_NAME, _llm_backend(self._wm))
        t0 = time.monotonic()
        response = await self._wm.message(
            worker_name=self.WORKER_NAME,
            message=prompt,
            system_prompt=system_prompt,
            model_override=model_override,
            max_tokens=max_tokens or self._max_tokens,
        )
        elapsed = time.monotonic() - t0

        usage = response.usage_metrics or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        logger.info(
            "LLM call completed in %.2fs (input_tokens=%d, output_tokens=%d)",
            elapsed,
            input_tokens,
            output_tokens,
        )
        if run is not None:
            run.add_tokens(input_tokens, output_tokens)
        text = response.output if isinstance(response.output, str) else str(response.output)
        if response.output is None or not text.strip():
            logger.warning(
                "LLM worker=%s returned empty output after %.1fs (likely aborted/truncated)",
                self.WORKER_NAME,
                elapsed,
            )
            raise RuntimeError(
                f"LLM worker {self.WORKER_NAME} returned empty output "
                f"(likely aborted/truncated after {elapsed:.1f}s)"
            )
        return LLMResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _build_full_page_prompt(
        self,
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
        return build_full_page_prompt(
            self._template_loader,
            url=url,
            html=html,
            page_type=page_type,
            style=style,
            findings=findings,
            brand=brand,
            site_analysis=site_analysis,
            style_description=style_description,
            context=context,
            image_manifest=image_manifest,
            greenfield=greenfield,
        )

    def _build_section_prompt(
        self,
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
        """Build a section-level prompt (delegates to prompt_builder)."""
        return build_section_prompt(
            self._template_loader,
            url=url,
            html=html,
            section=section,
            page_type=page_type,
            style=style,
            findings=findings,
            brand=brand,
            site_analysis=site_analysis,
            style_description=style_description,
            context=context,
            image_manifest=image_manifest,
            greenfield=greenfield,
        )

    def _build_component_prompt(
        self,
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
        """Build a component-level redesign prompt (delegates to prompt_builder)."""
        return build_component_prompt(
            url=url,
            component_html=component_html,
            component_type=component_type,
            page_type=page_type,
            style=style,
            findings=findings,
            brand=brand,
            site_analysis=site_analysis,
            style_description=style_description,
            context=context,
        )

    def _build_refinement_prompt(
        self,
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
        """Build a refinement prompt with previous HTML and feedback (delegates)."""
        return build_refinement_prompt(
            self._template_loader,
            url=url,
            previous_html=previous_html,
            feedback=feedback,
            page_type=page_type,
            style=style,
            brand=brand,
            site_analysis=site_analysis,
            style_description=style_description,
            focus_areas=focus_areas,
        )
