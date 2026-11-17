"""Deterministic page assembler — turns a block plan into self-contained HTML.

Given a *page plan* (an ordered list of ``{type, content}`` chosen by the LLM, or
built by hand in tests) this composes a complete, standalone HTML document:

* every block is rendered from its vetted Jinja partial through an **autoescaping**
  environment, so LLM/user content can never inject markup (XSS-safe by construction);
* the compiled DaisyUI/Tailwind CSS and (for interactive pages) htmx/Alpine are
  **inlined**, so the artifact renders identically under Playwright, in a browser
  live-preview iframe, and as a download — no external asset fetch, no CSP surprises;
* brand tokens are mapped to a DaisyUI theme deterministically (see ``theme.py``);
* ``embeds`` (the three.js seam) are injected as **sandboxed iframes** — the only
  external references a self-contained artifact carries.

The LLM never emits HTML/CSS here; it only selects blocks and fills text slots.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from w2v_common.net import SSRFError, validate_public_url

from src.config import get_settings
from src.services.catalog import integrity, registry
from src.services.catalog.icons import icon_svg
from src.services.catalog.theme import build_theme_css, resolve_font

logger = logging.getLogger(__name__)

_V2_DIR = Path(__file__).resolve().parent.parent.parent / "templates" / "web_design" / "v2"


@dataclass
class AssembledPage:
    """Result of assembling a plan."""

    html: str
    blocks_used: list[str]
    embeds_injected: int
    integrity_report: integrity.IntegrityReport | None = None


def _safe_url(value: Any) -> str:
    """Neutralise non-http(s)/mail/tel/relative URLs (defends against javascript: hrefs).

    Autoescape handles quoting; this handles the *scheme* — a ``javascript:`` or
    ``data:`` href survives escaping and would still execute, so it is dropped to
    ``#``.
    """
    if not value:
        return "#"
    v = str(value).strip()
    low = v.lower()
    if low.startswith(("http://", "https://", "mailto:", "tel:", "#", "/", "./", "../")):
        return v
    # A bare relative path ("about", "pricing.html") has no scheme before the
    # first slash → safe. Anything else with a scheme is rejected.
    first = v.split("/", 1)[0]
    return v if ":" not in first else "#"


# The only data: images an <img> may carry: base64 raster the service encoded
# itself (integrity.enforce_images has already required an exact own-value match).
_DATA_IMAGE_RE = re.compile(r"data:image/(?:jpeg|png|webp);base64,[A-Za-z0-9+/]+={0,2}")


def _safe_img(value: object) -> str:
    """``src`` for an ``<img>``: an http(s) URL or our own base64 raster, else empty."""
    v = str(value or "").strip()
    if v.lower().startswith(("http://", "https://")):
        return v
    return v if _DATA_IMAGE_RE.fullmatch(v) else ""


@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    """Autoescaping Jinja env for block partials + page shell.

    Autoescape is ON (unlike the prompt-template loader) because these render
    untrusted LLM/user content into HTML.
    """
    env = Environment(
        loader=FileSystemLoader([str(_V2_DIR / "blocks"), str(_V2_DIR)]),
        autoescape=select_autoescape(default=True, default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["safe_url"] = _safe_url
    env.filters["safe_img"] = _safe_img
    # Icons are code-built inline SVG — never a font ligature rendered as text.
    env.filters["icon"] = icon_svg
    return env


def _vendor_dir() -> Path:
    settings = get_settings()
    if settings.catalog.vendor_dir:
        return Path(settings.catalog.vendor_dir)
    return Path(__file__).resolve().parent.parent.parent / "static" / "vendor"


def _read_vendor(name: str) -> str:
    path = _vendor_dir() / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Vendored asset %s unavailable: %s", name, exc)
        return ""


def _coerce_content(manifest: registry.BlockManifest, raw: Any) -> dict[str, Any]:
    """Merge caller content over manifest defaults, dropping unknown slots.

    List slots are coerced to lists of dicts keeping only the declared item slots.
    Every leaf value stays a plain str/list — escaping happens in Jinja at render.
    """
    content = manifest.defaults()
    if not isinstance(raw, dict):
        return content
    for name, spec in manifest.slots.items():
        if name not in raw or raw[name] is None:
            continue
        value = raw[name]
        if spec.type == "list":
            items: list[dict[str, Any]] = []
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    items.append(
                        {k: item.get(k) for k in spec.item_slots if item.get(k) is not None}
                    )
            content[name] = items
        else:
            content[name] = value
    return content


def _render_block(
    env: Environment,
    manifest: registry.BlockManifest,
    content: dict,
    *,
    section_id: str = "",
    brand: str = "",
) -> str:
    template_name = f"{manifest.key}.html.j2"
    try:
        template = env.get_template(template_name)
    except Exception as exc:  # noqa: BLE001 — a missing/broken partial must not 500 the job
        logger.warning("Block partial %s failed to load: %s", template_name, exc)
        return ""
    return template.render(block=content, meta=manifest, section_id=section_id, brand=brand)


def _page_title(title: str, brand: str, resolved: list[tuple[registry.BlockManifest, dict]]) -> str:
    """The document <title>: caller's, else the brand, else the hero headline.

    Never a page-type slug: live run 01a156a6 shipped ``<title>landing</title>``
    because the caller fell back to ``job.page_type`` when the brand had no name.
    """
    if title.strip():
        return title.strip()
    headline = next(
        (str(c.get("headline") or "").strip() for m, c in resolved if m.key == "hero"), ""
    )
    if brand and headline:
        return f"{brand} — {headline}"
    return brand or headline or "Web design"


def _embed_iframe(embed: dict) -> str:
    """Render one embed as a sandboxed iframe (all values escaped)."""
    url = escape(str(embed.get("artifact_url", "")), quote=True)
    title = escape(str(embed.get("title") or "Embedded interactive artifact"), quote=True)
    ratio = escape(str(embed.get("aspect_ratio") or "16 / 9"), quote=True)
    sandbox = ' sandbox="allow-scripts"' if embed.get("sandbox", True) else ""
    return (
        '<section class="w-full py-8 px-4 flex justify-center bg-base-200">'
        f'<iframe src="{url}" title="{title}" loading="lazy"{sandbox} '
        'referrerpolicy="no-referrer" '
        f'style="width:100%;max-width:64rem;aspect-ratio:{ratio};border:0;'
        'border-radius:var(--radius-box,1rem)"></iframe></section>'
    )


def _normalize_plan(plan: list[dict]) -> list[tuple[registry.BlockManifest, dict]]:
    """Resolve plan entries to (manifest, content), skipping unknown block types.

    Singletons are de-duplicated (first wins) and hoisted: navbar to the front,
    footer to the back, so the page structure is always well-formed regardless of
    the order the LLM emitted.
    """
    max_blocks = get_settings().catalog.max_blocks
    resolved: list[tuple[registry.BlockManifest, dict]] = []
    seen_singletons: set[str] = set()
    for entry in plan:
        if len(resolved) >= max_blocks:
            break
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("type") or entry.get("block") or "").strip()
        manifest = registry.get_block(key)
        if manifest is None:
            logger.debug("Skipping unknown block type in plan: %r", key)
            continue
        if manifest.singleton:
            if key in seen_singletons:
                continue
            seen_singletons.add(key)
        resolved.append((manifest, _coerce_content(manifest, entry.get("content"))))

    def _rank(item: tuple[registry.BlockManifest, dict]) -> int:
        cat = item[0].category
        return 0 if cat == "nav" else (2 if cat == "footer" else 1)

    return sorted(resolved, key=_rank)


def _log_integrity(report: integrity.IntegrityReport) -> None:
    """Log what the integrity pass changed (one line per kind of change)."""
    if report.hours_dropped:
        logger.warning(
            "Assembly integrity: dropped %d opening-hours row(s)/FAQ answer(s) not stated "
            "in the brief",
            report.hours_dropped,
        )
    if report.testimonials_dropped or report.testimonial_blocks_removed:
        logger.warning(
            "Assembly integrity: dropped %d testimonial(s) not quoted in the brief; "
            "removed %d testimonials block(s) left empty",
            report.testimonials_dropped,
            report.testimonial_blocks_removed,
        )
    if report.brand_aliases or report.links_rewritten or report.links_removed:
        logger.info(
            "Assembly integrity: brand=%r aliases=%s links_rewritten=%d links_removed=%d "
            "images_dropped=%d",
            report.brand,
            report.brand_aliases,
            report.links_rewritten,
            report.links_removed,
            report.images_dropped,
        )
    if report.images_dropped:
        logger.warning(
            "Assembly integrity: dropped %d unverified external image(s); "
            "blocks render their designed fallback",
            report.images_dropped,
        )


def assemble_page(
    plan: list[dict],
    *,
    brand: dict | None = None,
    title: str = "",
    interactive: bool = False,
    embeds: list[dict] | None = None,
    lang: str = "en",
    brief: str = "",
    own_images: list[str] | None = None,
) -> AssembledPage:
    """Assemble a plan into a complete, self-contained HTML document.

    ``brief`` is the design brief text; it is read for a contact email/phone (so
    "Contact"/"Order" links become real ``mailto:``/``tel:`` targets) and as the
    ONLY source of opening hours (``integrity.enforce_hours``) and of testimonials
    (``integrity.enforce_testimonials``).

    ``own_images`` are the exact image values this service produced for the page
    (the inlined generated hero); they are the only ``data:`` images kept.
    """
    env = _jinja_env()
    settings = get_settings()
    resolved = _normalize_plan(plan)

    # Integrity pass (see ``integrity.py``): one brand, no dead links, no
    # unverified image hotlinks — enforced in code, not left to the LLM.
    report = integrity.IntegrityReport()
    brand_name = integrity.enforce_brand(resolved, brand, report)
    trusted_images = integrity.TrustedImages(
        hosts=integrity.trusted_image_hosts(settings.catalog.image_host_allowlist),
        own=frozenset(v for v in (own_images or []) if v),
    )
    integrity.enforce_images(resolved, trusted_images, report)
    integrity.enforce_hours(resolved, brief, report)
    integrity.enforce_testimonials(resolved, brief, report)
    section_ids = integrity.assign_section_ids([m.key for m, _ in resolved])
    integrity.enforce_links(resolved, section_ids, integrity.extract_contact(brief), report)
    _log_integrity(report)

    rendered: list[tuple[str, str]] = []  # (block_key, html)
    for (manifest, content), section_id in zip(resolved, section_ids, strict=True):
        html = _render_block(env, manifest, content, section_id=section_id, brand=brand_name)
        if html.strip():
            rendered.append((manifest.key, html))

    # Inject embeds. An embed targets a block key via ``mount``; it is inserted
    # immediately after that block, else after the hero, else at the top.
    embeds = embeds or []
    injected = 0
    for embed in embeds:
        if not isinstance(embed, dict) or not embed.get("artifact_url"):
            continue
        # Sink-side SSRF guard mirroring the `EmbedSpec.artifact_url` admission
        # check. `embeds` is read back from the `redesign_jobs.embeds` JSONB
        # column at render time, so schema validation does not cover this call —
        # a row written before that validator existed still reaches here. The
        # assembled page is rasterised by our own in-cluster Chromium, which
        # loads the iframe, so an internal target's pixels would be handed to
        # the tenant. Dropping the embed (the existing skip path) is the right
        # failure mode: a bad artifact must not fail an otherwise-good design.
        try:
            validate_public_url(str(embed["artifact_url"]))
        except (SSRFError, ValueError) as exc:
            logger.warning("Dropping embed with non-public artifact_url: %s", exc)
            continue
        frame = _embed_iframe(embed)
        mount = str(embed.get("mount") or "hero").strip()
        idx = next((i for i, (k, _) in enumerate(rendered) if k == mount), None)
        if idx is None:
            idx = next((i for i, (k, _) in enumerate(rendered) if k == "hero"), -1)
        rendered.insert(idx + 1, (f"embed:{embed.get('type', 'iframe')}", frame))
        injected += 1

    body = "\n".join(html for _, html in rendered)

    _family, google_font = resolve_font(brand)
    shell = env.get_template("page.html.j2")
    doc = shell.render(
        lang=lang,
        title=_page_title(title, brand_name, resolved),
        theme_css=build_theme_css(brand),  # marked |safe in the shell (we built it)
        base_theme=settings.catalog.default_theme,
        app_css=_read_vendor("app.css"),
        htmx_js=_read_vendor("htmx.min.js") if interactive else "",
        alpine_js=_read_vendor("alpine.min.js") if interactive else "",
        google_font=google_font,
        body=body,  # already-escaped rendered block HTML — marked |safe in the shell
    )
    return AssembledPage(
        html=doc,
        blocks_used=[k for k, _ in rendered if not k.startswith("embed:")],
        embeds_injected=injected,
        integrity_report=report,
    )
