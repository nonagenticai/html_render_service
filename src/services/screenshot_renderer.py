"""Render redesigned HTML files in Playwright and capture screenshots."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict
from urllib.parse import urlparse

from w2v_common.net import SSRFError, validate_public_url

from src.services.navigation_guard import require_ok_navigation

if TYPE_CHECKING:  # pragma: no cover — keeps the playwright import out of runtime scope
    from playwright.async_api import ElementHandle, FloatRect, Page, Route

logger = logging.getLogger(__name__)

# CSS-pixel viewports. ``desktop`` is the *delivery* viewport and is deliberately
# 1920x1080: every still captured here is cover-fit to a 1920x1080 video frame
# downstream, so a 1440x900 capture was upscaled 1.33x — the soft picture the
# owner reported. ``tablet``/``mobile`` are real device CSS sizes and MUST NOT
# change; they exist to exercise responsive breakpoints, not to fill a video frame.
VIEWPORT_SIZES: dict[str, dict[str, int]] = {
    "desktop": {"width": 1920, "height": 1080},
    "tablet": {"width": 768, "height": 1024},
    "mobile": {"width": 390, "height": 844},
}

# Supersampling factor for every *captured* page. A 1920x1080 CSS viewport at
# ``device_scale_factor=2`` rasterises at 3840x2160 and is resampled down to the
# 1920x1080 delivery frame, so text/vector edges land on the delivery grid with
# real detail behind them instead of being interpolated up from a smaller bitmap.
#
# This applies to the CSS -> bitmap step ONLY. Everything Playwright reports in
# CSS pixels (``bounding_box``, ``scrollHeight``, ``clip``) is unaffected, so the
# geometry thresholds below and the full-page height cap stay in CSS px. What it
# DOES multiply is bitmap bytes: 4x per CSS pixel. See ``_capture`` for the
# worst-case arithmetic against the pod's 4Gi limit.
#
# The tablet/mobile viewports get it too — a 2x DPR is what those devices
# actually have, so 1x was under-sampling them as well.
CAPTURE_DEVICE_SCALE_FACTOR = 2

# Passed to EVERY page/element ``screenshot()`` in this module. Capture fires
# immediately after ``networkidle``, with no settle wait, so a page with a hero
# fade-in or a slide-up reveal was captured mid-transition — a nondeterministic,
# half-opacity frame shipped into the video. ``animations="disabled"`` fast-
# forwards CSS animations and transitions to their finished state and freezes
# infinite ones at the first frame; ``caret="hide"`` suppresses the blinking text
# caret, which otherwise lands in the frame or not depending on timing.
_SHOT_STABILITY: dict[str, str] = {"animations": "disabled", "caret": "hide"}


class _ViewportSize(TypedDict):
    width: int
    height: int


class _NewPageArgs(TypedDict):
    """Keyword args accepted by Playwright's ``browser.new_page()`` that this
    module actually sets — not the full Playwright kwarg surface."""

    viewport: _ViewportSize
    device_scale_factor: int


def _page_args(size: dict[str, int]) -> _NewPageArgs:
    """Keyword args for ``browser.new_page()`` for a capture page.

    Single source of truth for viewport + supersampling, so a new capture site
    cannot silently open a 1x page (which is how ``device_scale_factor`` came to
    be set in exactly zero places across this repo).
    """
    return {
        "viewport": {"width": size["width"], "height": size["height"]},
        "device_scale_factor": CAPTURE_DEVICE_SCALE_FACTOR,
    }


def validate_viewport_names(names: list[str] | None) -> list[str] | None:
    """Reject viewport names the renderer cannot render.

    ``_render`` silently ``continue``s past an unknown name, so ``["Desktop"]``
    (wrong case) produced ZERO screenshots and the job still completed. The name
    set is closed and tiny — validate it at the boundary instead.
    """
    if names is None:
        return None
    unknown = [n for n in names if n not in VIEWPORT_SIZES]
    if unknown:
        raise ValueError(
            f"unknown screenshot viewport(s) {unknown!r}; supported: {sorted(VIEWPORT_SIZES)}"
        )
    return names


# Landmark elements that make good discrete redesign-section stills. ``main`` and
# ``body`` are deliberately excluded — they wrap the whole page, which is exactly
# the tall top-to-bottom scroll the per-section capture exists to replace.
_SECTION_SELECTOR = "section, header, footer, [data-section], [data-redesign-section]"
# A section worth a full 16:9 still must be a real full-width page band, not a
# nested fragment. The landmark selector matches tiny inner elements too — a
# logo-only <header> (323x107), a chat-card <section> (400x397), a thin spacer
# band (1440x172) — and cover-fitting any of those to 1920x1080 yields a blank/
# blurry frame (live E2E 019ea720, 2026-06-08). Keep only elements that are
# ~full content width AND tall enough to compose; everything else is dropped and,
# if nothing qualifies, we fall back to evenly-sliced viewport bands.
#
# Both thresholds are in CSS pixels, matched against ``bounding_box()`` which is
# also CSS px — ``device_scale_factor`` does NOT enter here. What DOES enter is
# the viewport WIDTH, because the real question either threshold asks is "is this
# band close enough to 16:9 that padding it does not swamp the content?", and
# that is a question about the band's ASPECT. So the height floor is derived from
# the desktop width rather than carried over as a literal: at 1440 the tuned
# value was 300 (a 4.8:1 band, padded to 16:9 => 63% letterbox), and holding that
# same aspect at 1920 gives 400. Hardcoding 300 at 1920 would have quietly
# admitted 6.4:1 slivers that pad to 72% letterbox.
_MIN_SECTION_HEIGHT_FRAC_OF_WIDTH = 300 / 1440
_MIN_SECTION_HEIGHT_PX = round(
    VIEWPORT_SIZES["desktop"]["width"] * _MIN_SECTION_HEIGHT_FRAC_OF_WIDTH
)
# Already relative (the caller multiplies it by the live viewport width), so it
# needs no re-derivation for 1920 — it means the same thing at any width.
_MIN_SECTION_WIDTH_FRAC = 0.6
# Two captured bands whose vertical spans overlap this much are the same content
# (a full-width parent section and its full-width child) — keep the first, skip
# the nested duplicate.
_SECTION_VERTICAL_OVERLAP_FRAC = 0.7
# Second-pass selector, used only when NO landmark clears the height floor. A thin
# page (an example.com redesign) has full-width ``data-section`` wrappers that are
# all short — the generator emits them, they are just under 400px. The height floor
# exists to reject anonymous slivers; an explicitly named, full-width section is a
# real section, and ``_capture_section`` pads it up to 16:9. Top-level only, so a
# nested named fragment cannot shadow its parent. Dropping these used to send the
# page to anonymous ``band-NN`` stills and starve the after-act script.
_TOP_LEVEL_DATA_SECTION_SELECTOR = "[data-section]:not([data-section] [data-section])"
# Height floor for that second pass: low enough to keep a short named section,
# high enough to reject a decorative full-width sliver (a divider, a hairline rule).
_MIN_NAMED_SECTION_HEIGHT_PX = 48
# Width floor for that second pass. A single-call full-page redesign lays its
# content out as a centred max-width column (~832 of 1920 CSS px = 43%, round 4
# example.com runs) between a full-width header and footer, so its named content
# blocks fail the 0.6 band floor and the page went to ``band-NN`` again. A named
# block that wide is a real section; it is accepted only when it is no taller than
# 16:9 of its own width, because then ``_pad_to_ratio`` PADS it (letterbox, no
# crop). A narrow TALL block would be cover-fit-cropped into an over-zoomed frame
# — that is still rejected and goes to (named) bands.
_MIN_NAMED_SECTION_WIDTH_FRAC = 0.35
# The ``data-section`` under a band's midpoint — names a fallback viewport band.
_BAND_LABEL_JS = """
([x, y]) => {
  const hit = document.elementFromPoint(x, y);
  const sec = hit ? hit.closest('[data-section]') : null;
  return sec ? sec.getAttribute('data-section') || '' : '';
}
"""
# Read an element's identity in one round-trip so the Python side can derive a
# stable, human-meaningful section name the orchestrator's LLM can map narration to.
_SECTION_LABEL_JS = """
(el) => {
  const heading = el.querySelector('h1,h2,h3,h4,h5,h6');
  return {
    id: el.id || '',
    tag: el.tagName.toLowerCase(),
    heading: heading ? heading.textContent.trim() : '',
    dataLabel: el.getAttribute('data-section') || el.getAttribute('aria-label') || '',
    bg: getComputedStyle(el).backgroundColor || '',
    bodyBg: getComputedStyle(document.body).backgroundColor || '',
  };
}
"""
# A short, full-width section band screenshots as a short, wide PNG. The video
# compile cover-fits it to 16:9, which scale-to-fills + crops — a ~2x over-zoom
# that cut the contact heading to "…t's talk" and turned the footer into
# whitespace (live E2E 019ea8b6, 2026-06-08). Pad the PNG up to 16:9 at capture
# time, centered, with the section's own background colour, so cover-fit stays
# ~1:1 with no crop and no bleed of the next section. Tall bands (>= 16:9) are
# left untouched — cover already frames them full-bleed.
_SECTION_TARGET_RATIO_W = 16
_SECTION_TARGET_RATIO_H = 9
# The source site can wrap hero/projects/writing in ONE landmark with no inner
# <section>s (a SPA's <main> of plain divs), so the redesign emits one greedy
# band that swallows every topic — the after-act video then has nothing per-topic
# to key narration to. When an accepted band is this many times taller than 16:9
# we split it on its inner top-level (h1/h2) heading offsets into per-topic stills
# (see _split_section_png). Bands at or below this stay one still (no regression).
_SECTION_SPLIT_MIN_RATIO = 1.6
# Read the inner top-level heading offsets WITHIN a band in one round-trip. Each
# offset is the heading's top relative to the band's OWN top — measuring both in
# the same getComputedStyle/getBoundingClientRect pass cancels scroll, so this is
# the band-local y the captured PNG is in (times the device scale, derived Python-
# side from the PNG width). Only h1/h2 = section-level titles, not card-level h3s, so a
# project grid doesn't shatter into per-card slivers.
_SECTION_HEADINGS_JS = """
(el) => {
  const r = el.getBoundingClientRect();
  return [...el.querySelectorAll('h1, h2')].map((h) => ({
    topCss: h.getBoundingClientRect().top - r.top,
    text: (h.textContent || '').trim(),
  }));
}
"""

# Browsers come from the process-wide pool (src/services/browser_pool.py): a
# bounded set of long-lived Chromium instances reused across renders. The pool's
# bounded size IS the concurrency gate (it replaced the old render semaphore), and
# it relaunches a browser after N leases to bound the memory ratchet.


async def _close_page_quietly(page: Page) -> None:
    """Close a Playwright page from a ``finally`` without losing it or the real error.

    Every render path below leases a browser from the pool and opens its own page.
    ``BrowserPool.lease()`` closes **nothing** on exit — it only re-queues the slot,
    and the slot's browser is relaunched once every ``browser_max_renders`` (200)
    leases — so a page this module fails to close stays alive for up to 200 renders.
    That is the multi-GB OOMKill path.

    Two failure modes, both fixed here, and neither is optional:

    * **A bare ``await page.close()`` masks the real error.** Playwright raises when
      the target is already gone (``TargetClosedError``, or a plain ``Error`` on a
      dead driver connection); raised from a ``finally`` that replaces whatever
      exception was actually propagating. ``suppress(Exception)`` keeps the caller's
      cause intact, and deliberately does **not** suppress ``BaseException`` — a
      cancelled render must still cancel, never degrade into a normal return.
    * **A bare await leaks the page under cancellation.** ``close()`` is a driver
      round-trip, so it suspends. In a cancelled scope (SIGTERM drain, job
      cancellation) the ``CancelledError`` is re-raised *at that suspension*, before
      Playwright has torn the page down, so the page leaks exactly as if the
      ``finally`` were not there. ``asyncio.shield`` detaches the close onto its own
      task so the teardown completes anyway, while the ``CancelledError`` from the
      shielded await still propagates.

    Do **not** "simplify" this back to ``await page.close()``.
    """
    with contextlib.suppress(Exception):
        await asyncio.shield(page.close())


def _parse_css_color(value: str | None) -> tuple[int, int, int] | None:
    """Parse a computed CSS ``rgb()/rgba()`` colour into an opaque ``(r,g,b)``.

    Returns ``None`` for empty, ``transparent``/``none``, an unparseable value, or
    a fully transparent colour (alpha ``0``) — the caller falls through to the next
    candidate. Chromium's ``getComputedStyle`` always reports rgb/rgba, so we only
    handle those two forms.
    """
    if not value:
        return None
    v = value.strip().lower()
    if v in ("transparent", "none"):
        return None
    m = re.match(r"rgba?\(([^)]+)\)", v)
    if not m:
        return None
    parts = [p.strip() for p in re.split(r"[,\s/]+", m.group(1)) if p.strip()]
    if len(parts) < 3:
        return None
    try:
        r, g, b = (round(float(parts[i])) for i in range(3))
        alpha = float(parts[3]) if len(parts) >= 4 else 1.0
    except ValueError:
        return None
    if alpha <= 0.0:
        return None
    clamp = lambda c: max(0, min(255, c))  # noqa: E731
    return (clamp(r), clamp(g), clamp(b))


def _resolve_bg(*candidates: str | None) -> tuple[int, int, int]:
    """First opaque candidate colour, else white — the letterbox pad colour."""
    for candidate in candidates:
        rgb = _parse_css_color(candidate)
        if rgb is not None:
            return rgb
    return (255, 255, 255)


def _pad_to_ratio(path: Path, bg: tuple[int, int, int]) -> None:
    """Pad a short section PNG up to 16:9, centered, with ``bg``.

    Computes the target height from the image's OWN width so the result is exactly
    16:9 regardless of capture-width quirks. No-op when the image is already at
    least 16:9 tall (a real-height/tall band) — those cover-fit cleanly already.
    Runs off the event loop (see caller) per the no-PIL-on-the-loop convention.
    """
    from PIL import Image

    with Image.open(path) as img:
        width, height = img.size
        target_h = round(width * _SECTION_TARGET_RATIO_H / _SECTION_TARGET_RATIO_W)
        if height >= target_h:
            return
        canvas = Image.new("RGB", (width, target_h), bg)
        canvas.paste(img.convert("RGB"), (0, (target_h - height) // 2))
        canvas.save(path)


def _plan_subbands(
    heads: Any, png_w: int, png_h: int, css_w: float, used: set[str]
) -> list[tuple[int, str]]:
    """Plan per-heading crop bands for a tall section PNG.

    ``heads`` is the ``_SECTION_HEADINGS_JS`` payload — an ordered list of
    ``{topCss, text}`` heading offsets from the band's top (CSS px). Returns an
    ordered ``[(start_y_px, unique_name)]`` of crop boundaries: the first band
    always starts at ``0`` (the lead/hero block, named by the first heading) and
    each later band starts at a heading whose gap to the previous start AND to the
    image bottom are both >= ``_MIN_SECTION_HEIGHT_PX`` (so no slivers). Returns
    ``[]`` when fewer than two clean bands result — the caller keeps the single
    band. Names share ``used`` so every emitted still is globally unique (a
    downstream dict-collapse silently drops duplicate names).
    """
    if not isinstance(heads, list) or not heads or css_w <= 0:
        return []
    scale = png_w / css_w
    min_px = round(_MIN_SECTION_HEIGHT_PX * scale)
    tops: list[tuple[int, str]] = []
    for h in heads:
        if not isinstance(h, dict):
            continue
        try:
            top = round(float(h.get("topCss") or 0) * scale)
        except TypeError, ValueError:
            continue
        tops.append((max(0, min(top, png_h)), str(h.get("text") or "")))
    if not tops:
        return []
    tops.sort(key=lambda t: t[0])
    # First band is the lead block [0, next cut), named by the first heading even
    # if it sits a few px below the band top (default heading margin).
    bands: list[tuple[int, str]] = [(0, tops[0][1])]
    prev = 0
    for top, text in tops[1:]:
        if top - prev >= min_px and png_h - top >= min_px:
            bands.append((top, text))
            prev = top
    if len(bands) < 2:
        return []
    return [
        (start, _section_name({"heading": text}, i, used)) for i, (start, text) in enumerate(bands)
    ]


def _split_section_png(
    path: Path, heads: Any, css_w: float, bg: tuple[int, int, int], used: set[str]
) -> list[tuple[str, str]]:
    """Crop a tall section PNG into per-heading sub-bands, each padded to >= 16:9.

    Returns ordered ``(name, png_path)`` pairs, or ``[]`` when the band yields
    fewer than two clean sub-bands (caller keeps it whole). Pure/synchronous —
    the caller runs it off the event loop (no-PIL-on-the-loop convention).
    """
    from PIL import Image

    with Image.open(path) as src:
        img = src.convert("RGB")
        png_w, png_h = img.size
        bands = _plan_subbands(heads, png_w, png_h, css_w, used)
        if not bands:
            return []
        starts = [start for start, _ in bands]
        out: list[tuple[str, str]] = []
        for i, (start, name) in enumerate(bands):
            end = starts[i + 1] if i + 1 < len(starts) else png_h
            top = max(0, min(start, png_h))
            bottom = max(top, min(end, png_h))
            if bottom <= top:
                continue
            subpath = path.with_name(f"{path.stem}_{i:02d}_{name}.png")
            img.crop((0, top, png_w, bottom)).save(subpath)
            out.append((name, str(subpath)))
    for _name, subpath in out:
        _pad_to_ratio(Path(subpath), bg)
    return out


async def _capture(page: Any, path: str, *, full_page: bool) -> None:
    """Screenshot a page, clipping height when a full-page capture would balloon.

    ``full_page=True`` on a pathologically tall SPA (10k-20k px) makes Chromium
    allocate the entire scroll-height bitmap and write a giant PNG — the real
    OOM source on the heavy path. When the page is taller than the configured
    cap we clip to that height instead of capturing the full scroll height; the
    screenshot is a coarse visual artifact, so the lost tail is acceptable.

    ``page`` is a Playwright ``Page`` (typed ``Any`` to keep the playwright import
    lazy, as the rest of this module does).

    **Units, because getting this wrong truncates slides.** ``max_full_page_height``,
    ``document.documentElement.scrollHeight`` and Playwright's ``clip`` are ALL in
    CSS pixels; ``device_scale_factor`` scales only the CSS -> bitmap step, which is
    why the cap is *not* doubled for ``CAPTURE_DEVICE_SCALE_FACTOR=2``. Doubling it
    would not undo a truncation (there is none to undo) — it would double the
    clipped CSS height and so QUADRUPLE the bitmap on the exact path this cap
    exists to bound.

    Measured 2026-09-12 on a 12000 CSS px page, clip height 8000: at 1440/dsf1
    the PNG is 1440x8000, at 1920/dsf2 it is 3840x16000 — exactly ``cap * dsf``
    both times, which is what pins the cap's units to CSS px.

    Peak Chromium RSS for that capped capture, same measurement: **+178 MB** over
    an idle browser at 1440/dsf1 and **+234 MB** at 1920/dsf2 (an UNCAPPED
    full-page shot of the same page costs +193 MB and +260 MB). So 2x costs about
    +56 MB at the worst point on the heaviest path, against a 4Gi container limit
    — the cap stays at 8000 on measurement, not on arithmetic.

    The taller 1080 CSS viewport does not enter either — the cap is a document
    scroll height, not a viewport multiple — and a 1920-wide layout is SHORTER in
    CSS px than the same content at 1440 (wider columns wrap less), so 8000 covers
    strictly more of a page than it did before, not less.

    ``animations="disabled"`` finishes CSS animations/transitions instead of
    capturing whatever frame happened to be on screen when ``networkidle`` fired,
    and ``caret="hide"`` removes the blinking text caret — both are nondeterminism
    baked straight into a delivered frame.
    """
    if not full_page:
        await page.screenshot(path=path, full_page=False, **_SHOT_STABILITY)
        return

    from src.config import get_settings

    max_height = get_settings().redesign.max_full_page_height
    scroll_height = await page.evaluate("document.documentElement.scrollHeight")
    if scroll_height and scroll_height > max_height:
        width = await page.evaluate("document.documentElement.scrollWidth")
        # ``full_page=True`` is REQUIRED alongside ``clip`` and is not redundant:
        # without it Chromium never enables capture-beyond-viewport, so the clip
        # rect is INTERSECTED WITH THE VIEWPORT and the "clipped" capture comes
        # back one viewport tall. Measured 2026-09-12 on a 12000px page:
        # clip(h=8000) alone -> 1440x900 PNG (9 KB); clip(h=8000) + full_page
        # -> 1440x8000 (77 KB). So every tall page was being truncated to the
        # fold while this branch logged that it had clipped to 8000px.
        await page.screenshot(
            path=path,
            full_page=True,
            clip={"x": 0, "y": 0, "width": width, "height": max_height},
            **_SHOT_STABILITY,
        )
        logger.info(
            "Clipped full-page screenshot: page %dpx exceeds cap %dpx", scroll_height, max_height
        )
        return
    await page.screenshot(path=path, full_page=True, **_SHOT_STABILITY)


async def render_screenshots(
    html_path: Path,
    output_dir: Path,
    viewports: list[str],
    timeout_ms: int = 30000,
    *,
    full_page: bool = True,
) -> dict[str, str]:
    """Render an HTML file at multiple viewport sizes and save screenshots.

    Args:
        html_path: Path to the self-contained HTML file.
        output_dir: Directory to write screenshot PNGs into.
        viewports: List of viewport names (desktop, tablet, mobile).
        timeout_ms: Page load timeout in milliseconds.
        full_page: Capture the entire scroll height (True) or just the
            viewport (False). Full-page on a tall page allocates a huge
            bitmap; viewport-framed is the lean default for above-the-fold
            scopes like hero.

    Returns:
        Mapping of viewport name to screenshot path string.
    """
    from src.services.browser_pool import get_browser_pool

    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots: dict[str, str] = {}

    file_url = f"file://{html_path.resolve()}"

    pool = await get_browser_pool()
    async with pool.lease() as browser:
        for viewport_name in viewports:
            size = VIEWPORT_SIZES.get(viewport_name)
            if not size:
                logger.warning("Unknown viewport '%s', skipping", viewport_name)
                continue

            page = await browser.new_page(**_page_args(size))
            try:
                blocked = await block_private_subresources(page, allow_hosts=_minio_allow_hosts())
                await page.goto(file_url, wait_until="networkidle", timeout=timeout_ms)
                _warn_blocked_subresources(blocked)
                screenshot_path = output_dir / f"{viewport_name}.png"
                await _capture(page, str(screenshot_path), full_page=full_page)
                screenshots[viewport_name] = str(screenshot_path)
                logger.debug("Screenshot saved: %s", screenshot_path)
            finally:
                await _close_page_quietly(page)

    return screenshots


async def render_from_html_string(
    html: str,
    output_dir: Path,
    viewports: list[str],
    filename_prefix: str = "screenshot",
    timeout_ms: int = 30000,
    *,
    full_page: bool = True,
) -> dict[str, str]:
    """Render HTML from a string (write temp file, then screenshot).

    Args:
        html: HTML content string.
        output_dir: Directory to write screenshot PNGs into.
        viewports: List of viewport names.
        filename_prefix: Prefix for screenshot filenames.
        timeout_ms: Page load timeout.
        full_page: Capture full scroll height (True) or just the viewport (False).

    Returns:
        Mapping of viewport name to screenshot path string.
    """
    import tempfile

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html)
        temp_path = Path(f.name)

    try:
        return await render_screenshots(
            temp_path, output_dir, viewports, timeout_ms, full_page=full_page
        )
    finally:
        temp_path.unlink(missing_ok=True)


def _minio_allow_hosts() -> list[str]:
    """The one presigned-artifact host the SSRF guard should exempt.

    Sourced from ``settings.storage.minio_endpoint`` (``STORAGE_MINIO_ENDPOINT``,
    ``host:port`` — e.g. ``minio.test:8389``) rather than hardcoded, so the
    exemption tracks whatever MinIO this deployment actually points at instead of
    baking in an IP/hostname that drifts. Returns ``[]`` when unset (local
    backend, or ``minio`` backend not yet configured) — the guard then blocks
    every private host, which is correct: there is no legitimate artifact host to
    allow yet. ``validate_public_url``'s ``allow_hosts`` matches by hostname (and
    optionally exact port); it never widens to a CIDR, so this does not open the
    guard to other private/CGNAT hosts, only to this one named endpoint.
    """
    from src.config import get_settings

    endpoint = (get_settings().storage.minio_endpoint or "").strip()
    return [endpoint] if endpoint else []


async def block_private_subresources(
    page: Page, *, allow_hosts: Iterable[str] | None = None
) -> list[str]:
    """Refuse every http(s) subresource the page asks for that is not PUBLIC.

    Installed on EVERY page that renders **caller-supplied** HTML from a
    ``file://`` temp file. That is all four sinks, not just the PDF one:
    ``render_screenshots``, ``render_pdf_from_html_string``,
    ``render_component_screenshot`` and ``render_section_screenshots``.

    ⚠️ The PDF route is the loudest case (the response body lands in bytes handed
    straight back to the caller), but it is NOT the only one. The screenshot sinks
    receive ``html_for_render`` — the LLM-generated redesigned document, fetched
    back out of storage in ``design_builder/finalize_phases.py:241`` — which a
    tenant's redesign prompt influences. Guarding only the PDF path would have left
    three siblings open, which is the recurring "guarded beside unguarded" shape in
    this fleet. If you add a fifth ``file://`` sink, install this on it too; the
    test ``test_every_file_url_sink_installs_the_subresource_guard`` fails if you
    do not.
    Those pages are navigated with the *unpinned* pool browser, so the
    ``--host-resolver-rules`` pin that protects ``capture_url_screenshots``
    does not apply and nothing else stood between an ``<img>``/``<iframe>``/
    ``fetch`` in the document and every address this pod can reach — MinIO on
    the tailnet, a ClusterIP, cloud metadata. The response body lands in the
    PDF bytes handed back to the caller, which makes the print endpoint a
    readable internal-network oracle, exactly the property
    ``test_render_target_ssrf`` closes for the URL sinks.

    ``PdfRenderRequest``'s docstring already claimed "no external assets"; this
    is the first thing that enforces anything. It is an interception, not an
    after-the-fact assertion: a blocked request is ``abort``ed, so the GET is
    never issued and blind side-effects are prevented too.

    WHAT IS ALLOWED THROUGH, deliberately:

    * ``file:``/``data:``/``about:``/``blob:`` — the temp document itself and
      inline assets. The route pattern below is scheme-scoped to http(s), so
      those never enter the handler at all and the main-frame navigation is
      untouched. NOTE: ``file://`` subresource reads (a ``<img
      src="file:///etc/passwd">``) are therefore NOT addressed here — whether
      Chromium permits them from a ``file://`` document is policy-dependent and
      unverified; this function makes no claim about them.
    * Any http(s) URL that passes ``validate_public_url`` — ordinary web fonts,
      CDN images, remote CSS. Real redesign documents use these, so blocking
      all remote assets would break ``/jobs/{id}/pdf``. The bar is "public",
      the same bar ``assemble_page`` already applies to embeds.
    * ``allow_hosts`` — an explicit trust-list forwarded straight to
      ``validate_public_url`` (bare host or ``host:port``). Callers pass
      ``_minio_allow_hosts()`` here so a presigned ``minio.test`` asset the
      fleet convention legitimately embeds is not treated as SSRF. Nothing else
      in the private/CGNAT range is exempted — an arbitrary internal host still
      hits the block below.

    Args:
        page: The page to install the route handler on, before ``goto``.
        allow_hosts: Hosts that legitimately resolve internal and should bypass
            the private/CGNAT block — see ``_minio_allow_hosts``.

    Returns:
        A list that accumulates the URLs blocked during the page's life. The
        caller can log it, and a test can assert on the *blocked request*
        rather than merely on the absence of an exception.
    """
    blocked: list[str] = []

    async def _guard(route: Route) -> None:
        url = route.request.url
        try:
            # Resolution is a blocking syscall; keep it off the event loop that
            # the browser connection is multiplexed over.
            await asyncio.to_thread(validate_public_url, url, allow_hosts=allow_hosts)
        except (SSRFError, ValueError) as exc:
            blocked.append(url)
            logger.warning("Blocked non-public subresource in rendered HTML: %s", exc)
            await route.abort("blockedbyclient")
            return
        await route.continue_()

    await page.route("http://**", _guard)
    await page.route("https://**", _guard)
    return blocked


class BlockedSubresourceError(RuntimeError):
    """A rendered document tried to fetch a non-public subresource.

    Every ``file://`` sink now grants the one legitimate exemption
    (``_minio_allow_hosts``), so anything that still reaches this point is
    neither a presigned MinIO asset nor ordinary public web content — it is a
    private/reserved/CGNAT host the guard correctly refused (issue #104).
    Logging that as a WARNING and shipping the screenshot/PDF anyway is the
    silent-broken-image failure mode this type replaces: nothing downstream
    (the job row, the API response, the video compile) ever learns the render
    is incomplete. Raising fails the render loudly instead — `render_pdf`
    surfaces it as a 500 and the queue worker's existing failure path
    (`_dispatch` / `mark_failed` in `job_worker.py`) marks the job failed
    rather than "succeeded" with a hole where an asset should be.
    """


def _warn_blocked_subresources(blocked: list[str]) -> None:
    """Escalate a blocked subresource from a silent warning to a loud failure.

    See ``BlockedSubresourceError`` for why this raises instead of merely
    logging. Kept as one call site (all four ``file://`` sinks call this
    immediately after ``page.goto``) so the policy is applied identically.
    """
    if not blocked:
        return
    logger.error(
        "Render blocked %d non-public subresource request(s) - failing the render: %s",
        len(blocked),
        blocked[:5],
    )
    raise BlockedSubresourceError(
        f"render referenced {len(blocked)} non-public subresource(s), e.g. {blocked[0]!r}"
    )


# A4 width in CSS px (210mm at 96dpi). Chromium lays a printed page out at
# ``paper width / scale`` CSS px, so scale = A4_WIDTH_CSS_PX / layout width.
A4_WIDTH_CSS_PX = 793.7
# Desktop layout width for a design-concept PDF. At the A4 default (~794px) every
# ``@media (min-width: 968px)`` desktop rule was dead: the nav wrapped and clipped,
# the hero split collapsed to one column and left a near-empty page. 1440 is the
# service's desktop capture width.
DESIGN_PDF_LAYOUT_WIDTH_PX = 1440


A4_HEIGHT_CSS_PX = 1122.5
# Never lay a concept out wider than this, however far a section overflows.
_MAX_PDF_LAYOUT_WIDTH_PX = 2400
# Anything that avoids a break inside and is taller than this fraction of a printed
# page may split; shorter pieces (a card, a figure) still move whole. At 0.5 a 900px
# band could still jump a page and leave ~45% of it empty.
_KEEP_WHOLE_MAX_PAGE_FRACTION = 0.33

_MEASURE_SCROLL_WIDTH_JS = "() => Math.ceil(document.documentElement.scrollWidth)"
# A flex row (typically the nav) whose content is wider than its own max-width box
# overflows to the right and is clipped at the paper edge; widening the layout does
# not help because the centred box moves with it. Tighten that row's own gaps and
# side padding (bounded: 6 rounds of x0.7, never below 8px gap / 16px padding).
_FIT_OVERFLOWING_ROWS_JS = """() => {
  const overflowing = (el) => el.scrollWidth > el.clientWidth + 1;
  const rows = [...document.querySelectorAll('[data-section] *')].filter((el) => {
    const d = getComputedStyle(el).display;
    return (d.includes('flex') || d.includes('grid')) && overflowing(el);
  });
  let rounds = 0;
  for (const row of rows) {
    for (let i = 0; i < 6 && overflowing(row); i++, rounds++) {
      for (const el of [row, ...row.querySelectorAll('*')]) {
        const cs = getComputedStyle(el);
        if (!(cs.display.includes('flex') || cs.display.includes('grid'))) continue;
        const gap = parseFloat(cs.columnGap) || 0;
        if (gap > 8) el.style.columnGap = Math.max(8, gap * 0.7) + 'px';
        if (el === row) {
          const pl = parseFloat(cs.paddingLeft) || 0, pr = parseFloat(cs.paddingRight) || 0;
          if (pl > 16) el.style.paddingLeft = Math.max(16, pl * 0.7) + 'px';
          if (pr > 16) el.style.paddingRight = Math.max(16, pr * 0.7) + 'px';
        }
      }
    }
  }
  return rounds;
}"""
# ANY element that avoids breaking inside and is too tall to keep whole may split. The
# LLM's own section CSS (``.testimonials-section``/``-grid`` matched the old
# ``[class*="testimonial"]`` hint) kept a 2877px band whole, so it jumped a page and
# left page 4 70% empty; relaxing only ``[data-section]`` never reached it.
_RELAX_TALL_SECTION_BREAKS_JS = """(maxHeight) => {
  let relaxed = 0;
  for (const el of document.querySelectorAll('[data-section], [data-section] *')) {
    const cs = getComputedStyle(el);
    const avoids = cs.breakInside === 'avoid' || cs.breakInside === 'avoid-page'
      || el.hasAttribute('data-section');
    if (avoids && el.getBoundingClientRect().height > maxHeight) {
      el.style.breakInside = 'auto'; el.style.pageBreakInside = 'auto'; relaxed++;
    }
  }
  return relaxed;
}"""
# Equal-height cards in a grid row stretch a short card to its tallest neighbour, and a
# ``margin-top: auto`` author block then sits under a gap of empty card (the recorded
# testimonials: ~250px of blank inside "Sheila Hogendorf"). On paper that reads as a
# broken card, so a row whose stretch opens a big gap inside a CARD (a box with its own
# border, shadow or background) is top-aligned instead.
_TIGHTEN_STRETCHED_CARDS_JS = """(minGap) => {
  const cardLike = (el) => {
    const cs = getComputedStyle(el);
    return parseFloat(cs.borderTopWidth) > 0 || cs.boxShadow !== 'none'
      || !/rgba\\(0, 0, 0, 0\\)|transparent/.test(cs.backgroundColor);
  };
  let tightened = 0;
  for (const row of document.querySelectorAll('[data-section] *')) {
    const cs = getComputedStyle(row);
    const isRow = cs.display.includes('grid')
      || (cs.display.includes('flex') && !cs.flexDirection.startsWith('column'));
    const kids = [...row.children].filter((k) => k.getBoundingClientRect().height > 0);
    if (!isRow || kids.length < 2 || !kids.some(cardLike)) continue;
    if (!['normal', 'stretch'].includes(cs.alignItems)) continue;
    const before = kids.map((k) => k.getBoundingClientRect().height);
    row.style.alignItems = 'start';
    const after = kids.map((k) => k.getBoundingClientRect().height);
    const gap = Math.max(...before.map((h, i) => h - after[i]));
    if (gap >= minGap) tightened++; else row.style.alignItems = '';
  }
  return tightened;
}"""
# A stretch opening at least this much blank space inside a card is a visible gap.
_STRETCHED_CARD_MIN_GAP_PX = 120


async def _prepare_desktop_print_layout(page: Page, layout_width_px: int) -> int:
    """Lay the page out for print at a desktop width; return the width to scale to.

    Emulates ``print`` media (what ``page.pdf`` lays out with) at ``layout_width_px``.
    If content still overflows horizontally (an over-wide nav), the layout is widened
    to the content, capped, so nothing is clipped at the paper edge. Then sections too
    tall to keep whole are allowed to split.
    """
    await page.emulate_media(media="print")
    await page.evaluate(_FIT_OVERFLOWING_ROWS_JS)
    width = layout_width_px
    for _ in range(2):
        measured = int(await page.evaluate(_MEASURE_SCROLL_WIDTH_JS))
        if measured <= width:
            break
        width = min(measured, _MAX_PDF_LAYOUT_WIDTH_PX)
        await page.set_viewport_size({"width": width, "height": 1024})
    page_height_css = A4_HEIGHT_CSS_PX * width / A4_WIDTH_CSS_PX
    await page.evaluate(_TIGHTEN_STRETCHED_CARDS_JS, _STRETCHED_CARD_MIN_GAP_PX)
    await page.evaluate(
        _RELAX_TALL_SECTION_BREAKS_JS, page_height_css * _KEEP_WHOLE_MAX_PAGE_FRACTION
    )
    return width


def _pdf_options(layout_width_px: int | None) -> dict:
    """``page.pdf`` options: plain A4, or A4 scaled so layout happens at a desktop width."""
    if not layout_width_px:
        return {"format": "A4", "print_background": True}
    scale = max(0.1, min(2.0, A4_WIDTH_CSS_PX / layout_width_px))
    return {
        "format": "A4",
        "print_background": True,
        "scale": scale,
        "margin": {"top": "0", "right": "0", "bottom": "0", "left": "0"},
    }


async def render_pdf_from_html_string(
    html: str, timeout_ms: int = 30000, *, layout_width_px: int | None = None
) -> bytes:
    """Render a self-contained HTML string to PDF bytes via headless Chromium.

    On-demand (no persistence): writes a temp file, prints it to PDF through the
    shared browser pool, and returns the bytes. Used by the design PDF download
    endpoint so a generated page can be shared/printed as a document.

    ``layout_width_px`` lays the document out at that CSS width (viewport + print
    scale) so a design concept prints its DESKTOP layout; ``None`` keeps plain A4
    for generic documents (``POST /render/pdf`` reports).

    ``html`` is caller-supplied, so every http(s) subresource it references is
    intercepted and refused unless it is public — see
    ``block_private_subresources`` for what is allowed and what is not covered.
    """
    import tempfile

    from src.services.browser_pool import get_browser_pool

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html)
        temp_path = Path(f.name)

    file_url = f"file://{temp_path.resolve()}"
    try:
        pool = await get_browser_pool()
        async with pool.lease() as browser:
            # A4 with background graphics so the brand colours/imagery survive print.
            page = await browser.new_page()
            try:
                if layout_width_px:
                    await page.set_viewport_size({"width": layout_width_px, "height": 1024})
                blocked = await block_private_subresources(page, allow_hosts=_minio_allow_hosts())
                await page.goto(file_url, wait_until="networkidle", timeout=timeout_ms)
                _warn_blocked_subresources(blocked)
                if layout_width_px:
                    layout_width_px = await _prepare_desktop_print_layout(page, layout_width_px)
                return await page.pdf(**_pdf_options(layout_width_px))
            finally:
                await _close_page_quietly(page)
    finally:
        temp_path.unlink(missing_ok=True)


def _assert_landed_host_is_public(
    requested_url: str, landed_url: str, pinned_host: str | None
) -> None:
    """Re-validate the URL the browser actually landed on after redirects.

    A ``30x`` to a *different* host escapes the resolver pin: ``MAP <host> <ip>``
    names one host, and Chromium resolves the redirect target normally (verified
    — a page on a pinned host redirecting to ``127.0.0.1`` was followed). So the
    pin alone would let a tenant hand us a public URL that immediately bounces to
    cloud metadata or a ClusterIP, and we would screenshot it.

    This closes the *exfiltration* half of that: the landed host is re-validated
    before anything is captured, so an internal page's content can never reach a
    tenant-visible artifact. A same-host landing (the overwhelmingly common case,
    including ``http``→``https`` upgrades) short-circuits, and a redirect to a
    different *public* host (``example.com``→``www.example.com``) still renders —
    a strict "host must not change" rule would blank those legitimate pages.

    Known residual, deliberately not fixed here (see PR "Blast radius"): the
    redirected GET has already been issued by the time we check, so blind
    side-effects are not prevented, and the redirect target is validated but
    **not** pinned — it keeps the narrow rebind window this change closes for the
    initial host.

    Args:
        requested_url: The URL originally validated and pinned.
        landed_url: ``page.url`` after navigation (post-redirect).
        pinned_host: Hostname of ``requested_url``.

    Raises:
        SSRFError: if the browser landed on a non-public host.
    """
    landed_host = urlparse(landed_url).hostname
    if not landed_host or landed_host == pinned_host:
        return
    logger.info(
        "Render target redirected across hosts: %s -> %s; re-validating", requested_url, landed_url
    )
    validate_public_url(landed_url)


async def capture_url_screenshots(
    url: str,
    output_dir: Path,
    viewports: list[str],
    timeout_ms: int = 30000,
    *,
    full_page: bool = True,
) -> dict[str, str]:
    """Capture screenshots of a live URL at multiple viewports.

    Args:
        url: HTTP(S) URL to screenshot.
        output_dir: Directory to write screenshot PNGs into.
        viewports: List of viewport names (desktop, tablet, mobile).
        timeout_ms: Page load timeout in milliseconds.
        full_page: Capture full scroll height (True) or just the viewport (False).

    Returns:
        Mapping of viewport name to screenshot path string.

    Raises:
        SSRFError: if ``url`` is not a public http(s) target, or if the browser
            was redirected onto a non-public host. Callers map this to the job's
            existing failure reporting — see ``_phase_screenshots``.
    """
    from src.services.browser_pool import get_browser_pool

    # Sink-side SSRF guard, deliberately duplicating the `RedesignRequest.url`
    # admission check rather than trusting it. `url` arrives here from the
    # `redesign_jobs.url` column — read back from the DB at render time, which
    # may be minutes after admission and may be a row written before this guard
    # existed (or by a future writer that bypasses the schema). Schema
    # validation therefore does not cover this call; the sink does. Same
    # defence-in-depth shape as `webhook_dispatcher.dispatch`, which pre-checks
    # and then re-validates inside `safe_post`.
    #
    # httpx-layer IP-pinning cannot reach Chromium: the browser resolves the
    # hostname ITSELF, so validating-then-``goto``ing the same string leaves a
    # DNS-rebind window (validate resolves to a public IP, Chromium re-resolves
    # milliseconds later to an internal one). The resolver-level lever is the
    # only correct pin for a browser: capture the address ``validate_public_url``
    # already vetted and force Chromium's own resolver to it via
    # ``--host-resolver-rules=MAP <host> <ip>`` (a launch arg), so the browser
    # cannot re-resolve. ``pinned_ip`` is ``None`` only for an allow-listed host
    # (never passed here) — that path keeps the shared pool unchanged.
    pinned_ip = validate_public_url(url)
    host = urlparse(url).hostname

    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots: dict[str, str] = {}

    pool = await get_browser_pool()
    lease_cm = pool.lease_pinned(host, pinned_ip) if pinned_ip and host else pool.lease()
    async with lease_cm as browser:
        for viewport_name in viewports:
            size = VIEWPORT_SIZES.get(viewport_name)
            if not size:
                logger.warning("Unknown viewport '%s', skipping", viewport_name)
                continue

            page = await browser.new_page(**_page_args(size))
            try:
                response = await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                # Same discarded-Response defect as `_fetch_page_html`: without
                # this, a 404/403/500 was screenshotted and shipped as the "before"
                # image of the tenant's site. This raises `PageFetchError`
                # (a RuntimeError), which every caller of this function already
                # treats as a SOFT capture failure via `capture_failure_types()` --
                # so a dead URL degrades to "no before image" rather than shipping
                # a picture of an error page as if it were the site.
                require_ok_navigation(response, url)
                # TODO(ssrf): a main-frame 30x is followed by page.goto before we
                # get here; the MAP rule only pins the ORIGINAL host, so a
                # cross-host redirect target is re-validated by the call below but
                # NOT re-pinned (rebind-via-redirect window remains). Closing it
                # needs request interception. See handoff + _assert_landed_host_is_public.
                _assert_landed_host_is_public(url, page.url, host)
                screenshot_path = output_dir / f"original_{viewport_name}.png"
                await _capture(page, str(screenshot_path), full_page=full_page)
                screenshots[viewport_name] = str(screenshot_path)
                logger.debug("Original screenshot saved: %s", screenshot_path)
            finally:
                await _close_page_quietly(page)

    return screenshots


async def render_component_screenshot(
    full_page_html: str,
    component_selector: str,
    output_path: Path,
    viewport: str = "desktop",
    timeout_ms: int = 30000,
) -> str | None:
    """Render a screenshot of a specific component on the page.

    Returns the screenshot path or None on failure.
    """
    import tempfile

    from src.services.browser_pool import get_browser_pool

    size = VIEWPORT_SIZES.get(viewport, VIEWPORT_SIZES["desktop"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(full_page_html)
        temp_path = Path(f.name)

    try:
        pool = await get_browser_pool()
        async with pool.lease() as browser:
            page = await browser.new_page(**_page_args(size))
            try:
                file_url = f"file://{temp_path.resolve()}"
                blocked = await block_private_subresources(page, allow_hosts=_minio_allow_hosts())
                await page.goto(file_url, wait_until="networkidle", timeout=timeout_ms)
                _warn_blocked_subresources(blocked)
                element = await page.query_selector(component_selector)
                if element:
                    await element.screenshot(path=str(output_path), **_SHOT_STABILITY)
                    return str(output_path)
                logger.warning("Component selector '%s' not found", component_selector)
                return None
            finally:
                await _close_page_quietly(page)
    finally:
        temp_path.unlink(missing_ok=True)


def _slugify_section(text: str) -> str:
    """Lowercase kebab slug capped at 48 chars (empty → '')."""
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:48]


def _section_name(meta: dict[str, Any], idx: int, used: set[str]) -> str:
    """Derive a unique, meaningful section name from an element's identity."""
    raw = (
        meta.get("dataLabel")
        or meta.get("heading")
        or meta.get("id")
        or meta.get("tag")
        or "section"
    )
    base = _slugify_section(str(raw)) or f"section-{idx + 1}"
    name = base
    suffix = 2
    while name in used:
        name = f"{base}-{suffix}"
        suffix += 1
    used.add(name)
    return name


async def _render_section_bands(
    page: Any, output_dir: Path, size: dict[str, int], *, max_sections: int
) -> list[tuple[str, str]]:
    """Fallback when a page has no clean section landmarks: slice it into evenly
    sized viewport-height bands by scrolling, so the after act still gets discrete,
    settled stills instead of one tall scroll. Each band is a current-viewport shot.

    Each band is named after the ``data-section`` under its vertical midpoint (so
    the orchestrator can still ground narration on it), de-duplicated with ``-2``
    suffixes, falling back to ``band-NN`` where no named section is hit.
    """
    results: list[tuple[str, str]] = []
    used: set[str] = set()
    viewport_h = size["height"]
    scroll_h = await page.evaluate("document.documentElement.scrollHeight") or viewport_h
    band_count = max(1, min(max_sections, math.ceil(scroll_h / viewport_h)))
    for i in range(band_count):
        await page.evaluate(f"window.scrollTo(0, {i * viewport_h})")
        try:
            label = await page.evaluate(_BAND_LABEL_JS, [size["width"] / 2, viewport_h / 2])
        except Exception as exc:  # noqa: BLE001 -- soft: unnamed band
            logger.warning("Band label %d failed: %s", i, exc)
            label = ""
        base = _slugify_section(str(label or "")) or f"band-{i + 1:02d}"
        name = base
        suffix = 2
        while name in used:
            name = f"{base}-{suffix}"
            suffix += 1
        used.add(name)
        path = output_dir / f"{i:02d}_{name}.png"
        try:
            await page.screenshot(path=str(path), full_page=False, **_SHOT_STABILITY)
            results.append((name, str(path)))
        except Exception as exc:  # noqa: BLE001 -- soft per-band skip
            logger.warning("Band screenshot %d failed: %s", i, exc)
    return results


async def _maybe_split_band(
    element: Any,
    path: Path,
    box: FloatRect,
    bg: tuple[int, int, int],
    used: set[str],
    idx: int,
) -> list[tuple[str, str]]:
    """Split a band that is much taller than 16:9 into per-heading stills, else ``[]``.

    The greedy one-landmark case (a SPA ``<main>`` of plain divs) yields a single
    tall band that swallows every topic; we crop it on its inner h1/h2 offsets so
    the after act has variety to key narration to. Soft — any failure returns ``[]``
    and the caller keeps the single padded band rather than dropping the section.
    """
    too_tall = box["width"] > 0 and box["height"] >= _SECTION_SPLIT_MIN_RATIO * (
        box["width"] * _SECTION_TARGET_RATIO_H / _SECTION_TARGET_RATIO_W
    )
    if not too_tall:
        return []
    try:
        heads = await element.evaluate(_SECTION_HEADINGS_JS)
        return await asyncio.to_thread(_split_section_png, path, heads, box["width"], bg, used)
    except Exception as exc:  # noqa: BLE001 -- soft split
        logger.warning("Section split %d failed: %s", idx, exc)
        return []


async def _capture_section(
    element: Any,
    *,
    box: FloatRect,
    idx: int,
    output_dir: Path,
    used: set[str],
    results: list[tuple[str, str]],
    max_sections: int,
    timeout_ms: int,
) -> None:
    """Screenshot one accepted band and append its still(s) to ``results``.

    A greedy too-tall band is split into per-topic crops (replacing the parent);
    otherwise the single band is padded up to 16:9 so the compile cover-fit does
    not over-zoom it. Both PIL steps run off the event loop.
    """
    meta = await element.evaluate(_SECTION_LABEL_JS)
    meta = meta if isinstance(meta, dict) else {}
    name = _section_name(meta, idx, used)
    path = output_dir / f"{len(results):02d}_{name}.png"
    await element.scroll_into_view_if_needed(timeout=timeout_ms)
    await element.screenshot(path=str(path), **_SHOT_STABILITY)
    bg = _resolve_bg(meta.get("bg"), meta.get("bodyBg"))
    split = await _maybe_split_band(element, path, box, bg, used, idx)
    if split:
        path.unlink(missing_ok=True)  # per-topic crops replace the greedy parent
        for sub_name, sub_path in split:
            if len(results) >= max_sections:
                break
            results.append((sub_name, sub_path))
        return
    try:
        await asyncio.to_thread(_pad_to_ratio, path, bg)
    except Exception as exc:  # noqa: BLE001 -- soft pad
        logger.warning("Section pad %d failed: %s", idx, exc)
    results.append((name, str(path)))


async def _capture_landmarks(
    page: Page,
    elements: list[ElementHandle],
    *,
    size: dict[str, int],
    min_height: float,
    output_dir: Path,
    used: set[str],
    results: list[tuple[str, str]],
    max_sections: int,
    timeout_ms: int,
    narrow_width_frac: float | None = None,
) -> None:
    """Capture every full-width, tall-enough, non-overlapping element in ``elements``.

    ``min_height`` is ``_MIN_SECTION_HEIGHT_PX`` for the generic landmark pass and
    ``_MIN_NAMED_SECTION_HEIGHT_PX`` for the named ``data-section`` second pass.
    ``narrow_width_frac`` (second pass only) also admits an element at least that
    fraction of the viewport wide when it is no taller than 16:9 of its width, so
    padding — never a cover-fit crop — brings it to 16:9. Soft per element.
    """
    min_width = size["width"] * _MIN_SECTION_WIDTH_FRAC
    narrow_min_width = size["width"] * narrow_width_frac if narrow_width_frac else min_width
    accepted_spans: list[tuple[float, float]] = []
    for idx, element in enumerate(elements):
        if len(results) >= max_sections:
            break
        try:
            box = await element.bounding_box()
            # Keep only real full-width page bands tall enough to
            # compose into a 16:9 still; tiny nested landmarks
            # (logos, cards, thin spacers) cover-fit to blank frames.
            if not box or box["height"] < min_height:
                continue
            pads_not_crops = (
                box["height"] * _SECTION_TARGET_RATIO_W <= box["width"] * _SECTION_TARGET_RATIO_H
            )
            if box["width"] < min_width and not (
                pads_not_crops and box["width"] >= narrow_min_width
            ):
                continue
            # Skip a section that mostly overlaps one already captured
            # (a full-width child nested in a full-width parent).
            # bounding_box() is VIEWPORT-relative, but each capture below
            # scroll_into_view's its element, leaving the page scrolled for
            # the next iteration's measurement — so compare in DOCUMENT-
            # absolute coordinates (add the live scrollY) or stacked
            # full-width bands spuriously overlap and get dropped.
            scroll_y = await page.evaluate("() => window.scrollY")
            top = box["y"] + scroll_y
            bottom = top + box["height"]
            if any(
                min(bottom, a_bottom) - max(top, a_top)
                >= _SECTION_VERTICAL_OVERLAP_FRAC * box["height"]
                for a_top, a_bottom in accepted_spans
            ):
                continue
            await _capture_section(
                element,
                box=box,
                idx=idx,
                output_dir=output_dir,
                used=used,
                results=results,
                max_sections=max_sections,
                timeout_ms=timeout_ms,
            )
            accepted_spans.append((top, bottom))
        except Exception as exc:  # noqa: BLE001 -- soft per-section skip
            logger.warning("Section screenshot %d failed: %s", idx, exc)
            continue


async def render_section_screenshots(
    html: str,
    output_dir: Path,
    viewport: str = "desktop",
    *,
    max_sections: int = 12,
    timeout_ms: int = 30000,
) -> list[tuple[str, str]]:
    """Render the composite redesign HTML and screenshot each top-level section.

    Returns an ordered list of ``(section_name, png_path)``. Built for the
    website_review AFTER act: discrete, settled per-section stills the video can
    key to the narration line that discusses each section, instead of one tall
    top-to-bottom scroll over a ~5000px page.

    Soft by contract — an individual section that fails to screenshot is skipped,
    and when no usable landmark sections are found we fall back to evenly-sliced
    viewport-height bands, so the caller always gets discrete stills (never zero,
    never one giant scroll). Reuses the process-wide browser pool.
    """
    import tempfile

    from src.services.browser_pool import get_browser_pool

    output_dir.mkdir(parents=True, exist_ok=True)
    size = VIEWPORT_SIZES.get(viewport, VIEWPORT_SIZES["desktop"])
    results: list[tuple[str, str]] = []

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html)
        temp_path = Path(f.name)

    try:
        pool = await get_browser_pool()
        async with pool.lease() as browser:
            page = await browser.new_page(**_page_args(size))
            try:
                file_url = f"file://{temp_path.resolve()}"
                blocked = await block_private_subresources(page, allow_hosts=_minio_allow_hosts())
                await page.goto(file_url, wait_until="networkidle", timeout=timeout_ms)
                _warn_blocked_subresources(blocked)
                used: set[str] = set()
                elements = await page.query_selector_all(_SECTION_SELECTOR)
                await _capture_landmarks(
                    page,
                    elements,
                    size=size,
                    min_height=_MIN_SECTION_HEIGHT_PX,
                    output_dir=output_dir,
                    used=used,
                    results=results,
                    max_sections=max_sections,
                    timeout_ms=timeout_ms,
                )
                if not results:
                    # No landmark is tall enough — but a thin page's named,
                    # full-width data-sections are still real sections.
                    named = await page.query_selector_all(_TOP_LEVEL_DATA_SECTION_SELECTOR)
                    await _capture_landmarks(
                        page,
                        named,
                        size=size,
                        min_height=_MIN_NAMED_SECTION_HEIGHT_PX,
                        output_dir=output_dir,
                        used=used,
                        results=results,
                        max_sections=max_sections,
                        timeout_ms=timeout_ms,
                        narrow_width_frac=_MIN_NAMED_SECTION_WIDTH_FRAC,
                    )
                if not results:
                    logger.info("No landmark sections found; falling back to viewport bands")
                    results = await _render_section_bands(
                        page, output_dir, size, max_sections=max_sections
                    )
            finally:
                await _close_page_quietly(page)
    finally:
        temp_path.unlink(missing_ok=True)

    return results
