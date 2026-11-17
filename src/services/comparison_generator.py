"""Visual before/after comparison image generation.

All image work here is synchronous, CPU-bound PIL — it MUST stay off the
event loop. Callers (``generate_comparisons``) hand it to
``asyncio.to_thread`` so a full-page screenshot of a tall SPA cannot block
the single asyncio loop long enough to trip the 1s liveness/readiness probe
(which is what killed the pod under the old per-pixel Python loop).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Full-page screenshots of tall marketing SPAs can be 8000px+ high; two of
# those decoded to RGB bitmaps plus a composite is what pressured the pod's
# memory. The comparison/diff is a coarse visual artifact (not pixel-exact),
# so down-scaling very tall captures before processing keeps the buffers
# bounded with no meaningful loss of usefulness.
MAX_COMPARISON_HEIGHT = 4000


def _open_capped(path: str) -> Image.Image:
    """Open an image, down-scaling it if it exceeds ``MAX_COMPARISON_HEIGHT``."""
    img = Image.open(path)
    if img.height > MAX_COMPARISON_HEIGHT:
        ratio = MAX_COMPARISON_HEIGHT / img.height
        img = img.resize(
            (max(1, int(img.width * ratio)), MAX_COMPARISON_HEIGHT),
            Image.Resampling.LANCZOS,
        )
    return img


def generate_side_by_side(
    original_path: str,
    redesigned_path: str,
    output_path: str,
    label_original: str = "BEFORE",
    label_redesigned: str = "AFTER",
) -> str:
    """Generate a labeled side-by-side comparison image.

    Returns the output path.
    """
    # Track every decoded bitmap so we can release them deterministically — at
    # full-page sizes these are tens of MB each; relying on GC lets them pile up.
    opened: list[Image.Image] = []
    try:
        original = _open_capped(original_path)
        redesigned = _open_capped(redesigned_path)
        opened += [original, redesigned]

        # Normalize heights
        max_height = max(original.height, redesigned.height)
        if original.height != max_height:
            ratio = max_height / original.height
            original = original.resize(
                (int(original.width * ratio), max_height), Image.Resampling.LANCZOS
            )
            opened.append(original)
        if redesigned.height != max_height:
            ratio = max_height / redesigned.height
            redesigned = redesigned.resize(
                (int(redesigned.width * ratio), max_height), Image.Resampling.LANCZOS
            )
            opened.append(redesigned)

        # Create composite with labels
        label_height = 40
        gap = 4
        total_width = original.width + gap + redesigned.width
        total_height = max_height + label_height

        composite = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        opened.append(composite)
        draw = ImageDraw.Draw(composite)

        # Draw labels
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except OSError:
            font = ImageFont.load_default()

        # Original label (red background)
        draw.rectangle(
            [(0, 0), (original.width, label_height)],
            fill=(220, 53, 69),
        )
        bbox = font.getbbox(label_original)
        text_w = bbox[2] - bbox[0]
        draw.text(
            ((original.width - text_w) // 2, 8),
            label_original,
            fill=(255, 255, 255),
            font=font,
        )

        # Redesigned label (green background)
        draw.rectangle(
            [(original.width + gap, 0), (total_width, label_height)],
            fill=(40, 167, 69),
        )
        bbox = font.getbbox(label_redesigned)
        text_w = bbox[2] - bbox[0]
        draw.text(
            (original.width + gap + (redesigned.width - text_w) // 2, 8),
            label_redesigned,
            fill=(255, 255, 255),
            font=font,
        )

        # Paste images
        composite.paste(original, (0, label_height))
        composite.paste(redesigned, (original.width + gap, label_height))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        composite.save(output_path, "PNG")
        logger.info("Side-by-side comparison saved: %s", output_path)
        return output_path
    finally:
        for img in opened:
            with contextlib.suppress(Exception):
                img.close()


def generate_pixel_diff(
    original_path: str,
    redesigned_path: str,
    output_path: str,
    threshold: int = 30,
) -> str:
    """Generate a pixel diff image highlighting changed areas in red.

    Vectorized with PIL band math (``ImageChops`` + LUT ``point``) so the work
    runs in C, not a per-pixel Python loop — on a full-page SPA the old loop
    was ~10M iterations that blocked the event loop for seconds. Semantics are
    preserved exactly: a pixel whose summed |Δ| across RGB exceeds ``threshold``
    is shown red-shifted from the redesign ``(min(255,r+100), max(0,g-50),
    max(0,b-50))``; unchanged pixels are dimmed toward white ``((c+200)//2)``.

    Returns the output path.
    """
    opened: list[Image.Image] = []
    try:
        original = _open_capped(original_path).convert("RGB")
        redesigned = _open_capped(redesigned_path).convert("RGB")
        opened += [original, redesigned]

        # Resize to match dimensions
        width = min(original.width, redesigned.width)
        height = min(original.height, redesigned.height)
        original = original.resize((width, height), Image.Resampling.LANCZOS)
        redesigned = redesigned.resize((width, height), Image.Resampling.LANCZOS)
        opened += [original, redesigned]

        rn, gn, bn = redesigned.split()

        # Summed per-channel |Δ|, thresholded to a 0/255 mask. ImageChops.add
        # clamps at 255, but the threshold (30) is far below 255 so a value is
        # >threshold before-clamp iff it is >threshold after-clamp — the mask is
        # identical to the old loop's ``abs(Δr)+abs(Δg)+abs(Δb) > threshold``.
        dr, dg, db = ImageChops.difference(original, redesigned).split()
        summed = ImageChops.add(ImageChops.add(dr, dg), db)
        mask = summed.point(lambda v: 255 if v > threshold else 0)

        # Changed layer: red-shifted redesign. Unchanged layer: dimmed redesign.
        changed = Image.merge(
            "RGB",
            (
                rn.point(lambda v: min(255, v + 100)),
                gn.point(lambda v: max(0, v - 50)),
                bn.point(lambda v: max(0, v - 50)),
            ),
        )
        dimmed = Image.merge(
            "RGB", tuple(band.point(lambda v: (v + 200) // 2) for band in (rn, gn, bn))
        )
        opened += [changed, dimmed]

        diff = Image.composite(changed, dimmed, mask)
        opened.append(diff)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        diff.save(output_path, "PNG")
        logger.info("Pixel diff saved: %s", output_path)
        return output_path
    finally:
        for img in opened:
            with contextlib.suppress(Exception):
                img.close()


async def generate_comparisons(
    original_screenshots: dict[str, str],
    redesigned_screenshots: dict[str, str],
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Generate side-by-side comparisons and pixel diffs for all viewports.

    Returns (comparison_paths, diff_paths) dicts keyed by viewport name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons: dict[str, str] = {}
    diffs: dict[str, str] = {}

    for viewport in original_screenshots:
        if viewport not in redesigned_screenshots:
            continue

        orig_path = original_screenshots[viewport]
        new_path = redesigned_screenshots[viewport]

        # Offload the CPU-bound PIL work to a worker thread so it never blocks
        # the event loop (and thus never starves the health probes).
        try:
            comp_path = str(output_dir / f"comparison_{viewport}.png")
            comparisons[viewport] = await asyncio.to_thread(
                generate_side_by_side, orig_path, new_path, comp_path
            )
        except Exception as exc:  # noqa: BLE001 — per-viewport absorb, see comment
            # Catch every exception type: the contract is "one bad viewport is
            # skipped, the rest still render". The previous tuple missed PIL's
            # `Image.DecompressionBombError`, which derives from `Exception`,
            # not `OSError` (unlike `UnidentifiedImageError` and the truncated-
            # file errors, which ARE `OSError` — that partial correctness is why
            # this looked right). One hostile screenshot therefore took out
            # every remaining viewport's comparison AND its diff.
            logger.warning("Failed to generate comparison for %s: %s", viewport, exc)

        try:
            diff_path = str(output_dir / f"diff_{viewport}.png")
            diffs[viewport] = await asyncio.to_thread(
                generate_pixel_diff, orig_path, new_path, diff_path
            )
        except Exception as exc:  # noqa: BLE001 — per-viewport absorb, same reasoning
            logger.warning("Failed to generate diff for %s: %s", viewport, exc)

    return comparisons, diffs
