"""Tests for the vectorized, off-loop comparison/diff generation.

The pixel diff used to be a per-pixel Python double loop run inline on the
event loop — on a full-page SPA that blocked the loop for seconds and starved
the 1s health probes. These tests pin two things: the vectorized rewrite
produces byte-identical output to the old loop, and the CPU-bound PIL work is
handed to a worker thread instead of blocking the loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image

from src.services import comparison_generator as cg


def _ref_pixel_diff(orig_path: str, new_path: str, threshold: int = 30) -> bytes:
    """Reference implementation of the original per-pixel loop, returning the
    raw RGB bytes of the expected diff image (byte-level keeps pixel typing
    unambiguous and mirrors generate_pixel_diff's preprocessing exactly)."""
    original = Image.open(orig_path).convert("RGB")
    redesigned = Image.open(new_path).convert("RGB")
    width = min(original.width, redesigned.width)
    height = min(original.height, redesigned.height)
    original = original.resize((width, height), Image.Resampling.LANCZOS)
    redesigned = redesigned.resize((width, height), Image.Resampling.LANCZOS)

    ob = original.tobytes()
    nb = redesigned.tobytes()
    out = bytearray(len(nb))
    for i in range(0, len(nb), 3):
        r1, g1, b1 = ob[i], ob[i + 1], ob[i + 2]
        r2, g2, b2 = nb[i], nb[i + 1], nb[i + 2]
        if abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2) > threshold:
            out[i], out[i + 1], out[i + 2] = min(255, r2 + 100), max(0, g2 - 50), max(0, b2 - 50)
        else:
            out[i], out[i + 1], out[i + 2] = (r2 + 200) // 2, (g2 + 200) // 2, (b2 + 200) // 2
    return bytes(out)


def _make_image(path: Path, pixels: list[list[tuple[int, int, int]]]) -> str:
    h = len(pixels)
    w = len(pixels[0])
    img = Image.new("RGB", (w, h))
    flat: list[tuple[int, int, int]] = [px for row in pixels for px in row]
    img.putdata(flat)
    img.save(path, "PNG")
    return str(path)


def test_pixel_diff_matches_reference_loop(tmp_path) -> None:
    # Mix of identical, slightly-different (<threshold) and very-different pixels
    # plus channel-clamp edge cases (near 0 and near 255) to exercise both
    # branches and the min/max clamps.
    orig = [
        [(0, 0, 0), (10, 10, 10), (200, 100, 50), (255, 255, 255)],
        [(250, 250, 250), (5, 5, 5), (128, 128, 128), (0, 128, 255)],
    ]
    new = [
        [(0, 0, 0), (12, 12, 12), (10, 220, 240), (255, 255, 255)],
        [(255, 200, 0), (5, 5, 30), (130, 130, 130), (40, 40, 40)],
    ]
    op = _make_image(tmp_path / "o.png", orig)
    npth = _make_image(tmp_path / "n.png", new)

    out = cg.generate_pixel_diff(op, npth, str(tmp_path / "diff.png"))
    actual = Image.open(out).convert("RGB").tobytes()
    expected = _ref_pixel_diff(op, npth)

    assert actual == expected


def test_open_capped_downscales_tall_images(tmp_path) -> None:
    tall = Image.new("RGB", (100, cg.MAX_COMPARISON_HEIGHT + 2000), (10, 20, 30))
    p = tmp_path / "tall.png"
    tall.save(p, "PNG")

    capped = cg._open_capped(str(p))
    assert capped.height == cg.MAX_COMPARISON_HEIGHT
    assert capped.width < 100  # aspect-preserved down-scale


def test_open_capped_leaves_short_images_untouched(tmp_path) -> None:
    short = Image.new("RGB", (100, 300), (10, 20, 30))
    p = tmp_path / "short.png"
    short.save(p, "PNG")

    capped = cg._open_capped(str(p))
    assert capped.size == (100, 300)


async def test_generate_comparisons_offloads_to_thread(tmp_path, monkeypatch) -> None:
    """The CPU-bound PIL work must run via asyncio.to_thread, not inline."""
    op = _make_image(tmp_path / "o.png", [[(0, 0, 0), (255, 255, 255)]])
    npth = _make_image(tmp_path / "n.png", [[(255, 255, 255), (0, 0, 0)]])

    offloaded: list[str] = []
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(fn, *args, **kwargs):
        offloaded.append(fn.__name__)
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(cg.asyncio, "to_thread", spy_to_thread)

    comparisons, diffs = await cg.generate_comparisons(
        {"desktop": op}, {"desktop": npth}, tmp_path / "out"
    )

    assert offloaded == ["generate_side_by_side", "generate_pixel_diff"]
    assert "desktop" in comparisons
    assert "desktop" in diffs
    assert Path(comparisons["desktop"]).exists()
    assert Path(diffs["desktop"]).exists()
