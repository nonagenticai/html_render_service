"""Encode a generated hero image as a small, self-contained ``data:`` URI.

The assembled page is self-contained (``catalog/assembler.py``): it is served
under ``img-src https: data:`` and viewed outside the cluster, so a
``http://minio.test`` reference — plain http, tailnet-only, presigned and
expiring — cannot appear in it. The generated hero is therefore inlined, after a
downscale + re-encode so it does not bloat the page.
"""

from __future__ import annotations

import base64
import io
import logging

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

MAX_WIDTH = 1600
MAX_BYTES = 300 * 1024
_QUALITIES = (82, 72, 62, 52)
_MIN_WIDTH = 640


def _encode(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()


def hero_data_uri(
    data: bytes, *, max_width: int = MAX_WIDTH, max_bytes: int = MAX_BYTES
) -> str | None:
    """Downscale to ``max_width`` and JPEG-encode to at most ``max_bytes``.

    Steps the quality down, then the width, until it fits. Returns
    ``data:image/jpeg;base64,...`` or ``None`` when the bytes are not a decodable
    image or cannot be brought under the budget at ``_MIN_WIDTH``.
    """
    try:
        with Image.open(io.BytesIO(data)) as src:
            src.load()
            img = src.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        logger.warning("hero image is not a decodable image: %s", exc)
        return None
    width = min(img.width, max_width)
    while width >= _MIN_WIDTH or width == img.width:
        frame = img
        if img.width > width:
            height = max(1, round(img.height * width / img.width))
            frame = img.resize((width, height), Image.Resampling.LANCZOS)
        for quality in _QUALITIES:
            encoded = _encode(frame, quality)
            if len(encoded) <= max_bytes:
                return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        if width <= _MIN_WIDTH:
            break
        width = max(_MIN_WIDTH, int(width * 0.8))
    logger.warning("hero image could not be encoded under %d bytes", max_bytes)
    return None
