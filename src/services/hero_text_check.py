"""Ask a vision model whether a generated hero image shows visible text.

Owner decision (live run 01a159b1, hero lettered a garbled shop sign "Cafiiti
Boker"): the negative prompt alone does not keep text out of a FLUX image, so the
hero is checked after generation. One check per image; the regenerate / gradient
decision lives with the caller (``design_builder.phases``).

The call goes through the service's shared :class:`WMClient` with ``content_blocks``
(text + base64 image) — the same multimodal path ``video_analyzer_service`` uses —
so it reaches inference-router with the job's bound identity, like every other
metered LLM call in this service. The image sent is the downscaled JPEG the page
would inline (``hero_data_uri``), not the multi-MB original.

The check NEVER blocks the page: any failure of the check itself (budget refused,
timeout, transport error, an unparseable verdict) comes back as
``HeroTextVerdict(has_text=False, error=...)``, and the caller records it as a
quality warning.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.hero_image import hero_data_uri
from src.services.html_gen.section_run import budget_window

if TYPE_CHECKING:
    from w2v_common.clients import WMClient

    from src.services.html_gen.section_run import GenerationBudget

logger = logging.getLogger(__name__)

HERO_TEXT_WORKER = "hero-text-check"

#: Upper bound on one check, independent of the job budget: a hung vision call
#: must not hold the page for the whole generation deadline.
HERO_TEXT_TIMEOUT_SECONDS = 90.0

HERO_TEXT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "has_text": {"type": "boolean"},
        "text_seen": {"type": "string"},
    },
    "required": ["has_text", "text_seen"],
    "additionalProperties": False,
}

HERO_TEXT_SYSTEM_PROMPT = (
    "You inspect photographs for visible text. Answer ONLY with the JSON object the "
    "schema asks for."
)

HERO_TEXT_INSTRUCTION = (
    "Does this image contain ANY visible text: letters, words, numbers, signage, "
    "shop signs, labels, logos with lettering, menus, captions or watermarks — "
    "including garbled or partial lettering? Set has_text to true if you can see any, "
    "and put what you can read (or a short description of where it is) in text_seen. "
    "Set has_text to false and text_seen to an empty string only if the image has no "
    "text at all."
)


@dataclass(frozen=True)
class HeroTextVerdict:
    """The outcome of ONE check. ``error`` set => the check itself failed (fail-open)."""

    has_text: bool
    text_seen: str = ""
    error: str | None = None


def _parse_verdict(output: object) -> HeroTextVerdict:
    data = json.loads(output) if isinstance(output, str) else output
    if not isinstance(data, dict):
        raise ValueError(f"verdict is {type(data).__name__}, not an object")
    has_text = data.get("has_text")
    if not isinstance(has_text, bool):
        raise ValueError(f"has_text is {has_text!r}, not a boolean")
    text_seen = data.get("text_seen")
    return HeroTextVerdict(has_text=has_text, text_seen=str(text_seen or "").strip()[:200])


class HeroTextChecker:
    """One vision call per image: ``{"has_text": bool, "text_seen": str}``."""

    def __init__(
        self, wm_client: WMClient, *, timeout_seconds: float = HERO_TEXT_TIMEOUT_SECONDS
    ) -> None:
        self._wm = wm_client
        self._timeout = timeout_seconds

    async def check(
        self, data: bytes, *, budget: GenerationBudget | None = None
    ) -> HeroTextVerdict:
        uri = hero_data_uri(data)
        if uri is None:
            return HeroTextVerdict(has_text=False, error="image not encodable for the check")
        b64 = uri.split(",", 1)[1]
        try:
            async with asyncio.timeout(self._timeout):
                if budget is not None:
                    async with budget_window(budget, extra=False):
                        response = await self._call(b64)
                else:
                    response = await self._call(b64)
            return _parse_verdict(response.output)
        except Exception as exc:  # noqa: BLE001 — the check must never block the page
            logger.warning("hero text check failed: %s: %s", type(exc).__name__, exc)
            return HeroTextVerdict(has_text=False, error=type(exc).__name__)

    async def _call(self, b64: str):  # noqa: ANN202 — WMResponse, kept out of the import graph
        return await self._wm.message(
            worker_name=HERO_TEXT_WORKER,
            content_blocks=[
                {"type": "text", "text": HERO_TEXT_INSTRUCTION},
                {"type": "image", "data": b64, "media_type": "image/jpeg"},
            ],
            system_prompt=HERO_TEXT_SYSTEM_PROMPT,
            output_schema=HERO_TEXT_SCHEMA,
            max_tokens=256,
        )
