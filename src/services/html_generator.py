"""Brand-faithful redesign generator (self-contained brand CSS) via worker_management.

This module was split into the ``src/services/html_gen/`` package (types /
html_utils / brand_css / images / prompt_builder / generator) for readability.
It is kept as a thin re-export facade so existing imports —
``from src.services.html_generator import HtmlGenerator | estimate_cost |
LLMResult | _extract_html_block | _absolutize_images | _resolve_palette |
_DEFAULT_PRIMARY`` — keep working unchanged.
"""

from __future__ import annotations

from src.services.html_gen import (
    _DEFAULT_PRIMARY,
    HtmlGenerator,
    LLMResult,
    _absolutize_images,
    _extract_html_block,
    _resolve_palette,
    estimate_cost,
)

__all__ = [
    "_DEFAULT_PRIMARY",
    "HtmlGenerator",
    "LLMResult",
    "_absolutize_images",
    "_extract_html_block",
    "_resolve_palette",
    "estimate_cost",
]
