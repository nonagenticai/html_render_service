"""HTML generation package: the ``HtmlGenerator`` facade + its helper modules.

Split out of the former 1319-line ``src/services/html_generator.py``. That module
is kept as a thin re-export facade so existing import paths keep working.
"""

from __future__ import annotations

from .brand_css import _DEFAULT_PRIMARY, _resolve_palette
from .generator import HtmlGenerator
from .html_utils import _extract_html_block
from .images import _absolutize_images
from .types import LLMResult, estimate_cost

__all__ = [
    "_DEFAULT_PRIMARY",
    "HtmlGenerator",
    "LLMResult",
    "_absolutize_images",
    "_extract_html_block",
    "_resolve_palette",
    "estimate_cost",
]
