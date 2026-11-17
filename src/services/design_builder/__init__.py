"""Redesign service package — public API preserved via re-exports.

Decomposed from a single ``redesign_service.py`` (1457 LOC) into focused
submodules; all identifiers previously importable as
``from src.services.design_builder import X`` remain
available at this same path. Names that test code patches at the
package level (``BatchManager``, ``JobManager``, ``select_pages``,
``resolve_context``, ``capture_url_screenshots``, ``render_from_html_string``,
``generate_comparisons``) are also re-exported here so that ``mock.patch``
on the package namespace continues to take effect.
"""

from __future__ import annotations

# Patchable symbols — submodules look these up via the package at call time.
from src.services.batch_manager import BatchManager  # noqa: F401
from src.services.comparison_generator import generate_comparisons  # noqa: F401
from src.services.context_resolver import resolve_context  # noqa: F401
from src.services.design_builder.service import RedesignService
from src.services.design_builder.types import (
    _A11yResult,
    _GenerationResult,
    _ScreenshotsResult,
    _ThemeDecision,
    _TokenAccumulator,
)
from src.services.job_manager import JobManager  # noqa: F401
from src.services.page_selector import select_pages  # noqa: F401
from src.services.screenshot_renderer import (  # noqa: F401
    capture_url_screenshots,
    render_from_html_string,
    render_section_screenshots,
)

__all__ = [
    "RedesignService",
    "_A11yResult",
    "_GenerationResult",
    "_ScreenshotsResult",
    "_ThemeDecision",
    "_TokenAccumulator",
]
