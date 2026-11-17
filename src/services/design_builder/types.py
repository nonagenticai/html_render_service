"""Phase value-objects (private dataclasses) used by ``RedesignService``."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.services.context_resolver import ResolvedContext


@dataclass(frozen=True)
class RenderOptions:
    """Per-request screenshot-shaping knobs.

    Defaults reproduce the historical behavior (full multi-viewport,
    full-page capture of both the live original and the redesign, plus a
    before/after comparison) so existing consumers (redesign UI /
    website_review) are unaffected. The promo fast-path opts into a lean
    shape — one viewport, viewport-framed, no original capture, no
    comparison — because it consumes a single hero image and the full set
    is what OOM-killed the pod (6 full-page Chromium captures of a heavy
    live SPA + a PIL diff at a 3Gi limit).
    """

    capture_original: bool = True
    generate_comparison: bool = True
    viewports: tuple[str, ...] | None = None  # None → settings.screenshot.viewports
    full_page: bool = True
    # When True, also capture discrete per-section stills of the composite redesign
    # (one PNG per landmark section) so a consumer can sync each section to the
    # narration line that discusses it instead of panning one tall scroll. Opt-in
    # (website_review's after act); the full-page screenshot is always captured too.
    section_screenshots: bool = False


@dataclass
class _ThemeDecision:
    """Result of style-resolution + optional accessibility-first auto-theming."""

    style: str
    style_description: str | None
    brand_info: dict | None
    context: ResolvedContext
    auto_theme_metadata: dict | None


@dataclass
class _GenerationResult:
    """Result of HTML generation phase (component / full_page / sections)."""

    redesigned_html: str
    redesigned_html_path: str | None
    section_html_paths: list[str] | None
    component_html_path: str | None
    input_tokens: int
    output_tokens: int
    # Compromises the generation made to finish within the job budget (see SectionRun).
    quality_warnings: list[str] = field(default_factory=list)
    # Assembly only: the generated hero image (inline data: URI), kept so
    # a verify/refine re-plan reuses it instead of paying for a second generation.
    # ``""`` = generation was attempted and failed (gradient); ``None`` = not attempted.
    hero_image_url: str | None = None
    # Assembly only: the outcome of the hero's visible-text check (owner decision),
    # persisted as ``html_validation_report.hero_image_check`` and carried across a
    # refine re-plan with ``hero_image_url``. ``None`` = no hero / no check.
    hero_image_check: dict[str, object] | None = None


@dataclass
class _A11yResult:
    """Result of optional accessibility scoring phase."""

    score: float | None
    report: dict | None
    redesigned_html_path: str | None  # set if auto-fix wrote a new file


@dataclass
class _ScreenshotsResult:
    """Result of screenshot capture/render/comparison phases."""

    original_paths: dict[str, str] | None
    screenshot_paths: dict[str, str] | None
    comparison_paths: dict[str, str] | None
    diff_paths: dict[str, str] | None
    screenshots_count: int
    # Discrete per-section redesign stills, section_name → stored path. None unless
    # RenderOptions.section_screenshots was set (website_review after act).
    section_paths: dict[str, str] | None = None


@dataclass
class _TokenAccumulator:
    """Mutable token accumulator threaded through phase methods."""

    input: int = 0
    output: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input += input_tokens
        self.output += output_tokens

    @property
    def total(self) -> int:
        return self.input + self.output

    def as_dict(self) -> dict[str, int] | None:
        if not (self.input or self.output):
            return None
        return {
            "prompt_tokens": self.input,
            "completion_tokens": self.output,
            "total_tokens": self.total,
        }
