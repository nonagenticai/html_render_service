"""Request/response Pydantic models for redesign_service."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator
from w2v_common.net import SSRFError, validate_public_url

# ============================================================================
# Shared Input Models
# ============================================================================


class LLMConfigOverride(BaseModel):
    """Optional LLM configuration override for per-request model switching."""

    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None


class CritiqueFindingInput(BaseModel):
    """A critique finding to address in the redesign."""

    severity: str = Field(
        default="medium", description="Finding severity: low, medium, high, critical"
    )
    title: str = Field(..., description="Short title of the finding")
    description: str = Field(..., description="Detailed description of the issue")
    recommendation: str | None = Field(default=None, description="Suggested fix")
    category: str | None = Field(
        default=None, description="Category: seo, security, a11y, performance, ux"
    )


class BrandInfoInput(BaseModel):
    """Brand identity data for consistent redesign."""

    name: str | None = Field(default=None, description="Brand/site name")
    primary_color: str = Field(default="#2563EB", description="Primary brand color (hex)")
    secondary_color: str = Field(default="#1E40AF", description="Secondary brand color (hex)")
    font_family: str | None = Field(
        default=None, description="Brand font family (display fallback)"
    )
    display_font: str | None = Field(default=None, description="Heading/display font family")
    body_font: str | None = Field(default=None, description="Body text font family")
    logo_url: str | None = Field(default=None, description="URL to brand logo")


class SiteAnalysisInput(BaseModel):
    """Condensed site analysis context for the redesign."""

    description: str | None = Field(default=None, description="Site description")
    target_audience: list[str] = Field(default_factory=list, description="Target audience segments")
    tone: str | None = Field(default=None, description="Brand tone: professional, casual, etc.")
    visual_identity: dict | None = Field(default=None, description="Visual identity attributes")


# ============================================================================
# Feature 2: User Context
# ============================================================================


class FeaturePriorityInput(BaseModel):
    """Priority weighting for a specific feature or improvement area."""

    feature: str = Field(..., description="Feature name (e.g., 'navigation', 'search')")
    priority: str = Field(default="medium", description="Priority: low, medium, high, critical")
    notes: str | None = Field(default=None, description="Additional context")


class VisualPreferencesInput(BaseModel):
    """Visual design preferences from the user."""

    color_scheme: str | None = Field(default=None, description="light, dark, colorful, monochrome")
    layout_style: str | None = Field(default=None, description="card-based, list, grid, magazine")
    animation_level: str | None = Field(default=None, description="none, subtle, moderate, dynamic")
    image_style: str | None = Field(
        default=None, description="photography, illustration, abstract, minimal"
    )


class UserContextInput(BaseModel):
    """Rich user context for better redesign targeting."""

    description: str | None = Field(default=None, description="Site/business description")
    site_type: str | None = Field(
        default=None, description="ecommerce, saas, blog, portfolio, corporate, landing"
    )
    target_audience: list[str] | None = Field(default=None, description="Target audience segments")
    goals: list[str] | None = Field(
        default=None, description="Redesign goals (e.g., increase_conversions)"
    )
    focus_pages: list[str] | None = Field(default=None, description="URLs to prioritize in batch")
    exclude_pages: list[str] | None = Field(default=None, description="URLs to skip in batch")
    feature_priorities: list[FeaturePriorityInput] | None = Field(
        default=None, description="Feature priority list"
    )
    visual_preferences: VisualPreferencesInput | None = Field(
        default=None, description="Visual design preferences"
    )


# ============================================================================
# Feature 3: Custom Style Definitions
# ============================================================================


class ColorPaletteInput(BaseModel):
    """Color palette for a custom style."""

    primary: str | None = Field(default=None, description="Primary color hex")
    secondary: str | None = Field(default=None, description="Secondary color hex")
    accent: str | None = Field(default=None, description="Accent color hex")
    background: str | None = Field(default=None, description="Background color hex")
    text: str | None = Field(default=None, description="Text color hex")


class TypographyInput(BaseModel):
    """Typography settings for a custom style."""

    heading_font: str | None = Field(default=None, description="Heading font family")
    body_font: str | None = Field(default=None, description="Body text font family")
    scale: str | None = Field(default=None, description="Type scale: compact, default, spacious")


class LayoutPreferencesInput(BaseModel):
    """Layout preferences for a custom style."""

    max_width: str | None = Field(default=None, description="Max content width (e.g., '1200px')")
    spacing: str | None = Field(
        default=None, description="Spacing density: tight, normal, generous"
    )
    grid_columns: int | None = Field(default=None, description="Grid column count")
    border_radius: str | None = Field(
        default=None, description="Border radius: none, small, medium, large, pill"
    )


class CreateCustomStyleRequest(BaseModel):
    """Request to create a custom style definition."""

    name: str = Field(..., max_length=100, description="Style name")
    description: str | None = Field(default=None, max_length=500, description="Style description")
    color_palette: ColorPaletteInput | None = Field(default=None)
    typography: TypographyInput | None = Field(default=None)
    layout: LayoutPreferencesInput | None = Field(default=None)
    example_references: list[str] | None = Field(
        default=None, max_length=5, description="Reference site URLs"
    )


class UpdateCustomStyleRequest(BaseModel):
    """Request to update a custom style definition."""

    name: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    color_palette: ColorPaletteInput | None = Field(default=None)
    typography: TypographyInput | None = Field(default=None)
    layout: LayoutPreferencesInput | None = Field(default=None)
    example_references: list[str] | None = Field(default=None, max_length=5)
    is_active: bool | None = Field(default=None)


class CustomStyleResponse(BaseModel):
    """Response for a custom style."""

    id: str
    name: str
    slug: str
    description: str | None = None
    color_palette: dict | None = None
    typography: dict | None = None
    layout: dict | None = None
    example_references: list[str] | None = None
    is_active: bool = True
    created_at: str | None = None


class StyleListResponse(BaseModel):
    """Response listing custom styles."""

    styles: list[CustomStyleResponse]
    count: int


class StylePreviewRequest(BaseModel):
    """Request to preview a style's effect."""

    html_snippet: str = Field(..., max_length=5000, description="HTML to preview the style on")


class StylePreviewResponse(BaseModel):
    """Response with styled HTML preview."""

    styled_html: str
    style_description: str


# ============================================================================
# Feature 6: Component Redesign
# ============================================================================


class ComponentTargetInput(BaseModel):
    """Target component for component-level redesign."""

    component_type: str = Field(
        ...,
        description="Component type: navbar, hero, footer, cta, pricing, testimonials, features, contact, sidebar",
    )
    component_selector: str | None = Field(default=None, description="Custom CSS selector override")


# ============================================================================
# Computed-assembly inputs (mode / interactivity / embeds)
# ============================================================================


class EmbedSpec(BaseModel):
    """An external interactive artifact embedded into an assembled page.

    The three.js input seam: a future three.js service (or any producer) yields a
    fetchable ``artifact_url`` and this spec tells the assembler where and how to
    mount it. Injection is deterministic — a sandboxed ``<iframe>`` placed at the
    named mount point — so the LLM never emits the embed markup itself.
    """

    type: str = Field(
        default="iframe",
        max_length=32,
        description="Embed kind (e.g. 'threejs', 'iframe'). Rendered as a sandboxed iframe.",
    )
    artifact_url: str = Field(
        ..., max_length=2048, description="Fetchable URL of the artifact to embed."
    )
    mount: str = Field(
        default="hero",
        max_length=64,
        description="Named mount point / block slot where the embed is injected.",
    )
    title: str | None = Field(default=None, max_length=200, description="Accessible iframe title.")
    sandbox: bool = Field(
        default=True, description="Wrap the embed in a sandboxed iframe (strongly recommended)."
    )
    aspect_ratio: str = Field(
        default="16 / 9", max_length=16, description="CSS aspect-ratio for the embed frame."
    )

    @field_validator("artifact_url")
    @classmethod
    def _require_public_http_url(cls, v: str) -> str:
        """Reject embed artifacts that point at estate-internal addresses.

        WHY this is an SSRF sink and not an inert string: the embed is injected
        into the assembled page as an ``<iframe src=...>``, and that page is then
        rasterised by our own headless Chromium (``render_from_html_string``)
        inside the cluster. Chromium loads the iframe, so the *pixels* of
        whatever ``artifact_url`` points at are baked into a screenshot that is
        handed straight back to the tenant. ``sandbox="allow-scripts"`` stops
        script-level exfiltration but does nothing about the raster.

        ``embeds`` is purely tenant-supplied — the orchestrator passes
        ``brief["embeds"]`` through verbatim (``web_design_v2/design.py``) and
        nothing in this service ever constructs an ``EmbedSpec`` server-side, so
        guarding it cannot break an S2S/MinIO artifact flow. The previous check
        was ``startswith("http")``, which is a scheme check, not an SSRF guard:
        ``http://100.64.0.5:5432/`` passed it, and RFC 6598 CGNAT is this
        fleet's tailnet (``ipaddress.is_private`` is False for it).

        Raises:
            ValueError: non-http(s) scheme, unparseable host, or a
                blocked/internal target. Pydantic surfaces it as a 422.
        """
        try:
            validate_public_url(v)
        except (SSRFError, ValueError) as exc:
            raise ValueError(f"artifact_url is not a valid public http(s) URL: {exc}") from exc
        return v


# ============================================================================
# Main Redesign Request (all features integrated)
# ============================================================================


class RedesignRequest(BaseModel):
    """Request body for a redesign job."""

    url: str = Field(
        default="",
        description="URL of the page to design/redesign (optional for scope=landing)",
    )
    html_content: str | None = Field(
        default=None, max_length=100000, description="Raw HTML content (if not fetching live)"
    )
    scope: str = Field(
        default="full_page",
        pattern=r"^(full_page|sections|section|component|hero|landing|redesign)$",
        description=(
            "Design scope. full_page/redesign = whole-page (chunked) generation; "
            "section(s) = per-section; component = a single component (needs "
            "component_target); hero = fast single hero treatment (the promo fast "
            "path); landing = from-scratch landing page (url optional)."
        ),
    )
    style: str = Field(default="modern", max_length=100, description="Design style name or slug")
    page_type: str | None = Field(
        default=None, description="Page type: homepage, landing, product, etc."
    )
    critique_findings: list[CritiqueFindingInput] | None = Field(
        default=None, description="Critique findings to address"
    )
    brand: BrandInfoInput | None = Field(default=None, description="Brand identity data")
    site_analysis: SiteAnalysisInput | None = Field(
        default=None, description="Site analysis context"
    )
    generate_screenshots: bool = Field(
        default=True, description="Render screenshots of the redesign"
    )
    generate_concept_images: bool = Field(default=False, description="Generate AI concept images")

    # Screenshot shaping (defaults reproduce full historical behavior; the promo
    # fast-path opts into a lean single-shot to avoid OOMing on heavy live SPAs).
    capture_original: bool = Field(
        default=True, description="Also screenshot the live original URL (for before/after)"
    )
    generate_comparison: bool = Field(
        default=True, description="Generate before/after comparison + diff images"
    )
    screenshot_viewports: list[str] | None = Field(
        default=None,
        description=(
            "Override viewports to render (e.g. ['desktop']); None = service default. "
            "Must be known viewport names — an unknown one is rejected here rather "
            "than silently skipped by the renderer (which yields zero screenshots)."
        ),
    )
    full_page: bool = Field(
        default=True,
        description="Capture full scroll height (True) or just the viewport (False)",
    )
    section_screenshots: bool = Field(
        default=False,
        description=(
            "Also capture a discrete still per landmark section of the composite "
            "redesign (surfaced as redesign_section:<name> visual assets) so a "
            "consumer can sync each section to the narration line that discusses it"
        ),
    )

    # Feature 2: User context
    user_context: UserContextInput | None = Field(default=None, description="Rich user context")

    # Feature 4: A/B variants
    styles: list[str] | None = Field(
        default=None, description="Multiple styles for A/B variant generation (max 4)"
    )

    # Feature 6: Component redesign
    component_target: ComponentTargetInput | None = Field(
        default=None, description="Component target for scope=component"
    )

    # Feature 7: Accessibility
    run_accessibility_check: bool = Field(
        default=False, description="Run accessibility scoring after generation"
    )
    auto_fix_accessibility: bool = Field(default=False, description="Auto-fix accessibility issues")
    accessibility_threshold: float = Field(
        default=70.0, description="Minimum accessibility score (0-100)"
    )
    accessibility_first_auto_theme: bool = Field(
        default=False,
        description=(
            "Apply accessibility-first auto theme (style/palette tuning from critique findings) "
            "and revalidate with critique adapters"
        ),
    )

    # Feature 9: Webhooks
    webhook_url: str | None = Field(
        default=None, max_length=2048, description="Webhook URL for job notifications"
    )
    webhook_secret: str | None = Field(
        default=None, max_length=512, description="Webhook HMAC secret"
    )

    # LLM config override
    llm_config: LLMConfigOverride | None = Field(
        default=None, description="Optional LLM configuration override"
    )

    # ---- Computed-assembly + interactivity (defaults reproduce legacy behavior) ----
    mode: str = Field(
        default="freeform",
        pattern=r"^(freeform|assembly)$",
        description=(
            "Generation mode. 'freeform' = the legacy LLM-writes-CSS path (default, "
            "byte-identical for existing callers). 'assembly' = compose the page from "
            "the vetted open-source component catalog (LLM selects blocks + fills "
            "content; scaffolding/theming/assets are deterministic)."
        ),
    )
    interactivity: str = Field(
        default="static",
        pattern=r"^(static|interactive)$",
        description=(
            "'static' = no client-side runtime. 'interactive' = allow vendored "
            "htmx/Alpine + inline JS in assembled output (assembly mode only)."
        ),
    )
    component_set: str | None = Field(
        default=None,
        max_length=64,
        description="Component library variant for assembly mode (default: settings.catalog).",
    )
    embeds: list[EmbedSpec] = Field(
        default_factory=list,
        description="External interactive artifacts (three.js seam) injected as sandboxed iframes.",
    )

    @field_validator("url")
    @classmethod
    def validate_render_target(cls, v: str) -> str:
        """Reject render targets that point at estate-internal addresses.

        WHY: ``url`` is the tenant-supplied page this service navigates to with
        headless Chromium — ``_fetch_page_html`` (to read the DOM the LLM works
        from) and ``capture_url_screenshots`` (the before/after original). Both
        run *inside* the cluster and both hand their output back to the tenant,
        so an unguarded value is a full SSRF read primitive against the estate.
        RFC 6598 ``100.64.0.0/10`` is this fleet's tailnet (prod Postgres, the
        registry, the k3s control plane) and ``ipaddress.is_private`` is False
        for it, which is why this must use the shared ``validate_public_url``
        rather than a hand-rolled private-range check.

        Empty is legal and is NOT validated: ``url`` defaults to ``""`` and
        ``validate_scope_requirements`` deliberately permits a missing source
        for ``scope='landing'`` and for ``mode='assembly'`` (both generate from
        a brief, greenfield, with no page to fetch). ``_phase_resolve_html``
        short-circuits on a falsy ``url`` for exactly that reason. Only a
        non-empty value describes a target we will actually navigate to.

        Raises:
            ValueError: non-http(s) scheme, unparseable host, or a
                blocked/internal target. Pydantic surfaces it as a 422.
        """
        if not v or not v.strip():
            return v
        try:
            validate_public_url(v.strip())
        except (SSRFError, ValueError) as exc:
            raise ValueError(f"url is not a valid public http(s) URL: {exc}") from exc
        return v

    @field_validator("screenshot_viewports")
    @classmethod
    def validate_screenshot_viewports(cls, v: list[str] | None) -> list[str] | None:
        """An unknown viewport must 422 here, not silently render zero screenshots."""
        from src.services.screenshot_renderer import validate_viewport_names

        return validate_viewport_names(v)

    @field_validator("styles")
    @classmethod
    def validate_styles(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            if len(v) > 4:
                raise ValueError("Maximum 4 style variants allowed")
            if len(v) != len(set(v)):
                raise ValueError("Duplicate styles not allowed")
        return v

    @model_validator(mode="after")
    def validate_scope_requirements(self) -> RedesignRequest:
        # scope=component needs an explicit target; scope=hero auto-targets the
        # hero section (no target required). landing can generate from scratch
        # (url optional); every other scope needs a url or inline html_content.
        if self.scope == "component" and self.component_target is None:
            raise ValueError("component_target is required when scope is 'component'")
        # Assembly mode is greenfield-from-brief; it never requires a source URL.
        if (
            self.mode != "assembly"
            and self.scope != "landing"
            and not self.url
            and not self.html_content
        ):
            raise ValueError("url (or html_content) is required unless scope is 'landing'")
        return self


# ============================================================================
# Job Responses
# ============================================================================


class AsyncJobResponse(BaseModel):
    """Response when a redesign job is created."""

    job_id: str
    status: str = "pending"
    poll_url: str
    webhook_configured: bool = False


class ScreenshotInfo(BaseModel):
    """Screenshot file info."""

    viewport: str
    path: str
    width: int
    height: int


class AccessibilityFindingResponse(BaseModel):
    """A single accessibility finding."""

    category: str
    severity: str
    description: str
    element: str | None = None
    recommendation: str | None = None
    wcag_criterion: str | None = None


class AccessibilityCategoryResponse(BaseModel):
    """Score for an accessibility category."""

    category: str
    # None = this category was never scored (the LLM reply omitted it). A number
    # here would be indistinguishable from a real measurement.
    score: float | None = None
    findings: list[AccessibilityFindingResponse] = []


class AccessibilityScoreResponse(BaseModel):
    """Accessibility assessment results."""

    # None = UNSCORED (assessment did not happen). Never substitute a number.
    overall_score: float | None = None
    unscored_reason: str | None = None
    categories: list[AccessibilityCategoryResponse] = []
    findings: list[AccessibilityFindingResponse] = []
    html_was_fixed: bool = False
    auto_theme: dict | None = None
    critique_adapter_validation: dict | None = None


class VariantResult(BaseModel):
    """Result for a single A/B variant."""

    variant_id: str
    style: str
    status: str
    redesigned_html_path: str | None = None
    screenshot_paths: dict[str, str] | None = None
    error: str | None = None


class VisualAssetResponse(BaseModel):
    """Structured visual asset metadata for downstream consumers."""

    asset_id: str
    title: str
    kind: str
    role: str
    url: str
    source: str
    page_url: str | None = None
    metadata: dict | None = None


class ConceptImageAssetResponse(BaseModel):
    """Resolved concept-image asset metadata from ai_content_generation_service."""

    job_id: str | None = None
    asset_id: str
    provider: str | None = None
    status: str = "completed"
    prompt: str | None = None
    url: str
    download_url: str | None = None
    presigned_url: str | None = None
    image_url: str | None = None
    target_usage: str | None = None
    usage_context: dict | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    file_size_bytes: int | None = None
    cost: float | None = None
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    metadata: dict | None = None


class RedesignResultResponse(BaseModel):
    """Full result of a completed redesign job."""

    job_id: str
    url: str
    scope: str
    style: str
    redesigned_html_path: str | None = None
    section_html_paths: list[str] | None = None
    screenshots: dict[str, str] | None = None
    # Discrete per-section redesign stills, section_name → fetchable URL (also
    # surfaced individually in visual_assets as redesign_section:<name>).
    section_screenshots: dict[str, str] | None = None
    concept_image_assets: list[ConceptImageAssetResponse] | None = None
    # True + reason when concept images were requested and generation FAILED —
    # so an empty concept_image_assets is not mistaken for "none requested".
    concept_image_degraded: bool = False
    concept_image_error: str | None = None
    visual_assets: list[VisualAssetResponse] | None = None

    # Feature 4: Variants
    variants: list[VariantResult] | None = None

    # Feature 6: Component
    component_html_path: str | None = None

    # Feature 7: Accessibility
    accessibility: AccessibilityScoreResponse | None = None

    # Feature 8: Comparisons
    original_screenshots: dict[str, str] | None = None
    comparisons: dict[str, str] | None = None
    diffs: dict[str, str] | None = None

    # Enhancement 4: Deterministic HTML validation
    html_validation: dict | None = None

    # Automated verification lane (assembly mode): passed/failed/skipped + report
    verification_status: str | None = None
    verification_report: dict | None = None

    # Token usage tracking
    token_usage: dict | None = None


class IterationSummary(BaseModel):
    """Summary of a single iteration in the refinement chain."""

    job_id: str
    iteration_number: int
    status: str
    feedback: str | None = None
    created_at: str | None = None


class JobStatusResponse(BaseModel):
    """Response for job status polling."""

    job_id: str
    status: str
    url: str | None = None
    scope: str | None = None
    style: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result_url: str | None = None

    # Concept-image degradation: True + reason when concept images were requested
    # and generation FAILED (distinct from "none requested"). The job still
    # completed — the main HTML/screenshots are available.
    concept_image_degraded: bool = False
    concept_image_error: str | None = None

    # Feature 5: Iteration
    iteration_number: int | None = None
    parent_job_id: str | None = None
    root_job_id: str | None = None
    refine_url: str | None = None

    # Feature 7: Accessibility
    accessibility_score: float | None = None

    # Feature 10: Batch
    batch_id: str | None = None

    # Token usage tracking
    token_usage: dict | None = None


# ============================================================================
# Feature 4: Variant Comparison
# ============================================================================


class VariantComparisonResponse(BaseModel):
    """Side-by-side comparison of all variants for a job."""

    job_id: str
    variants: list[VariantResult]


# ============================================================================
# Feature 5: Iteration/Refinement
# ============================================================================


class RefineRequest(BaseModel):
    """Request to refine an existing redesign with feedback."""

    feedback: str = Field(..., max_length=2000, description="Textual feedback for refinement")
    generate_screenshots: bool = Field(default=True)
    generate_concept_images: bool = Field(default=False)
    focus_areas: list[str] | None = Field(
        default=None,
        description="Structured feedback areas: layout, colors, typography, content, spacing",
    )
    llm_config: LLMConfigOverride | None = Field(
        default=None, description="Optional LLM configuration override"
    )


class IterationHistoryResponse(BaseModel):
    """Full iteration history for a root job."""

    root_job_id: str
    iterations: list[IterationSummary]
    total_iterations: int


# ============================================================================
# Feature 9: Webhooks
# ============================================================================


class WebhookEventPayload(BaseModel):
    """Standard webhook event payload."""

    event: str
    job_id: str
    timestamp: str
    status: str | None = None
    error: str | None = None
    result_url: str | None = None
    batch_id: str | None = None
    variant_id: str | None = None
    accessibility_score: float | None = None


class WebhookDeliveryStatus(BaseModel):
    """Status of a webhook delivery attempt."""

    attempt_number: int
    event_type: str
    status: str
    response_status: int | None = None
    created_at: str | None = None


# ============================================================================
# Feature 10: Batch Processing
# ============================================================================


class BatchPageInput(BaseModel):
    """A single page in a batch redesign request."""

    url: str = Field(..., description="Page URL")
    page_type: str | None = Field(default=None, description="Page type")
    html_content: str | None = Field(default=None, max_length=100000)

    @field_validator("url")
    @classmethod
    def validate_render_target(cls, v: str) -> str:
        """Reject batch render targets that point at estate-internal addresses.

        WHY here as well as on ``RedesignRequest.url``: each ``BatchPageInput``
        becomes its own ``redesign_jobs`` row and reaches the same headless-
        Chromium sinks, so leaving it unguarded would make the batch route a
        trivially-scriptable bypass of the single-job admission check (and a
        20-target-per-request internal port scanner). Unlike ``RedesignRequest``
        this field is required, so an empty value is a caller error — but it is
        still short-circuited rather than handed to a resolver, matching the
        single-job validator exactly.

        Raises:
            ValueError: non-http(s) scheme, unparseable host, or a
                blocked/internal target. Pydantic surfaces it as a 422.
        """
        if not v or not v.strip():
            return v
        try:
            validate_public_url(v.strip())
        except (SSRFError, ValueError) as exc:
            raise ValueError(f"url is not a valid public http(s) URL: {exc}") from exc
        return v


class BatchRedesignRequest(BaseModel):
    """Request for batch multi-page redesign."""

    pages: list[BatchPageInput] = Field(
        ..., min_length=1, max_length=20, description="Pages to redesign"
    )
    style: str = Field(default="modern", max_length=100)
    scope: str = Field(default="full_page", pattern=r"^(full_page|sections)$")
    generate_screenshots: bool = Field(default=True)
    generate_concept_images: bool = Field(default=False)
    critique_findings: list[CritiqueFindingInput] | None = Field(default=None)
    brand: BrandInfoInput | None = Field(default=None)
    site_analysis: SiteAnalysisInput | None = Field(default=None)
    user_context: UserContextInput | None = Field(default=None)
    webhook_url: str | None = Field(default=None, max_length=2048)
    webhook_secret: str | None = Field(default=None, max_length=512)


class BatchPageJobInfo(BaseModel):
    """Info about a single page job within a batch."""

    job_id: str
    url: str
    page_type: str | None = None
    priority_score: float | None = None
    status: str = "pending"


class BatchJobResponse(BaseModel):
    """Response when a batch job is created."""

    batch_id: str
    status: str = "pending"
    total_pages: int
    page_jobs: list[BatchPageJobInfo]
    poll_url: str


class BatchPageStatusInfo(BaseModel):
    """Status of a single page within a batch."""

    job_id: str
    url: str
    status: str
    error: str | None = None


class BatchStatusResponse(BaseModel):
    """Status of a batch redesign job."""

    batch_id: str
    status: str
    total_pages: int
    completed_pages: int
    failed_pages: int
    page_statuses: list[BatchPageStatusInfo]
    # The batch-level reason. A terminal batch used to arrive as a bare
    # ``failed`` with the explanation dropped on the floor, and an ABANDONED
    # batch never arrived at all — it stayed ``processing`` forever. Both now
    # carry why. Optional so existing clients are unaffected.
    error: str | None = None


class BatchPageErrorInfo(BaseModel):
    """Error info for a failed page in a batch."""

    job_id: str
    url: str
    error: str


class BatchResultsResponse(BaseModel):
    """Results of a completed batch job."""

    batch_id: str
    status: str
    total_pages: int
    completed_pages: int
    failed_pages: int
    results: list[RedesignResultResponse]
    errors: list[BatchPageErrorInfo]


# ============================================================================
# Enhancement 2: Job Cancellation & Listing
# ============================================================================


class JobListItem(BaseModel):
    """Summary of a job in a listing."""

    job_id: str
    url: str
    status: str
    scope: str | None = None
    style: str | None = None
    page_type: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    jobs: list[JobListItem]
    total: int
    limit: int
    offset: int


class CancelJobResponse(BaseModel):
    """Response when a job is cancelled."""

    job_id: str
    status: str = "cancelled"
    cancelled_at: str


# ============================================================================
# Enhancement 11: Batch Enhancements
# ============================================================================


class BatchListItem(BaseModel):
    """Summary of a batch in a listing."""

    batch_id: str
    status: str
    total_pages: int
    completed_pages: int
    failed_pages: int
    created_at: str | None = None


class BatchListResponse(BaseModel):
    """List of batches for a tenant."""

    batches: list[BatchListItem]
    total: int


class BatchRetryResponse(BaseModel):
    """Response when retrying failed batch pages."""

    batch_id: str
    retried_jobs: list[str]
    count: int
