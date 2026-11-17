"""SQLAlchemy ORM models for html_render_service.

Lifted from analysis_service's redesign domain. Table names are preserved
(``redesign_jobs`` etc.) so the cutover can REASSIGN/keep prod data if needed;
the service-internal vocabulary is "design jobs" but the schema is unchanged.
"""

from __future__ import annotations

import datetime as dt
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID  # noqa: N811
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from w2v_common.database.rls import register_rls_ddl


class Base(DeclarativeBase):
    """Declarative base for all html_render_service ORM models."""


#: ``tenant_id`` is a native Postgres ``uuid`` column (migration
#: ``0010_tenant_id_uuid``) — never varchar, never a sentinel. Fleet ruling.
#:
#: ``as_uuid=False`` IS LOAD-BEARING, NOT STYLE. With ``as_uuid=True`` SQLAlchemy
#: hands back ``uuid.UUID`` objects, and ``uuid.UUID(t) == t`` is ``False`` for
#: every ``str`` ``t``. This service compares tenant strings in ~20 places —
#: ``_get_tenant_id`` (``src/api/routes/_helpers.py``) returns the canonical
#: ``str(uuid)`` and it is threaded through ``JobManager``/``StyleManager``/
#: ``BatchManager`` filters, the ``(tenant_id, slug)`` style-description cache key
#: in ``design_builder/phases.py``, and the ``ClaimedJob`` snapshot the worker
#: re-binds identity from. None of those comparisons is a TYPE error, so ``ty``
#: and ``ruff`` stay green while ownership guards 404 the owner on their own rows
#: and scope predicates match nothing. ``as_uuid=False`` keeps the Postgres column
#: a real ``uuid`` (the entire point) while the Python-side value stays a ``str``,
#: so no comparison site changes. Pinned by ``tests/unit/test_tenant_id_column_type.py``.
TENANT_ID_TYPE = PgUUID(as_uuid=False)


class RedesignJob(Base):
    """Design/redesign job tracking."""

    __tablename__ = "redesign_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(TENANT_ID_TYPE, index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)

    # Input parameters
    url: Mapped[str] = mapped_column(String(2048))
    html_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope: Mapped[str] = mapped_column(String(20), default="full_page")
    style: Mapped[str] = mapped_column(String(255), default="modern")
    page_type: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Computed-assembly inputs (server_default keeps legacy rows on the freeform
    # path). ``mode='assembly'`` composes the page from the component catalog;
    # ``interactivity='interactive'`` inlines htmx/Alpine; ``embeds`` are the
    # three.js-seam sandboxed iframes.
    mode: Mapped[str] = mapped_column(String(20), default="freeform", server_default="freeform")
    interactivity: Mapped[str] = mapped_column(
        String(20), default="static", server_default="static"
    )
    component_set: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embeds: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # Automated verification lane (assembly mode gate). ``verification_status`` is
    # passed/failed/skipped; ``verification_report`` is the structured findings.
    verification_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    verification_report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Context data (JSON)
    critique_findings: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    brand_info: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    site_analysis: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Feature 1: Prompt templates
    template_version: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Feature 2: User context
    user_context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Feature 4: A/B variants
    styles: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    variant_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Feature 5: Iteration/refinement
    parent_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    root_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    iteration_number: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Feature 6: Component redesign
    component_target: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    component_html_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Feature 7: Accessibility scoring
    accessibility_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    accessibility_report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Feature 8: Visual comparison
    original_screenshot_paths: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)
    comparison_paths: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)
    diff_paths: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)

    # Feature 9: Webhooks
    webhook_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    webhook_secret: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Feature 10: Batch processing
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    priority_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Output
    redesigned_html_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    section_html_paths: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    screenshot_paths: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)
    # Discrete per-section redesign stills, section_name → stored path (migration
    # 0004). Populated only when the request opts into section_screenshots
    # (website_review after act); NULL for every other scope/consumer.
    section_screenshot_paths: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)
    concept_image_assets: Mapped[list[dict[str, object]] | None] = mapped_column(
        JSONB, nullable=True
    )
    # Set only when concept images were REQUESTED and generation FAILED (submit/
    # poll error, timeout, failed downstream batch). NULL = not requested,
    # succeeded, or legitimately produced zero — so a genuine failure is no longer
    # indistinguishable from "none requested". The job itself still completes.
    concept_image_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Deterministic HTML validation (Enhancement 4)
    html_validation_report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # LLM token tracking (Enhancement 14)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Metadata
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    pages_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Durable claim-queue (migration 0002). A row with job_options set is owned
    # by the worker pool; claimed_by/lease_expires_at/attempts drive lease-based
    # claiming and reclaim of jobs orphaned by a crashed/restarted pod.
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    job_options: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_redesign_jobs_status_created", "status", "created_at"),
        # Mirrors migration 0002's partial index so create_all-based tests build
        # the same shape the claim query relies on.
        Index(
            "ix_redesign_jobs_claimable",
            "status",
            "lease_expires_at",
            postgresql_where=text("job_options IS NOT NULL"),
        ),
    )


class RedesignUsage(Base):
    """Daily design usage aggregation per tenant."""

    __tablename__ = "redesign_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(TENANT_ID_TYPE, index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    pages_redesigned: Mapped[int] = mapped_column(Integer, default=0)
    screenshots_rendered: Mapped[int] = mapped_column(Integer, default=0)
    concept_images_generated: Mapped[int] = mapped_column(Integer, default=0)
    llm_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    variants_generated: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (UniqueConstraint("tenant_id", "date", name="uq_redesign_usage_tenant_date"),)


class RedesignVariant(Base):
    """Individual style variant within an A/B test job."""

    __tablename__ = "redesign_variants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    style: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), default="pending")

    redesigned_html_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    screenshot_paths: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)
    concept_image_assets: Mapped[list[dict[str, object]] | None] = mapped_column(
        JSONB, nullable=True
    )

    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CustomStyle(Base):
    """Tenant-scoped custom style definition."""

    __tablename__ = "custom_styles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    style_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(TENANT_ID_TYPE, index=True)
    name: Mapped[str] = mapped_column(String(100))
    slug: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    color_palette: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    typography: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    layout: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    example_references: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, onupdate=func.now()
    )

    # UNIQUE, not merely indexed: ``slug`` is the key every generation-path
    # lookup resolves a style by (``get_style_by_slug``, and the cache key that
    # now mirrors it), so two rows sharing one ``(tenant_id, slug)`` made the
    # winner arbitrary — and a slug-keyed cache would cross tenants outright.
    # Same index NAME as before so migration 0009 converts it in place.
    __table_args__ = (Index("ix_custom_styles_tenant_slug", "tenant_id", "slug", unique=True),)


class BatchJob(Base):
    """Batch design job for multi-page processing."""

    __tablename__ = "batch_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(TENANT_ID_TYPE, index=True)
    status: Mapped[str] = mapped_column(String(30), default="pending")

    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    completed_pages: Mapped[int] = mapped_column(Integer, default=0)
    failed_pages: Mapped[int] = mapped_column(Integer, default=0)
    shared_settings: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Why a batch needs its own reason column: ``mark_batch_failed`` has always
    # ACCEPTED an ``error_message`` and silently dropped it, so every terminal
    # batch failure — including the reaped-as-abandoned one — reached the client
    # as a bare ``failed`` with nothing to explain it. Mirrors
    # ``redesign_jobs.error_message``.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # THE PROGRESS CLOCK, and the reason the reaper is not a wall-clock ceiling.
    #
    # ``started_at`` is written exactly once (``mark_processing``) and never
    # again: ``increment_completed`` / ``increment_failed`` bump counters and
    # touch no timestamp. So an age measured from ``started_at`` is the age of
    # the WHOLE batch, and any ceiling built on it must exceed the longest
    # legitimate batch — which here is 20 pages (``BatchRedesignRequest.pages``
    # max_length) at ``batch_max_concurrency`` 3 = 7 waves, with NO wall-clock
    # cap on an in-process batch child at all (``max_job_lifetime_seconds``
    # bounds only the queue path). Every such number is a guess, and a guess that
    # is too small reaps a HEALTHY batch.
    #
    # This column removes the guess. It is bumped on every committed page
    # outcome, so the reaper asks "has this batch made no progress for a while",
    # which is the question it actually means — and the window no longer has to
    # scale with how many pages the client sent.
    last_progress_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WebhookDeliveryLog(Base):
    """Log of webhook delivery attempts."""

    __tablename__ = "webhook_delivery_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(50))
    request_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    response_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# Phase D RLS. (1) Strict per-table policies on the 4 tenant_id tables (redesign_jobs,
# redesign_usage, custom_styles, batch_jobs) via w2v_common. (2) Parent-derived join
# policies on the child tables (redesign_variants, webhook_delivery_log — no tenant_id).
# (3) The SECURITY DEFINER claim resolver for the cross-tenant lease queue. All
# registered as after_create events so tests' create_all matches the 0005_rls migration.
# Inert until the runtime connects as restricted w2v_html_render_app with
# TENANT_RLS_MODE=enforce.
register_rls_ddl(Base.metadata)

from src.database.rls_policies import register_child_policies  # noqa: E402
from src.database.rls_resolvers import register_claim_fn  # noqa: E402

register_child_policies(Base.metadata)
register_claim_fn(Base.metadata)
