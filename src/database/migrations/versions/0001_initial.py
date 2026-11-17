"""Initial schema for html_render_service.

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-03 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "redesign_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), server_default="pending", nullable=False),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("html_content", sa.Text(), nullable=True),
        sa.Column("scope", sa.String(20), server_default="full_page", nullable=False),
        sa.Column("style", sa.String(50), server_default="modern", nullable=False),
        sa.Column("page_type", sa.String(50), nullable=True),
        sa.Column("critique_findings", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("brand_info", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("site_analysis", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("template_version", sa.String(20), nullable=True),
        sa.Column("user_context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("styles", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("variant_count", sa.Integer(), nullable=True),
        sa.Column("parent_job_id", sa.String(36), nullable=True),
        sa.Column("root_job_id", sa.String(36), nullable=True),
        sa.Column("iteration_number", sa.Integer(), server_default="0", nullable=True),
        sa.Column("feedback_text", sa.Text(), nullable=True),
        sa.Column("component_target", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("component_html_path", sa.String(500), nullable=True),
        sa.Column("accessibility_score", sa.Float(), nullable=True),
        sa.Column("accessibility_report", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "original_screenshot_paths",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("comparison_paths", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("diff_paths", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("webhook_url", sa.String(2048), nullable=True),
        sa.Column("webhook_secret", sa.String(512), nullable=True),
        sa.Column("batch_id", sa.String(36), nullable=True),
        sa.Column("priority_score", sa.Float(), nullable=True),
        sa.Column("redesigned_html_path", sa.String(500), nullable=True),
        sa.Column("section_html_paths", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("screenshot_paths", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("concept_image_assets", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("html_validation_report", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("pages_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index(op.f("ix_redesign_jobs_job_id"), "redesign_jobs", ["job_id"], unique=True)
    op.create_index(
        op.f("ix_redesign_jobs_tenant_id"), "redesign_jobs", ["tenant_id"], unique=False
    )
    op.create_index(op.f("ix_redesign_jobs_status"), "redesign_jobs", ["status"], unique=False)
    op.create_index(
        op.f("ix_redesign_jobs_parent_job_id"),
        "redesign_jobs",
        ["parent_job_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_redesign_jobs_root_job_id"),
        "redesign_jobs",
        ["root_job_id"],
        unique=False,
    )
    op.create_index(op.f("ix_redesign_jobs_batch_id"), "redesign_jobs", ["batch_id"], unique=False)
    op.create_index(
        "ix_redesign_jobs_status_created",
        "redesign_jobs",
        ["status", "created_at"],
        unique=False,
    )

    op.create_table(
        "redesign_usage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("request_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("pages_redesigned", sa.Integer(), server_default="0", nullable=False),
        sa.Column("screenshots_rendered", sa.Integer(), server_default="0", nullable=False),
        sa.Column("concept_images_generated", sa.Integer(), server_default="0", nullable=False),
        sa.Column("llm_tokens_used", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("variants_generated", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "date", name="uq_redesign_usage_tenant_date"),
    )
    op.create_index(
        op.f("ix_redesign_usage_tenant_id"), "redesign_usage", ["tenant_id"], unique=False
    )
    op.create_index(op.f("ix_redesign_usage_date"), "redesign_usage", ["date"], unique=False)

    op.create_table(
        "redesign_variants",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("variant_id", sa.String(36), nullable=False),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("style", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), server_default="pending", nullable=False),
        sa.Column("redesigned_html_path", sa.String(500), nullable=True),
        sa.Column("screenshot_paths", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("concept_image_assets", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("variant_id"),
    )
    op.create_index(
        op.f("ix_redesign_variants_variant_id"),
        "redesign_variants",
        ["variant_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_redesign_variants_job_id"), "redesign_variants", ["job_id"], unique=False
    )

    op.create_table(
        "custom_styles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("style_id", sa.String(36), nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("slug", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("color_palette", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("typography", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("layout", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("example_references", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("style_id"),
    )
    op.create_index(op.f("ix_custom_styles_style_id"), "custom_styles", ["style_id"], unique=True)
    op.create_index(
        op.f("ix_custom_styles_tenant_id"), "custom_styles", ["tenant_id"], unique=False
    )
    op.create_index(
        "ix_custom_styles_tenant_slug",
        "custom_styles",
        ["tenant_id", "slug"],
        unique=False,
    )

    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("batch_id", sa.String(36), nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(30), server_default="pending", nullable=False),
        sa.Column("total_pages", sa.Integer(), server_default="0", nullable=False),
        sa.Column("completed_pages", sa.Integer(), server_default="0", nullable=False),
        sa.Column("failed_pages", sa.Integer(), server_default="0", nullable=False),
        sa.Column("shared_settings", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("batch_id"),
    )
    op.create_index(op.f("ix_batch_jobs_batch_id"), "batch_jobs", ["batch_id"], unique=True)
    op.create_index(op.f("ix_batch_jobs_tenant_id"), "batch_jobs", ["tenant_id"], unique=False)

    op.create_table(
        "webhook_delivery_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("request_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("response_status", sa.Integer(), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), server_default="pending", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_webhook_delivery_log_job_id"),
        "webhook_delivery_log",
        ["job_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_webhook_delivery_log_job_id"), table_name="webhook_delivery_log")
    op.drop_table("webhook_delivery_log")

    op.drop_index(op.f("ix_batch_jobs_tenant_id"), table_name="batch_jobs")
    op.drop_index(op.f("ix_batch_jobs_batch_id"), table_name="batch_jobs")
    op.drop_table("batch_jobs")

    op.drop_index("ix_custom_styles_tenant_slug", table_name="custom_styles")
    op.drop_index(op.f("ix_custom_styles_tenant_id"), table_name="custom_styles")
    op.drop_index(op.f("ix_custom_styles_style_id"), table_name="custom_styles")
    op.drop_table("custom_styles")

    op.drop_index(op.f("ix_redesign_variants_job_id"), table_name="redesign_variants")
    op.drop_index(op.f("ix_redesign_variants_variant_id"), table_name="redesign_variants")
    op.drop_table("redesign_variants")

    op.drop_index(op.f("ix_redesign_usage_date"), table_name="redesign_usage")
    op.drop_index(op.f("ix_redesign_usage_tenant_id"), table_name="redesign_usage")
    op.drop_table("redesign_usage")

    op.drop_index("ix_redesign_jobs_status_created", table_name="redesign_jobs")
    op.drop_index(op.f("ix_redesign_jobs_batch_id"), table_name="redesign_jobs")
    op.drop_index(op.f("ix_redesign_jobs_root_job_id"), table_name="redesign_jobs")
    op.drop_index(op.f("ix_redesign_jobs_parent_job_id"), table_name="redesign_jobs")
    op.drop_index(op.f("ix_redesign_jobs_status"), table_name="redesign_jobs")
    op.drop_index(op.f("ix_redesign_jobs_tenant_id"), table_name="redesign_jobs")
    op.drop_index(op.f("ix_redesign_jobs_job_id"), table_name="redesign_jobs")
    op.drop_table("redesign_jobs")
