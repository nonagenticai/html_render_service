"""Add ``redesign_jobs.section_screenshot_paths`` (per-section redesign stills).

The website_review video's after act keys every narration line to the redesign's
ONE full-page screenshot panned top-to-bottom (``fit=scroll``), which restarts on
each line and reads as an unwatchable repeating scroll. To sync each line to the
section it discusses, the redesign job now also captures a discrete still per
landmark section (opt-in ``section_screenshots``) and surfaces them as
``redesign_section:<name>`` visual assets. This column persists those stills as a
``section_name → stored_path`` JSONB map, mirroring ``screenshot_paths``.

Nullable + additive: NULL for every job that doesn't opt in, so existing rows and
all other scopes/consumers are unaffected. ``ADD COLUMN IF NOT EXISTS`` keeps it
re-runnable against a create_all-built test DB. The service role owns the table
(created in 0001), so the ALTER's ownership check passes.

Revision ID: 0004_section_screenshot_paths
Revises: 0003_widen_style
Create Date: 2026-06-08 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_section_screenshot_paths"
down_revision: str | None = "0003_widen_style"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE redesign_jobs ADD COLUMN IF NOT EXISTS section_screenshot_paths JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS section_screenshot_paths")
