"""Add computed-assembly columns to ``redesign_jobs`` (mode/interactivity/embeds).

Assembly mode composes a page from the vetted component catalog instead of the
LLM hand-writing CSS. These columns persist the per-job design inputs:

* ``mode`` — ``freeform`` (legacy LLM path) or ``assembly`` (catalog assembly).
* ``interactivity`` — ``static`` or ``interactive`` (inline htmx/Alpine).
* ``component_set`` — component-library variant for assembly mode.
* ``embeds`` — the three.js-seam sandboxed-iframe specs (JSONB).

Additive + defaulted: ``mode``/``interactivity`` carry server defaults so every
existing row and all non-assembly callers stay byte-identical on the freeform
path. ``ADD COLUMN IF NOT EXISTS`` keeps it re-runnable against a create_all-built
test DB. The service role owns the table (0001), so the ALTER's ownership check
passes.

Revision ID: 0006_assembly_fields
Revises: 0005_rls
Create Date: 2026-07-01 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006_assembly_fields"
down_revision: str | None = "0005_rls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE redesign_jobs "
        "ADD COLUMN IF NOT EXISTS mode VARCHAR(20) NOT NULL DEFAULT 'freeform'"
    )
    op.execute(
        "ALTER TABLE redesign_jobs "
        "ADD COLUMN IF NOT EXISTS interactivity VARCHAR(20) NOT NULL DEFAULT 'static'"
    )
    op.execute("ALTER TABLE redesign_jobs ADD COLUMN IF NOT EXISTS component_set VARCHAR(64)")
    op.execute("ALTER TABLE redesign_jobs ADD COLUMN IF NOT EXISTS embeds JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS embeds")
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS component_set")
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS interactivity")
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS mode")
