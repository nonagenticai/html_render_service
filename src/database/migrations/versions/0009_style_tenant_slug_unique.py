"""Make ``custom_styles (tenant_id, slug)`` UNIQUE, deduping first.

WHY
---
``slug`` — not ``style_id`` — is the key every generation path resolves a custom
style by: ``RedesignJob.style`` carries the style token the caller submitted, and
``StyleManager.get_style_by_slug(db, tenant_id, slug)`` is what turns it into a
row (design_builder ``phases``, ``variants`` and ``lifecycle`` all go through it).
``0001`` created ``ix_custom_styles_tenant_slug`` **non-unique**, so nothing
stopped one tenant owning two styles with the same slug — ``slugify()`` maps
"House Style", "house style" and "House  Style!" to the identical
``house-style`` — and ``scalar_one_or_none()`` then raised
``MultipleResultsFound`` or, worse, the pair silently made "which style did my
redesign use?" a coin flip.

It is also the precondition for the style-description cache fix landing in this
same change. That cache is keyed ``(tenant_id, slug)``; a key is only a correct
cache key if it identifies at most one row.

DUPLICATE DATA
--------------
A bare ``CREATE UNIQUE INDEX`` fails on any prod row set that already contains a
collision, which would crashloop the migrate init-container. So duplicates are
handled EXPLICITLY and non-destructively:

* Within each ``(tenant_id, slug)`` group the OLDEST row (``created_at``, then
  ``style_id`` as a deterministic tiebreak) keeps the slug. It is the one whose
  slug callers have been resolving, so keeping it preserves current behaviour.
* Every later duplicate is RENAMED, not deleted: its slug gets an 8-char
  ``md5(style_id)`` suffix. No style definition is lost, the row stays visible
  in ``GET /styles`` and stays addressable by ``style_id``; only its slug moves.
  ``left(slug, 100) || '-' || <8 chars>`` is at most 109 chars, inside the
  ``VARCHAR(120)`` column.

  **The rename is not free, and only the WINNER's behaviour is preserved.** The
  slug is the token generation resolves by, so any ``RedesignJob.style``
  pointing at a renamed row — a queued job, or a stored job later refined or
  re-run — no longer matches:
  ``StyleManager.get_style_by_slug`` misses,
  ``RedesignPhasesMixin._lookup_style_description`` returns ``None``, and the
  job renders with a BUILT-IN style rather than the tenant's custom one. The
  job still succeeds; only the design is wrong. That silent downgrade is now
  logged at WARNING by ``_lookup_style_description`` (it names the tenant and
  the unresolved token), which is the signal an operator should grep for after
  running this migration on a populated table.

  Measured before shipping: prod ``custom_styles`` holds **0 rows**, so this
  migration renames nothing there and the consequence above is theoretical for
  the current deployment. It is real for any environment that does hold
  colliding rows. The alternative — deleting duplicates — loses the definition
  outright, and the third option, refusing to migrate, leaves the cache key
  ambiguous; the rename is the least-bad of the three, not a free one.
* A post-dedupe assertion then RAISES with a clear message if any duplicate
  survives (only reachable via an md5-suffix collision), so the index creation
  can never be the thing that fails.

Re-runnable: the dedupe is a no-op once no duplicates remain, and the index is
dropped-then-recreated by the same name, converting ``0001``'s non-unique index
in place. ``models.py`` declares the same name with ``unique=True`` so a
``create_all``-built test DB agrees. The service role owns ``custom_styles``
(created in ``0001``), so the DDL's ownership check passes.

Revision ID: 0009_style_tenant_slug_unique
Revises: 0008_concept_image_error
Create Date: 2026-08-08 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009_style_tenant_slug_unique"
down_revision: str | None = "0008_concept_image_error"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


#: Rename every non-canonical duplicate so ``(tenant_id, slug)`` becomes unique.
#: Exported so the test suite can apply exactly the SQL that ships, rather than a
#: paraphrase of it.
DEDUPE_SQL: str = """
WITH ranked AS (
    SELECT
        style_id,
        row_number() OVER (
            PARTITION BY tenant_id, slug
            ORDER BY created_at, style_id
        ) AS rn
    FROM custom_styles
)
UPDATE custom_styles AS cs
SET slug = left(cs.slug, 100) || '-' || substr(md5(cs.style_id), 1, 8)
FROM ranked AS r
WHERE cs.style_id = r.style_id
  AND r.rn > 1
"""

#: Fail loudly (never silently create a half-enforced constraint) if the dedupe
#: did not fully resolve — the only way in is an md5-suffix collision.
#:
#: Deliberately free of any ``%`` character: ``op.execute`` routes the string
#: through SQLAlchemy ``text()`` on a pyformat-paramstyle driver, where a literal
#: ``%`` would have to be doubled. So the message is built with ``||`` instead of
#: ``format()``, and raised via ``USING MESSAGE =`` instead of ``RAISE ... , arg``
#: (both of which would need ``%s`` placeholders).
ASSERT_NO_DUPLICATES_SQL: str = """
DO $$
DECLARE
    offending text;
BEGIN
    SELECT string_agg(tenant_id || ' / ' || slug || ' (x' || n || ')', ', ')
    INTO offending
    FROM (
        SELECT tenant_id, slug, count(*) AS n
        FROM custom_styles
        GROUP BY tenant_id, slug
        HAVING count(*) > 1
    ) AS dups;

    IF offending IS NOT NULL THEN
        RAISE EXCEPTION USING MESSAGE =
            'custom_styles still has duplicate (tenant_id, slug) rows after dedupe: '
            || offending
            || '. Resolve them by hand (rename or delete the extras), then re-run '
            || 'migration 0009_style_tenant_slug_unique.';
    END IF;
END
$$
"""

#: Convert 0001's non-unique index in place, keeping the name so ``models.py``
#: and a ``create_all``-built test DB describe the same object. Two separate
#: statements — one ``op.execute`` per statement, so this does not depend on
#: multi-statement support in whichever driver runs the migration.
DROP_INDEX_SQL: str = "DROP INDEX IF EXISTS ix_custom_styles_tenant_slug"

CREATE_UNIQUE_INDEX_SQL: str = (
    "CREATE UNIQUE INDEX ix_custom_styles_tenant_slug ON custom_styles (tenant_id, slug)"
)

#: Ordered DDL for the whole upgrade — dedupe, assert, then enforce.
UPGRADE_SQL: tuple[str, ...] = (
    DEDUPE_SQL,
    ASSERT_NO_DUPLICATES_SQL,
    DROP_INDEX_SQL,
    CREATE_UNIQUE_INDEX_SQL,
)


def upgrade() -> None:
    for statement in UPGRADE_SQL:
        op.execute(statement)


def downgrade() -> None:
    # Back to 0001's non-unique index. The renamed duplicate slugs are NOT
    # restored: the original slug is not recoverable from the suffixed one, and
    # re-colliding them on the way down would only recreate the ambiguity.
    op.execute(DROP_INDEX_SQL)
    op.execute("CREATE INDEX ix_custom_styles_tenant_slug ON custom_styles (tenant_id, slug)")
