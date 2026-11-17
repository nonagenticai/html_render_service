"""Custom style definition management with CRUD and LLM prompt composition."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, cast

from sqlalchemy import CursorResult, delete, select, update
from sqlalchemy import func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import CustomStyle

logger = logging.getLogger(__name__)


def slugify(name: str) -> str:
    """Convert a style name to a URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return re.sub(r"-+", "-", slug).strip("-")


class StyleManager:
    """CRUD operations for custom style definitions."""

    @staticmethod
    async def create_style(
        db: AsyncSession,
        *,
        tenant_id: str,
        name: str,
        description: str | None = None,
        color_palette: dict | None = None,
        typography: dict | None = None,
        layout: dict | None = None,
        example_references: list[str] | None = None,
    ) -> CustomStyle:
        """Create a new custom style definition.

        Args:
            db: Async database session.
            tenant_id: Owning tenant identifier.
            name: Human-readable style name.
            description: Optional textual description.
            color_palette: Color palette dict.
            typography: Typography settings dict.
            layout: Layout preferences dict.
            example_references: Reference site URLs.

        Returns:
            The newly created CustomStyle instance.
        """
        style = CustomStyle(
            style_id=str(uuid.uuid7()),
            tenant_id=tenant_id,
            name=name,
            slug=slugify(name),
            description=description,
            color_palette=color_palette,
            typography=typography,
            layout=layout,
            example_references=example_references,
            is_active=True,
        )
        db.add(style)
        await db.flush()
        logger.info(
            "Created custom style '%s' (id=%s) for tenant %s", name, style.style_id, tenant_id
        )
        return style

    @staticmethod
    async def get_style(
        db: AsyncSession,
        style_id: str,
        *,
        tenant_id: str | None,
        include_inactive: bool = False,
    ) -> CustomStyle | None:
        """Fetch a custom style by its unique ID, scoped to ``tenant_id``.

        ``tenant_id=None`` means deliberately cross-tenant and must be an explicit
        decision at the call site; there is no such call site today. A mismatch
        returns ``None``, which every route turns into the same 404 a genuine
        miss produces — never a 403, which would confirm the id exists.

        Deactivated styles are EXCLUDED by default, matching
        ``get_style_by_slug`` / ``list_styles`` / every generation-path lookup.
        This method was the one reader that ignored ``is_active``, so a style the
        tenant had switched off still rendered through ``POST
        /styles/{id}/preview`` while being invisible everywhere else.

        ``include_inactive=True`` is the explicit opt-in for the MANAGEMENT
        reads — the existence check and the post-write re-read inside ``PUT
        /styles/{id}``. Without it, deactivating a style would make it
        permanently unreachable: the very request that would flip ``is_active``
        back to True 404s on its own guard, stranding the row (and, since
        ``count_styles`` now counts it, its quota slot) forever.
        """
        stmt = select(CustomStyle).where(CustomStyle.style_id == style_id)
        if tenant_id is not None:
            stmt = stmt.where(CustomStyle.tenant_id == tenant_id)
        if not include_inactive:
            stmt = stmt.where(CustomStyle.is_active.is_(True))
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_style_by_slug(db: AsyncSession, tenant_id: str, slug: str) -> CustomStyle | None:
        """Fetch an active custom style by tenant and slug, or None if not found."""
        result = await db.execute(
            select(CustomStyle).where(
                CustomStyle.tenant_id == tenant_id,
                CustomStyle.slug == slug,
                CustomStyle.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_styles(
        db: AsyncSession,
        tenant_id: str,
        *,
        include_inactive: bool = False,
    ) -> list[CustomStyle]:
        """List custom styles for a tenant, newest first.

        Active-only by default, matching ``get_style_by_slug`` and every
        generation-path lookup.

        ``include_inactive=True`` is the MANAGEMENT listing, and it is what
        ``GET /styles`` uses. It has to: ``count_styles`` charges quota for
        stored rows, active or not, so an active-only listing would let a tenant
        fill the cap, deactivate everything, and then be told "maximum reached"
        by ``POST`` while ``GET`` showed an empty list — with ``PUT``/``DELETE``
        both needing a ``style_id`` no API could still hand out. The quota-
        consuming set and the discoverable set must be the same set.
        """
        conditions = [CustomStyle.tenant_id == tenant_id]
        if not include_inactive:
            conditions.append(CustomStyle.is_active.is_(True))
        result = await db.execute(
            select(CustomStyle).where(*conditions).order_by(CustomStyle.created_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def update_style(
        db: AsyncSession,
        style_id: str,
        *,
        tenant_id: str | None,
        **kwargs: object,
    ) -> bool:
        """Update allowed fields on a custom style, scoped to ``tenant_id``.

        The predicate is on the UPDATE statement itself, not merely on a read
        above it: a caller that reaches this method with someone else's id
        writes nothing and gets ``False``.

        Args:
            db: Async database session.
            style_id: Style to update.
            tenant_id: Owning tenant; ``None`` = deliberately cross-tenant.
            **kwargs: Fields to update (name, description, color_palette, etc.).

        Returns:
            True if a row was updated, False otherwise.
        """
        allowed = {
            "name",
            "description",
            "color_palette",
            "typography",
            "layout",
            "example_references",
            "is_active",
        }
        values = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if "name" in values:
            values["slug"] = slugify(str(values["name"]))
        if not values:
            return False
        conditions = [CustomStyle.style_id == style_id]
        if tenant_id is not None:
            conditions.append(CustomStyle.tenant_id == tenant_id)
        cursor = cast(
            CursorResult[Any],
            await db.execute(update(CustomStyle).where(*conditions).values(**values)),
        )
        return (cursor.rowcount or 0) > 0

    @staticmethod
    async def count_styles(db: AsyncSession, tenant_id: str) -> int:
        """Count ALL stored styles for a tenant — active and deactivated alike.

        This is the quota counter (``redesign.max_custom_styles_per_tenant``), and
        the quota bounds ROWS STORED, not rows currently switched on. Counting
        only ``is_active`` rows made the cap trivially bypassable: create up to
        the limit, ``PUT {"is_active": false}`` on each, and the count drops back
        to zero while every row — description, palette, typography, reference
        URLs — is still stored and still reactivatable. Repeat for unbounded
        storage under a cap that reads as enforced.

        Deleting a style still frees a slot; deactivating one no longer does —
        which is why ``GET /styles`` lists deactivated rows too (see
        ``list_styles``). Everything that consumes quota must stay discoverable,
        or a tenant that deactivated its way to the cap could never find the
        ``style_id`` needed to delete anything.
        """
        result = await db.execute(
            select(sa_func.count())
            .select_from(CustomStyle)
            .where(CustomStyle.tenant_id == tenant_id)
        )
        return result.scalar() or 0

    @staticmethod
    async def delete_style(db: AsyncSession, style_id: str, *, tenant_id: str | None) -> bool:
        """Delete a custom style by ID, scoped to ``tenant_id``.

        The predicate is on the DELETE statement itself: a mismatched tenant
        removes no row and gets ``False``.
        """
        conditions = [CustomStyle.style_id == style_id]
        if tenant_id is not None:
            conditions.append(CustomStyle.tenant_id == tenant_id)
        cursor = cast(
            CursorResult[Any],
            await db.execute(delete(CustomStyle).where(*conditions)),
        )
        return (cursor.rowcount or 0) > 0

    @staticmethod
    def _palette_description(cp: dict) -> str:
        labels = (
            ("primary", "primary"),
            ("secondary", "secondary"),
            ("accent", "accent"),
            ("background", "background"),
            ("text", "text"),
        )
        colors = [f"{label}={cp[key]}" for key, label in labels if cp.get(key)]
        return f"Color palette: {', '.join(colors)}." if colors else ""

    @staticmethod
    def _typography_description(tp: dict) -> str:
        items: list[str] = []
        if tp.get("heading_font"):
            items.append(f"headings in {tp['heading_font']}")
        if tp.get("body_font"):
            items.append(f"body text in {tp['body_font']}")
        if tp.get("scale"):
            items.append(f"{tp['scale']} type scale")
        return f"Typography: {', '.join(items)}." if items else ""

    @staticmethod
    def _layout_description(lp: dict) -> str:
        items: list[str] = []
        if lp.get("max_width"):
            items.append(f"max width {lp['max_width']}")
        if lp.get("spacing"):
            items.append(f"{lp['spacing']} spacing")
        if lp.get("grid_columns"):
            items.append(f"{lp['grid_columns']}-column grid")
        if lp.get("border_radius"):
            items.append(f"{lp['border_radius']} border radius")
        return f"Layout: {', '.join(items)}." if items else ""

    @staticmethod
    def resolve_style_description(style: CustomStyle) -> str:
        """Compose a rich LLM prompt description from custom style fields."""
        parts: list[str] = []
        if style.description:
            parts.append(style.description)
        if style.color_palette:
            desc = StyleManager._palette_description(style.color_palette)
            if desc:
                parts.append(desc)
        if style.typography:
            desc = StyleManager._typography_description(style.typography)
            if desc:
                parts.append(desc)
        if style.layout:
            desc = StyleManager._layout_description(style.layout)
            if desc:
                parts.append(desc)
        if style.example_references:
            refs = ", ".join(str(r) for r in style.example_references[:3])
            parts.append(f"Reference sites for inspiration: {refs}.")
        return " ".join(parts) if parts else f"Custom style: {style.name}"
