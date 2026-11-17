"""Custom style CRUD + preview endpoints (Feature 3, Enhancement 7)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    CreateCustomStyleRequest,
    CustomStyleResponse,
    StyleListResponse,
    StylePreviewRequest,
    StylePreviewResponse,
    UpdateCustomStyleRequest,
)
from src.database.session import get_db
from src.services.style_manager import StyleManager

from ._helpers import _get_redesign_service, _get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/styles", operation_id="list_redesign_styles")
async def list_styles(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> StyleListResponse:
    """List all custom styles for the tenant, deactivated ones included.

    A MANAGEMENT listing, like ``GET /styles/{id}``: every row carries
    ``is_active`` so a client can filter for itself. It must show deactivated
    rows because ``count_styles`` charges quota for them — an active-only list
    would let a tenant hit "maximum custom styles reached" on ``POST`` while
    ``GET`` reported ``count: 0``, with no way left to obtain the ``style_id``
    that ``PUT`` (to reactivate) and ``DELETE`` (to free the slot) both require.
    That was a permanent, API-inescapable lockout.
    """
    tenant_id = _get_tenant_id(request)
    logger.debug("List styles request for tenant=%s", tenant_id)
    styles = await StyleManager.list_styles(db, tenant_id, include_inactive=True)
    return StyleListResponse(
        styles=[
            CustomStyleResponse(
                id=s.style_id,
                name=s.name,
                slug=s.slug,
                description=s.description,
                color_palette=s.color_palette,
                typography=s.typography,
                layout=s.layout,
                example_references=s.example_references,
                is_active=s.is_active,
                created_at=s.created_at.isoformat() if s.created_at else None,
            )
            for s in styles
        ],
        count=len(styles),
    )


@router.post(
    "/styles",
    responses={
        409: {"description": "Maximum custom styles reached"},
        422: {"description": "Validation error"},
    },
    operation_id="create_redesign_style",
)
async def create_style(
    body: CreateCustomStyleRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CustomStyleResponse:
    """Create a new custom style."""
    tenant_id = _get_tenant_id(request)
    logger.info("Create style request from tenant=%s name=%s", tenant_id, body.name)

    from src.config import get_settings as _get_settings

    max_styles = _get_settings().redesign.max_custom_styles_per_tenant
    # Counts deactivated styles too — the quota bounds rows stored, and
    # "deactivate to free headroom" was an unbounded-storage bypass.
    current_count = await StyleManager.count_styles(db, tenant_id)
    if current_count >= max_styles:
        raise HTTPException(
            status_code=409,
            detail=f"Maximum custom styles ({max_styles}) reached for this tenant",
        )

    try:
        style = await StyleManager.create_style(
            db,
            tenant_id=tenant_id,
            name=body.name,
            description=body.description,
            color_palette=body.color_palette.model_dump(exclude_none=True)
            if body.color_palette
            else None,
            typography=body.typography.model_dump(exclude_none=True) if body.typography else None,
            layout=body.layout.model_dump(exclude_none=True) if body.layout else None,
            example_references=body.example_references,
        )
        await db.commit()
    except IntegrityError as exc:
        # ``(tenant_id, slug)`` is now UNIQUE (migration 0009): the slug is the
        # key every generation-path lookup resolves a style by, so two rows
        # sharing one made the winner arbitrary. Surface the collision as a 409
        # instead of the 500 a raw IntegrityError would produce.
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"A style with this name already exists for this tenant: {body.name}",
        ) from exc
    return CustomStyleResponse(
        id=style.style_id,
        name=style.name,
        slug=style.slug,
        description=style.description,
        color_palette=style.color_palette,
        typography=style.typography,
        layout=style.layout,
        example_references=style.example_references,
        is_active=style.is_active,
        created_at=style.created_at.isoformat() if style.created_at else None,
    )


@router.get(
    "/styles/{style_id}",
    responses={
        404: {"description": "Style not found"},
    },
    operation_id="get_redesign_style",
)
async def get_style(
    style_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CustomStyleResponse:
    """Get a custom style by ID.

    A MANAGEMENT read: it opts into deactivated styles (``include_inactive``)
    because the response body carries ``is_active`` precisely so the owner can
    see and manage a switched-off style. Hiding it here while ``PUT`` can still
    reactivate it — and while it still occupies a quota slot — would leave the
    row invisible but chargeable.
    """
    tenant_id = _get_tenant_id(request)
    logger.debug("Get style request for style_id=%s tenant=%s", style_id, tenant_id)
    style = await StyleManager.get_style(db, style_id, tenant_id=tenant_id, include_inactive=True)
    if style is None:
        raise HTTPException(status_code=404, detail=f"Style not found: {style_id}")
    return CustomStyleResponse(
        id=style.style_id,
        name=style.name,
        slug=style.slug,
        description=style.description,
        color_palette=style.color_palette,
        typography=style.typography,
        layout=style.layout,
        example_references=style.example_references,
        is_active=style.is_active,
        created_at=style.created_at.isoformat() if style.created_at else None,
    )


@router.put(
    "/styles/{style_id}",
    responses={
        404: {"description": "Style not found"},
        422: {"description": "Validation error"},
    },
    operation_id="update_redesign_style",
)
async def update_style(
    style_id: str,
    body: UpdateCustomStyleRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CustomStyleResponse:
    """Update a custom style.

    Both reads opt into deactivated styles: this is the ONLY route that can flip
    ``is_active`` back to True, so filtering them out here would strand a
    deactivated row permanently (404 on its own guard) while it still consumed a
    quota slot.
    """
    tenant_id = _get_tenant_id(request)
    logger.info("Update style request for style_id=%s tenant=%s", style_id, tenant_id)
    existing = await StyleManager.get_style(
        db, style_id, tenant_id=tenant_id, include_inactive=True
    )
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Style not found: {style_id}")
    previous_slug = existing.slug

    update_data = body.model_dump(exclude_none=True)
    if "color_palette" in update_data and body.color_palette:
        update_data["color_palette"] = body.color_palette.model_dump(exclude_none=True)
    if "typography" in update_data and body.typography:
        update_data["typography"] = body.typography.model_dump(exclude_none=True)
    if "layout" in update_data and body.layout:
        update_data["layout"] = body.layout.model_dump(exclude_none=True)

    try:
        await StyleManager.update_style(db, style_id, tenant_id=tenant_id, **update_data)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="A style with this name already exists for this tenant",
        ) from exc

    updated = await StyleManager.get_style(db, style_id, tenant_id=tenant_id, include_inactive=True)
    if updated is None:
        raise HTTPException(status_code=404, detail="Style not found after update")

    # Invalidate under the CACHE key, which is (tenant_id, slug) — and under BOTH
    # slugs, because renaming re-slugifies: dropping only the new slug would
    # leave the old one serving the pre-edit description until its TTL expired.
    #
    # A failed invalidation is not fatal but it IS a wrong answer with a
    # deadline: the pre-edit description keeps being served to the generation
    # path for the rest of its 600s TTL. Log it so the stale window is
    # diagnosable instead of silent.
    cache_service = getattr(request.app.state, "cache_service", None)
    if cache_service:
        for slug in {previous_slug, updated.slug}:
            if not await cache_service.invalidate_style(tenant_id, slug):
                logger.warning(
                    "Style cache invalidation FAILED after update: tenant=%s slug=%s "
                    "style_id=%s — the pre-update description may still be served "
                    "to generation for up to the cache TTL",
                    tenant_id,
                    slug,
                    style_id,
                )
    return CustomStyleResponse(
        id=updated.style_id,
        name=updated.name,
        slug=updated.slug,
        description=updated.description,
        color_palette=updated.color_palette,
        typography=updated.typography,
        layout=updated.layout,
        example_references=updated.example_references,
        is_active=updated.is_active,
        created_at=updated.created_at.isoformat() if updated.created_at else None,
    )


@router.delete(
    "/styles/{style_id}",
    responses={
        404: {"description": "Style not found"},
    },
    operation_id="delete_redesign_style",
)
async def delete_style(
    style_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a custom style.

    Reads the row first — with ``include_inactive`` so a deactivated style is
    still deletable — purely to learn its ``slug``: the cache key is
    ``(tenant_id, slug)`` and after the DELETE there is nothing left to derive it
    from. The DELETE keeps its own tenant predicate (it is not relying on this
    read for scoping); the read only turns a miss into the same 404 as before.
    """
    tenant_id = _get_tenant_id(request)
    logger.info("Delete style request for style_id=%s tenant=%s", style_id, tenant_id)
    existing = await StyleManager.get_style(
        db, style_id, tenant_id=tenant_id, include_inactive=True
    )
    # Materialise the slug as a plain str BEFORE the DELETE + commit: the ORM
    # instance is expired by ``commit()`` and refreshing it would re-query a row
    # that no longer exists.
    deleted_slug = existing.slug if existing is not None else None

    deleted = await StyleManager.delete_style(db, style_id, tenant_id=tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Style not found: {style_id}")
    await db.commit()

    cache_service = getattr(request.app.state, "cache_service", None)
    # Worse here than on update: the row is gone, so nothing will ever
    # re-populate or correct the entry — a deleted style keeps rendering until
    # the TTL expires. Never swallow the failure. (Short-circuit order matters:
    # the invalidation is only attempted once there is a cache AND a slug.)
    if (
        cache_service
        and deleted_slug is not None
        and not await cache_service.invalidate_style(tenant_id, deleted_slug)
    ):
        logger.warning(
            "Style cache invalidation FAILED after delete: tenant=%s slug=%s "
            "style_id=%s — the deleted style's description may still be served "
            "to generation for up to the cache TTL",
            tenant_id,
            deleted_slug,
            style_id,
        )

    return {"deleted": True, "style_id": style_id}


@router.post(
    "/styles/{style_id}/preview",
    responses={
        404: {"description": "Style not found"},
        422: {"description": "Validation error"},
        504: {"description": "Style preview generation timed out"},
    },
    operation_id="preview_redesign_style",
)
async def preview_style(
    style_id: str,
    body: StylePreviewRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> StylePreviewResponse:
    """Preview a custom style applied to an HTML snippet.

    A CONSUMPTION route, not a management one: it renders the style, exactly as
    the generation path does. So it takes the default active-only ``get_style``
    and now 404s on a deactivated style, matching ``get_style_by_slug`` — which
    every real redesign/variant/refinement lookup goes through. Previously this
    was the one place a switched-off style still produced output (and still spent
    an LLM call).
    """
    tenant_id = _get_tenant_id(request)
    logger.info("Style preview request for style_id=%s tenant=%s", style_id, tenant_id)
    style = await StyleManager.get_style(db, style_id, tenant_id=tenant_id)
    if style is None:
        raise HTTPException(status_code=404, detail=f"Style not found: {style_id}")

    style_desc = StyleManager.resolve_style_description(style)
    service = _get_redesign_service(request)

    prompt = (
        f"Apply the following design style to this HTML snippet.\n\n"
        f"## Style\n{style_desc}\n\n"
        f"## HTML Snippet\n```html\n{body.html_snippet}\n```\n\n"
        f"Output ONLY the styled HTML using Tailwind CSS classes in a ```html code block."
    )
    try:
        llm_result = await asyncio.wait_for(
            service._html_gen._call_claude(prompt, temperature=0.4),
            timeout=15.0,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Style preview generation timed out") from exc

    from src.services.html_generator import _extract_html_block

    styled_html = _extract_html_block(llm_result.text)

    return StylePreviewResponse(
        styled_html=styled_html,
        style_description=style_desc,
    )
