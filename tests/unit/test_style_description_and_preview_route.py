"""The VALUE the ``(tenant_id, slug)`` cache stores, and the routes that mint it.

``test_style_cache_and_active_filters.py`` pins the KEY: which entry is written,
which is read, which is evicted. It never asserts what is *in* the entry. But the
cached payload is ``StyleManager.resolve_style_description(style)`` — the tenant's
palette, typography, layout and reference sites, composed into the string that is
handed to the design LLM and returned verbatim in ``StylePreviewResponse``. A
cache that stores the right key against a description missing half the brand is
just as wrong as a cache that crosses tenants, and nothing tested that composition.

Three groups here, all on this branch's surface:

1. ``resolve_style_description`` — the cached/previewed payload itself.
2. ``StyleManager.update_style``'s empty-update guard — the branch made
   ``is_active`` a settable field, and ``False`` must survive the ``is not None``
   filter that guard sits behind.
3. ``POST /styles/{id}/preview`` — the route whose read this branch deliberately
   left at the DEFAULT (active-only) ``get_style``, making it the one route where
   ``include_inactive`` is *absent* on purpose. Section 3 of
   ``test_style_cache_and_active_filters`` proves the management routes opt IN;
   nothing proved this one stayed opted OUT.
"""

from __future__ import annotations

import asyncio
import importlib
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import text

from src.api.routes.styles import create_style, preview_style
from src.api.schemas import CreateCustomStyleRequest, StylePreviewRequest
from src.cache.redis_client import RedisClient
from src.database.models import CustomStyle
from src.services.cache_service import CacheService
from src.services.style_manager import StyleManager
from tests.tenants import TENANT_1, TENANT_2

_MIGRATION = importlib.import_module(
    "src.database.migrations.versions.0009_style_tenant_slug_unique"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _style(**kwargs: Any) -> CustomStyle:
    """An unpersisted ``CustomStyle``.

    ``resolve_style_description`` is a pure function of the instance's
    attributes, so no session is needed — and keeping it out of the DB keeps
    these assertions about composition rather than about persistence.
    """
    defaults: dict[str, Any] = {
        "style_id": "s1",
        "tenant_id": TENANT_1,
        "name": "House Style",
        "slug": "house-style",
        "description": None,
        "color_palette": None,
        "typography": None,
        "layout": None,
        "example_references": None,
        "is_active": True,
    }
    return CustomStyle(**(defaults | kwargs))


def _request(tenant_id: str, *, service: Any = None) -> Any:
    """Request stand-in with a REAL string tenant on ``.state`` (never a
    MagicMock — ``_get_tenant_id`` would then scope every query to a MagicMock
    and the tenant assertions would pass for the wrong reason)."""
    return SimpleNamespace(
        state=SimpleNamespace(tenant_id=tenant_id),
        app=SimpleNamespace(state=SimpleNamespace(redesign_service=service, cache_service=None)),
    )


def _llm(text: str = "```html\n<p>styled</p>\n```") -> Any:
    service = MagicMock()
    service._html_gen._call_claude = AsyncMock(return_value=SimpleNamespace(text=text))
    return service


# ---------------------------------------------------------------------------
# 1. resolve_style_description — the payload behind the cache key
# ---------------------------------------------------------------------------


def test_description_carries_every_configured_facet() -> None:
    """A fully-configured style contributes ALL of its facets to the prompt.

    MUTATION KILLED: dropping any entry from ``_palette_description``'s
    ``labels`` tuple (e.g. ``accent``), or dropping any of the
    ``_typography_description`` / ``_layout_description`` clauses — each removal
    silently strips part of the tenant's brand from the prompt AND from the
    string that gets cached under ``(tenant_id, slug)``, and every one of the
    assertions below names the facet it protects.
    """
    desc = StyleManager.resolve_style_description(
        _style(
            description="Brutalist, high contrast.",
            color_palette={
                "primary": "#111111",
                "secondary": "#222222",
                "accent": "#ff0055",
                "background": "#ffffff",
                "text": "#000000",
            },
            typography={"heading_font": "Inter", "body_font": "Georgia", "scale": "major-third"},
            layout={
                "max_width": "1200px",
                "spacing": "generous",
                "grid_columns": 12,
                "border_radius": "0px",
            },
        )
    )

    assert "Brutalist, high contrast." in desc
    for fragment in (
        "primary=#111111",
        "secondary=#222222",
        "accent=#ff0055",
        "background=#ffffff",
        "text=#000000",
    ):
        assert fragment in desc
    for fragment in ("headings in Inter", "body text in Georgia", "major-third type scale"):
        assert fragment in desc
    for fragment in ("max width 1200px", "generous spacing", "12-column grid", "0px border radius"):
        assert fragment in desc


def test_unset_facets_are_omitted_not_rendered_empty() -> None:
    """Partially-configured styles must not emit dangling labels.

    ``Color palette: .`` / ``Typography: .`` in an LLM prompt is noise the model
    will happily invent around.

    MUTATION KILLED: replacing the ``if colors``/``if items`` guards in the three
    ``_*_description`` helpers with an unconditional f-string — the empty-dict
    style below would then carry three empty section headers.
    """
    partial = StyleManager.resolve_style_description(
        _style(description="Just a note.", color_palette={"primary": "#abcdef"})
    )
    assert "primary=#abcdef" in partial
    assert "secondary" not in partial
    assert "Typography" not in partial
    assert "Layout" not in partial

    empty_dicts = StyleManager.resolve_style_description(
        _style(description="Just a note.", color_palette={}, typography={}, layout={})
    )
    assert empty_dicts == "Just a note."


def test_reference_sites_are_capped_at_three() -> None:
    """``example_references`` is caller-supplied and unbounded; the prompt takes
    the first three.

    MUTATION KILLED: removing the ``[:3]`` slice — the fourth URL, which is
    asserted absent, would be pasted into every prompt and every cached entry.
    """
    desc = StyleManager.resolve_style_description(
        _style(
            example_references=[
                "https://a.example",
                "https://b.example",
                "https://c.example",
                "https://d.example",
            ]
        )
    )
    assert "https://a.example" in desc
    assert "https://c.example" in desc
    assert "https://d.example" not in desc


def test_a_style_with_nothing_configured_falls_back_to_its_name() -> None:
    """An empty description must never be produced: ``_lookup_style_description``
    only caches (``if self._cache and style_description``) and only overrides the
    built-in look when the string is truthy, so an empty compose would silently
    downgrade the job to built-in styling.

    MUTATION KILLED: replacing the ``f"Custom style: {style.name}"`` fallback
    with ``""`` — this assertion fails, and so does the "the name survives"
    guarantee that makes a bare style usable at all.
    """
    assert StyleManager.resolve_style_description(_style(name="Bare")) == "Custom style: Bare"


# ---------------------------------------------------------------------------
# 2. update_style's empty-update guard
# ---------------------------------------------------------------------------


async def test_update_with_no_recognised_field_writes_nothing(db_session) -> None:
    """Unknown/None-only kwargs must short-circuit BEFORE the UPDATE statement.

    ``update(CustomStyle).values()`` with an empty mapping is not a harmless
    no-op — SQLAlchemy compiles it to an UPDATE with no SET clause, which is a
    syntax error on postgres. The guard is the only thing between a caller
    passing an unknown field and a 500.

    MUTATION KILLED: deleting ``if not values: return False`` — this test raises
    instead of returning False, and the row-identity assertions below never run.
    """
    style = await StyleManager.create_style(db_session, tenant_id=TENANT_1, name="House Style")
    await db_session.commit()

    assert (
        await StyleManager.update_style(
            db_session, style.style_id, tenant_id=TENANT_1, tenant_id_override=TENANT_2, name=None
        )
        is False
    )
    await db_session.commit()

    row = await StyleManager.get_style(db_session, style.style_id, tenant_id=TENANT_1)
    assert row is not None
    assert row.name == "House Style"
    assert row.tenant_id == TENANT_1


async def test_is_active_false_is_a_real_update_not_a_dropped_falsy(db_session) -> None:
    """``is_active=False`` is the ONE update whose value is falsy, and the whole
    deactivation story this branch is about depends on it landing.

    MUTATION KILLED: narrowing the kwargs filter from ``v is not None`` to ``v``
    — deactivation silently becomes an empty update, ``update_style`` returns
    False, the row stays active, and every ``include_inactive`` test in
    ``test_style_cache_and_active_filters`` would go green while the feature was
    inoperative.
    """
    style = await StyleManager.create_style(db_session, tenant_id=TENANT_1, name="House Style")
    await db_session.commit()

    assert (
        await StyleManager.update_style(
            db_session, style.style_id, tenant_id=TENANT_1, is_active=False
        )
        is True
    )
    await db_session.commit()

    assert await StyleManager.get_style(db_session, style.style_id, tenant_id=TENANT_1) is None
    deactivated = await StyleManager.get_style(
        db_session, style.style_id, tenant_id=TENANT_1, include_inactive=True
    )
    assert deactivated is not None
    assert deactivated.is_active is False


# ---------------------------------------------------------------------------
# 3. POST /styles/{id}/preview — the route that stayed active-only ON PURPOSE
# ---------------------------------------------------------------------------


async def test_preview_refuses_a_deactivated_style_and_spends_no_llm_call(db_session) -> None:
    """Preview is a CONSUMPTION route: it must behave like the generation path,
    which resolves through ``get_style_by_slug`` and never sees a deactivated
    style. It was previously the one place a switched-off style still rendered —
    and still burned an LLM call doing it.

    MUTATION KILLED: adding ``include_inactive=True`` to preview's ``get_style``
    (the same opt-in the three management routes carry, which is exactly why
    someone would "fix" this one for consistency) — the 404 becomes a 200 and
    ``_call_claude`` is invoked, both asserted here.
    """
    style = await StyleManager.create_style(db_session, tenant_id=TENANT_1, name="House Style")
    await db_session.commit()
    service = _llm()
    body = StylePreviewRequest(html_snippet="<p>hi</p>")

    live = await preview_style(
        style.style_id, body, _request(TENANT_1, service=service), db=db_session
    )
    assert live.styled_html == "<p>styled</p>"
    assert live.style_description == "Custom style: House Style"
    assert service._html_gen._call_claude.await_count == 1

    await StyleManager.update_style(db_session, style.style_id, tenant_id=TENANT_1, is_active=False)
    await db_session.commit()

    with pytest.raises(HTTPException) as exc:
        await preview_style(
            style.style_id, body, _request(TENANT_1, service=service), db=db_session
        )
    assert exc.value.status_code == 404
    assert exc.value.detail == f"Style not found: {style.style_id}"
    # Still 1: the refusal happened before the LLM was reached.
    assert service._html_gen._call_claude.await_count == 1


async def test_preview_prompt_carries_the_resolved_description_and_the_snippet(
    db_session,
) -> None:
    """The composed description is what the preview actually sends — the same
    string the cache stores — not merely what it echoes back in the response.

    MUTATION KILLED: dropping ``## Style\\n{style_desc}`` from the prompt (leaving
    the response field populated) — the preview would render an unstyled snippet
    while reporting the style, and only an assertion on the prompt sent to
    ``_call_claude`` catches it.
    """
    style = await StyleManager.create_style(
        db_session,
        tenant_id=TENANT_1,
        name="House Style",
        description="Brutalist.",
        color_palette={"accent": "#ff0055"},
    )
    await db_session.commit()
    service = _llm()

    result = await preview_style(
        style.style_id,
        StylePreviewRequest(html_snippet="<p>SNIPPET-MARKER</p>"),
        _request(TENANT_1, service=service),
        db=db_session,
    )

    prompt = service._html_gen._call_claude.await_args.args[0]
    assert "Brutalist." in prompt
    assert "accent=#ff0055" in prompt
    assert "SNIPPET-MARKER" in prompt
    assert result.style_description == StyleManager.resolve_style_description(style)


async def test_preview_timeout_is_a_504_not_a_hang(db_session, monkeypatch) -> None:
    """The 15s ``asyncio.wait_for`` around the LLM call is the only thing keeping
    a stalled provider from holding a request worker open indefinitely.

    MUTATION KILLED: removing the ``asyncio.wait_for`` wrapper (or the
    ``TimeoutError`` handler) — the call below hangs on the never-returning stub
    instead of raising 504, so the test fails by timeout rather than passing.
    """
    style = await StyleManager.create_style(db_session, tenant_id=TENANT_1, name="House Style")
    await db_session.commit()

    async def _never_returns(*_args: object, **_kwargs: object) -> None:
        await asyncio.Event().wait()

    service = MagicMock()
    service._html_gen._call_claude = _never_returns

    # Shorten the route's own 15s budget rather than replace the mechanism: the
    # call still goes through ``asyncio.wait_for``, so deleting that wrapper from
    # the route makes this test hang on ``_never_returns`` instead of passing.
    real_wait_for = asyncio.wait_for

    async def _short_budget(awaitable: Any, **_kwargs: object) -> Any:
        return await real_wait_for(awaitable, timeout=0.05)

    monkeypatch.setattr(asyncio, "wait_for", _short_budget)

    with pytest.raises(HTTPException) as exc:
        await preview_style(
            style.style_id,
            StylePreviewRequest(html_snippet="<p>hi</p>"),
            _request(TENANT_1, service=service),
            db=db_session,
        )

    assert exc.value.status_code == 504
    assert exc.value.detail == "Style preview generation timed out"


# ---------------------------------------------------------------------------
# 4. POST /styles — the two 409s this branch is responsible for
# ---------------------------------------------------------------------------


@pytest.fixture
async def unique_index_enforced(db_engine):
    """Apply migration 0009's DDL so the starting schema is deterministic.

    ``conftest``'s ``create_all`` is ``checkfirst``: a long-lived local test
    database still carrying 0001's NON-unique index of the same name keeps it,
    and the duplicate-slug test would then fail for a reason unrelated to the
    code. The ``lock_timeout`` turns the "another session is idle-in-transaction
    on custom_styles" case into one readable error instead of a suite-wide hang.
    """
    async with db_engine.begin() as conn:
        await conn.execute(text("SET lock_timeout = '5s'"))
        for statement in _MIGRATION.UPGRADE_SQL:
            await conn.execute(text(statement))


def _create_body(name: str) -> CreateCustomStyleRequest:
    return CreateCustomStyleRequest(name=name)


async def test_create_style_quota_counts_deactivated_rows(db_session, monkeypatch) -> None:
    """The route's 409 is charged on ``count_styles``, which this branch changed
    to count deactivated rows — so deactivating must NOT buy headroom back.

    MUTATION KILLED: restoring ``count_styles``' ``is_active.is_(True)``
    predicate — the deactivated row stops counting, the second POST succeeds,
    and a tenant can store unbounded styles under a cap that reads as enforced.
    """
    from src.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("REDESIGN_MAX_CUSTOM_STYLES_PER_TENANT", "1")
    get_settings.cache_clear()
    try:
        created = await create_style(_create_body("First"), _request(TENANT_1), db=db_session)
        assert created.slug == "first"

        await StyleManager.update_style(db_session, created.id, tenant_id=TENANT_1, is_active=False)
        await db_session.commit()

        with pytest.raises(HTTPException) as exc:
            await create_style(_create_body("Second"), _request(TENANT_1), db=db_session)
        assert exc.value.status_code == 409
        assert "Maximum custom styles (1) reached" in exc.value.detail

        # The cap is per-tenant, so t2 is unaffected — proving the 409 above is
        # the quota rule and not a broken session.
        assert (
            await create_style(_create_body("First"), _request(TENANT_2), db=db_session)
        ).slug == ("first")
    finally:
        get_settings.cache_clear()


async def test_create_style_duplicate_slug_is_409_not_500(
    unique_index_enforced, db_session
) -> None:
    """``(tenant_id, slug)`` is UNIQUE as of migration 0009, and ``slugify``
    collapses "House Style", "house style" and "House  Style!" onto one slug — so
    a perfectly ordinary second POST now hits a DB constraint. It must surface as
    a 409 with the offending name, not as the raw ``IntegrityError`` (a 500) the
    constraint would otherwise produce.

    MUTATION KILLED: removing the ``except IntegrityError`` arm from
    ``create_style`` — this call raises ``IntegrityError`` instead of
    ``HTTPException``, and the session is left dirty (the ``rollback`` in that
    arm is what lets the final assertion below run at all).
    """
    await create_style(_create_body("House Style"), _request(TENANT_1), db=db_session)

    with pytest.raises(HTTPException) as exc:
        await create_style(_create_body("House  Style!"), _request(TENANT_1), db=db_session)
    assert exc.value.status_code == 409
    assert "House  Style!" in exc.value.detail

    # The rollback in that handler leaves the session usable, and the SAME slug
    # under another tenant stays legal — which is precisely why the cache key
    # cannot be the slug alone.
    other = await create_style(_create_body("House Style"), _request(TENANT_2), db=db_session)
    assert other.slug == "house-style"


# ---------------------------------------------------------------------------
# 5. CacheService degrades; it never raises into the caller
# ---------------------------------------------------------------------------


class _ExplodingRedis:
    """A Redis whose every call fails with the transport errors redis-py raises.

    The style helpers go through ``RedisClient.set``/``.get``; the HTML helpers
    go through ``_ensure_connected()`` and the raw client. Both shapes fail here.

    THE EXCEPTIONS MUST COME FROM ``redis.exceptions``, NOT THE BUILTINS. This
    class raised the builtin ``ConnectionError`` for its whole life — an
    ``OSError`` subclass that a real Redis never raises. ``CacheService`` caught
    exactly that builtin tuple, so the test and the code agreed with each other
    and with nothing else: the "MUTATION KILLED" claim below was true only
    against an exception no deployment can produce, and the test PINNED the
    defect as correct. ``redis.exceptions.ConnectionError``/``TimeoutError``
    derive from ``RedisError(Exception)`` and are not ``OSError`` subclasses.
    """

    def _ensure_connected(self) -> _ExplodingRedis:
        raise RedisConnectionError("redis is down")

    async def set(self, *_args: object, **_kwargs: object) -> bool:
        raise RedisTimeoutError("redis timed out")

    async def get(self, *_args: object, **_kwargs: object) -> Any:
        raise RedisConnectionError("redis is down")


class _RecordingRedis:
    """Records the exact keys touched, so key-shape assertions are real."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def _ensure_connected(self) -> _RecordingRedis:
        return self

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        self.store[key] = value
        self.ttl = ttl
        return True

    async def get(self, key: str) -> Any:
        return self.store.get(key)


async def test_html_cache_is_keyed_by_url_digest_and_decodes_bytes() -> None:
    """Two different URLs must never share an entry, and a bytes reply from a
    real Redis must come back as ``str``.

    MUTATION KILLED (a): keying on the raw URL instead of ``_url_hash`` — the
    stored key would contain "https://a.example" and the digest assertion fails.
    MUTATION KILLED (b): dropping the ``isinstance(value, bytes)`` decode arm —
    the second assertion gets ``"b'<b>bytes</b>'"`` instead of the HTML.
    """
    redis = _RecordingRedis()
    cache = CacheService(cast(RedisClient, redis))

    assert await cache.cache_html("https://a.example", "<p>A</p>") is True
    assert await cache.get_cached_html("https://a.example") == "<p>A</p>"
    # Distinct URL, distinct entry — no collision, and no miss-as-hit.
    assert await cache.get_cached_html("https://b.example") is None

    stored_key = next(iter(redis.store))
    assert "https://a.example" not in stored_key
    assert stored_key == f"redesign:html:{sha256(b'https://a.example').hexdigest()}"

    redis.store[stored_key] = b"<b>bytes</b>"
    assert await cache.get_cached_html("https://a.example") == "<b>bytes</b>"


async def test_every_cache_read_and_write_degrades_when_redis_is_down() -> None:
    """A Redis outage must degrade the cache to a miss, never propagate.

    Both style helpers sit directly on the generation path
    (``_lookup_style_description``): if a read raised, an unreachable Redis would
    turn every redesign job into a failed job rather than a slower one.

    MUTATION KILLED: deleting any of the four ``except`` arms in
    ``CacheService`` (``cache_html``, ``get_cached_html``,
    ``cache_style_description``, ``get_cached_style_description``), OR narrowing
    any of them back to the builtin ``(OSError, ConnectionError, TimeoutError,
    RuntimeError)`` tuple — the matching call below then raises
    ``redis.exceptions.ConnectionError``/``TimeoutError`` instead of returning
    the False/None degraded value. The second half of that mutation is the one
    that matters: it is the shipped bug this test used to certify as correct.
    """
    cache = CacheService(cast(RedisClient, _ExplodingRedis()))

    assert await cache.cache_html("https://a.example", "<p>A</p>") is False
    assert await cache.get_cached_html("https://a.example") is None
    assert await cache.cache_style_description(TENANT_1, "house-style", "desc") is False
    assert await cache.get_cached_style_description(TENANT_1, "house-style") is None


# ---------------------------------------------------------------------------
# 6. Migration 0009's upgrade()/downgrade() bodies
# ---------------------------------------------------------------------------


async def test_migration_upgrade_runs_its_statements_in_the_shipped_order() -> None:
    """``upgrade()`` must dedupe, THEN assert, THEN swap the index.

    The order is the whole design: creating the unique index before the dedupe
    fails on any colliding prod row set, which crashloops the migrate
    init-container. The SQL itself is exercised against real postgres in
    ``test_style_cache_and_active_filters``; what is pinned here is that
    ``upgrade()`` actually issues it, all of it, in that order.

    MUTATION KILLED: reordering ``UPGRADE_SQL`` (index before dedupe), or
    dropping a statement from the loop — the recorded sequence stops matching.
    """
    executed: list[str] = []
    with mock.patch.object(_MIGRATION.op, "execute", side_effect=executed.append):
        _MIGRATION.upgrade()

    assert executed == [
        _MIGRATION.DEDUPE_SQL,
        _MIGRATION.ASSERT_NO_DUPLICATES_SQL,
        _MIGRATION.DROP_INDEX_SQL,
        _MIGRATION.CREATE_UNIQUE_INDEX_SQL,
    ]


async def test_migration_downgrade_restores_a_non_unique_index() -> None:
    """Down must land back on 0001's plain index. Recreating it UNIQUE would
    leave the constraint in place under a revision that claims to have removed
    it — the worst outcome, because a rollback would then fail on the same
    duplicate rows it was rolling back to allow.

    MUTATION KILLED: reusing ``CREATE_UNIQUE_INDEX_SQL`` in ``downgrade()`` —
    the ``"UNIQUE" not in ...`` assertion fails.
    """
    executed: list[str] = []
    with mock.patch.object(_MIGRATION.op, "execute", side_effect=executed.append):
        _MIGRATION.downgrade()

    assert executed[0] == _MIGRATION.DROP_INDEX_SQL
    assert len(executed) == 2
    assert "ix_custom_styles_tenant_slug" in executed[1]
    assert "UNIQUE" not in executed[1].upper()
