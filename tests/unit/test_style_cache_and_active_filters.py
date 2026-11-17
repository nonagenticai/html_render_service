"""The style cache actually caches — and does NOT cross tenants doing it.

Four defects, all intra-tenant in effect but one of which becomes a genuine
cross-tenant leak if fixed the obvious way:

1. ``_lookup_style_description`` READ ``get_cached_style_description(slug)`` and
   WROTE ``cache_style_description(style_id)``. Same key prefix, different key —
   so the read could never find the write and the cache was a 100% miss for its
   entire life. The naive repair (key on the slug) is a SECURITY BUG: a slug is
   unique only within a tenant, so tenant A's brand description would be served
   to tenant B. The fix is a COMPOSITE ``(tenant_id, slug)`` key on both sides,
   and ``test_same_slug_different_tenant_is_a_cache_miss`` is the test that
   pins it.
2. ``StyleManager.get_style`` ignored ``is_active`` while every other reader
   applied it.
3. ``StyleManager.count_styles`` counted only active rows, so the per-tenant
   quota was bypassable by deactivating.
4. (In ``test_tenant_id_fail_closed.py``.) ``_get_tenant_id`` defaulted to the
   literal ``"anonymous"``. It now requires a real, non-nil UUID and 401s.

And the three follow-on defects that (3) and the migration created, covered in
sections 5-7 below:

5. Charging quota for deactivated rows while ``GET /styles`` still hid them was
   a permanent SELF-LOCKOUT: fill the cap, deactivate everything, and ``POST``
   says "maximum reached" while ``GET`` says ``count: 0`` and no API can hand
   back the ``style_id`` that ``DELETE`` needs.
6. A style token that resolves to nothing renders with a BUILT-IN style and
   returns success — the silent wrong-render migration ``0009``'s duplicate
   rename can cause.
7. ``invalidate_style`` failing on a Redis blip left the stale description
   served for the full TTL with no signal at all.

Plus the constraint that makes (1) sound at all: ``(tenant_id, slug)`` is now
UNIQUE, and migration ``0009`` is exercised here against real postgres —
including its explicit duplicate handling, because a bare ``CREATE UNIQUE
INDEX`` would crashloop the migrate init-container on any colliding prod row.
"""

from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from redis.exceptions import RedisError
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from src.api.routes.styles import delete_style, list_styles, update_style
from src.api.schemas import UpdateCustomStyleRequest
from src.cache.redis_client import RedisClient
from src.database.models import CustomStyle
from src.services.cache_service import CacheService, style_cache_key
from src.services.design_builder.phases import RedesignPhasesMixin
from src.services.style_manager import StyleManager
from tests.tenants import TENANT_1, TENANT_2

_MIGRATION = importlib.import_module(
    "src.database.migrations.versions.0009_style_tenant_slug_unique"
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _FakeRedis:
    """In-memory stand-in for ``RedisClient`` covering the three calls
    ``CacheService`` makes for styles. Deliberately NOT a MagicMock: the whole
    point of these tests is which KEY is touched, and a mock's auto-attributes
    would let a wrong key pass silently."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def _ensure_connected(self) -> _FakeRedis:
        """``invalidate_style`` deletes through the underlying client rather than
        ``RedisClient.delete`` (which swallows ``RedisError``), so the fake plays
        both roles."""
        return self

    async def get(self, key: str) -> Any | None:
        return self.store.get(key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        self.store[key] = value
        return True

    async def delete(self, key: str) -> bool:
        return self.store.pop(key, None) is not None


class _BrokenRedis(_FakeRedis):
    """A reachable-then-not Redis: writes land, the DELETE blows up.

    Models the exact D4 scenario — a blip during ``PUT``/``DELETE /styles/{id}``
    that leaves the pre-edit description cached for the rest of its TTL.
    """

    async def delete(self, key: str) -> bool:
        raise RedisError("redis is down")


def _cache(redis: _FakeRedis) -> CacheService:
    """Wrap a fake in a real ``CacheService``.

    ``cast`` rather than a per-call-site ``# type: ignore[arg-type]``: ty (which
    CI runs) does not honour mypy's rule codes, so those suppressions were inert
    and every construction below was a ``ty check`` error.
    """
    return CacheService(cast(RedisClient, redis))


def _warnings(caplog: pytest.LogCaptureFixture, logger_name: str) -> list[str]:
    """WARNING messages from ONE logger.

    Filtering by name, not just by level: an unrelated SQLAlchemy/httpx warning
    emitted during the same test would otherwise satisfy (or break) these
    assertions and the log-signal tests would stop meaning anything.
    """
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == logger_name
    ]


class _Phases(RedesignPhasesMixin):
    """Bare host for ``_lookup_style_description`` — the mixin needs nothing but
    ``_cache`` for this path, so nothing else is stood up."""

    def __init__(self, cache: CacheService | None) -> None:
        self._cache = cache


async def _seed(db, tenant_id: str, name: str, description: str) -> CustomStyle:
    style = await StyleManager.create_style(
        db, tenant_id=tenant_id, name=name, description=description
    )
    await db.commit()
    return style


# ---------------------------------------------------------------------------
# 1. The cache is no longer dead
# ---------------------------------------------------------------------------


async def test_style_description_read_after_write_hits(db_session) -> None:
    """First lookup populates the cache; the second is served FROM it.

    Proved by poisoning the cached value with a sentinel between the two calls:
    if the second lookup returns the sentinel it can only have come from the
    cache, and if it returns the DB text the read key still misses the write key
    (the original bug, which no amount of "the cache is configured" asserting
    would have caught)."""
    style = await _seed(db_session, TENANT_1, "House Style", "t1 house description")
    redis = _FakeRedis()
    phases = _Phases(_cache(redis))

    first = await phases._lookup_style_description(db_session, TENANT_1, "house-style")
    assert first is not None
    assert "t1 house description" in first

    # Exactly one key, and it is the composite one.
    key = style_cache_key(TENANT_1, style.slug)
    assert list(redis.store) == [key]

    redis.store[key] = "SENTINEL-FROM-CACHE"
    second = await phases._lookup_style_description(db_session, TENANT_1, "house-style")
    assert second == "SENTINEL-FROM-CACHE"


async def test_same_slug_different_tenant_is_a_cache_miss(db_session) -> None:
    """THE leak test. Two tenants, same style name -> same slug.

    A slug-keyed cache (the tempting one-line "fix") returns t1's brand
    description — palette, typography, reference sites — to t2 verbatim. The
    composite key must miss and fall through to t2's own row."""
    await _seed(db_session, TENANT_1, "House Style", "T1 CONFIDENTIAL BRAND")
    await _seed(db_session, TENANT_2, "House Style", "t2 own description")

    redis = _FakeRedis()
    phases = _Phases(_cache(redis))

    t1_desc = await phases._lookup_style_description(db_session, TENANT_1, "house-style")
    assert t1_desc is not None
    assert "T1 CONFIDENTIAL BRAND" in t1_desc

    t2_desc = await phases._lookup_style_description(db_session, TENANT_2, "house-style")
    assert t2_desc is not None
    assert "T1 CONFIDENTIAL BRAND" not in t2_desc
    assert "t2 own description" in t2_desc

    # Two rows, two tenants, two distinct cache keys.
    assert set(redis.store) == {
        style_cache_key(TENANT_1, "house-style"),
        style_cache_key(TENANT_2, "house-style"),
    }


async def test_cache_service_style_keys_are_tenant_scoped() -> None:
    """The same isolation at the CacheService level, with no DB in the way."""
    cache = _cache(_FakeRedis())

    await cache.cache_style_description(TENANT_1, "house-style", "t1 secret")

    assert await cache.get_cached_style_description(TENANT_1, "house-style") == "t1 secret"
    assert await cache.get_cached_style_description(TENANT_2, "house-style") is None

    # Invalidation is scoped the same way: t2's invalidate touches its OWN key
    # and leaves t1's entry intact. Asserted by reading, not by the return value
    # — the return value now reports "did the DELETE succeed", and deleting a key
    # that was never there succeeds (see
    # ``test_invalidate_style_returns_false_only_on_redis_failure``).
    assert await cache.invalidate_style(TENANT_2, "house-style") is True
    assert await cache.get_cached_style_description(TENANT_1, "house-style") == "t1 secret"
    # ...and the owner really does evict.
    assert await cache.invalidate_style(TENANT_1, "house-style") is True
    assert await cache.get_cached_style_description(TENANT_1, "house-style") is None


async def test_style_cache_key_is_injective() -> None:
    """Percent-encoding both halves keeps the key unambiguous: without it,
    ``("a:b", "c")`` and ``("a", "b:c")`` would collide on one string."""
    assert style_cache_key("a:b", "c") != style_cache_key("a", "b:c")


# ---------------------------------------------------------------------------
# 2. get_style respects is_active (with an explicit opt-in for management)
# ---------------------------------------------------------------------------


async def test_get_style_excludes_deactivated_by_default(db_session) -> None:
    style = await _seed(db_session, TENANT_1, "Retired Style", "old brand")
    assert await StyleManager.get_style(db_session, style.style_id, tenant_id=TENANT_1) is not None

    await StyleManager.update_style(db_session, style.style_id, tenant_id=TENANT_1, is_active=False)
    await db_session.commit()

    assert await StyleManager.get_style(db_session, style.style_id, tenant_id=TENANT_1) is None
    # ...matching every other reader, which is the consistency this restores.
    assert await StyleManager.get_style_by_slug(db_session, TENANT_1, "retired-style") is None
    assert await StyleManager.list_styles(db_session, TENANT_1) == []


async def test_get_style_include_inactive_is_the_management_opt_in(db_session) -> None:
    """Without the opt-in a deactivated style could never be reactivated: the
    PUT route's own existence guard would 404 on it forever."""
    style = await _seed(db_session, TENANT_1, "Retired Style", "old brand")
    await StyleManager.update_style(db_session, style.style_id, tenant_id=TENANT_1, is_active=False)
    await db_session.commit()

    found = await StyleManager.get_style(
        db_session, style.style_id, tenant_id=TENANT_1, include_inactive=True
    )
    assert found is not None
    assert found.is_active is False

    # The opt-in does NOT weaken the tenant gate.
    assert (
        await StyleManager.get_style(
            db_session, style.style_id, tenant_id=TENANT_2, include_inactive=True
        )
        is None
    )


# ---------------------------------------------------------------------------
# 3. count_styles counts deactivated rows (the quota bypass)
# ---------------------------------------------------------------------------


async def test_count_styles_counts_deactivated_rows(db_session) -> None:
    """Deactivating must not hand back quota headroom.

    The bypass was: create up to the cap, ``PUT {"is_active": false}`` on each,
    watch the count fall to zero, repeat — unbounded stored styles under a cap
    that reads as enforced. Every row is still stored and still reactivatable,
    so every row still counts."""
    a = await _seed(db_session, TENANT_1, "Style A", "a")
    b = await _seed(db_session, TENANT_1, "Style B", "b")
    await _seed(db_session, TENANT_2, "Style C", "c")  # another tenant: never counted

    assert await StyleManager.count_styles(db_session, TENANT_1) == 2

    for style in (a, b):
        await StyleManager.update_style(
            db_session, style.style_id, tenant_id=TENANT_1, is_active=False
        )
    await db_session.commit()

    assert await StyleManager.count_styles(db_session, TENANT_1) == 2
    assert await StyleManager.count_styles(db_session, TENANT_2) == 1

    # Deleting — unlike deactivating — really does free a slot.
    await StyleManager.delete_style(db_session, a.style_id, tenant_id=TENANT_1)
    await db_session.commit()
    assert await StyleManager.count_styles(db_session, TENANT_1) == 1


# ---------------------------------------------------------------------------
# 4. The (tenant_id, slug) unique constraint + migration 0009
# ---------------------------------------------------------------------------


async def _run_ddl(db_engine, *statements: str) -> None:
    """Execute schema DDL on its own connection, under a short ``lock_timeout``.

    The timeout is not decoration. ``DROP INDEX`` / ``CREATE UNIQUE INDEX`` need
    ACCESS EXCLUSIVE on ``custom_styles``, so ANY other session sitting
    idle-in-transaction holding even ACCESS SHARE blocks it — and, because
    postgres queues later lock requests behind the waiter, that one stall wedges
    every subsequent test in the file AND in ``test_rls_isolation``. Failing
    after five seconds turns a whole-suite hang into one readable error.
    """
    async with db_engine.begin() as conn:
        await conn.execute(text("SET lock_timeout = '5s'"))
        for statement in statements:
            await conn.execute(text(statement))


@pytest.fixture
async def unique_index_enforced(db_engine):
    """Apply migration 0009's DDL before the test.

    ``conftest``'s ``create_all`` is ``checkfirst``: on a long-lived local test
    database that already carries 0001's NON-unique index of the same name, it
    would leave the old one in place and this test would fail for a reason that
    has nothing to do with the code. Applying the shipped migration makes the
    starting schema deterministic on a fresh CI database and a stale local one
    alike."""
    await _run_ddl(db_engine, *_MIGRATION.UPGRADE_SQL)


async def test_duplicate_tenant_slug_is_rejected(unique_index_enforced, db_session) -> None:
    """The constraint the composite cache key depends on for correctness."""
    await _seed(db_session, TENANT_1, "House Style", "first")

    with pytest.raises(IntegrityError):
        await StyleManager.create_style(db_session, tenant_id=TENANT_1, name="House  Style!")
    await db_session.rollback()

    # Same slug under a DIFFERENT tenant stays legal — that is exactly why the
    # cache key cannot be the slug alone.
    other = await StyleManager.create_style(db_session, tenant_id=TENANT_2, name="House Style")
    await db_session.commit()
    assert other.slug == "house-style"


@pytest.fixture
async def pre_0009_index(db_engine):
    """Put the schema back to its pre-0009 shape (non-unique index) so duplicate
    rows can be seeded, then re-apply the real migration on teardown.

    Tests using this fixture must NOT hold an open ORM session across it — see
    ``_run_ddl``. They read through ``db_engine`` connections for that reason.
    """
    await _run_ddl(
        db_engine,
        _MIGRATION.DROP_INDEX_SQL,
        "CREATE INDEX ix_custom_styles_tenant_slug ON custom_styles (tenant_id, slug)",
    )
    yield
    await _run_ddl(db_engine, *_MIGRATION.UPGRADE_SQL)


async def _insert_raw(
    conn, *, style_id: str, tenant_id: str, name: str, slug: str, created_at: str
) -> None:
    """Insert a ``custom_styles`` row with an EXPLICIT ``created_at``.

    Not optional: ``created_at`` defaults to ``now()``, which in postgres is the
    TRANSACTION timestamp — so rows seeded in one transaction are all born at the
    same instant and the dedupe's ``ORDER BY created_at`` would be decided
    entirely by its ``style_id`` tiebreak. Pinning the timestamps is what makes
    "the oldest row keeps the slug" an actual assertion rather than an
    alphabetical accident.
    """
    await conn.execute(
        text(
            "INSERT INTO custom_styles (style_id, tenant_id, name, slug, is_active, created_at) "
            "VALUES (:sid, :tid, :name, :slug, TRUE, CAST(:created_at AS timestamptz))"
        ),
        {
            "sid": style_id,
            "tid": tenant_id,
            "name": name,
            "slug": slug,
            "created_at": created_at,
        },
    )


async def test_migration_0009_dedupes_then_enforces_uniqueness(db_engine, pre_0009_index) -> None:
    """Duplicates are RENAMED, never dropped, and the index lands unique.

    A bare ``CREATE UNIQUE INDEX`` on real data with a collision fails, and a
    failed migration is a crashlooping init-container. So the shipped SQL is run
    here verbatim (imported from the migration module, not paraphrased) against
    a deliberately-colliding row set.

    Reads go through short-lived ``db_engine`` connections rather than the
    ``db_session`` fixture: a session left idle-in-transaction on
    ``custom_styles`` blocks the fixture's teardown DDL, which then wedges every
    later test that touches the table."""
    # ``s-keep`` sorts LAST by style_id but was created FIRST. If the dedupe ever
    # regressed to ordering by id, this pair would fail — which is the point.
    async with db_engine.begin() as conn:
        await _insert_raw(
            conn,
            style_id="s-keep",
            tenant_id=TENANT_1,
            name="A",
            slug="house-style",
            created_at="2026-01-01T00:00:00Z",
        )
        await _insert_raw(
            conn,
            style_id="s-dupe",
            tenant_id=TENANT_1,
            name="B",
            slug="house-style",
            created_at="2026-06-01T00:00:00Z",
        )
        await _insert_raw(
            conn,
            style_id="s-solo",
            tenant_id=TENANT_2,
            name="C",
            slug="house-style",
            created_at="2026-06-01T00:00:00Z",
        )

    await _run_ddl(db_engine, *_MIGRATION.UPGRADE_SQL)

    async with db_engine.connect() as conn:
        result = await conn.execute(select(CustomStyle.style_id, CustomStyle.slug))
        rows = {r.style_id: r.slug for r in result.all()}
        unique = (
            await conn.execute(
                text(
                    "SELECT indisunique FROM pg_index i "
                    "JOIN pg_class c ON c.oid = i.indexrelid "
                    "WHERE c.relname = 'ix_custom_styles_tenant_slug'"
                )
            )
        ).scalar_one()

    # Nothing deleted.
    assert set(rows) == {"s-keep", "s-dupe", "s-solo"}
    # The OLDEST row in the colliding group keeps the slug callers resolve —
    # despite sorting last by style_id.
    assert rows["s-keep"] == "house-style"
    # The duplicate is renamed with a deterministic suffix, not destroyed.
    assert rows["s-dupe"].startswith("house-style-")
    assert rows["s-dupe"] != "house-style"
    # The other tenant is untouched — its slug never collided.
    assert rows["s-solo"] == "house-style"

    # And the index really is unique now.
    assert unique is True


async def test_migration_0009_fails_loudly_on_unresolvable_duplicates(
    db_engine, pre_0009_index
) -> None:
    """The guard between the dedupe and the index: if any duplicate survives,
    the migration raises a message that names the offending pair — it never
    proceeds to a ``CREATE UNIQUE INDEX`` that would fail with postgres's own
    opaque error, and never leaves the constraint half-applied."""
    async with db_engine.begin() as conn:
        await _insert_raw(
            conn,
            style_id="d1",
            tenant_id=TENANT_1,
            name="A",
            slug="house-style",
            created_at="2026-01-01T00:00:00Z",
        )
        await _insert_raw(
            conn,
            style_id="d2",
            tenant_id=TENANT_1,
            name="B",
            slug="house-style",
            created_at="2026-06-01T00:00:00Z",
        )

    with pytest.raises(DBAPIError) as exc:
        async with db_engine.begin() as conn:
            await conn.execute(text(_MIGRATION.ASSERT_NO_DUPLICATES_SQL))

    message = str(exc.value)
    assert "duplicate (tenant_id, slug)" in message
    assert f"{TENANT_1} / house-style" in message


# ---------------------------------------------------------------------------
# 5. The quota set and the discoverable set are the SAME set (the self-lockout)
# ---------------------------------------------------------------------------


def _request(tenant_id: str, *, cache_service: Any = None) -> Any:
    """Request stand-in with a REAL string tenant on ``.state``.

    Not a ``MagicMock``: ``_get_tenant_id`` does ``getattr(request.state,
    "tenant_id", ...)`` and a MagicMock would hand back a MagicMock, so every
    tenant-scoped query would miss and these tests would pass for the wrong
    reason. ``app.state`` carries ``cache_service`` because the routes read it
    with ``getattr(..., None)``.
    """
    return SimpleNamespace(
        state=SimpleNamespace(tenant_id=tenant_id),
        app=SimpleNamespace(state=SimpleNamespace(cache_service=cache_service)),
    )


async def test_list_styles_include_inactive_matches_what_count_styles_charges(
    db_session,
) -> None:
    """``list_styles(include_inactive=True)`` must return exactly the rows
    ``count_styles`` bills for — otherwise quota is charged for rows no caller
    can enumerate.

    MUTATION KILLED: dropping the ``include_inactive`` branch from
    ``StyleManager.list_styles`` (i.e. always appending the
    ``is_active.is_(True)`` predicate) — the management list drops to 0 while
    the count stays 2.
    """
    a = await _seed(db_session, TENANT_1, "Style A", "a")
    b = await _seed(db_session, TENANT_1, "Style B", "b")
    await _seed(db_session, TENANT_2, "Style C", "c")  # another tenant: never listed

    for style in (a, b):
        await StyleManager.update_style(
            db_session, style.style_id, tenant_id=TENANT_1, is_active=False
        )
    await db_session.commit()

    # Default (generation-path) listing still hides them...
    assert await StyleManager.list_styles(db_session, TENANT_1) == []
    # ...and the management listing shows exactly the billed set.
    managed = await StyleManager.list_styles(db_session, TENANT_1, include_inactive=True)
    assert {s.style_id for s in managed} == {a.style_id, b.style_id}
    assert len(managed) == await StyleManager.count_styles(db_session, TENANT_1)
    # The opt-in does not widen the tenant gate.
    assert await StyleManager.list_styles(db_session, TENANT_2, include_inactive=True) != []
    assert all(s.tenant_id == TENANT_1 for s in managed)


async def test_get_styles_route_can_always_escape_the_quota(db_session) -> None:
    """The end-to-end lockout, driven through the real routes.

    A tenant deactivates everything it owns. ``count_styles`` still bills for
    both rows (so ``POST`` would 409 "maximum reached"), therefore ``GET
    /styles`` MUST still return them — and the ``style_id`` it returns must
    actually work against ``DELETE``. Without that, the tenant is wedged with no
    API escape at all.

    MUTATION KILLED: removing ``include_inactive=True`` from the ``GET /styles``
    route (``styles.py``) — the listing comes back empty and count 0 while the
    quota still charges for both rows, and the delete leg below has no id to
    use.
    """
    a = await _seed(db_session, TENANT_1, "Style A", "a")
    b = await _seed(db_session, TENANT_1, "Style B", "b")
    for style in (a, b):
        await StyleManager.update_style(
            db_session, style.style_id, tenant_id=TENANT_1, is_active=False
        )
    await db_session.commit()

    listed = await list_styles(_request(TENANT_1), db=db_session)
    assert listed.count == 2
    assert {s.id for s in listed.styles} == {a.style_id, b.style_id}
    # The flag is in the payload, so a client that wants active-only can filter.
    assert all(s.is_active is False for s in listed.styles)
    # Another tenant's rows are still not in there.
    assert await StyleManager.count_styles(db_session, TENANT_1) == 2

    # The escape hatch actually opens: an id from the listing frees its slot.
    freed = await delete_style(listed.styles[0].id, _request(TENANT_1), db=db_session)
    assert freed == {"deleted": True, "style_id": listed.styles[0].id}
    assert await StyleManager.count_styles(db_session, TENANT_1) == 1


# ---------------------------------------------------------------------------
# 6. An unresolvable style token is LOUD, not silent
# ---------------------------------------------------------------------------


async def test_unresolved_style_token_warns_before_falling_back_to_builtin(
    db_session, caplog: pytest.LogCaptureFixture
) -> None:
    """A stored ``RedesignJob.style`` that no longer matches a row renders with a
    built-in style and still reports success. Migration ``0009`` can cause
    exactly this by re-slugging a duplicate under a queued job (prod
    ``custom_styles`` is empty, so it is a no-op there — but the same silent
    downgrade happens on delete/deactivate/typo). It must at least be greppable.

    MUTATION KILLED: deleting the ``logger.warning`` from
    ``RedesignPhasesMixin._lookup_style_description``'s ``not custom_style``
    branch — the return value is ``None`` either way, so ONLY the log record
    distinguishes the fixed code from the broken code.
    """
    phases = _Phases(None)

    # The shape migration 0009 produces for a losing duplicate: slug + md5 suffix.
    renamed_token = "house-style-a1b2c3d4"
    with caplog.at_level(logging.WARNING, logger="src.services.design_builder.phases"):
        assert await phases._lookup_style_description(db_session, TENANT_1, renamed_token) is None

    warnings = _warnings(caplog, "src.services.design_builder.phases")
    assert len(warnings) == 1
    message = warnings[0]
    # Names the token AND the tenant — both are needed to find the row by hand.
    assert "house-style-a1b2c3d4" in message
    assert TENANT_1 in message


async def test_resolvable_and_builtin_style_tokens_do_not_warn(
    db_session, caplog: pytest.LogCaptureFixture
) -> None:
    """The paired negative: the warning must fire on the failure only.

    Without this, ``logger.warning("...")`` moved above the ``if`` — or made
    unconditional — would still pass the test above while making the signal
    worthless (it would fire on every successful job).

    MUTATION KILLED: hoisting the warning out of the ``not custom_style``
    branch, i.e. logging on every lookup.
    """
    await _seed(db_session, TENANT_1, "House Style", "t1 house description")
    phases = _Phases(None)

    with caplog.at_level(logging.WARNING, logger="src.services.design_builder.phases"):
        resolved = await phases._lookup_style_description(db_session, TENANT_1, "house-style")
        # A built-in token short-circuits before the DB and is NOT a failure.
        assert await phases._lookup_style_description(db_session, TENANT_1, "modern") is None

    assert resolved is not None
    assert "t1 house description" in resolved
    assert _warnings(caplog, "src.services.design_builder.phases") == []


# ---------------------------------------------------------------------------
# 7. A failed cache invalidation is reported, not swallowed
# ---------------------------------------------------------------------------


async def test_invalidate_style_returns_false_only_on_redis_failure() -> None:
    """``False`` must mean "the stale entry may still be served", nothing else.

    ``RedisClient.delete`` returns False both for "key absent" and for "Redis
    unreachable" (it catches ``RedisError`` itself), so a caller that routed
    through it either warned on every uncached update or never warned at all.

    MUTATION KILLED: reverting ``invalidate_style`` to ``return await
    self._redis.delete(key)`` — the absent-key case below flips to False (making
    the callers' warning fire constantly on healthy Redis), and the broken-Redis
    case ALSO returns False via the helper's own swallow, so the two become
    indistinguishable.
    """
    healthy = _cache(_FakeRedis())
    # Nothing cached under this key: a no-op delete is still a successful one.
    assert await healthy.invalidate_style(TENANT_1, "never-cached") is True

    await healthy.cache_style_description(TENANT_1, "house-style", "t1 secret")
    assert await healthy.invalidate_style(TENANT_1, "house-style") is True
    assert await healthy.get_cached_style_description(TENANT_1, "house-style") is None

    # Redis blows up mid-delete -> the entry may survive its TTL -> False.
    broken = _cache(_BrokenRedis())
    assert await broken.invalidate_style(TENANT_1, "house-style") is False


async def test_update_route_warns_when_invalidation_fails(
    db_session, caplog: pytest.LogCaptureFixture
) -> None:
    """``PUT /styles/{id}`` must not swallow a failed invalidation.

    MUTATION KILLED: dropping the ``if not await
    cache_service.invalidate_style(...)`` guard in the update route (back to a
    bare ``await``) — the route still returns 200 with the new name, so the
    response body proves nothing; only the WARNING distinguishes the two.
    """
    style = await _seed(db_session, TENANT_1, "House Style", "old brand")
    request = _request(TENANT_1, cache_service=_cache(_BrokenRedis()))

    with caplog.at_level(logging.WARNING, logger="src.api.routes.styles"):
        updated = await update_style(
            style.style_id,
            UpdateCustomStyleRequest(name="Renamed Style"),
            request,
            db=db_session,
        )

    assert updated.slug == "renamed-style"  # paired success — the write happened
    messages = _warnings(caplog, "src.api.routes.styles")
    # BOTH slugs are invalidated on a rename, so both failures are reported.
    assert len(messages) == 2
    assert {m for m in messages if "house-style" in m}
    assert {m for m in messages if "renamed-style" in m}
    assert all(TENANT_1 in m and style.style_id in m for m in messages)


async def test_delete_route_warns_when_invalidation_fails(
    db_session, caplog: pytest.LogCaptureFixture
) -> None:
    """``DELETE /styles/{id}`` is the worse case: the row is gone, so nothing
    will ever correct the cached description — it is served until the TTL runs
    out.

    MUTATION KILLED: dropping the ``if not ...`` guard in the delete route — the
    response is ``{"deleted": True, ...}`` either way, so only the WARNING
    separates "invalidated" from "stale for the next 600s".
    """
    style = await _seed(db_session, TENANT_1, "House Style", "old brand")
    request = _request(TENANT_1, cache_service=_cache(_BrokenRedis()))

    with caplog.at_level(logging.WARNING, logger="src.api.routes.styles"):
        result = await delete_style(style.style_id, request, db=db_session)

    assert result == {"deleted": True, "style_id": style.style_id}
    messages = _warnings(caplog, "src.api.routes.styles")
    assert len(messages) == 1
    assert "house-style" in messages[0]
    assert TENANT_1 in messages[0]
    assert style.style_id in messages[0]


async def test_healthy_invalidation_stays_quiet(
    db_session, caplog: pytest.LogCaptureFixture
) -> None:
    """The paired negative for both routes: a working Redis logs NOTHING.

    MUTATION KILLED: inverting the guard (``if await
    cache_service.invalidate_style(...)``) or warning unconditionally — either
    would pass the two tests above while burying real staleness in noise on
    every successful update and delete.
    """
    style = await _seed(db_session, TENANT_1, "House Style", "old brand")
    request = _request(TENANT_1, cache_service=_cache(_FakeRedis()))

    with caplog.at_level(logging.WARNING, logger="src.api.routes.styles"):
        await update_style(
            style.style_id,
            UpdateCustomStyleRequest(name="Renamed Style"),
            request,
            db=db_session,
        )
        await delete_style(style.style_id, request, db=db_session)

    assert _warnings(caplog, "src.api.routes.styles") == []
