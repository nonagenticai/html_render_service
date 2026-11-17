"""Application-level caching service wrapping RedisClient."""

from __future__ import annotations

import hashlib
import logging
from urllib.parse import quote

from redis.exceptions import RedisError

from src.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)

# Key prefix constants
_HTML_PREFIX = "redesign:html:"
_STYLE_PREFIX = "redesign:style:"
# A FAILED fetch lives in its own keyspace, never in `_HTML_PREFIX`. The two are
# separate on purpose and the read order in `_phase_resolve_html` matters: a
# known-good body is returned before the failure keyspace is consulted, so a
# transient 503 can never shadow HTML we already have. Keeping that structurally
# impossible beats relying on a conditional. Same shape as
# `modality_gateway/src/services/resolve_cache.py`, which splits its good and
# fail-open route caches for exactly this reason.
_HTML_FAILED_PREFIX = "redesign:html:failed:"

# Good HTML is cached for `cache_html`'s 3600s default. A FAILED fetch gets its
# own, much shorter TTL: the cache is the amplifier in this defect: pinning a bad
# result for an hour turns one blip into an hour of every job for that URL
# inheriting it. 60s keeps the only property worth keeping -- a hot URL that is
# hard-down does not pay a 30s Playwright navigation per job -- while bounding
# the exposure to roughly the blip. It is deliberately NOT tunable via config:
# there is no operational reason to want this longer.
_FAILED_FETCH_TTL_SECONDS = 60


def _url_hash(url: str) -> str:
    """Compute a SHA-256 hex digest of a URL for use as a cache key."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def style_cache_key(tenant_id: str, slug: str) -> str:
    """Build the cache key for a resolved style description.

    The key is COMPOSITE — ``(tenant_id, slug)`` — and both halves are
    percent-encoded so the mapping is injective (a tenant id or slug that
    itself contains ``:`` cannot forge another tenant's key).

    Why the composite and not the bare slug: ``custom_styles`` is keyed by
    ``(tenant_id, slug)``, never by slug alone. Two tenants may each own a
    style called "House Style" -> slug ``house-style``. A slug-keyed cache
    would therefore hand tenant A's brand description (palette, typography,
    reference sites) straight to tenant B on the next lookup — a cross-tenant
    leak introduced by "fixing" the read/write key mismatch the lazy way.

    Why not the ``style_id``: the read side only ever holds the slug
    (``RedesignJob.style`` is the free-text style token the caller submitted),
    so a ``style_id``-keyed entry can never be found. That mismatch is what
    made this cache a 100% miss for its whole life.
    """
    return f"{_STYLE_PREFIX}{quote(tenant_id, safe='')}:{quote(slug, safe='')}"


class CacheService:
    """High-level caching facade for the redesign service.

    Wraps a ``RedisClient`` instance and provides typed helpers for
    caching fetched HTML pages and style descriptions with appropriate
    key patterns and TTLs.

    EVERY handler below catches ``OSError, RedisError, RuntimeError`` — the same
    triple as ``invalidate_style``, and for the same reason: redis-py raises
    ``redis.exceptions.ConnectionError``/``TimeoutError``/``ResponseError``,
    which derive from ``RedisError(Exception)`` and are **not** ``OSError``
    subclasses. The builtin ``ConnectionError``/``TimeoutError`` these clauses
    used to name are ``OSError`` subclasses that a real Redis never raises, so
    the whole degrade-to-miss contract was decorative on the two HTML helpers,
    which call the raw client via ``_ensure_connected()``. A ``RedisError``
    escaping ``get_cached_html`` propagates out of ``_phase_resolve_html`` into
    ``upstream_failure`` classification, which only treats
    ``httpx.TimeoutException`` as transient — so a Redis blip TERMINALLY failed
    a tenant's paid redesign instead of costing it one cache miss.

    The two style helpers go through ``RedisClient.set``/``.get``, which absorb
    ``RedisError`` themselves, so they were never reachable that way; they name
    the same triple anyway so the class has one rule rather than two, and so a
    future refactor onto the raw client cannot silently reopen the hole.
    """

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client

    # ------------------------------------------------------------------
    # HTML caching
    # ------------------------------------------------------------------

    async def cache_html(
        self,
        url: str,
        html: str,
        ttl: int = 3600,
    ) -> bool:
        """Cache fetched HTML content keyed by the SHA-256 of *url*."""
        key = f"{_HTML_PREFIX}{_url_hash(url)}"
        try:
            client = self._redis._ensure_connected()
            result = await client.setex(key, ttl, html)
            return bool(result)
        except OSError, RedisError, RuntimeError:
            logger.warning("Failed to cache HTML for url=%s", url, exc_info=True)
            return False

    async def get_cached_html(self, url: str) -> str | None:
        """Retrieve previously cached HTML for *url*."""
        key = f"{_HTML_PREFIX}{_url_hash(url)}"
        try:
            client = self._redis._ensure_connected()
            value = await client.get(key)
            if value is None:
                return None
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)
        except OSError, RedisError, RuntimeError:
            logger.warning(
                "Failed to retrieve cached HTML for url=%s",
                url,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Failed-fetch caching (separate keyspace, short TTL — see the prefix note)
    # ------------------------------------------------------------------

    async def cache_failed_fetch(
        self,
        url: str,
        reason: str,
        ttl: int = _FAILED_FETCH_TTL_SECONDS,
    ) -> bool:
        """Record that fetching *url* did NOT yield the site, for a short window.

        ``reason`` is the human-readable failure message; it is replayed verbatim
        by the next job so the second failure is as legible as the first rather
        than a bare "cached failure".
        """
        key = f"{_HTML_FAILED_PREFIX}{_url_hash(url)}"
        try:
            client = self._redis._ensure_connected()
            result = await client.setex(key, ttl, reason)
            return bool(result)
        except OSError, RedisError, RuntimeError:
            logger.warning("Failed to cache fetch-failure for url=%s", url, exc_info=True)
            return False

    async def get_failed_fetch(self, url: str) -> str | None:
        """Return the recorded failure reason for *url*, or None if not recorded."""
        key = f"{_HTML_FAILED_PREFIX}{_url_hash(url)}"
        try:
            client = self._redis._ensure_connected()
            value = await client.get(key)
            if value is None:
                return None
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)
        except OSError, RedisError, RuntimeError:
            logger.warning(
                "Failed to retrieve cached fetch-failure for url=%s",
                url,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Style description caching
    # ------------------------------------------------------------------

    async def cache_style_description(
        self,
        tenant_id: str,
        slug: str,
        description: str,
        ttl: int = 600,
    ) -> bool:
        """Cache a resolved style description under ``(tenant_id, slug)``."""
        key = style_cache_key(tenant_id, slug)
        try:
            return await self._redis.set(key, description, ttl=ttl)
        except OSError, RedisError, RuntimeError:
            logger.warning(
                "Failed to cache style description for tenant=%s slug=%s",
                tenant_id,
                slug,
                exc_info=True,
            )
            return False

    async def get_cached_style_description(
        self,
        tenant_id: str,
        slug: str,
    ) -> str | None:
        """Return a cached style description for ``(tenant_id, slug)``, or ``None`` on miss."""
        key = style_cache_key(tenant_id, slug)
        try:
            value = await self._redis.get(key)
            if isinstance(value, str):
                return value
            return None
        except OSError, RedisError, RuntimeError:
            logger.warning(
                "Failed to retrieve cached style for tenant=%s slug=%s",
                tenant_id,
                slug,
                exc_info=True,
            )
            return None

    async def invalidate_style(self, tenant_id: str, slug: str) -> bool:
        """Remove the cached style description for ``(tenant_id, slug)``.

        Returns True when the entry is known to be GONE — which includes the
        ordinary case where nothing was cached — and False ONLY when the delete
        could not be carried out, i.e. a stale description may still be served
        for the rest of its TTL. Callers (``PUT``/``DELETE /styles/{id}``) log
        that False, so the two cases must not be conflated.

        Hence the DELETE goes to the underlying redis client rather than through
        ``RedisClient.delete``: that helper returns False both for "key absent"
        and for "Redis unreachable" (it catches ``RedisError`` itself and only
        logs), so routing through it would either make the caller's warning fire
        on every uncached update or hide the real failure entirely. ``RedisError``
        is caught explicitly here for the same reason — it is not an ``OSError``
        subclass, so the builtin tuple alone would let it escape.
        """
        key = style_cache_key(tenant_id, slug)
        try:
            client = self._redis._ensure_connected()
            await client.delete(key)
            return True
        except OSError, RedisError, RuntimeError:
            logger.warning(
                "Failed to invalidate style cache for tenant=%s slug=%s",
                tenant_id,
                slug,
                exc_info=True,
            )
            return False
