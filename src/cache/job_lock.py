"""Distributed job locking via Redis SET NX to prevent duplicate processing."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable
from typing import TYPE_CHECKING, cast

from redis.exceptions import RedisError

if TYPE_CHECKING:
    from w2v_common.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)

_LOCK_PREFIX = "slide:lock:job:"
_OWNER_ID = uuid.uuid7().hex[-12:]

# Lua script: release only if this pod owns the lock
_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


async def acquire_job_lock(redis_client: RedisClient | None, job_id: str, ttl: int = 300) -> bool:
    """Attempt to acquire a distributed lock for a job.

    Fail-open: returns True if Redis is unavailable (allows processing).

    ``RedisError`` is in the net because these helpers drive the RAW redis client
    (``redis_client._ensure_connected()``), not ``RedisClient.get``/``set`` which
    swallow it internally. A redis ``ConnectionError``/``TimeoutError`` is a
    ``RedisError`` — an ``Exception``, **not** an ``OSError`` — so the previous
    ``(OSError, RuntimeError)`` tuple never fired and an unreachable Redis
    *crashed the job* here, the exact opposite of the documented fail-open. The
    builtins stay: a socket failure below the client can still surface as those.
    Only the base ``RedisError`` is named, never its subclasses — any redis-side
    failure means "we could not determine ownership", which is one decision.
    """
    if redis_client is None:
        return True
    try:
        client = redis_client._ensure_connected()
        key = f"{_LOCK_PREFIX}{job_id}"
        result = await client.set(key, _OWNER_ID, nx=True, ex=ttl)
        if result:
            logger.debug("Acquired lock for job %s", job_id)
            return True
        logger.info("Job %s already locked by another pod", job_id)
        return False
    except (OSError, RuntimeError, RedisError) as exc:
        logger.warning("Redis lock acquisition failed for job %s: %s (fail-open)", job_id, exc)
        return True


async def release_job_lock(redis_client: RedisClient | None, job_id: str) -> None:
    """Release a distributed lock for a job.

    Uses a Lua script to ensure only the lock owner can release it.

    Same ``RedisError`` reasoning as :func:`acquire_job_lock`: release runs in the
    finally-shaped tail of a finished job, so an uncaught ``RedisError`` here
    would mark a job that already succeeded as failed. Degrading to a warning is
    safe — the lock carries a TTL and expires on its own.
    """
    if redis_client is None:
        return
    try:
        client = redis_client._ensure_connected()
        key = f"{_LOCK_PREFIX}{job_id}"
        await cast(Awaitable[str], client.eval(_RELEASE_SCRIPT, 1, key, _OWNER_ID))
        logger.debug("Released lock for job %s", job_id)
    except (OSError, RuntimeError, RedisError) as exc:
        logger.warning("Redis lock release failed for job %s: %s", job_id, exc)
