"""Webhook notification dispatcher with HMAC signing and retry logic."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from datetime import UTC, datetime

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.net import (
    ResponseTooLargeError,
    SSRFError,
    safe_post,
    validate_public_url,
)

from src.database.models import WebhookDeliveryLog

# The webhook response body is attacker-controlled, so it is read under a hard
# cap and aborted the moment the running total crosses it — it is never fully
# materialised. Only a short snippet is kept, in the delivery log.
_MAX_RESPONSE_BYTES = 64 * 1024
_MAX_RESPONSE_TEXT_CHARS = 1000

logger = logging.getLogger(__name__)


class WebhookDispatcher:
    """Dispatch webhook notifications with HMAC-SHA256 signing and retries."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        timeout: float = 10.0,
    ) -> None:
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._timeout = timeout

    async def dispatch(
        self,
        db: AsyncSession,
        *,
        job_id: str,
        webhook_url: str,
        webhook_secret: str | None = None,
        event_type: str,
        payload: dict,
    ) -> bool:
        """Dispatch a webhook notification with retry logic.

        Args:
            db: Async database session for delivery logging.
            job_id: Associated job identifier.
            webhook_url: Target URL for the webhook POST.
            webhook_secret: Optional HMAC-SHA256 signing secret.
            event_type: Event name (e.g., 'job.completed').
            payload: Event payload dict.

        Returns:
            True on successful delivery, False after all retries fail.
        """
        logger.info(
            "Dispatching webhook event=%s for job %s to %s", event_type, job_id, webhook_url
        )

        # SSRF pre-check so an invalid target is rejected (and logged as
        # "blocked") before we bother building the signed body. The actual
        # send below goes through ``safe_post``, which re-validates and — the
        # part that matters — PINS the connection to the address it resolved.
        # Validating here and then handing the *hostname* to httpx would let it
        # resolve a second time: the DNS-rebinding TOCTOU window this guard
        # exists to close. No allow-list: a webhook must never reach an
        # internal host.
        try:
            validate_public_url(webhook_url)
        except (SSRFError, ValueError) as exc:
            logger.error(
                "Refusing webhook for job %s — target %s is not a valid public URL: %s",
                job_id,
                webhook_url,
                exc,
            )
            await self._log_delivery(
                db,
                job_id=job_id,
                attempt=1,
                event_type=event_type,
                payload=payload,
                response_status=None,
                response_text=f"blocked: {exc}",
                status="blocked",
            )
            return False

        body = json.dumps(payload, default=str)

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-Webhook-Event": event_type,
            "X-Webhook-Delivery": str(uuid.uuid7()),
        }

        if webhook_secret:
            signature = hmac.new(
                webhook_secret.encode("utf-8"),
                body.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        for attempt in range(1, self._max_retries + 1):
            status = "pending"
            response_text = None
            response_status = None

            try:
                # `safe_post` resolves the host once and pins the connection to
                # that address, so the transport cannot re-resolve to an
                # internal one. It also bounds the (attacker-controlled)
                # response body via `max_fetch_bytes` — streamed, aborted the
                # moment the running total crosses the cap, so an oversized
                # body is never fully materialised.
                #
                # max_redirects=0: this body carries an HMAC signature and must
                # never be replayed to a redirect target.
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await safe_post(
                        client,
                        webhook_url,
                        content=body,
                        headers=headers,
                        max_redirects=0,
                        max_fetch_bytes=_MAX_RESPONSE_BYTES,
                    )
                    response_status = response.status_code
                    response_text = response.text[:_MAX_RESPONSE_TEXT_CHARS]

                    if response.is_success:
                        status = "delivered"
                        logger.info(
                            "Webhook delivered successfully for job %s (attempt %d)",
                            job_id,
                            attempt,
                        )
                        await self._log_delivery(
                            db,
                            job_id=job_id,
                            attempt=attempt,
                            event_type=event_type,
                            payload=payload,
                            response_status=response_status,
                            response_text=response_text,
                            status=status,
                        )
                        return True

                    status = "failed"
                    logger.warning(
                        "Webhook attempt %d/%d failed for job %s: HTTP %d",
                        attempt,
                        self._max_retries,
                        job_id,
                        response_status,
                    )

            except SSRFError as exc:
                # The target answered with a 30x. `max_redirects=0` refuses to
                # follow it, so this is the target trying to bounce a *signed*
                # body somewhere we will not send it — terminal, not transient.
                # Retrying cannot help, and letting SSRFError (a ValueError, so
                # NOT caught by the transport tuple below) propagate would crash
                # the caller instead of failing this one delivery.
                logger.error(
                    "Refusing webhook for job %s — target %s redirected: %s",
                    job_id,
                    webhook_url,
                    exc,
                )
                await self._log_delivery(
                    db,
                    job_id=job_id,
                    attempt=attempt,
                    event_type=event_type,
                    payload=payload,
                    response_status=None,
                    response_text=f"blocked: {exc}",
                    status="blocked",
                )
                return False

            except (
                OSError,
                TimeoutError,
                RuntimeError,
                httpx.HTTPError,
                ResponseTooLargeError,
            ) as exc:
                # httpx transport/status errors (ConnectError, ReadTimeout,
                # TimeoutException, …) subclass httpx.HTTPError, NOT OSError —
                # without it a webhook to a resolvable-but-down host escapes the
                # retry loop uncaught and crashes dispatch.
                status = "error"
                response_text = str(exc)[:500]
                logger.warning(
                    "Webhook attempt %d/%d error for job %s: %s",
                    attempt,
                    self._max_retries,
                    job_id,
                    exc,
                )

            await self._log_delivery(
                db,
                job_id=job_id,
                attempt=attempt,
                event_type=event_type,
                payload=payload,
                response_status=response_status,
                response_text=response_text,
                status=status,
            )

            if attempt < self._max_retries:
                backoff = self._backoff_base**attempt
                await asyncio.sleep(backoff)

        return False

    async def _log_delivery(
        self,
        db: AsyncSession,
        *,
        job_id: str,
        attempt: int,
        event_type: str,
        payload: dict,
        response_status: int | None,
        response_text: str | None,
        status: str,
    ) -> None:
        """Log a webhook delivery attempt."""
        try:
            log_entry = WebhookDeliveryLog(
                job_id=job_id,
                attempt_number=attempt,
                event_type=event_type,
                request_payload=payload,
                response_status=response_status,
                response_text=response_text,
                status=status,
                created_at=datetime.now(UTC),
            )
            db.add(log_entry)
            await db.flush()
        except Exception as exc:  # noqa: BLE001 — audit logging must never fail the caller
            # Catch every exception type. This method only WRITES AN AUDIT ROW;
            # it has no bearing on whether the webhook was delivered, so there
            # is no failure here worth propagating to `dispatch` and from there
            # into the job. The previous tuple could not see the one thing this
            # body actually raises: `db.flush()` raises
            # `sqlalchemy.exc.SQLAlchemyError`, which derives from `Exception`,
            # not `OSError` — so a routine DB hiccup while logging a delivery
            # escaped `_log_delivery`, escaped `dispatch`, and took down
            # whatever was dispatching.
            logger.warning("Failed to log webhook delivery: %s", exc)

    @staticmethod
    def build_event_payload(
        event_type: str,
        job_id: str,
        *,
        status: str | None = None,
        error: str | None = None,
        result_url: str | None = None,
        batch_id: str | None = None,
        variant_id: str | None = None,
        accessibility_score: float | None = None,
    ) -> dict:
        """Build a standard webhook event payload."""
        payload: dict = {
            "event": event_type,
            "job_id": job_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if status:
            payload["status"] = status
        if error:
            payload["error"] = error
        if result_url:
            payload["result_url"] = result_url
        if batch_id:
            payload["batch_id"] = batch_id
        if variant_id:
            payload["variant_id"] = variant_id
        if accessibility_score is not None:
            payload["accessibility_score"] = accessibility_score
        return payload
