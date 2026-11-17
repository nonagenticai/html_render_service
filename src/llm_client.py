"""Shared WMClient factory for analysis_service.

All LLM calls in this service go through worker_management's
``POST /workers/{worker_name}/message`` endpoint — never openai /
anthropic / inference-router directly. See memory ``wm-as-sole-llm-gateway.md``
and the parent CLAUDE.md.

A single :class:`WMClient` is constructed at app startup (see
``asgi.py``) and reused across all call sites: its underlying
``httpx.AsyncClient`` is connection-pooled, and the
``identity_event_hook`` reads request-scoped ContextVars per outbound
request so the same client serves concurrent end-user requests with the
correct per-user identity headers.
"""

from __future__ import annotations

from w2v_common.clients import WMClient

from src.config import get_settings


def get_wm_client() -> WMClient:
    """Build a new :class:`WMClient` (used by tests + lifespan startup).

    Production paths SHOULD reuse the shared instance attached to
    ``app.state.wm_client`` rather than calling this helper repeatedly.

    The per-request timeout is taken from ``WMSettings.timeout_seconds``.
    It must cover the slowest worker: a full-page ``redesign-html`` generation
    (~16k output tokens) runs well past WMClient's 120s default, so the default
    silently ReadTimeout'd every real redesign.
    """
    return WMClient(timeout=get_settings().wm.timeout_seconds)
