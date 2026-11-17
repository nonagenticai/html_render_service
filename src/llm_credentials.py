"""Compatibility shim for the legacy LLM-credential helpers.

Historic call sites and tests imported :class:`LLMConfigLike` and the
``resolve_*`` helpers from this module. The credential resolver path is
gone — every LLM call now flows through :class:`w2v_common.clients.WMClient`,
which carries identity via httpx event hooks instead of per-call API
keys. The Protocol survives so domain schemas (``LLMConfigOverride`` in
brand / review / redesign / analyze / critique) can still be passed
around as a structurally typed hint when callers want to forward an
optional ``model_override``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMConfigLike(Protocol):
    """Structural protocol matching any domain's ``LLMConfigOverride``.

    Only the ``model`` field is meaningful now — it maps onto
    :meth:`w2v_common.clients.WMClient.message`'s ``model_override``
    parameter. ``base_url`` and ``api_key_env`` are accepted as
    request-shape compatibility only; the values are ignored because the
    upstream gateway (``worker_management``) decides routing.
    """

    model: str | None


def model_override_from(llm_config: LLMConfigLike | None) -> str | None:
    """Extract the ``model`` field from any ``LLMConfigOverride``-shaped object."""
    if llm_config is None:
        return None
    return llm_config.model
