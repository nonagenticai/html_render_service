"""Client wrapper for downstream AI concept image generation."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from w2v_common.auth import get_tenant_id_from_context, is_tenant_vouched
from w2v_common.auth.cell_jwt import CELL_TOKEN_HEADER
from w2v_common.clients.base import BaseServiceClient
from w2v_common.discovery import service_url
from w2v_common.tenant import MissingTenantError, TenantError, UnvouchedTenantError

from src.config import get_settings
from src.services.prompt_template import PromptTemplateLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConceptImageResult:
    """Outcome of a concept-image generation attempt.

    ``degraded`` distinguishes a genuine FAILURE (submit/poll error, timeout, or
    a failed downstream batch) — where a fuller result may exist once the image
    service recovers — from a legitimate empty result (feature disabled, or the
    batch completed with zero usable assets). Both yield ``assets == []``, which
    is exactly why the empty list alone was previously ambiguous.
    """

    assets: list[dict[str, object]] = field(default_factory=list)
    degraded: bool = False
    reason: str | None = None


class CellCredentialError(RuntimeError):
    """This pod could not mint the cell token for an in-cell ACG call.

    The cell token is the hop's only credential, so a request without one would
    leave as an anonymous call ACG must refuse. Raised instead (no signing key /
    keyring misconfiguration) and re-raised by the generator, never degraded.
    """


async def _require_cell_token(request: httpx.Request) -> None:
    """Refuse to send an ACG request that carries no ``X-W2V-Cell-Token``.

    Runs AFTER ``identity_event_hook``. The identity hook silently skips the
    mint for an unvouched tenant or an unconfigured signer (correct for its
    many dual-send callers); with no licence on this hop that would be a
    credential-less request, so fail loudly here instead.
    """
    if request.headers.get(CELL_TOKEN_HEADER):
        return
    if get_tenant_id_from_context() is not None and not is_tenant_vouched():
        raise UnvouchedTenantError(
            f"refusing {request.method} {request.url.path} to {ACG_SERVICE}: the bound "
            "tenant's provenance is not vouched, so no cell token can be minted"
        )
    raise CellCredentialError(
        f"refusing {request.method} {request.url.path} to {ACG_SERVICE}: no cell token "
        "minted (CELL_JWT_SIGNING_KEY / CELL_SERVICE_NAME / CELL_DEPLOYMENT_ID unset?)"
    )


#: Discovery name of the callee. Its in-cell identity (the cell token's ``aud``)
#: is the Kubernetes Service name ``cell_services()`` maps it to.
ACG_SERVICE = "ai_content_generation"


class ContentGenerationClient(BaseServiceClient):
    """In-cell client for ai_content_generation_service's concept-image API.

    A ``BaseServiceClient`` so every send goes through w2v_common's
    ``identity_event_hook``: the bound tenant / actor / w2v run ride as headers,
    and — the point of P7 — a fresh ``X-W2V-Cell-Token`` (aud = the ACG Service
    name) is minted per request once this pod has a cell signing key. This used
    to be a raw ``httpx.AsyncClient`` with a static ``Authorization`` header and
    no hook, i.e. a call that carried no tenant at all.

    The cell token is the ONLY credential on this hop. ACG lists
    ``html-render-service`` in ``CELL_ALLOWED_CALLERS`` and runs ``dual`` /
    ``enforce``: a request with no ``Bearer`` credential goes straight to the
    cell lane, which authenticates the SIGNED tenant (``TenantSource.CELL_JWT``,
    vouched) and skips the licence re-gate. The shared service-account
    ``DOWNSTREAM_LICENSE_TOKEN`` (and the old ``CONTENT_GENERATION_API_TOKEN``)
    is deliberately NOT read or sent: a static fleet licence on an in-cell hop is
    what P7 replaces, and dual-sending it kept ACG on the legacy lane.
    """

    def __init__(self, *, timeout: float) -> None:
        super().__init__(
            base_url=service_url(ACG_SERVICE),
            service_name=ACG_SERVICE,
            timeout=timeout,
            license_token=None,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        client = await super()._get_client()
        hooks = client.event_hooks["request"]
        if _require_cell_token not in hooks:
            # Appended, so it sees the headers identity_event_hook stamped.
            client.event_hooks = {**client.event_hooks, "request": [*hooks, _require_cell_token]}
        return client

    async def submit_batch(self, payload: dict[str, object]) -> dict[str, object]:
        """``POST /api/concept-images/batches``.

        No ``Idempotency-Key``, so a 5xx is NOT replayed: each item is a paid
        image generation.
        """
        return await self._request("POST", "/api/concept-images/batches", json=payload)

    async def get_batch(self, batch_id: str) -> dict[str, object]:
        """``GET /api/concept-images/batches/{batch_id}``."""
        return await self._request("GET", f"/api/concept-images/batches/{batch_id}")

    async def download(self, path: str) -> httpx.Response:
        """``GET`` an asset's bytes (``/api/concept-images/assets/{id}/download``).

        Same client, so the same identity hook + cell token as submit/poll: ACG
        resolves the asset under the SIGNED tenant, never an anonymous read.
        """
        return await self._request_raw("GET", path)


#: Upper bound on a downloaded hero image. A provider image is ~1-4 MB; anything
#: far beyond that is not an image this page should inline-reference.
_MAX_HERO_BYTES = 15 * 1024 * 1024

#: Hero canvas: wide landscape (~16:9), multiples of 64 so the mflux backend keeps
#: it as-is, and a size dall-e-3 also accepts. Independent of ``DALLE_SIZE`` (the
#: concept-mockup size) — a hero is always a wide banner.
HERO_IMAGE_SIZE = "1792x1024"

#: Sent as the ACG ``negative_prompt`` (forwarded natively to the mflux backend,
#: appended as "Avoid:" by the OpenAI provider). The words that made the live run
#: (01a15968, "Bakkerij De Molen") paint a whole fake LANDING PAGE — nav bar,
#: buttons, garbled "Prestengares" — live HERE and never in the positive prompt.
HERO_NEGATIVE_PROMPT = (
    "text, letters, words, typography, captions, signage, logos, watermarks, "
    "website, web page, landing page, homepage, user interface, UI, UX, navigation bar, "
    "menu, buttons, screenshot, mockup, wireframe, template, device frame, laptop, "
    "phone, tablet, monitor, screen, browser window"
)

#: Added to :data:`HERO_NEGATIVE_PROMPT` for the ONE abstract retry after the first
#: hero showed text (live run 01a159b1: a garbled shop sign "Cafiiti Boker"). Shops,
#: rooms and street scenes are where FLUX letters things, so the retry excludes them.
HERO_ABSTRACT_NEGATIVE_EXTRA = (
    ", storefront, shop front, shop sign, facade, building exterior, street, "
    "interior, room, counter, menu board, chalkboard, packaging, labels, price tags"
)

#: Words that steer an image model towards drawing a WEBSITE. Stripped from the brief
#: before it enters the positive prompt ("Make a website for my bakery" -> "Make a
#: for my bakery" still describes the bakery; a model only needs the subject).
_WEB_TERMS_RE = re.compile(
    r"\b(?:web\s*-?\s*sites?|web\s*-?\s*pages?|landing\s*-?\s*pages?|home\s*-?\s*pages?|"
    r"sites?|pages?|user\s+interfaces?|ui|ux|mock\s*-?\s*ups?|screenshots?|wireframes?|"
    r"templates?|apps?|layouts?|navigation|nav\s*bars?|buttons?)\b",
    re.IGNORECASE,
)


def hero_subject(brand_name: str | None, brief: str | None) -> str:
    """The business's real-world subject, from the brief, with no web/UI vocabulary.

    The brand NAME is replaced by "the business": a literal name in an image prompt is
    an instruction to letter it onto a sign (the live hero lettered "De Molen").
    """
    text = " ".join(str(brief or "").split())
    name = (brand_name or "").strip()
    if name:
        text = re.sub(re.escape(name), "the business", text, flags=re.IGNORECASE)
    text = _WEB_TERMS_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(text.split()))
    return text[:600].strip()


def abstract_hero_prompt(subject: str, primary_color: str) -> str:
    """The retry prompt: a tight close-up of product / material / hands, no place at all.

    Used only after the first hero showed visible text. A storefront, a shop interior or
    a street is where an image model paints signs and menus, so the retry frames the
    THING the business makes or works with, filling the frame.
    """
    lines = ["A professional close-up macro photograph, tightly framed."]
    if subject:
        lines.append(f"Subject — what the business makes or works with: {subject}")
    lines.append(
        "Show only the product itself, its raw materials, or a pair of hands at work on it, "
        "filling the frame: textures, surfaces, craft detail. No storefront, no shop front, "
        "no building, no street, no room or interior, no counter, no menu, no packaging, "
        f"no labels. Natural light, a colour mood that complements {primary_color}, wide "
        "landscape 16:9, shallow depth of field, soft abstract background. Photorealistic. "
        "Absolutely no text, no letters, no words, no numbers, no signage, no logos, "
        "no watermarks."
    )
    return " ".join(lines)


class ConceptImageGenerator:
    """Build concept-image prompts and delegate rendering to the content service."""

    def __init__(self, template_loader: PromptTemplateLoader | None = None) -> None:
        settings = get_settings()
        self._template_loader = template_loader
        self._base_url = service_url(ACG_SERVICE).rstrip("/")
        self._enabled = settings.content_generation.enabled
        self._provider = settings.content_generation.default_provider
        self._timeout = settings.content_generation.timeout
        self._poll_interval_s = settings.content_generation.poll_interval_s
        self._poll_timeout_s = settings.content_generation.poll_timeout_s
        self._size = settings.dalle.size
        self._quality = (
            "high" if str(settings.dalle.quality).lower() == "hd" else settings.dalle.quality
        )
        self._count = settings.dalle.count

        self._client = ContentGenerationClient(timeout=self._timeout)

    async def aclose(self) -> None:
        """Release the long-lived client's connection pool.

        The singleton is built once in ``asgi._startup`` and kept on
        ``app.state.redesign_service.concept_generator``; without this hook,
        ``asgi._shutdown`` only knew about the top-level ``http_client`` and
        ``wm_client`` and never closed the one nested inside RedesignService.
        Net result was a per-pod-lifetime FD/connection-pool leak that also
        held the asyncio loop alive past intended shutdown.

        Defensive ``getattr`` so tests that patch ``__init__`` (and therefore
        never set ``_client``) can still exercise the shutdown path.
        """
        client = getattr(self, "_client", None)
        if client is not None:
            await client.close()

    async def is_available(self) -> bool:
        """Check if downstream image generation is configured."""
        return self._enabled and bool(self._base_url)

    async def generate(
        self,
        *,
        site_description: str | None = None,
        target_audience: list[str] | None = None,
        brand_name: str | None = None,
        primary_color: str = "#2563EB",
        style: str = "modern",
        style_description: str | None = None,
        count: int | None = None,
        output_dir: Path,
    ) -> ConceptImageResult:
        """Generate concept images via the downstream content service.

        ``output_dir`` is accepted for interface compatibility but no longer used for storage.

        Returns a :class:`ConceptImageResult`. A submit/poll error, timeout, or a
        failed downstream batch yields ``degraded=True`` with a reason; a disabled
        feature or a batch that completed with zero assets yields a non-degraded
        empty result.

        Raises :class:`~w2v_common.tenant.TenantError` — NOT a degraded result —
        when this execution context has no (usable) tenant. Every concept-image
        call acts for the job's tenant; a call without one would go out on the
        service licence alone (anonymous w.r.t. tenant, and refused outright
        once ACG enforces cell tokens). That is an identity defect in the
        caller, so it fails the job loudly instead of hiding as "degraded".
        """
        del output_dir
        if not await self.is_available():
            return ConceptImageResult()
        if get_tenant_id_from_context() is None:
            raise MissingTenantError(
                "concept image generation: no tenant bound to this execution context; "
                "refusing an anonymous call to ai_content_generation"
            )

        prompt = self._build_prompt(
            site_description=site_description,
            target_audience=target_audience,
            brand_name=brand_name,
            primary_color=primary_color,
            style=style,
            style_description=style_description,
        )

        requested_count = max(1, count or self._count or 1)
        usage_context = {
            "site_description": site_description,
            "target_audience": target_audience or [],
            "brand_name": brand_name,
            "primary_color": primary_color,
            "style_description": style_description,
            "style": style,
        }

        return await self._submit_and_wait(
            prompt=prompt,
            style=style,
            usage_context=usage_context,
            target_usage="redesign_concept",
            requested_count=requested_count,
        )

    async def generate_hero(
        self,
        *,
        brand_name: str | None,
        brief: str | None,
        primary_color: str = "#2563EB",
        style: str = "modern",
        abstract: bool = False,
        seed: int | None = None,
    ) -> ConceptImageResult:
        """Generate ONE hero photograph for an assembled page (``target_usage=hero_image``).

        ``abstract=True`` is the single retry after a hero came back with visible
        text: a close-up of the product / material / hands, never a storefront,
        signage or interior. ``seed`` is forwarded as ``extra_params.seed`` (the mflux
        backend honours it), so the retry is a genuinely different draw.

        Same downstream, same client, same identity rules as :meth:`generate`
        (a paid call that must act for the job's tenant — no tenant raises
        :class:`MissingTenantError`); only the prompt differs: an on-brief
        photograph of the business's subject with no text, where the concept prompt
        asks for a laptop mockup of the whole site. The page headline is deliberately
        NOT an input: page copy in an image prompt is an instruction to letter it.
        """
        if not await self.is_available():
            return ConceptImageResult(reason="content generation disabled")
        if get_tenant_id_from_context() is None:
            raise MissingTenantError(
                "hero image generation: no tenant bound to this execution context; "
                "refusing an anonymous call to ai_content_generation"
            )
        prompt = self._build_hero_prompt(
            brand_name=brand_name, brief=brief, primary_color=primary_color, abstract=abstract
        )
        negative = HERO_NEGATIVE_PROMPT + (HERO_ABSTRACT_NEGATIVE_EXTRA if abstract else "")
        # usage_context is appended to the prompt verbatim by the OpenAI provider, so it
        # carries no brand name / headline (text bait) — only the photographic mood.
        return await self._submit_and_wait(
            prompt=prompt,
            style=style,
            usage_context={"primary_color": primary_color, "style": style},
            target_usage="hero_image",
            requested_count=1,
            size=HERO_IMAGE_SIZE,
            negative_prompt=negative,
            extra_params={"seed": seed} if seed is not None else None,
        )

    async def download_asset(self, asset: dict[str, object]) -> tuple[bytes, str] | None:
        """Fetch a generated asset's bytes from ai_content_generation.

        Returns ``(data, content_type)`` or ``None`` when the asset is not
        downloadable from OUR content service (a provider-CDN ``image_url`` on an
        arbitrary host is never fetched — this pod reaches the tailnet), the
        response is not an image, or it is empty / oversized.
        """
        url = str(asset.get("download_url") or asset.get("url") or "").strip()
        base = self._base_url
        if not (base and url.startswith(f"{base}/")):
            logger.warning("hero image asset is not served by %s; not fetching", ACG_SERVICE)
            return None
        path = url[len(base) :]
        response = await self._client.download(path)
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
        data = response.content
        if not content_type.startswith("image/"):
            logger.warning("hero image download returned %r, not an image", content_type)
            return None
        if not data or len(data) > _MAX_HERO_BYTES:
            logger.warning("hero image download size %d out of bounds", len(data))
            return None
        return data, content_type

    async def _submit_and_wait(
        self,
        *,
        prompt: str,
        style: str,
        usage_context: dict[str, object],
        target_usage: str,
        requested_count: int,
        size: str | None = None,
        negative_prompt: str | None = None,
        extra_params: dict[str, object] | None = None,
    ) -> ConceptImageResult:
        """Submit one batch and poll it to a terminal state (shared by both generators)."""
        item: dict[str, object] = {
            "prompt": prompt,
            "provider": self._provider,
            "style": style,
            "size": size or self._size,
            "quality": self._quality,
            "usage_context": usage_context,
            "target_usage": target_usage,
        }
        if negative_prompt:
            item["negative_prompt"] = negative_prompt
        if extra_params:
            item["extra_params"] = dict(extra_params)
        payload: dict[str, object] = {"items": [dict(item) for _ in range(requested_count)]}

        try:
            response_payload = await self._client.submit_batch(payload)
        except TenantError:
            # Raised by the identity hook / _require_cell_token (blank / unvouched tenant). A
            # TenantError IS a ValueError, so without this it would be
            # swallowed below as an ordinary degraded result.
            raise
        except CellCredentialError:
            raise
        except (OSError, TimeoutError, ValueError, RuntimeError, httpx.HTTPError) as exc:
            logger.warning("Failed to submit concept image batch: %s", exc)
            return ConceptImageResult(degraded=True, reason=f"submit failed: {exc}")

        batch_id = response_payload.get("batch_id")
        if not (isinstance(batch_id, str) and batch_id.strip()):
            logger.warning("Concept image batch submit returned no batch_id")
            return ConceptImageResult(degraded=True, reason="submit returned no batch_id")

        try:
            assets, unusable_reason = await self._wait_for_batch(
                batch_id, requested_count=requested_count
            )
        except TenantError, CellCredentialError:
            raise
        except (OSError, TimeoutError, ValueError, RuntimeError, httpx.HTTPError) as exc:
            logger.warning("Failed to resolve concept image batch %s: %s", batch_id, exc)
            return ConceptImageResult(degraded=True, reason=f"batch {batch_id} failed: {exc}")
        if unusable_reason is not None:
            # The batch reported success but handed back nothing this service can
            # use. That is a downstream/contract failure, NOT "zero images were
            # asked for" — and the two were previously byte-identical results.
            logger.warning("Concept image batch %s: %s", batch_id, unusable_reason)
            return ConceptImageResult(degraded=True, reason=f"batch {batch_id}: {unusable_reason}")
        # Batch resolved cleanly — assets may legitimately be empty (zero usable
        # images), which is NOT degraded.
        return ConceptImageResult(assets=assets)

    async def _wait_for_batch(
        self,
        batch_id: str,
        *,
        requested_count: int,
    ) -> tuple[list[dict[str, object]], str | None]:
        """Poll the batch to a terminal state.

        Returns ``(assets, unusable_reason)``. ``unusable_reason`` is ``None`` for
        a clean resolution — INCLUDING a batch that genuinely produced nothing —
        and a human-readable string when the batch claimed completion while
        returning payloads none of which survived normalisation. Without that
        second element the caller could not tell "the provider returned nothing"
        from "the provider returned N items and every one of them was missing an
        ``asset_id`` or a url": both collapse to ``assets == []``, and the batch
        was reported as a clean, non-degraded empty. The counts are the whole
        point of the distinction, so they are carried in the reason.
        """
        deadline = time.monotonic() + self._poll_timeout_s
        while time.monotonic() < deadline:
            data = await self._client.get_batch(batch_id)
            status = str(data.get("status") or "").lower()
            if status in {"completed", "partial"}:
                results = data.get("results", [])
                if not isinstance(results, list):
                    # Same fail-open class: a malformed payload is not an empty one.
                    return [], f"malformed batch payload: results was {type(results).__name__}"
                assets = [
                    asset
                    for asset in (self._normalize_asset_payload(item) for item in results)
                    if asset
                ]
                if assets:
                    return assets[:requested_count], None
                if status == "completed":
                    if results:
                        return [], f"{len(results)} item(s) returned, 0 normalizable"
                    return [], None
            if status in {"failed", "error", "cancelled"}:
                raise RuntimeError(data.get("error") or f"Concept image batch {batch_id} failed")
            await asyncio.sleep(self._poll_interval_s)
        raise TimeoutError(f"Timed out waiting for concept image batch {batch_id}")

    def _normalize_asset_payload(self, payload: object) -> dict[str, object] | None:
        if not isinstance(payload, dict):
            return None
        data: dict[str, object] = {str(k): v for k, v in payload.items()}
        url = self._resolve_asset_url(
            data.get("url")
            or data.get("presigned_url")
            or data.get("download_url")
            or data.get("image_url")
        )
        asset_id = str(data.get("asset_id") or "").strip()
        if not url or not asset_id:
            return None
        job_id = str(data.get("job_id") or "").strip()
        provider = str(data.get("provider") or "").strip()
        target_usage = str(data.get("target_usage") or "").strip()
        raw_meta = data.get("metadata")
        metadata: dict[str, object] = {
            **({str(k): v for k, v in raw_meta.items()} if isinstance(raw_meta, dict) else {}),
            "download_url": self._resolve_asset_url(data.get("download_url")),
            "presigned_url": self._resolve_asset_url(data.get("presigned_url")),
            "image_url": self._resolve_asset_url(data.get("image_url")),
            "provenance": {
                "source": "ai_content_generation_service",
                "job_id": job_id or None,
                "asset_id": asset_id,
                "provider": provider or None,
                "target_usage": target_usage or None,
            },
        }
        return {
            "job_id": job_id or None,
            "asset_id": asset_id,
            "provider": provider or None,
            "status": str(data.get("status") or "completed"),
            "prompt": str(data.get("prompt") or "").strip() or None,
            "url": url,
            "download_url": metadata["download_url"],
            "presigned_url": metadata["presigned_url"],
            "image_url": metadata["image_url"],
            "target_usage": target_usage or None,
            "usage_context": data.get("usage_context")
            if isinstance(data.get("usage_context"), dict)
            else None,
            "mime_type": str(data.get("mime_type") or "").strip() or None,
            "width": data.get("width") if isinstance(data.get("width"), int) else None,
            "height": data.get("height") if isinstance(data.get("height"), int) else None,
            "file_size_bytes": data.get("file_size_bytes")
            if isinstance(data.get("file_size_bytes"), int)
            else None,
            "cost": float(cost_val)
            if isinstance((cost_val := data.get("cost")), int | float)
            else None,
            "error": str(data.get("error") or "").strip() or None,
            "created_at": str(data.get("created_at") or "").strip() or None,
            "completed_at": str(data.get("completed_at") or "").strip() or None,
            "metadata": {key: value for key, value in metadata.items() if value is not None},
        }

    def _resolve_asset_url(self, value: object) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        if text.startswith(("http://", "https://")):
            return text
        if text.startswith("/"):
            return f"{self._base_url}{text}"
        return f"{self._base_url}/{text.lstrip('/')}"

    @staticmethod
    def _build_hero_prompt(
        *,
        brand_name: str | None,
        brief: str | None,
        primary_color: str,
        abstract: bool = False,
    ) -> str:
        """A photographic SCENE of the business's subject — never a picture of a page.

        The previous prompt opened "hero image for the website of X" and quoted the page
        headline; the model drew exactly that — a whole landing page with a nav bar,
        buttons and garbled lettering. Now: the subject (from the brief, web/UI words
        stripped, brand name neutralised), the photographic treatment, and explicit
        exclusions; the web/UI exclusion words go in :data:`HERO_NEGATIVE_PROMPT`.
        """
        subject = hero_subject(brand_name, brief)
        if abstract:
            return abstract_hero_prompt(subject, primary_color)
        lines = ["A professional editorial photograph: a single real-world scene."]
        if subject:
            lines.append(f"Subject — the business and what it offers: {subject}")
        lines.append(
            "Show its products, its premises or its people at work, as a camera would "
            f"capture them, in natural light, with a colour mood that complements {primary_color}. "
            "Wide landscape 16:9 composition with a clear focal point and calm negative "
            "space on one side. Photorealistic, shallow depth of field. "
            "Absolutely no text, no letters, no words, no numbers, no signage, no logos, "
            "no watermarks, no screens, no interface elements, no device frames."
        )
        return " ".join(lines)

    def _build_prompt(
        self,
        *,
        site_description: str | None,
        target_audience: list[str] | None,
        brand_name: str | None,
        primary_color: str,
        style: str,
        style_description: str | None = None,
    ) -> str:
        """Build a presentation-ready prompt for concept image generation."""
        description = site_description or "a professional website"
        audience = ", ".join(target_audience[:3]) if target_audience else "general audience"
        name = brand_name or "the website"

        style_descriptions = {
            "modern": "clean, modern design with generous whitespace, subtle gradients, and crisp typography",
            "minimalist": "minimalist aesthetic with ample negative space, monochromatic palette, and elegant simplicity",
            "bold": "bold and vibrant design with strong colors, dynamic layouts, and impactful visual hierarchy",
            "elegant": "sophisticated and elegant design with refined typography, muted tones, and premium feel",
        }
        style_desc = style_description or style_descriptions.get(
            style, style_descriptions["modern"]
        )

        if self._template_loader and hasattr(self._template_loader, "render"):
            rendered = self._template_loader.render(
                "concept_image.md.j2",
                brand_name=name,
                description=description,
                audience=audience,
                style_desc=style_desc,
                primary_color=primary_color,
            )
            if rendered:
                return rendered

        return (
            f"A professional website redesign concept for {name}, "
            f"which is {description}. The target audience is {audience}. "
            f"Design direction: {style_desc}. "
            f"Primary brand color: {primary_color}. "
            f"Show a high-fidelity mockup of a modern, conversion-optimized web page "
            f"displayed on a laptop screen in an isometric view. "
            f"The design should look realistic and professional, suitable for a "
            f"client presentation. No text overlays or labels."
        )
