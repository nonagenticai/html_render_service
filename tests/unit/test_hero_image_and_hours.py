"""Assembled-page hero image + opening hours (owner decision after a live web_design_v2 run).

Live run: the hero shipped as a fallback card holding ONE brand letter (the planner
is told to leave ``image`` empty and ``enforce_images`` blanks any URL it cannot
vouch for), and opening hours existed only as an INVENTED FAQ answer.

Now:
* the service generates ONE hero image per job (ai_content_generation, same client
  and identity as concept images), keeps the original in its bucket for provenance,
  and inlines a downscaled JPEG ``data:`` URI — the page never references MinIO
  (http, tailnet-only, expiring, and blocked by the served ``img-src https: data:``);
* ``enforce_images`` trusts exactly that self-produced value, never an LLM ``data:``;
* a failure renders a brand-colour gradient with no letter and never fails the page;
* an ``hours`` block shows ONLY ranges stated in the brief (parsed, normalised,
  per day), else "Opening hours on request"; an FAQ answer with invented hours goes.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from bs4 import BeautifulSoup
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image
from w2v_common.auth import clear_identity_context, set_identity_context
from w2v_common.auth.cell_jwt import CELL_TOKEN_HEADER, reset_cell_state
from w2v_common.clients import base as client_base
from w2v_common.tenant import TenantSource

import src.services.design_builder as design_builder
from src.services.catalog import assemble_page
from src.services.catalog.integrity import (
    IntegrityReport,
    TrustedImages,
    enforce_hours,
    enforce_images,
)
from src.services.catalog.registry import get_block
from src.services.concept_image_generator import ConceptImageGenerator, ConceptImageResult
from src.services.design_builder.phases import RedesignPhasesMixin
from src.services.design_builder.types import _ThemeDecision
from src.services.design_builder.variants import RedesignVariantsMixin
from src.services.hero_image import MAX_BYTES, hero_data_uri
from src.services.html_gen.section_run import GenerationBudget, SectionRun
from tests.tenants import TENANT_1

_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _png(width: int = 2400, height: int = 1600) -> bytes:
    """A noisy PNG (noise defeats JPEG compression, so the size budget is exercised)."""
    img = Image.effect_noise((width, height), 80).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_PNG = _png()


# --------------------------------------------------------------------------- #
# Hero: data: URI encoding
# --------------------------------------------------------------------------- #


def test_hero_data_uri_is_downscaled_jpeg_under_budget():
    uri = hero_data_uri(_PNG)
    assert uri is not None
    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert len(raw) <= MAX_BYTES
    with Image.open(io.BytesIO(raw)) as img:
        assert img.format == "JPEG"
        assert img.width <= 1600


def test_hero_data_uri_rejects_non_image_bytes():
    assert hero_data_uri(b"<html>not an image</html>") is None


# --------------------------------------------------------------------------- #
# Hero: phase wiring (generate -> provenance copy -> inline data: URI)
# --------------------------------------------------------------------------- #


class _FakeStorage:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.stored: list[tuple[str, bytes, str]] = []

    async def store_bytes(self, job_id: str, data: bytes, filename: str) -> str:
        if self.fail:
            raise OSError("minio down")
        self.stored.append((job_id, data, filename))
        return f"/app/data/{job_id}/{filename}"

    async def presigned_url(self, *_a: Any, **_k: Any) -> str | None:  # pragma: no cover
        raise AssertionError("the page must never reference a presigned MinIO URL")


class _FakeGen:
    def __init__(self, *, fail: BaseException | None = None, empty: bool = False) -> None:
        self.fail = fail
        self.empty = empty
        self.calls: list[dict[str, Any]] = []

    async def generate_hero(self, **kwargs: Any) -> ConceptImageResult:
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        if self.empty:
            return ConceptImageResult(degraded=True, reason="submit failed: boom")
        return ConceptImageResult(assets=[{"asset_id": "a1", "url": "x"}])

    async def download_asset(self, asset: dict) -> tuple[bytes, str] | None:
        return _PNG, "image/png"


class _Phases(RedesignPhasesMixin):
    def __init__(self, gen: _FakeGen, storage: _FakeStorage | None = None) -> None:
        self._concept_gen = cast("ConceptImageGenerator", gen)
        self._storage = storage or _FakeStorage()


def _theme() -> _ThemeDecision:
    return _ThemeDecision(
        style="modern",
        style_description=None,
        brand_info={"name": "Hearth & Crumb", "primary_color": "#b45309"},
        context=cast("Any", None),
        auto_theme_metadata=None,
    )


def _plan(image: str = "https://images.unsplash.com/x.jpg") -> list[dict]:
    return [
        {"type": "hero", "content": {"headline": "Bread baked at dawn", "image": image}},
        {"type": "footer", "content": {"brand": "Hearth & Crumb"}},
    ]


def _run() -> SectionRun:
    return SectionRun(budget=GenerationBudget.from_settings())


@pytest.mark.asyncio
async def test_hero_success_inlines_data_uri_and_keeps_original_for_provenance():
    phases = _Phases(_FakeGen())
    plan = _plan()
    run = _run()
    uri, check = await phases._assembly_hero_image(
        "job-1", plan, _theme(), "A sourdough bakery", run=run
    )
    assert check is None  # no checker wired on this fake

    assert uri is not None
    assert uri.startswith("data:image/jpeg;base64,")
    assert plan[0]["content"]["image"] == uri  # the LLM's hotlink is replaced
    storage = cast("_FakeStorage", phases._storage)
    assert storage.stored == [("job-1", _PNG, "hero/original.png")]
    call = cast("_FakeGen", phases._concept_gen).calls[0]
    assert call["brand_name"] == "Hearth & Crumb"
    assert "headline" not in call  # page copy never reaches the image prompt
    assert call["brief"] == "A sourdough bakery"
    assert run.degradations == []

    html = assemble_page(plan, own_images=[uri]).html
    img = _soup(html).select_one("#home img")
    assert img is not None
    assert img["src"] == uri
    assert "minio" not in html


@pytest.mark.asyncio
async def test_hero_provenance_store_failure_does_not_lose_the_image():
    phases = _Phases(_FakeGen(), _FakeStorage(fail=True))
    plan = _plan()
    uri, _ = await phases._assembly_hero_image("job-1", plan, _theme(), "b", run=_run())
    assert uri and uri.startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize(
    "gen",
    [_FakeGen(fail=RuntimeError("acg down")), _FakeGen(empty=True)],
    ids=["raises", "degraded"],
)
@pytest.mark.asyncio
async def test_hero_failure_renders_gradient_without_letter(gen):
    phases = _Phases(gen)
    plan = _plan()
    run = _run()
    url, _ = await phases._assembly_hero_image("job-1", plan, _theme(), "brief", run=run)

    assert url == ""  # attempted and failed: a refine must not pay again
    assert plan[0]["content"]["image"] == ""
    assert any("hero image not generated" in d for d in run.degradations)

    res = assemble_page(plan, brand={"name": "Hearth & Crumb"})
    soup = _soup(res.html)
    assert soup.find("img") is None
    art = soup.select_one("#home .hero-art")
    assert art is not None
    assert art.get_text(strip=True) == ""
    assert "hero-art-mark" not in res.html


@pytest.mark.asyncio
async def test_hero_refine_reuses_first_outcome_without_a_second_paid_call():
    gen = _FakeGen()
    phases = _Phases(gen)
    for reuse in ("data:image/jpeg;base64,AAAA", ""):
        plan = _plan()
        got, _ = await phases._assembly_hero_image(
            "job-1", plan, _theme(), "brief", run=_run(), reuse=reuse
        )
        assert got == reuse
        assert plan[0]["content"]["image"] == reuse
    assert gen.calls == []


@pytest.mark.asyncio
async def test_no_hero_block_means_no_generation():
    gen = _FakeGen()
    plan = [{"type": "footer", "content": {}}]
    got = await _Phases(gen)._assembly_hero_image("j", plan, _theme(), "b", run=_run())
    assert got == (None, None)
    assert gen.calls == []


@pytest.mark.asyncio
async def test_ab_variants_never_generate_or_overwrite_a_hero(monkeypatch):
    """A/B variants (``job.styles`` > 1) render via ``generate_page_chunked``, not the
    assembly path, so they never call ``generate_hero`` nor write ``hero/*`` — the
    hero is generated at most once per job, by the single-page assembly path."""
    jm = MagicMock(
        create_variant=AsyncMock(side_effect=lambda *_a, **k: MagicMock(variant_id=k["style"])),
        mark_variant_completed=AsyncMock(),
    )
    monkeypatch.setattr(design_builder, "JobManager", jm)

    gen = _FakeGen()
    storage = MagicMock(
        store_html=AsyncMock(return_value="/app/data/j/v.html"), store_bytes=AsyncMock()
    )
    obj = RedesignVariantsMixin.__new__(RedesignVariantsMixin)
    obj._check_cancelled = AsyncMock(return_value=False)  # type: ignore[method-assign]
    obj._dispatch_webhook = AsyncMock()  # type: ignore[method-assign]
    obj.__dict__["_concept_gen"] = gen  # what a combined service would expose
    obj._storage = storage
    obj._html_gen = MagicMock(
        generate_page_chunked=AsyncMock(
            return_value=MagicMock(text="<html></html>", input_tokens=1, output_tokens=1)
        )
    )
    job = MagicMock(job_id="j", tenant_id="t", mode="assembly", styles=["modern", "bold"])
    db = MagicMock(commit=AsyncMock(), rollback=AsyncMock())
    for style in job.styles:
        assert (
            await obj._process_variant(db, job, "", style, MagicMock(), MagicMock(), False) is None
        )

    assert jm.mark_variant_completed.await_count == 2
    assert gen.calls == []
    storage.store_bytes.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Hero: enforce_images trusts exactly our own produced value
# --------------------------------------------------------------------------- #

_OWN = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ=="


@pytest.mark.parametrize(
    ("url", "kept"),
    [
        (_OWN, True),
        # an LLM-written data: URI (or ours, altered) is never trusted by shape
        ("data:image/jpeg;base64,/9j/EVIL", False),
        (_OWN + "A", False),
        ("data:image/png;base64,iVBORw0KGgo=", False),
        # MinIO is no longer an image source the page may reference
        ("http://minio.test:8389/html-render-service/j/hero/hero.png", False),
        ("https://minio.test:8389/html-render-service/j/hero/hero.png", False),
        ("https://images.unsplash.com/photo.jpg", False),
    ],
)
def test_enforce_images_keeps_only_our_exact_value(url, kept):
    manifest = get_block("hero")
    assert manifest is not None
    content: dict[str, Any] = {"image": url}
    report = IntegrityReport()
    enforce_images([(manifest, content)], TrustedImages(own=frozenset({_OWN})), report)
    assert (content["image"] == url) is kept
    assert report.images_dropped == (0 if kept else 1)


def test_llm_data_uri_is_dropped_from_the_assembled_page():
    plan = [{"type": "hero", "content": {"headline": "H", "image": _OWN}}]
    html = assemble_page(plan).html  # no own_images: the plan's data: is not ours
    assert _soup(html).find("img") is None


# --------------------------------------------------------------------------- #
# Hero: the generator's hero call (real client, identity hooks intact)
# --------------------------------------------------------------------------- #


@pytest.fixture
def _cell_tenant(monkeypatch):
    pem = (
        Ed25519PrivateKey.generate()
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", pem)
    monkeypatch.setenv("CELL_JWT_KID", "html-hero-kid")
    monkeypatch.setenv("CELL_SERVICE_NAME", "html-render-service")
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", "test-cell")
    reset_cell_state()
    monkeypatch.setattr(client_base, "_RETRY_DELAY", 0.0)
    set_identity_context(TENANT_1, "lic", "sub", tenant_source=TenantSource.LICENSE_TOKEN)
    yield
    clear_identity_context()
    reset_cell_state()


def _gen_with(monkeypatch, handler) -> ConceptImageGenerator:
    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(client_base.httpx, "AsyncClient", _client)
    gen = ConceptImageGenerator()
    gen._enabled = True
    gen._poll_interval_s = 0.0
    return gen


@pytest.mark.asyncio
async def test_generate_hero_is_one_hero_item_and_download_carries_cell_token(
    monkeypatch, _cell_tenant
):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"batch_id": "b1"})
        if request.url.path.endswith("/download"):
            return httpx.Response(200, content=b"img", headers={"content-type": "image/png"})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "results": [
                    {"asset_id": "a1", "download_url": "/api/concept-images/assets/a1/download"}
                ],
            },
        )

    gen = _gen_with(monkeypatch, handler)
    try:
        result = await gen.generate_hero(brand_name="Hearth & Crumb", brief="A bakery")
        assert len(result.assets) == 1
        fetched = await gen.download_asset(result.assets[0])
    finally:
        await gen.aclose()

    assert fetched == (b"img", "image/png")
    items = json.loads(seen[0].content)["items"]
    assert len(items) == 1
    assert items[0]["target_usage"] == "hero_image"
    assert "no text" in items[0]["prompt"].lower()
    download = seen[-1]
    assert download.url.path == "/api/concept-images/assets/a1/download"
    assert download.headers.get(CELL_TOKEN_HEADER)  # the job's signed tenant identity


@pytest.mark.asyncio
async def test_download_asset_never_fetches_a_foreign_host(monkeypatch, _cell_tenant):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError(f"unexpected fetch {request.url}")

    gen = _gen_with(monkeypatch, handler)
    try:
        got = await gen.download_asset({"url": "https://cdn.provider.test/img.png"})
    finally:
        await gen.aclose()
    assert got is None


# --------------------------------------------------------------------------- #
# Hours block
# --------------------------------------------------------------------------- #

_BRIEF_WITH_HOURS = (
    "Hearth & Crumb is a sourdough bakery at 12 Mill Lane, Bristol. "
    "Open Tuesday to Saturday 7am - 3pm, Sunday 8:30am - 1pm, closed Monday."
)
_BRIEF_NO_HOURS = "Hearth & Crumb is a sourdough bakery in Bristol, baking since 1998."


def _hours_plan() -> list[dict]:
    return [
        {
            "type": "hero",
            "content": {
                "headline": "Bread baked at dawn",
                "primary_label": "Order now",
                "secondary_label": "View Opening Hours",
                "secondary_href": "#",
            },
        },
        {
            "type": "hours",
            "content": {
                "headline": "Visit us",
                "items": [
                    {"day": "Tuesday - Saturday", "time": "7am - 3pm"},
                    {"day": "Sunday", "time": "8:30am - 1pm"},
                    {"day": "Monday", "time": "Closed"},
                    {"day": "Friday late", "time": "6pm - 10pm"},  # invented
                ],
                "address": "12 Mill Lane, Bristol",
            },
        },
        {
            "type": "faq",
            "content": {
                "items": [
                    {"question": "When are you open?", "answer": "Daily 9:00 - 17:00."},
                    {"question": "Do you deliver?", "answer": "Yes, across Bristol."},
                ]
            },
        },
        {"type": "footer", "content": {"brand": "Hearth & Crumb"}},
    ]


def test_hours_from_brief_render_as_items_and_invented_rows_are_dropped():
    res = assemble_page(_hours_plan(), brief=_BRIEF_WITH_HOURS)
    soup = _soup(res.html)
    section = soup.select_one("section#hours")
    assert section is not None
    days = [h.get_text(strip=True) for h in section.select(".card-title")]
    times = [p.get_text(strip=True) for p in section.select(".card-body p")]
    assert days == ["Tuesday - Saturday", "Sunday", "Monday"]
    assert times == ["7am - 3pm", "8:30am - 1pm", "Closed"]
    assert "6pm" not in res.html
    assert "12 Mill Lane, Bristol" in section.get_text()
    assert section.select_one("[data-hours-on-request]") is None
    # The hours-ish hero CTA lands on the hours block, not the footer.
    hero_links = [a["href"] for a in soup.select("#home a")]
    assert "#hours" in hero_links
    # The FAQ invented 9:00-17:00 — not in the brief — so that Q&A is gone.
    assert "When are you open?" not in res.html
    assert "Do you deliver?" in res.html
    assert res.integrity_report is not None
    assert res.integrity_report.hours_dropped == 2


def test_no_hours_in_brief_renders_on_request_and_no_times():
    res = assemble_page(_hours_plan(), brief=_BRIEF_NO_HOURS)
    soup = _soup(res.html)
    section = soup.select_one("section#hours")
    assert section is not None
    assert section.select(".card") == []
    on_request = section.select_one("[data-hours-on-request]")
    assert on_request is not None
    assert on_request.get_text(strip=True) == "Opening hours on request"
    text = section.get_text(" ", strip=True)
    for invented in ("7am", "3pm", "8:30", "Closed", "6pm", "Mill Lane"):
        assert invented not in text
    assert "When are you open?" not in res.html  # FAQ may not invent hours either
    assert "Do you deliver?" in res.html


def test_enforce_hours_ignores_non_time_numbers_in_faq():
    manifest = get_block("faq")
    assert manifest is not None
    content: dict[str, Any] = {
        "items": [
            {"question": "How long are you open for business?", "answer": "Since 1998."},
            {"question": "Price?", "answer": "A loaf is £3.50."},
        ]
    }
    report = IntegrityReport()
    enforce_hours([(manifest, content)], _BRIEF_NO_HOURS, report)
    assert len(content["items"]) == 2
    assert report.hours_dropped == 0


def test_fill_prompts_forbid_invented_hours():
    from src.services.assembly_planner import ASSEMBLY_FILL_SYSTEM_PROMPT, ASSEMBLY_SYSTEM_PROMPT

    for prompt in (ASSEMBLY_FILL_SYSTEM_PROMPT, ASSEMBLY_SYSTEM_PROMPT):
        assert "Opening hours" in prompt
        assert "FAQ answer" in prompt
        assert "Never invent them" in prompt


# --------------------------------------------------------------------------- #
# Hours: parsed, normalised ranges — never bare-number comparison
# --------------------------------------------------------------------------- #


def _kept_rows(brief: str, rows: list[dict]) -> list[dict]:
    manifest = get_block("hours")
    assert manifest is not None
    content: dict[str, Any] = {"items": rows}
    enforce_hours([(manifest, content)], brief, IntegrityReport())
    return content["items"]


def test_hours_12h_brief_keeps_equivalent_24h_row():
    """Brief "9am\u20135pm", model "09:00\u201317:00": the SAME range (540, 1020) — kept."""
    rows = [{"day": "Monday - Friday", "time": "09:00\u201317:00"}]
    assert _kept_rows("Open Monday to Friday 9am\u20135pm.", rows) == rows


def test_hours_same_digits_different_range_is_dropped():
    """Brief "9:00\u201317:30", model "9:30\u201317:00": same digits, different range — dropped."""
    rows = [{"day": "Monday", "time": "9:30\u201317:00"}]
    assert _kept_rows("We are open Monday 9:00\u201317:30.", rows) == []
    # ...while the faithful copy survives.
    good = [{"day": "Monday", "time": "9:00 - 17:30"}]
    assert _kept_rows("We are open Monday 9:00\u201317:30.", good) == good


def test_hours_bare_9_to_5_means_nine_to_five_pm():
    """Brief "9-5": end < start, so 9:00\u201317:00 — a correct row is kept, not dropped."""
    rows = [
        {"day": "Weekdays", "time": "9:00 - 17:00"},
        {"day": "Monday", "time": "9am - 5pm"},
    ]
    assert _kept_rows("Open weekdays 9-5.", rows) == rows
    assert _kept_rows("Open weekdays 9-5.", [{"day": "Saturday", "time": "9-5"}]) == []
