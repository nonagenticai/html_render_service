"""The LLM's reply must be markup before it is stored as the page.

Two ways an LLM prose reply used to become ``redesign.html``:

1. ``refine_full_page`` extracted the code block but never applied ``_require_html``
   the way its siblings ``generate_full_page`` / ``generate_sections`` do. With no
   fence in the reply ``_extract_html_block`` returns the text VERBATIM, so a
   refusal/apology was stored, screenshotted and the job marked completed.
2. ``_require_html`` only asked ``"<" in text``, so an HTML *comment*
   (``<!-- could not complete -->``) passed as "markup" and shipped as a blank
   white PNG. The freeform mode (the film-pipeline default) returns before
   ``verify_html``, so nothing downstream caught it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from src.services.html_gen.html_utils import _require_html

if TYPE_CHECKING:
    from w2v_common.clients import WMClient


class _StubWMResponse:
    def __init__(self, output: object) -> None:
        self.output = output
        self.usage_metrics = {"prompt_tokens": 1, "completion_tokens": 2}


class _StubWM:
    def __init__(self, output: object) -> None:
        self._output = output

    async def message(self, **kwargs: object) -> _StubWMResponse:
        return _StubWMResponse(self._output)


_PROSE = (
    "I'm sorry, but I can't complete this redesign request. "
    "Consider simplifying the navigation and increasing contrast instead."
)


# ---------------------------------------------------------------------------
# 1. refine must not store prose
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_refine_full_page_rejects_unfenced_prose() -> None:
    from src.services.html_generator import HtmlGenerator

    gen = HtmlGenerator(wm_client=cast("WMClient", _StubWM(_PROSE)))
    with pytest.raises(RuntimeError, match="refine_full_page"):
        await gen.refine_full_page(
            url="https://example.com",
            previous_html="<html><body><h1>old</h1></body></html>",
            feedback="make it bolder",
        )


@pytest.mark.asyncio
async def test_refine_full_page_still_returns_real_html() -> None:
    from src.services.html_generator import HtmlGenerator

    page = "<!DOCTYPE html><html><body><h1>new</h1></body></html>"
    gen = HtmlGenerator(wm_client=cast("WMClient", _StubWM(f"```html\n{page}\n```")))
    result = await gen.refine_full_page(
        url="https://example.com",
        previous_html="<html><body><h1>old</h1></body></html>",
        feedback="make it bolder",
    )
    assert result.text == page


@pytest.mark.asyncio
async def test_refine_job_does_not_store_prose_as_redesign_html() -> None:
    """Pin the production call site: the refine lifecycle must not reach store_html."""
    from src.services.html_generator import HtmlGenerator

    gen = HtmlGenerator(wm_client=cast("WMClient", _StubWM(_PROSE)))
    # The lifecycle stores ``llm_result.text`` verbatim, so the guard has to live
    # inside refine_full_page — assert it raises before any text is produced.
    with pytest.raises(RuntimeError):
        await gen.refine_full_page(
            url="https://example.com", previous_html="<html></html>", feedback="x"
        )


# ---------------------------------------------------------------------------
# 2. _require_html must demand an element, not just a '<'
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "<!-- could not complete -->",
        "<!DOCTYPE html>",
        "a < b and c > d",
    ],
)
def test_require_html_rejects_non_element_text(text: str) -> None:
    with pytest.raises(RuntimeError, match="no HTML markup"):
        _require_html(text, "unit")


@pytest.mark.parametrize(
    "text",
    [
        "<!DOCTYPE html><html><body><h1>hi</h1></body></html>",
        "<section class='hero'><h2>Hi</h2></section>",
        "<!-- leading comment --><div>ok</div>",
    ],
)
def test_require_html_accepts_real_markup(text: str) -> None:
    assert _require_html(text, "unit") == text


@pytest.mark.asyncio
async def test_generate_full_page_rejects_a_comment_only_reply() -> None:
    """Freeform (the film-pipeline default) returns before verify_html — guard here."""
    from src.services.html_generator import HtmlGenerator

    gen = HtmlGenerator(wm_client=cast("WMClient", _StubWM("<!-- could not complete -->")))
    with pytest.raises(RuntimeError, match="generate_full_page"):
        await gen.generate_full_page(url="https://example.com", html="<html><body>x</body></html>")
