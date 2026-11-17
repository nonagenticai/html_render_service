"""A fenced or truncated LLM reply must never become the delivered ``redesign.html``.

Observed on a delivered ``web_design_v1`` artifact on 2026-09-10: the page began with
a literal ```` ```html ```` fence rendered as visible text AND ended mid-``<ul>`` with
no ``</body></html>``. The job reported ``completed``.

Both symptoms are ONE root cause. ``_extract_html_block`` only knew the *pair-matched*
fence ``` ```html … ``` ```; when the reply is cut off at the output-token ceiling
there is no closing fence, so the regex missed and the raw text — leading fence and
all — was returned verbatim and stored. ``_require_html`` then passed it, because it
only asked "is there an element tag in here".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from src.services.accessibility_scorer import _reject_reason as _scorer_reject_reason
from src.services.html_gen.html_utils import (
    _extract_html_block,
    _reject_reason,
    _require_html,
)

if TYPE_CHECKING:
    from w2v_common.clients import WMClient


_COMPLETE_PAGE = (
    '<!DOCTYPE html>\n<html lang="en">\n<head><title>Acme</title></head>\n'
    "<body><main><h1>Acme</h1><ul><li>One</li><li>Two</li></ul></main></body>\n</html>"
)

# The shape that actually shipped: opening fence, no closing fence, cut mid-<ul>.
_TRUNCATED_REPLY = (
    '```html\n<!DOCTYPE html>\n<html lang="en">\n<head><title>Acme</title></head>\n'
    "<body><main><h1>Acme</h1><ul><li>One</li><li>Tw"
)


class _StubWMResponse:
    def __init__(self, output: object) -> None:
        self.output = output
        self.usage_metrics = {"prompt_tokens": 1, "completion_tokens": 2}


class _StubWM:
    def __init__(self, output: object) -> None:
        self._output = output

    async def message(self, **kwargs: object) -> _StubWMResponse:
        return _StubWMResponse(self._output)


# ---------------------------------------------------------------------------
# 1. A fenced reply is stored UNFENCED — including an UNTERMINATED fence
# ---------------------------------------------------------------------------
def test_closed_fence_is_stripped() -> None:
    assert _extract_html_block(f"```html\n{_COMPLETE_PAGE}\n```") == _COMPLETE_PAGE


def test_unterminated_fence_is_stripped() -> None:
    """The regression: no closing fence, so the pair-matching regexes both miss."""
    extracted = _extract_html_block(f"```html\n{_COMPLETE_PAGE}")
    assert not extracted.startswith("`")
    assert extracted == _COMPLETE_PAGE


def test_unterminated_generic_fence_is_stripped() -> None:
    extracted = _extract_html_block(f"```\n{_COMPLETE_PAGE}")
    assert not extracted.startswith("`")


def test_unfenced_reply_is_returned_unchanged() -> None:
    assert _extract_html_block(_COMPLETE_PAGE) == _COMPLETE_PAGE


def test_trailing_prose_after_the_document_is_dropped() -> None:
    """A complete document + commentary is COMPLETE; it must not be failed later."""
    extracted = _extract_html_block(f"{_COMPLETE_PAGE}\n\nLet me know if you want changes!")
    assert extracted == _COMPLETE_PAGE
    assert _reject_reason(extracted) is None


# ---------------------------------------------------------------------------
# 2. A structurally truncated reply is REJECTED
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<html><body><div class=", "does not end on a closed tag (truncated)"),
        ("<!DOCTYPE html><html><body><h1>hi</h1></body>", "unterminated <html> element"),
        ("I'm sorry, I cannot complete this.", "no HTML markup"),
        ("<!-- could not complete -->", "no HTML markup"),
    ],
)
def test_reject_reason_names_the_defect(html: str, expected: str) -> None:
    reason = _reject_reason(html)
    assert reason is not None and expected in reason


@pytest.mark.parametrize(
    "html",
    [
        _COMPLETE_PAGE,
        "<section class='hero'><h2>Hi</h2></section>",  # per-section fragment
        "<!-- leading comment --><div>ok</div>",
        "<div>ok</div><!-- trailing comment -->",
    ],
)
def test_complete_documents_and_fragments_pass_untouched(html: str) -> None:
    """False-positive guard: this protects every currently-working run."""
    assert _reject_reason(html) is None
    assert _require_html(html, "unit") == html


def test_require_html_raises_on_truncation() -> None:
    with pytest.raises(RuntimeError, match="truncated"):
        _require_html("<html><body><ul><li>One</li><li>Tw", "unit")


def test_scorer_reuses_the_same_checker() -> None:
    """No second implementation: the a11y auto-fix gate IS this function."""
    assert _scorer_reject_reason is _reject_reason


# ---------------------------------------------------------------------------
# 3. The production call sites fail the job rather than storing the artifact
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_full_page_rejects_the_shipped_reply() -> None:
    from src.services.html_generator import HtmlGenerator

    gen = HtmlGenerator(wm_client=cast("WMClient", _StubWM(_TRUNCATED_REPLY)))
    with pytest.raises(RuntimeError, match=r"generate_full_page.*truncated"):
        await gen.generate_full_page(url="https://example.com", html="<html><body>x</body></html>")


@pytest.mark.asyncio
async def test_refine_full_page_rejects_the_shipped_reply() -> None:
    from src.services.html_generator import HtmlGenerator

    gen = HtmlGenerator(wm_client=cast("WMClient", _StubWM(_TRUNCATED_REPLY)))
    with pytest.raises(RuntimeError, match=r"refine_full_page.*truncated"):
        await gen.refine_full_page(
            url="https://example.com",
            previous_html="<html><body><h1>old</h1></body></html>",
            feedback="make it bolder",
        )


@pytest.mark.asyncio
async def test_generate_component_rejects_a_truncated_reply() -> None:
    from src.services.html_generator import HtmlGenerator

    gen = HtmlGenerator(wm_client=cast("WMClient", _StubWM("```html\n<nav class=")))
    with pytest.raises(RuntimeError, match="generate_component"):
        await gen.generate_component(
            url="https://example.com", component_html="<nav>old</nav>", component_type="nav"
        )


@pytest.mark.asyncio
async def test_a_good_full_page_reply_is_still_returned_unfenced() -> None:
    """The false-positive guard at the call site, not just on the helper."""
    from src.services.html_generator import HtmlGenerator

    gen = HtmlGenerator(wm_client=cast("WMClient", _StubWM(f"```html\n{_COMPLETE_PAGE}\n```")))
    result = await gen.generate_full_page(
        url="https://example.com", html="<html><body>x</body></html>"
    )
    assert result.text.startswith("<!DOCTYPE html>")
    assert result.text.rstrip().endswith("</html>")
