"""HTML text utilities: truncation, code-block extraction, fragment balancing."""

from __future__ import annotations

import re

__all__ = [
    "CEILING_RATIO",
    "_balance_section_fragment",
    "_extract_html_block",
    "_reject_reason",
    "_require_html",
    "_strip_document_wrapper",
    "_token_budget_note",
    "_truncate_html",
    "degenerate_repetition",
    "section_reply_defect",
]


def _truncate_html(html: str, max_chars: int = 15000) -> str:
    """Truncate HTML to a reasonable size for the LLM prompt."""
    if len(html) <= max_chars:
        return html
    return html[:max_chars] + "\n<!-- ... truncated ... -->"


# An opening markdown fence with an optional language tag, at the very start of the
# reply: ```` ```html ```` / ```` ``` ````.
_OPEN_FENCE_RE = re.compile(r"\A\s*```[A-Za-z0-9_+-]*[ \t]*\r?\n")
# A dangling closing fence at the very end (possibly with trailing whitespace).
_CLOSE_FENCE_RE = re.compile(r"\r?\n\s*```\s*\Z")


def _extract_html_block(response: str) -> str:
    """Extract HTML code block from LLM response.

    Both the CLOSED and the UNCLOSED fence must be handled. The pair-matching
    regexes below need a *closing* ```` ``` ````; when the reply is cut off at the
    output-token ceiling there is none, so they both miss and the raw text —
    ```` ```html ```` and all — used to be returned verbatim and stored as
    ``redesign.html``. That is exactly the artifact delivered on 2026-09-10: a page
    whose first visible characters were the literal fence, ending mid-``<ul>``.
    So: after the pair-matching attempts, strip a leading fence (and a dangling
    trailing one) unconditionally. Truncation itself is caught by
    :func:`_require_html`; this function only removes the wrapper.
    """
    pattern = r"```html\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return _trim_after_document(match.group(1).strip())
    # Try generic code block
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return _trim_after_document(match.group(1).strip())
    # Unterminated fence (truncated reply, or a reply that opened a fence and
    # never closed it): drop the wrapper we can see.
    stripped = _OPEN_FENCE_RE.sub("", response)
    stripped = _CLOSE_FENCE_RE.sub("", stripped)
    return _trim_after_document(stripped.strip())


def _trim_after_document(text: str) -> str:
    """Drop any commentary the model appended AFTER ``</html>``.

    A complete document followed by "Let me know if…" is a *complete* document, and
    must not be failed as truncated by :func:`_reject_reason` (which requires the
    text to end on a closed tag). Documents without ``</html>`` are untouched.
    """
    idx = text.lower().rfind("</html>")
    if idx == -1:
        return text
    return text[: idx + len("</html>")]


# An opening element tag: ``<div>``, ``<section class=…>``, ``<br/>``. Deliberately
# does NOT match ``<!-- comment -->`` or ``<!DOCTYPE html>``.
_ELEMENT_TAG_RE = re.compile(r"<[A-Za-z][A-Za-z0-9-]*[\s/>]")


def _reject_reason(html: str) -> str | None:
    """Return why ``html`` must not be stored/shipped as a page, else ``None``.

    Cheap structural checks only — enough to catch the failures that actually
    shipped: a document truncated mid-tag (``…<div class=``), a document whose
    ``<html>`` is never closed, or a reply with no element in it at all. Deliberately
    fragment-tolerant: per-section generations return bare ``<section>…</section>``
    snippets, which are complete even though they are not documents.

    Single source of truth — ``accessibility_scorer`` re-exports this to gate the
    auto-fixer's output, and ``_require_html`` applies it to every generated page.
    """
    text = html.strip()
    if not _ELEMENT_TAG_RE.search(text):
        return "no HTML markup (no element tag)"
    if not text.endswith(">"):
        return "does not end on a closed tag (truncated)"
    if "<html" in text.lower() and "</html>" not in text.lower():
        return "unterminated <html> element (truncated)"
    return None


# A completion within this fraction of the requested cap is at the ceiling, not a
# model that chose to stop. vLLM stops exactly AT the cap, but the cap that applies
# may be lower than the one requested (worker_management clamps a self-hosted
# provider's budget to a wall-clock time budget), so this is a ratio, not equality.
CEILING_RATIO = 0.98
# A structurally truncated reply that spent at least this share of the budget ran out
# of it (see section_reply_defect).
NEAR_CEILING_RATIO = 0.85


def _token_budget_note(completion_tokens: int | None, max_tokens: int | None) -> str:
    """Explain a truncation in terms of the output-token budget, or return ``""``.

    "does not end on a closed tag (truncated)" cannot, on its own, distinguish
    *hit the ceiling* from *the model stopped early* — and those have opposite fixes.
    The generator already receives ``usage_metrics.completion_tokens``; comparing it
    against the cap the call requested makes the next failure diagnosable in one read.

    ``finish_reason`` would settle it outright, but worker_management does not put it
    on the wire, so the token comparison is the strongest signal available here.
    """
    if not completion_tokens or not max_tokens or completion_tokens <= 0 or max_tokens <= 0:
        return ""
    if completion_tokens >= max_tokens * CEILING_RATIO:
        return (
            f" — truncated at the {completion_tokens}-token ceiling "
            f"(completion_tokens={completion_tokens} of requested max_tokens={max_tokens}); "
            "the reply did not fit the output budget, so generate less per call"
        )
    return (
        f" — stopped BELOW the requested budget (completion_tokens={completion_tokens} of "
        f"max_tokens={max_tokens}); either the model stopped early, or the provider clamped "
        "the budget below the request (a self-hosted provider's time budget does this)"
    )


# A unit of 2-40 chars repeated back-to-back >= 10 more times: the candidate shape of the
# sampling loop a model falls into (``--ct-reverse-reverse-reverse-…``, run
# 01a0a045-602a, 2026-09-14: ``-reverse`` x 2400, 19200 chars, no markup at all).
_REPEAT_RE = re.compile(r"(.{2,40}?)\1{10,}", re.DOTALL)
# A LOOP is long. Legitimate repetition in a design is a handful of identical ELEMENTS
# (12 carousel dots, 16 empty grid cells, 12 identical table rows) — each unit carries
# its own tags, and the run is a few hundred characters. A degenerate loop repeats a
# token INSIDE an identifier/value/sentence — no tag in the unit — for as long as the
# budget lasts.
_MIN_LOOP_RUN_CHARS = 400
_MIN_LOOP_REPEATS = 20
# Whole ELEMENTS repeating is only a loop at a size no design has.
_MIN_MARKUP_LOOP_REPEATS = 200
_OPEN_TAG_NAME_RE = re.compile(r"<([A-Za-z][A-Za-z0-9-]*)[\s/>]")


def degenerate_repetition(text: str) -> str | None:
    """The first degenerate repetition LOOP in ``text`` (truncated for display), or ``None``.

    A loop is a long exact repetition of a short character n-gram that carries no
    structure: the unit has a letter and at least two distinct characters, contains no
    tag (``<``/``>``), repeats >= 20 times and spans >= 400 chars. Repeated ELEMENTS
    (dots, grid cells, table rows, stars) are legitimate and never count — unless an
    element repeats >= 200 times, which no page design does. A CSS divider comment
    (``/* ====== */``) or ``0 0 0 0`` path data has no letter / one distinct character.
    """
    for m in _REPEAT_RE.finditer(text):
        unit, run = m.group(1), m.group(0)
        repeats = len(run) // len(unit)
        if len(set(unit)) < 2 or not any(ch.isalpha() for ch in unit):
            continue
        if "<" in unit or ">" in unit:
            if repeats < _MIN_MARKUP_LOOP_REPEATS:
                continue
        elif len(run) < _MIN_LOOP_RUN_CHARS or repeats < _MIN_LOOP_REPEATS:
            continue
        return f"{unit!r} repeated {repeats} times"
    return None


def section_reply_defect(
    text: str, *, completion_tokens: int | None = None, max_tokens: int | None = None
) -> str | None:
    """Why ONE section reply is unusable (truncated / degenerate / unbalanced), else ``None``.

    Per-section generation can afford to retry a bad reply, so this is stricter than
    :func:`_reject_reason` (which must stay fragment-tolerant for whole pages):

    * the reply is degenerate — a sampling loop (even one that later closes its tags);
    * it is structurally truncated (:func:`_reject_reason`);
    * it spent the output ceiling (``usage_metrics.completion_tokens`` within 2% of the
      requested ``max_tokens``) — worker_management does not return ``finish_reason``,
      so this is the strongest "cut off" signal on the wire;
    * a ``<style>`` or the section's outermost element is opened more often than
      closed — a reply cut right after a closing tag ends on ``>`` and slipped through.
    """
    loop = degenerate_repetition(text)
    if loop:
        return f"degenerate output: the reply fell into a repetition loop ({loop})"
    reason = _reject_reason(text)
    if reason is not None:
        if (
            completion_tokens
            and max_tokens
            and max_tokens * NEAR_CEILING_RATIO <= completion_tokens < max_tokens * CEILING_RATIO
        ):
            # The time-budget clamp lands a little under the token cap (prefill time
            # is spent from the same wall clock): 11225 of 12069 is the ceiling, not a
            # model that chose to stop mid-declaration.
            return (
                f"{reason} — ran into the provider's output ceiling (completion_tokens="
                f"{completion_tokens} of max_tokens={max_tokens}); the reply was too long"
            )
        return reason + _token_budget_note(completion_tokens, max_tokens)
    if completion_tokens and max_tokens and completion_tokens >= max_tokens * CEILING_RATIO:
        return "reply reached the output-token ceiling (truncated)" + _token_budget_note(
            completion_tokens, max_tokens
        )
    lowered = text.lower()
    if lowered.count("<style") > lowered.count("</style"):
        return "a <style> element is never closed (truncated)"
    first = next(
        (
            m.group(1).lower()
            for m in _OPEN_TAG_NAME_RE.finditer(text)
            if m.group(1).lower() not in ("style", "script", "link", "meta")
        ),
        None,
    )
    if first in ("section", "header", "footer", "nav", "main", "aside", "article", "div"):
        opens = len(re.findall(rf"<{first}[\s>]", lowered))
        closes = lowered.count(f"</{first}>") + lowered.count(f"</{first} ")
        if opens > closes:
            return f"the outermost <{first}> is never closed (truncated)"
    return None


def _require_html(
    text: str,
    where: str,
    *,
    completion_tokens: int | None = None,
    max_tokens: int | None = None,
) -> str:
    """Return ``text`` if it contains markup, else raise.

    Guards against an aborted/empty/garbage LLM response being stored as the
    redesigned page. A genuine timeout/abort surfaces upstream as an httpx error,
    but a 200 with no extractable HTML would otherwise pass through silently.

    "Markup" means at least one opening *element* tag. A bare ``"<" in text``
    check let ``<!-- could not complete -->`` (and a lone ``<!DOCTYPE html>``)
    through: freeform mode returns before ``verify_html``, so such a reply was
    stored and rendered as a blank white PNG.

    It ALSO rejects a structurally truncated document (see :func:`_reject_reason`).
    Raising is the deliberate failure mode: every caller runs inside a job phase
    whose ``except Exception`` seam (``design_builder/service.py``) records the
    reason and marks the row failed. Logging instead would re-ship the broken
    artifact under a ``completed`` status, which is the defect being fixed.
    """
    reason = _reject_reason(text)
    if reason is not None:
        note = _token_budget_note(completion_tokens, max_tokens) if "truncated" in reason else ""
        raise RuntimeError(
            f"{where}: LLM reply is not a storable page ({reason}){note}: "
            f"starts {text[:80]!r} ends {text[-80:]!r}"
        )
    return text


def _strip_document_wrapper(snippet: str) -> str:
    """Strip a full-document wrapper from a section snippet.

    Section generations occasionally return a whole ``<!DOCTYPE><html>…`` document
    instead of a bare section. When that happens, keep only the inner-body markup so
    composing several sections doesn't nest documents. Snippets without a ``<body>``
    are returned unchanged (the common, correct case).
    """
    match = re.search(r"<body[^>]*>(.*?)</body>", snippet, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return snippet


def _balance_section_fragment(snippet: str) -> str:
    """Strip any document wrapper AND close unbalanced tags in a section fragment.

    Per-section generations occasionally emit an unclosed ``<section>``/``<div>``.
    Composed flat, that dangling open tag consumes the *next* section wrapper's
    ``</section>`` and swallows every following sibling — they render nested inside
    one greedy band at fractional width and never qualify as standalone capture
    bands (live E2E run 019ea7e2: only ``main`` survived because contact/footer/nav
    got nested into it at ~709px). Round-tripping the inner markup through the
    lenient HTML parser closes the dangling tags so each ``<section data-section>``
    wrapper stays a self-contained, full-width direct child of ``<body>``.
    """
    from bs4 import BeautifulSoup

    return str(BeautifulSoup(_strip_document_wrapper(snippet), "html.parser"))
