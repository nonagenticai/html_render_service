"""Display-vs-body font distinctness, the typographic twin of ``ensure_distinct_accent``.

A redesign's ``:root`` carries two font tokens, ``--font-display`` (headings) and
``--font-body`` (running text). They are resolved independently — from the context
resolver, the brand dict, or ``extract_fonts`` over the source markup — and nothing
checked that the two answers differ.

The cliffordtraining.nl artifact is what that costs: ``--font-body: "Fraunces"`` — the
serif DISPLAY face — while Inter was loaded in ``<head>`` and never used. Every
paragraph on the page rendered in the headline face, and the pairing the source site
actually had was silently discarded. The colour side of the same contract has a real
validator (:func:`accessibility_theme.ensure_distinct_accent`, which keeps a supplied
accent only when it clears a contrast bar and otherwise derives one); this is that
shape for fonts.

Deliberately name-agnostic: no font list, no serif/sans classification table. The
judgement is IDENTITY against the display face, and the REPLACEMENT — like
``ensure_distinct_accent``'s — is taken from what the SOURCE SITE itself loads
(``google_font_links``) before anything is derived. On the owner's artifact those links
carry Inter, the face that was loaded and never used, so the pairing is repaired rather
than flattened.

Two traps this deliberately avoids:

* **identity alone is not the bar.** With no display face resolved, ``body_font`` equal
  to the FIRST loaded family is the same defect wearing a different hat — that is the
  role order ``extract_fonts`` itself assumes for ``family=`` URLs (display, then body),
  and the artifact's serif body text survives an identity-only check. So when the
  display face is unknown, the first loaded family stands in for it — but only when the
  site loads MORE THAN ONE family, because that is the only evidence a second face was
  ever intended.
* **a single-family pairing is legitimate.** Inter-700 headings over Inter-400 body text
  is indistinguishable from the bug under an identity test, so a site that loads exactly
  one family is left alone. Degrading it to ``system-ui`` would destroy a real design
  decision to "fix" a page that has none of the defect.

Only when the site offers no alternative family is ``None`` returned, which drops the
caller to its generic body stack.
"""

from __future__ import annotations

import re

__all__ = ["ensure_distinct_body_font", "normalize_font_family"]

# ``"Fraunces", serif`` / ``'Fraunces'`` / ``  Fraunces  `` -> ``fraunces``.
_QUOTES = "\"'"
_WS_RE = re.compile(r"\s+")


def normalize_font_family(family: str | None) -> str | None:
    """The comparable identity of a font family value: first family, unquoted, folded.

    A resolved value may arrive as a bare family (``Fraunces``), a quoted one, or an
    entire stack (``"Fraunces", Georgia, serif``). Only the FIRST family is the face the
    page actually asks for, so that is what is compared.
    """
    if not family:
        return None
    first = family.split(",", 1)[0].strip().strip(_QUOTES).strip()
    if not first:
        return None
    return _WS_RE.sub(" ", first).casefold()


def ensure_distinct_body_font(
    display_font: str | None,
    body_font: str | None,
    google_links: list[str] | None = None,
) -> str | None:
    """A body face that is a different family from the display face, or ``None``.

    Mirrors :func:`accessibility_theme.ensure_distinct_accent`: a supplied value that
    clears the bar is kept verbatim; a value that fails it is replaced from the SOURCE
    SITE's own evidence (the families its Google-Fonts links load) before anything is
    given up. ``None`` means "no brand body face" — the caller falls back to its generic
    body stack, which happens only when the site offers no second family.

    When ``display_font`` is unknown, the first loaded family stands in for it, and only
    if the site loads more than one family — see the module docstring for why a
    single-family site is left alone.
    """
    if not body_font:
        return body_font
    loaded = _loaded_families(google_links)
    display = normalize_font_family(display_font)
    if display is None:
        if len(loaded) < 2:
            return body_font
        display = normalize_font_family(loaded[0])
    body = normalize_font_family(body_font)
    if display is None or body is None or display != body:
        return body_font
    for family in loaded:
        if normalize_font_family(family) not in (display, body):
            return family
    return None


def _loaded_families(google_links: list[str] | None) -> list[str]:
    """The families the source page's Google-Fonts links load: in order, deduped."""
    if not google_links:
        return []
    from src.services.context_resolver import _families_from_google_links

    out: list[str] = []
    seen: set[str] = set()
    for family in _families_from_google_links(google_links):
        key = normalize_font_family(family)
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(family)
    return out
