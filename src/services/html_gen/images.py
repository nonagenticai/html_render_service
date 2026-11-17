"""Image-manifest formatting and deterministic image-URL absolutization."""

from __future__ import annotations

import re
from urllib.parse import urljoin

__all__ = [
    "_abs_srcset",
    "_abs_url",
    "_absolutize_images",
    "_format_image_manifest",
]


def _format_image_manifest(manifest: list[dict] | None) -> str:
    """Render the absolutized image list the model MUST reuse verbatim.

    ``None`` and ``[]`` are DIFFERENT states and are rendered differently. They
    used to collapse into the same empty string, which is what made the gap
    silent:

    * ``None`` — no manifest applies (greenfield: there is no source page). The
      prompt says nothing about reusing images, which is correct.
    * ``[]``  — a real page was parsed and yielded no ``<img src>``. Very common:
      CSS background images, ``data-src`` lazy-loading, ``<picture>`` with only
      ``srcset``. Returning ``""`` here deleted the "never blank out a real
      image" instruction from the prompt, while the Requirements block below it
      still told the model to "reuse the exact absolute image URLs listed above".
      Pointed at a list that was not there, the model filled the hole with
      ``https://placehold.co/`` boxes — on a redesign of a site whose images
      exist and are simply not in an ``<img src>``.

    So an empty extraction now STATES itself. Absence of a manifest is not
    permission to invent images.
    """
    if manifest is None:
        return ""
    if not manifest:
        return (
            "## Images — none could be extracted from the source page\n"
            "No `<img src>` URLs were extracted. This does NOT mean the page has no "
            "images — it usually means they are CSS backgrounds, lazy-loaded "
            "(`data-src`), or `srcset`-only. Treat this as MISSING INFORMATION, not "
            "as an empty page:\n"
            "- Do NOT invent `https://placehold.co/` boxes to fill the gap.\n"
            "- Prefer CSS-only visuals (gradients, shapes, type) and layouts that do "
            "not depend on a photo you do not have.\n"
            "- Keep any image URL that appears in the original markup excerpt above, "
            "even though it is not in this list."
        )
    lines = ["## Images — reuse these EXACT absolute URLs (do not rewrite to relative paths)"]
    for img in manifest:
        alt = f' (alt: "{img["alt"]}")' if img.get("alt") else ""
        lines.append(f"- [{img.get('role', 'content')}] {img['url']}{alt}")
    lines.append(
        "Only use a `https://placehold.co/` placeholder (tinted to the brand) for an "
        "image that is genuinely missing — never blank out a real image above."
    )
    return "\n".join(lines)


def _abs_url(raw: str, by_filename: dict[str, str], base_url: str) -> str:
    """Absolutize one URL: keep absolute/data, else map by filename or urljoin."""
    raw = raw.strip()
    if not raw or raw.startswith(("data:", "http://", "https://", "//")):
        return raw
    return by_filename.get(raw.rsplit("/", 1)[-1].split("?")[0]) or urljoin(base_url, raw)


def _abs_srcset(value: str, by_filename: dict[str, str], base_url: str) -> str:
    """Absolutize every candidate URL in a srcset attribute."""
    out: list[str] = []
    for cand in value.split(","):
        bits = cand.strip().split(" ", 1)
        bits[0] = _abs_url(bits[0], by_filename, base_url)
        out.append(" ".join(bits))
    return ", ".join(out)


def _absolutize_images(
    generated_html: str, manifest: list[dict] | None, base_url: str | None
) -> str:
    """Deterministically rewrite relative image URLs in the model output to absolute.

    LLM-independent insurance: even if the model echoes a relative ``/assets/x.png``
    (which 404s when the standalone redesign is screenshotted off-origin), rewrite it
    against ``base_url``. Also maps a generated src to the original by filename when the
    model paraphrased the path. Leaves ``data:`` and already-absolute URLs untouched.
    """
    if not generated_html or not base_url:
        return generated_html
    # Cheap pre-check: only parse/reserialize when a relative-looking src exists.
    if not re.search(r'src(?:set)?\s*=\s*["\'](?!https?:|data:|//)', generated_html):
        return generated_html
    from bs4 import BeautifulSoup

    by_filename: dict[str, str] = {}
    for img in manifest or []:
        url = img.get("url", "")
        if url and not url.startswith("data:"):
            by_filename.setdefault(url.rsplit("/", 1)[-1].split("?")[0], url)

    soup = BeautifulSoup(generated_html, "html.parser")
    for img in soup.find_all("img"):
        src = img.get("src")
        if isinstance(src, str) and src:
            img["src"] = _abs_url(src, by_filename, base_url)
        srcset = img.get("srcset")
        if isinstance(srcset, str) and srcset:
            img["srcset"] = _abs_srcset(srcset, by_filename, base_url)
    for source in soup.find_all("source"):
        srcset = source.get("srcset")
        if isinstance(srcset, str) and srcset:
            source["srcset"] = _abs_srcset(srcset, by_filename, base_url)
    return str(soup)
