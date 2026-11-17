"""Image-manifest absolutization + font extraction + deterministic src rewrite.

Regression cover for the blank-image bug: a relative ``<img src="/assets/cover.png">``
must reach the model AND the generated output as an ABSOLUTE URL, or it 404s when the
standalone redesign is screenshotted off-origin.
"""

from __future__ import annotations

from src.services.context_resolver import extract_fonts, extract_image_manifest
from src.services.html_generator import _absolutize_images


def test_manifest_absolutizes_relative_src() -> None:
    html = '<html><body><img src="/assets/book/cover.png" alt="Cover"></body></html>'
    manifest = extract_image_manifest(html, "https://nonagentic.ai/")
    urls = {m["url"] for m in manifest}
    assert "https://nonagentic.ai/assets/book/cover.png" in urls
    cover = next(m for m in manifest if m["url"].endswith("cover.png"))
    assert cover["alt"] == "Cover"


def test_manifest_keeps_absolute_and_data_uris() -> None:
    html = (
        '<img src="https://cdn.x.com/a.png">'
        '<img src="data:image/png;base64,AAAA">'
        '<meta property="og:image" content="/og.png">'
    )
    urls = {m["url"] for m in extract_image_manifest(html, "https://x.com/page")}
    assert "https://cdn.x.com/a.png" in urls
    assert "data:image/png;base64,AAAA" in urls
    assert "https://x.com/og.png" in urls  # og:image absolutized


def test_manifest_logo_role_from_header() -> None:
    html = '<header><img src="/logo.svg" alt="Brand"></header>'
    manifest = extract_image_manifest(html, "https://x.com/")
    assert manifest[0]["role"] == "logo"


def test_manifest_picks_first_srcset_when_no_src() -> None:
    html = '<img srcset="/small.png 480w, /big.png 1080w">'
    urls = {m["url"] for m in extract_image_manifest(html, "https://x.com/")}
    assert "https://x.com/small.png" in urls


def test_extract_fonts_from_google_link_and_css_vars() -> None:
    html = (
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces&family=IBM+Plex+Sans">'
        "<style>:root{--font-display:'Fraunces',serif;--font-sans:'IBM Plex Sans',sans-serif}</style>"
    )
    fonts = extract_fonts(html)
    assert any("fonts.googleapis.com" in h for h in fonts["google_links"])
    assert fonts["display"] == "Fraunces"
    assert fonts["body"] == "IBM Plex Sans"


def test_extract_fonts_fallback_to_google_family_order() -> None:
    html = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:wght@400&family=Inter">'
    fonts = extract_fonts(html)
    assert fonts["display"] == "Fraunces"
    assert fonts["body"] == "Inter"


def test_absolutize_images_rewrites_relative_output() -> None:
    generated = '<html><body><img src="/assets/book/cover.png" alt="c"></body></html>'
    out = _absolutize_images(generated, manifest=None, base_url="https://nonagentic.ai/")
    assert 'src="https://nonagentic.ai/assets/book/cover.png"' in out


def test_absolutize_images_prefers_manifest_by_filename() -> None:
    generated = '<img src="cover.png">'
    manifest = [
        {"role": "content", "url": "https://nonagentic.ai/assets/book/cover.png", "alt": None}
    ]
    out = _absolutize_images(generated, manifest, base_url="https://nonagentic.ai/page/")
    assert "https://nonagentic.ai/assets/book/cover.png" in out


def test_absolutize_images_leaves_absolute_untouched() -> None:
    generated = '<img src="https://cdn.x.com/a.png"><img src="data:image/png;base64,AAAA">'
    out = _absolutize_images(generated, manifest=None, base_url="https://x.com/")
    # Cheap pre-check returns input unchanged when no relative src exists.
    assert out == generated
