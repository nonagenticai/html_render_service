"""Feature icons must never render as a visible ligature word.

Regression for live web_design_v2 run 01a1588f-2686: the ``features`` block's
``icon`` slot was free text, the model filled it with Material Symbols ligature
names (``bolt``, ``shield``), and the assembled page — self-contained, no icon
font ever loaded — showed those names as literal words on every feature card.
"""

from __future__ import annotations

import re

import pytest
from bs4 import BeautifulSoup

from src.services.catalog import assemble_page
from src.services.catalog.icons import DEFAULT_ICON, ICON_NAMES, icon_svg, resolve_icon
from src.services.catalog.registry import get_block

# Values the live model emitted, plus the other shapes models produce.
_LIVE_ICON_VALUES = [
    "bolt",
    "shield",
    "rocket_launch",
    "material-symbols-outlined verified_user",
    "support_agent",
    "Local Shipping",
    "eco_outlined",
    "fa-lock",
    "⚡️",
    "totally_unknown_word",
]

_LIGATURE_WORD = re.compile(r"^[a-z][a-z0-9_ -]*$", re.IGNORECASE)


def _features_page(icons: list[str]) -> BeautifulSoup:
    items = [
        {"icon": icon, "title": f"Feature {i}", "body": "Body"} for i, icon in enumerate(icons)
    ]
    page = assemble_page(
        [{"type": "features", "content": {"headline": "Why us", "items": items}}],
        brand={"name": "Acme"},
    )
    return BeautifulSoup(page.html, "html.parser")


def test_no_feature_icon_renders_its_ligature_name_as_text() -> None:
    soup = _features_page(_LIVE_ICON_VALUES)
    cards = soup.select(".card-body")
    assert len(cards) == len(_LIVE_ICON_VALUES)
    for card, value in zip(cards, _LIVE_ICON_VALUES, strict=True):
        icon_box = card.find("div", attrs={"aria-hidden": "true"})
        assert icon_box is not None, f"icon {value!r} produced no icon element"
        visible = icon_box.get_text(strip=True)
        assert visible == "", f"icon {value!r} rendered visible text {visible!r}"
        assert icon_box.find("svg") is not None, f"icon {value!r} is not an inline SVG"


def test_page_never_depends_on_an_icon_font() -> None:
    html = str(_features_page(_LIVE_ICON_VALUES))
    assert "material-symbols" not in html.lower()
    assert "material+symbols" not in html.lower()
    assert "material-icons" not in html.lower()
    # No element anywhere on the page shows a bare ligature word from the input.
    soup = BeautifulSoup(html, "html.parser")
    for text in soup.find_all(string=True):
        word = text.strip()
        if word and _LIGATURE_WORD.match(word):
            assert word not in _LIVE_ICON_VALUES, f"ligature name {word!r} is visible"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("bolt", "bolt"),
        ("shield", "shield"),
        ("rocket_launch", "rocket"),
        ("material-symbols-outlined verified_user", "shield"),
        ("Local Shipping", "truck"),
        ("eco_outlined", "leaf"),
        ("fa-lock", "lock"),
        ("⚡️", "bolt"),
        ("totally_unknown_word", DEFAULT_ICON),
        ("\U0001f984", DEFAULT_ICON),
        ("", None),
        (None, None),
    ],
)
def test_resolve_icon(value: object, expected: str | None) -> None:
    assert resolve_icon(value) == expected


def test_icon_svg_is_decorative_and_empty_for_no_icon() -> None:
    assert str(icon_svg("")) == ""
    svg = str(icon_svg("<script>alert(1)</script>"))
    assert "<script" not in svg
    assert 'aria-hidden="true"' in svg
    assert f'data-icon="{DEFAULT_ICON}"' in svg


def test_fill_contract_offers_the_icon_vocabulary() -> None:
    features = get_block("features")
    assert features is not None
    items = features.slots["items"]
    assert items.item_slots["icon"] == "icon"
    for name in ICON_NAMES:
        assert name in items.description
