"""Component catalog + deterministic assembler for assembly-mode web design."""

from __future__ import annotations

from src.services.catalog.assembler import AssembledPage, assemble_page
from src.services.catalog.registry import (
    BlockManifest,
    all_blocks,
    block_keys,
    catalog_brief,
    get_block,
)
from src.services.catalog.theme import build_theme_css

__all__ = [
    "AssembledPage",
    "BlockManifest",
    "all_blocks",
    "assemble_page",
    "block_keys",
    "build_theme_css",
    "catalog_brief",
    "get_block",
]
