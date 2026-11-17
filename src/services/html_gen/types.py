"""LLM result container + cost estimation for the HTML generator."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["LLMResult", "estimate_cost"]


@dataclass
class LLMResult:
    """Result from an LLM call with token usage tracking."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Quality compromises the render made to deliver within its budget (a dropped
    # section, an unrepaired contrast failure). Empty = none.
    quality_warnings: list[str] = field(default_factory=list)


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "") -> float:
    """Estimate cost in USD based on token usage."""
    # Approximate pricing for Claude Sonnet
    input_cost_per_1k = 0.003
    output_cost_per_1k = 0.015
    return (input_tokens / 1000 * input_cost_per_1k) + (output_tokens / 1000 * output_cost_per_1k)
