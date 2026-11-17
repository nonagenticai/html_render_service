"""Per-job generation state: the budget, the section jobs, tokens and degradations.

WHY THIS IS NOT ON ``HtmlGenerator``. One ``HtmlGenerator`` is built per process
(``asgi.py``) and shared by every job the pod runs: the queue's worker loops, a
batch's pages (``batch_max_concurrency`` at once) and a job's style variants. State
kept on the instance is therefore shared by concurrent renders — page A's
``generate_sections`` would overwrite page B's section prompts, and B's rendered repair
would regenerate from A's prompt or fail on a key A never had. Everything a render
needs to remember lives HERE, and is passed explicitly from the call that created it.

THE BUDGET is job-scoped (a job's variants share one), because the caller waiting on
the job — w2v_orchestrator's ``website_redesign`` stage — waits on the whole job:

* a wall-clock DEADLINE; every LLM call runs under the time remaining, so an in-flight
  call cannot carry the job past it;
* a cap on EXTRA calls (retries and repair regenerations). A first draft per section is
  never refused by the cap — a page needs every section — only by the deadline.

When the budget is spent the render stops retrying/repairing and finishes with what it
has; every such decision is recorded in :attr:`SectionRun.degradations`, which the job
persists as quality warnings instead of failing the deliverable.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from src.llm_credentials import LLMConfigLike

__all__ = ["BudgetExhausted", "GenerationBudget", "SectionJob", "SectionRun", "budget_window"]

# Never START an LLM call with less time than this left: it could not finish.
MIN_CALL_SECONDS = 30.0


class BudgetExhausted(Exception):  # noqa: N818 — a condition, not an error
    """The job's generation budget does not allow this call."""


@dataclass
class GenerationBudget:
    """Wall-clock deadline + extra-call cap for ONE job (see module docstring)."""

    deadline: float | None = None  # ``clock()`` value; None = no deadline
    max_extra_calls: int | None = None  # None = uncapped
    clock: Callable[[], float] = time.monotonic
    calls: int = 0
    extra_calls: int = 0

    @classmethod
    def from_settings(cls, *, clock: Callable[[], float] = time.monotonic) -> GenerationBudget:
        from src.config import get_settings

        cfg = get_settings().redesign
        return cls(
            deadline=clock() + cfg.generation_deadline_seconds,
            max_extra_calls=cfg.max_extra_llm_calls,
            clock=clock,
        )

    def remaining(self) -> float | None:
        return None if self.deadline is None else self.deadline - self.clock()

    def why_not(self, *, extra: bool) -> str | None:
        """Why a call may NOT start now, else ``None``."""
        left = self.remaining()
        if left is not None and left < MIN_CALL_SECONDS:
            return f"generation deadline reached ({max(left, 0):.0f}s left)"
        if extra and self.max_extra_calls is not None and self.extra_calls >= self.max_extra_calls:
            return f"retry/repair budget spent ({self.extra_calls}/{self.max_extra_calls} calls)"
        return None

    def spend(self, *, extra: bool) -> float | None:
        """Account one call; returns its timeout (seconds) or raises :class:`BudgetExhausted`."""
        reason = self.why_not(extra=extra)
        if reason is not None:
            raise BudgetExhausted(reason)
        self.calls += 1
        if extra:
            self.extra_calls += 1
        return self.remaining()


@asynccontextmanager
async def budget_window(budget: GenerationBudget, *, extra: bool) -> AsyncIterator[None]:
    """Account ONE LLM call and bound it by the time the job's budget has left.

    The single enforcement seam for every dispatch path that consults a
    :class:`GenerationBudget` — see call sites for the full list (generation,
    accessibility scoring, refinement).

    Raises :class:`BudgetExhausted` when the call may not start, or when the job
    deadline expires while it is in flight (the call is cancelled — a reply that
    lands after the caller stopped waiting is worth nothing).
    """
    timeout = budget.spend(extra=extra)
    deadline = asyncio.timeout(timeout)
    try:
        async with deadline:
            yield
    except TimeoutError as exc:
        if deadline.expired():
            raise BudgetExhausted("generation deadline reached during an LLM call") from exc
        raise


@dataclass(frozen=True)
class SectionJob:
    """What it takes to regenerate one section after the page is rendered."""

    prompt: str
    where: str
    url: str
    manifest: list[dict]
    llm_config: LLMConfigLike | None
    greenfield: bool
    source_image_counts: dict[str, int]


@dataclass
class SectionRun:
    """The state of ONE page render (``generate_sections`` -> ``repair_rendered_page``)."""

    budget: GenerationBudget = field(default_factory=GenerationBudget)
    jobs: dict[str, SectionJob] = field(default_factory=dict)
    manifest: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    degradations: list[str] = field(default_factory=list)

    def add_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def degrade(self, message: str) -> None:
        if message not in self.degradations:
            self.degradations.append(message)
