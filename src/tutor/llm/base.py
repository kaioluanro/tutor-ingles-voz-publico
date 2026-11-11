"""Backend-agnostic contract every LLM client must satisfy.

Two shapes, because the product needs both:

* :meth:`LLMClient.stream` — the interactive path. Tokens are yielded as they
  arrive so the sentence chunker can start TTS before the answer is finished.
* :meth:`LLMClient.complete` — the fire-and-forget path used by the background
  feedback worker, where a single opaque blob is enough.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

Message = dict[str, str]


@runtime_checkable
class LLMClient(Protocol):
    """Any OpenAI-compatible chat completion client."""

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield content deltas in arrival order."""
        ...

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        """Return the whole completion as a single string."""
        ...
