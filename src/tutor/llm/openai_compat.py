"""OpenAI-compatible chat client used for both the local and the remote backend.

Everything that is *local-only* (`cache_prompt`, `chat_template_kwargs`) is
stripped before a request leaves for a remote backend, so a provider never sees
llama.cpp-specific fields.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from openai import AsyncOpenAI

from tutor.llm.base import Message

# llama.cpp extensions; sending them to a hosted provider is at best noise and
# at worst a rejected request.
LOCAL_ONLY_EXTRA_KEYS = frozenset({"cache_prompt", "chat_template_kwargs"})


def filter_extra_body(extra_body: Mapping[str, Any] | None, *, local_backend: bool) -> dict[str, Any]:
    """Drop backend-private keys unless the request is aimed at that backend."""
    body = dict(extra_body or {})
    if not local_backend:
        for key in LOCAL_ONLY_EXTRA_KEYS:
            body.pop(key, None)
    return body


class OpenAICompatClient:
    """Async chat client speaking ``POST /v1/chat/completions``."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        extra_body: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        timeout_s: float | None = None,
        local_backend: bool = True,
        max_retries: int = 0,
    ) -> None:
        self.model = model
        self.local_backend = local_backend
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.extra_body = filter_extra_body(extra_body, local_backend=local_backend)
        # Fail fast: a hung request must surface so the router can fall back
        # (remote) or so the user sees it immediately (local).
        self._client = AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key or "none",
            timeout=timeout_s,
            max_retries=max_retries,
        )

    async def aclose(self) -> None:
        await self._client.close()

    def build_request(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float | None = None,
        json_mode: bool = False,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Return the exact kwargs handed to ``chat.completions.create``.

        Kept public so tests can assert on the wire format without a server.
        """
        extra = dict(self.extra_body)
        if self.top_k is not None:
            # `top_k` is not part of the OpenAI spec; llama.cpp reads it from
            # the JSON body, so it travels inside extra_body.
            extra.setdefault("top_k", self.top_k)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if extra:
            kwargs["extra_body"] = extra

        effective = self.temperature if temperature is None else temperature
        if effective is not None:
            kwargs["temperature"] = effective
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float | None = None,
    ):
        kwargs = self.build_request(
            messages, max_tokens=max_tokens, temperature=temperature, stream=True
        )
        chunk_stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in chunk_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                print(
                    "[AVISO] reasoning_content recebido - thinking nao esta desligado neste backend.",
                    file=sys.stderr,
                )

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        kwargs = self.build_request(
            messages, max_tokens=max_tokens, json_mode=json_mode, stream=False
        )
        response = await self._client.chat.completions.create(**kwargs)
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""
