"""Which backend failures are worth waiting for.

The session needs to tell *"the server is down"* from *"our request is wrong"*:
only the first one is worth a reconnect, the second would loop forever.
"""

from __future__ import annotations

import openai


def is_llm_unavailable(exc: BaseException) -> bool:
    """True when the failure means **the backend is not reachable right now**.

    * :class:`openai.APIConnectionError` — connection reset / refused / timed
      out (``APITimeoutError`` is a subclass). The typical "llama-server was
      never started" case.
    * :class:`openai.APIStatusError` with **5xx** — the process answered but
      could not serve us (model unloaded, GPU out of memory).
    * :class:`ConnectionError` / :class:`TimeoutError` — anything raised by a
      future client that does not speak the openai hierarchy.

    A **4xx** is our bug (bad model name, malformed body): retrying with the
    same payload cannot fix it, so it stays unhandled and surfaces.
    """
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return isinstance(exc, (ConnectionError, TimeoutError))


def describe(exc: BaseException) -> str:
    """Short Portuguese reason, fit for the status panel."""
    if isinstance(exc, openai.APIStatusError):
        return f"o servidor respondeu HTTP {exc.status_code}"
    if isinstance(exc, openai.APIConnectionError):
        return "sem conexao"
    if isinstance(exc, TimeoutError):
        return "tempo esgotado"
    return type(exc).__name__
