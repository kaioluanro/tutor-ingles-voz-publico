"""Liveness probe for an OpenAI-compatible server, used to reconnect.

llama-server exposes ``GET /health`` on its **root** (the ``/v1`` prefix is the
API, not the server): ``{"status":"ok"}`` with HTTP 200, 503 while the model is
still loading. It stays 200 while a request is being generated — verified
against the running server — so polling it is a safe "can I speak now?".

Remote providers are out of scope: they have no equivalent endpoint, and the
feedback worker already falls back on its own.
"""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request

DEFAULT_TIMEOUT_S = 20.0
DEFAULT_POLL_S = 1.0
PROBE_TIMEOUT_S = 2.0


def health_url(base_url: str) -> str:
    """``http://host:port/v1`` -> ``http://host:port/health``.

    Trailing slashes and an ``/v1`` suffix are stripped; any other path is left
    alone (a hosted provider's ``/api/v1`` never has ``/health`` anyway).
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/health"


def probe(url: str, *, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """GET ``url`` and report whether it answered 200. Blocking."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - config url
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False


async def wait_for_server(
    base_url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
) -> bool:
    """Poll ``base_url``'s liveness until it answers 200, or ``timeout`` runs out.

    ``True`` means "go ahead and retry the turn"; ``False`` means give up and
    tell the learner. The probe runs in a worker thread so a hung socket never
    freezes the event loop.
    """
    url = health_url(base_url)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if await loop.run_in_executor(None, probe, url):
            return True
        if loop.time() + poll_s >= deadline:
            return False
        await asyncio.sleep(poll_s)
