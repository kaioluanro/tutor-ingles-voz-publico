"""LLM streaming check: time to first token + tokens/second.

Uses the real project client (``tutor.llm``), so a green run here proves the
Phase 2 client talks to the local backend.

Usage: python scripts/check_llm.py [--prompt "..."] [--max-tokens 80]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from _common import ROOT

sys.path.insert(0, str(ROOT / "src"))

from tutor.config import ConfigError, load_config  # noqa: E402
from tutor.llm.router import get_conversation_llm  # noqa: E402


async def _run(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 2

    if args.url:
        cfg.llm_local.base_url = args.url
    client = get_conversation_llm(cfg)

    messages = [{"role": "user", "content": args.prompt}]
    print(f"cliente: {type(client).__name__}  backend local  model={client.model}")
    print(f"base_url: {cfg.llm_local.base_url}")
    print(f"extra_body: {client.extra_body}\n")

    t0 = time.perf_counter()
    first_token_at: float | None = None
    tokens = 0
    pieces: list[str] = []

    try:
        async for content in client.stream(messages, max_tokens=args.max_tokens):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            tokens += 1
            pieces.append(content)
            print(content, end="", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERRO] servidor LLM indisponivel em {cfg.llm_local.base_url}: {exc}",
              file=sys.stderr)
        print("Suba o servidor: veja README.md secao 3.", file=sys.stderr)
        return 1
    finally:
        await client.aclose()

    total = time.perf_counter() - t0
    print("\n")
    if first_token_at is None:
        print("[ERRO] nenhum token de conteudo recebido.", file=sys.stderr)
        return 1

    ttft = first_token_at - t0
    gen = total - ttft
    print(f"tempo ate o 1o token (TTFT): {ttft * 1000:.0f} ms")
    print(f"tokens de conteudo: {tokens}")
    if gen > 0:
        print(f"tokens/s apos o 1o: {tokens / gen:.1f}")
    print(f"total: {total * 1000:.0f} ms")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica streaming do LLM local.")
    parser.add_argument("--prompt", default="Say hello in exactly five words.")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--config", default=None, help="caminho do config.yaml")
    parser.add_argument("--url", default=None, help="sobrescreve llm_local.base_url")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
