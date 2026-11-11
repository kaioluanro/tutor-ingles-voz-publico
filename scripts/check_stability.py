"""Stability soak (Phase 11): 15 minutes of turns, watching for leaks.

Replays recorded utterances through the **whole** pipeline — VAD, STT, LLM,
TTS, playback and background feedback — for ``--minutes``, sampling process
RSS and GPU VRAM after every turn. It fails (exit 1) if either keeps growing,
which is the only honest way to catch a leak that a single turn never shows.

Samples are compared as *last vs first*, not *max vs first*: one noisy frame
must not fail the run, a steady climb must.

Usage:
    python scripts/check_stability.py --minutes 15 --feedback
    python scripts/check_stability.py --minutes 0.5 --wav tests/fixtures/speech_a.wav
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console

from _common import ROOT

sys.path.insert(0, str(ROOT / "src"))

from bench_latency import (  # noqa: E402
    _load_wav,
    _queue_for_vad,
    _wait_unmuted,
)
from tutor.config import AppConfig, ConfigError, load_all  # noqa: E402
from tutor.core.session import Session  # noqa: E402

DEFAULT_WAV = ROOT / "tests" / "fixtures" / "speech_b.wav"
#: Growth tolerated before a run is called a leak (RSS in bytes, VRAM in MiB).
RSS_TOLERANCE = 64 * 1024 * 1024
VRAM_TOLERANCE = 128
PAGE_SIZE = 4096


def rss_bytes() -> int | None:
    """Current resident set size. ``None`` where /proc is unavailable."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * PAGE_SIZE
    except (OSError, IndexError, ValueError):
        return None


def vram_mib() -> int | None:
    """GPU memory in use by this machine. ``None`` without nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    values = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    try:
        return max(int(value) for value in values)
    except ValueError:
        return None


def _mib(size: int | None) -> str:
    return "?" if size is None else f"{size // (1024 * 1024)} MiB"


def _drift(first: int | None, last: int | None, tolerance: int) -> bool:
    """True when the growth is inside ``tolerance`` (or unknown)."""
    if first is None or last is None:
        return True
    return last - first <= tolerance


async def _run(
    args: argparse.Namespace,
    cfg: AppConfig,
    levels: dict,
    console: Console,
) -> int:
    level = levels[args.level]
    wavs = [Path(p) for p in args.wavs]
    deadline = time.monotonic() + args.minutes * 60.0

    rss_series: list[int] = []
    vram_series: list[int] = []
    turns = 0

    async with Session(
        cfg,
        level_name=args.level,
        level=level,
        scenario=args.scenario,
        console=console,
        listen=False,
        streaming=True,
        feedback=args.feedback,
    ) as session:
        started = time.monotonic()
        try:
            while time.monotonic() < deadline:
                for path in wavs:
                    if time.monotonic() >= deadline:
                        break
                    audio = _load_wav(path, cfg.audio.capture_sample_rate)
                    if not await _wait_unmuted(session.detector):
                        console.print("[yellow]mic mudo (half-duplex) - parando.[/yellow]")
                        return 1
                    budget = _queue_for_vad(session, audio)
                    try:
                        heard = await session.listen(timeout=budget)
                    except TimeoutError:
                        console.print("[yellow]VAD nao fechou - pulando.[/yellow]")
                        continue
                    tl = await session.turn(heard)
                    if tl is None:
                        continue
                    turns += 1
                    current_rss = rss_bytes()
                    current_vram = vram_mib()
                    if current_rss is not None:
                        rss_series.append(current_rss)
                    if current_vram is not None:
                        vram_series.append(current_vram)
                    vram_text = (
                        "?" if current_vram is None else f"{current_vram} MiB"
                    )
                    console.print(
                        f"[dim]turno {turns:4d}  "
                        f"rss {_mib(current_rss)}  "
                        f"vram {vram_text}  "
                        f"fila {session.capture.queue.qsize()}  "
                        f"latencia {tl.latency_ms:.0f} ms[/dim]"
                    )
        except KeyboardInterrupt:
            console.print("\ninterrompido pelo usuario.")

        if session.feedback is not None:
            await session.summarize()
        elapsed = time.monotonic() - started

    return _report(console, turns, elapsed, rss_series, vram_series)


def _report(
    console: Console,
    turns: int,
    elapsed: float,
    rss: list[int],
    vram: list[int],
) -> int:
    if turns < 2:
        console.print("[yellow]menos de 2 turnos - nada conclusivo.[/yellow]")
        return 1

    lines = [
        "",
        "[bold]estabilidade[/bold]",
        f"  turnos      {turns} em {elapsed / 60:.1f} min "
        f"({elapsed / turns:.1f} s/turno)",
    ]
    ok = True

    if rss:
        first, last, peak = rss[0], rss[-1], max(rss)
        lines.append(
            f"  RSS         {_mib(first)} -> {_mib(last)}  "
            f"(pico {_mib(peak)}, drift {(last - first) // (1024 * 1024)} MiB)"
        )
        ok = ok and _drift(first, last, RSS_TOLERANCE)
    else:
        lines.append("  RSS         indisponivel")

    if vram:
        first, last, peak = vram[0], vram[-1], max(vram)
        lines.append(
            f"  VRAM        {first} -> {last} MiB  (pico {peak} MiB, "
            f"drift {last - first} MiB)"
        )
        ok = ok and _drift(first, last, VRAM_TOLERANCE)
        ok = ok and last < 5632  # DoD: < 5.5 GB
    else:
        lines.append("  VRAM        indisponivel")

    console.print("\n".join(lines))
    if ok:
        console.print("[green]sem vazamento detectado.[/green]")
        return 0
    console.print("[red]crescimento acima do tolerado - ver series acima.[/red]")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sessao longa medindo RSS/VRAM por turno (Fase 11)."
    )
    parser.add_argument("--config", default=None, help="caminho do config.yaml")
    parser.add_argument("--levels", default=None, help="caminho do levels.yaml")
    parser.add_argument("--level", default="beginner", help="nivel (levels.yaml)")
    parser.add_argument(
        "--minutes", type=float, default=15.0, help="duracao da sessao em minutos"
    )
    parser.add_argument(
        "--wav",
        dest="wav",
        action="append",
        default=[],
        help="wav de uma fala (repetir o argumento); padrao speech_b.wav",
    )
    parser.add_argument(
        "--scenario", default=None, metavar="NAME", help="cenario (padrao free_chat)"
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        default=False,
        help="liga o feedback de correcao em segundo plano (Fase 8)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    args.wavs = args.wav or [str(DEFAULT_WAV)]

    try:
        cfg, levels = load_all(args.config, args.levels)
    except ConfigError as exc:
        console.print(f"[red]ERRO:[/red] {exc}")
        return 2
    if args.level not in levels:
        console.print(f"[red]ERRO:[/red] nivel '{args.level}' inexistente.")
        return 2

    try:
        return asyncio.run(_run(args, cfg, levels, console))
    except ConfigError as exc:
        console.print(f"[red]ERRO:[/red] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
