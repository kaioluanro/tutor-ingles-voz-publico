"""Latency benchmark: N turns through the pipeline, then the §9 table.

Two modes:

* ``live`` (default) — a real conversation through the microphone.
* ``file`` — recorded utterances pushed into the **capture queue**, so they go
  through the real Silero VAD (pre-roll, ``min_utterance_ms`` and the
  ``end_silence_ms`` wait) and then through STT, LLM, TTS and playback. Only
  the PortAudio input callback is replaced, which makes the number reproducible.

Every accepted utterance can be written to ``--save-wavs`` so the same speech
can be replayed later.

``--pipeline`` picks the turn implementation: ``sequential`` (Phase 5 baseline)
or ``streaming`` (Phase 6, three concurrent tasks). ``--no-greet`` skips the
AI's opening line (turn 0).

Usage:
    python scripts/bench_latency.py --turns 5 --level beginner
    python scripts/bench_latency.py --mode file --wav /tmp/bench/turn_1.wav \
        --wav /tmp/bench/turn_2.wav
    python scripts/bench_latency.py --mode file --pipeline streaming --turns 10 ...
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console

from _common import ROOT

sys.path.insert(0, str(ROOT / "src"))

from tutor.config import AppConfig, ConfigError, LevelConfig, load_all  # noqa: E402
from tutor.core.session import Session  # noqa: E402
from tutor.timeline import TurnTimeline, render  # noqa: E402

# The half-duplex mute is armed by Playback and released a fixed time after the
# tutor stops talking; audio queued before that would be discarded silently.
UNMUTE_TIMEOUT_S = 3.0


def _load_wav(path: Path, sample_rate: int) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != sample_rate:
        raise SystemExit(
            f"{path.name}: {rate} Hz gravado, o pipeline espera {sample_rate} Hz"
        )
    return np.asarray(audio, dtype=np.float32)


async def _wait_unmuted(detector: object) -> bool:
    """Wait for half-duplex to release the mic; False if it never does."""
    deadline = time.monotonic() + UNMUTE_TIMEOUT_S
    while getattr(detector, "muted", False):
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.02)
    return True


def _queue_for_vad(session: Session, audio: np.ndarray) -> float:
    """Push a recording into the capture queue the way the microphone would.

    Trailing silence is appended because the utterances were cut by a VAD and
    end right after the speech; without it the detector would never see the
    ``end_silence_ms`` of quiet it needs to close the turn. Returns how long
    the caller should allow for :meth:`Session.listen`.
    """
    cfg = session.cfg.audio
    pad_s = session.level.end_silence_ms / 1000.0 + 0.75
    stream = np.concatenate([audio, np.zeros(int(cfg.capture_sample_rate * pad_s), np.float32)])
    chunk = cfg.vad_chunk
    queue = session.capture.queue
    for i in range(0, stream.size, chunk):
        block = stream[i : i + chunk]
        if block.size < chunk:
            # Silero rejects anything shorter than one chunk; the hardware
            # always delivers exactly `vad_chunk` samples, so pad the tail.
            block = np.pad(block, (0, chunk - block.size))
        queue.put_nowait(block)
    return audio.size / cfg.capture_sample_rate + pad_s + 2.0


async def _run(
    args: argparse.Namespace,
    cfg: AppConfig,
    levels: dict[str, LevelConfig],
    console: Console,
) -> int:
    level = levels[args.level]
    wavs = [Path(p) for p in (args.wavs or [])]
    if args.mode == "live" and wavs:
        console.print("[red]--wav so faz sentido com --mode file.[/red]")
        return 2
    if args.mode == "file" and not wavs:
        console.print("[red]--mode file exige pelo menos um --wav.[/red]")
        return 2

    timelines: list[TurnTimeline] = []
    save_dir = Path(args.save_wavs) if args.save_wavs else None

    async with Session(
        cfg,
        level_name=args.level,
        level=level,
        scenario=args.scenario,
        input_device=args.device,
        console=console,
        save_dir=save_dir,
        # file mode still needs the capture queue, just not the device.
        listen=args.mode == "live",
        streaming=args.pipeline == "streaming",
        feedback=args.feedback,
    ) as session:
        try:
            if args.greet:
                await session.greet()
            if args.mode == "live":
                console.print(
                    f"[bold]Fale {args.turns} vezes.[/bold] Ctrl+C encerra antes da hora."
                )
                await session.run(args.turns, out=timelines)
            for path in wavs:
                if len(timelines) >= args.turns:
                    break
                audio = _load_wav(path, cfg.audio.capture_sample_rate)
                console.print(f"\n[dim]{path.name}[/dim]")
                if not await _wait_unmuted(session.detector):
                    console.print("[yellow]mic mudo (half-duplex) - pulando.[/yellow]")
                    break
                budget = _queue_for_vad(session, audio)
                try:
                    heard = await session.listen(timeout=budget)
                except TimeoutError:
                    console.print(
                        "[yellow]VAD nao fechou o turno (fala curta?) - pulando.[/yellow]"
                    )
                    continue
                tl = await session.turn(heard)
                if tl is None:
                    continue
                timelines.append(tl)
                session.report(tl)
        except KeyboardInterrupt:
            console.print("\ninterrompido pelo usuario.")
        if session.feedback is not None:
            summary = await session.summarize()
            if summary:
                session.ui.summary(summary)
            worker = session.feedback
            console.print(
                "\n[dim]feedback: "
                f"{len(worker.results)} correcao(oes)  "
                f"{worker.dropped} descartado(s)  "
                f"{worker.pending} pendente(s)[/dim]"
            )

    if not timelines:
        console.print("[yellow]nenhum turno concluido.[/yellow]")
        return 1
    console.print()
    render(timelines, console)
    _report_overlap(timelines, console)
    if save_dir is not None:
        console.print(f"\n[dim]falas gravadas em {save_dir}[/dim]")
    return 0


def _report_overlap(timelines: list[TurnTimeline], console: Console) -> None:
    """Phase 6's first acceptance: sound leaves before the model stops writing."""
    usable = [
        tl
        for tl in timelines
        if tl.marks.get("t_playback_start") is not None
        and tl.marks.get("t_llm_done") is not None
    ]
    if not usable:
        return
    ahead = sum(
        1
        for tl in usable
        if tl.marks["t_playback_start"] < tl.marks["t_llm_done"]
    )
    style = "green" if ahead == len(usable) else "yellow"
    console.print(
        f"\n[{style}]playback antes do fim do LLM: {ahead}/{len(usable)} turnos[/{style}]"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mede a latencia por turno.")
    parser.add_argument("--config", default=None, help="caminho do config.yaml")
    parser.add_argument("--levels", default=None, help="caminho do levels.yaml")
    parser.add_argument("--level", default="beginner", help="nivel (levels.yaml)")
    parser.add_argument("--turns", type=int, default=5, help="turnos a executar")
    parser.add_argument(
        "--mode",
        choices=("live", "file"),
        default="live",
        help="live = microfone; file = wav pela fila de captura (VAD real)",
    )
    parser.add_argument(
        "--wav",
        dest="wav",
        action="append",
        default=[],
        help="wav de uma fala para --mode file (repetir o argumento)",
    )
    parser.add_argument(
        "--save-wavs",
        default=None,
        metavar="DIR",
        help="grava a fala de cada turno nesse diretorio",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        metavar="NAME",
        help="cenario (prompts/scenarios/*.md); padrao free_chat",
    )
    parser.add_argument("--device", default=None, help="indice/nome do microfone")
    parser.add_argument(
        "--pipeline",
        choices=("sequential", "streaming"),
        default="sequential",
        help="sequential = baseline da Fase 5; streaming = Fase 6",
    )
    parser.add_argument(
        "--no-greet",
        dest="greet",
        action="store_false",
        default=True,
        help="pula a abertura da IA (turno 0)",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        default=False,
        help="liga o feedback de correcao em segundo plano (Fase 8) + resumo final",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    args.wavs = args.wav

    try:
        cfg, levels = load_all(args.config, args.levels)
    except ConfigError as exc:
        console.print(f"[red]ERRO:[/red] {exc}")
        return 2

    if args.level not in levels:
        console.print(
            f"[red]ERRO:[/red] nivel '{args.level}' inexistente. "
            f"Disponiveis: {', '.join(sorted(levels))}"
        )
        return 2

    try:
        return asyncio.run(_run(args, cfg, levels, console))
    except ConfigError as exc:
        console.print(f"[red]ERRO:[/red] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
