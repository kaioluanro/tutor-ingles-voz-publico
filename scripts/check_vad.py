"""Live VAD check: prints START/END events and the duration of each utterance.

Uses the real project classes (``AudioCapture`` + ``TurnDetector``), so a green
run here validates the same code the pipeline will use.

Usage: python scripts/check_vad.py [--seconds 20] [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

from _common import ROOT, load_yaml

sys.path.insert(0, str(ROOT / "src"))

from tutor.audio.capture import AudioCapture  # noqa: E402
from tutor.audio.vad import TurnDetector  # noqa: E402
from tutor.config import AppConfig, ConfigError, load_config  # noqa: E402

SAVE_DIR = Path("/tmp")


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf

        sf.write(path, audio, sample_rate, subtype="PCM_16")
        print(f"          salvo em {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"          (nao foi possivel salvar wav: {exc})", file=sys.stderr)


async def _run(args: argparse.Namespace, cfg: AppConfig) -> int:
    levels = load_yaml("levels.yaml")
    if args.level not in levels:
        print(f"[ERRO] nivel '{args.level}' inexistente", file=sys.stderr)
        return 2
    level = levels[args.level]

    audio = cfg.audio
    if audio.capture_sample_rate != 16000:
        print(
            f"[ERRO] silero-vad exige 16000 Hz, config diz {audio.capture_sample_rate}",
            file=sys.stderr,
        )
        return 1

    print(
        f"VAD: threshold={args.threshold} end_silence={args.end_silence_ms}ms "
        f"chunk={audio.vad_chunk} nivel={args.level}"
    )
    print(f"Gravando por {args.seconds:.0f}s - fale, pause e fale de novo.\n")

    detector = TurnDetector(
        sample_rate=audio.capture_sample_rate,
        threshold=args.threshold,
        chunk_samples=audio.vad_chunk,
        min_utterance_ms=audio.min_utterance_ms,
        max_utterance_s=audio.max_utterance_s,
        pre_roll_ms=audio.pre_roll_ms,
        end_silence_ms=args.end_silence_ms,
    )
    capture = AudioCapture(
        sample_rate=audio.capture_sample_rate,
        blocksize=audio.vad_chunk,
        device=args.device,
    )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + args.seconds
    started_at: float | None = None
    turns = 0

    print("=== falando ===")
    try:
        async with capture:
            while loop.time() < deadline:
                try:
                    block = await asyncio.wait_for(capture.get(), timeout=1.0)
                except TimeoutError:
                    print(
                        "[AVISO] sem blocos de audio em 1 s - microfone mudo?",
                        file=sys.stderr,
                    )
                    continue
                was_collecting = detector.collecting
                utterance = detector.process(block)

                if detector.collecting and not was_collecting:
                    started_at = time.perf_counter()
                    print(
                        f"  START  sample={detector.position:7d}  "
                        f"({detector.position / audio.capture_sample_rate * 1000:8.1f} ms)"
                    )

                if utterance is not None:
                    turns += 1
                    duration_ms = len(utterance) / audio.capture_sample_rate * 1000
                    wall_ms = (time.perf_counter() - started_at) * 1000 if started_at else 0.0
                    print(
                        f"  END    sample={detector.position:7d}  dur={duration_ms:8.0f} ms"
                        f"  (wall {wall_ms:6.0f} ms)  [{turns}o turno]"
                    )
                    _save_wav(
                        SAVE_DIR / f"vad_utterance_{turns}.wav",
                        utterance,
                        audio.capture_sample_rate,
                    )
                    started_at = None
    except KeyboardInterrupt:
        print("\ninterrompido pelo usuario.")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERRO] audio: {exc}", file=sys.stderr)
        return 1

    print(
        f"\nresumo: {turns} turno(s) aceitos / {detector.turn_count} detectado(s) / "
        f"{detector.discarded_short} descartado(s) por ser curto / "
        f"{capture.overflows} overflow(s) de fila"
    )
    if turns == 0:
        print(
            "[AVISO] nenhum turno aceito - fale mais alto, baixe o threshold "
            f"(atual {args.threshold}) ou verifique o ganho do microfone (README 5.2)."
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verifica eventos de VAD ao vivo.")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--end-silence-ms",
        type=int,
        default=None,
        help="min_silence_duration_ms (padrao: end_silence_ms do nivel)",
    )
    parser.add_argument("--level", default="beginner", help="nivel usado para end_silence_ms")
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=None, help="caminho do config.yaml")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 2

    if args.device is None:
        args.device = cfg.audio.input_device
    if args.threshold is None:
        args.threshold = cfg.audio.vad_threshold
    if args.end_silence_ms is None:
        levels = load_yaml("levels.yaml")
        args.end_silence_ms = int(levels[args.level]["end_silence_ms"])

    try:
        return asyncio.run(_run(args, cfg))
    except ConfigError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
