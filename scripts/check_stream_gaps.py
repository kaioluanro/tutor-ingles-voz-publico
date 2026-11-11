"""Measure the gap the streaming pipeline puts between two sentences.

Phase 6 synthesises sentence by sentence, because the text is not finished yet.
Kokoro pads every call with a little silence at both ends, so the boundary
between two sentences can end up longer than it would have been if the whole
answer had been synthesised in one call — an audible stutter in the middle of
a reply.

This script synthesises the same answer both ways and reports the silence
*inside* the audio (leading and trailing padding of the whole answer is
expected and harmless). Exit status is 1 when the per-sentence path adds more
than ``--max-extra-gap`` of silence at any boundary.

Usage: python scripts/check_stream_gaps.py [--text "..."] [--max-extra-gap 400]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from contextlib import aclosing
from pathlib import Path

import numpy as np

from _common import ROOT, get_config, load_yaml, preload_cuda_libs

DEFAULT_TEXT = (
    "Hello! My name is Anna. I am your English tutor for today. "
    "Let's practice a short conversation together."
)

FRAME_MS = 10  # RMS window; short enough to keep pauses honest
MERGE_MS = 50  # gaps shorter than this stay inside one speech segment


def _speech_segments(audio: np.ndarray, sample_rate: int, ratio: float) -> list[tuple[int, int]]:
    """Sample ranges that hold speech, after gluing short breaths back on."""
    frame = max(1, int(sample_rate * FRAME_MS / 1000))
    usable = (len(audio) // frame) * frame
    if usable == 0:
        return []
    rms = np.sqrt(np.mean(audio[:usable].reshape(-1, frame) ** 2, axis=1))
    threshold = max(float(rms.max()) * ratio, 1e-5)

    merge_frames = max(1, int(MERGE_MS / FRAME_MS))
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, loud in enumerate(rms > threshold):
        if loud and start is None:
            start = i
        elif not loud and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, int(rms.size)))

    merged: list[tuple[int, int]] = []
    for seg_start, seg_end in segments:
        if merged and seg_start - merged[-1][1] <= merge_frames:
            merged[-1] = (merged[-1][0], seg_end)
        else:
            merged.append((seg_start, seg_end))
    return [(a * frame, b * frame) for a, b in merged]


def _inner_gaps_ms(audio: np.ndarray, sample_rate: int, *, ratio: float) -> list[float]:
    """Silence between speech segments (the answer's own padding excluded)."""
    segments = _speech_segments(audio, sample_rate, ratio)
    return [
        (b_start - a_end) * 1000.0 / sample_rate
        for (_, a_end), (b_start, _) in zip(segments, segments[1:])
    ]


async def _synthesize(tts: object, text: str, speed: float) -> np.ndarray:
    chunks: list[np.ndarray] = []
    async with aclosing(tts.synthesize_stream(text, speed=speed)) as stream:
        async for chunk in stream:
            chunks.append(np.asarray(chunk, dtype=np.float32))
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def _split_sentences(text: str) -> list[str]:
    from tutor.core.chunker import SentenceChunker

    chunker = SentenceChunker(min_words=1)
    out: list[str] = []
    for char in text:
        out.extend(chunker.feed(char))
    out.extend(chunker.flush())
    return out


def _words(text: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]


def _verify_nothing_lost(
    raw: np.ndarray, trimmed: np.ndarray, rate: int, args: argparse.Namespace
) -> int:
    """Transcribe both and fail if the trim dropped a word.

    Trimming with a threshold is only safe if speech is still speech
    afterwards; a transcript is the cheapest proof.
    """
    del args  # signature kept uniform with main()
    preload_cuda_libs()
    from _common import get_config as _cfg

    stt_cfg = _cfg().get("stt", {})
    from tutor.stt import WhisperSTT

    asr = WhisperSTT(
        model=stt_cfg.get("model", "small.en"),
        device=stt_cfg.get("device", "cuda"),
        compute_type=stt_cfg.get("compute_type", "int8_float16"),
        beam_size=int(stt_cfg.get("beam_size", 1)),
    )
    asr.load()
    try:
        before = _words(asr.transcribe_sync(_to_16k(raw, rate)))
        after = _words(asr.transcribe_sync(_to_16k(trimmed, rate)))
    finally:
        asr.close()

    missing = [w for w in before if w not in after]
    print(f"antes : {' '.join(before)}")
    print(f"depois: {' '.join(after)}")
    if missing:
        print(f"ERRO: o trim apagou {len(missing)} palavra(s): {missing}")
        return 1
    print("verificado: nenhuma palavra perdida no trim")
    return 0


def _to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return np.asarray(audio, dtype=np.float32)
    n16 = int(len(audio) * 16000 / rate)
    idx = np.linspace(0, len(audio) - 1, n16)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mede o silencio entre frases.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="resposta multi-frase")
    parser.add_argument("--speed", type=float, default=None, help="tts_speed do nivel")
    parser.add_argument("--level", default="beginner", help="nivel (levels.yaml)")
    parser.add_argument("--max-extra-gap", type=float, default=400.0)
    parser.add_argument(
        "--ratio", type=float, default=0.08, help="fracao do pico usada como limiar"
    )
    parser.add_argument("--save", default=None, help="prefixo para gravar os dois WAVs")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="transcreve o audio trimmado e confirma que nenhuma palavra sumiu",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="mede o audio cru do Kokoro, sem o trim do pipeline",
    )
    args = parser.parse_args()

    cfg = get_config().get("tts", {})
    if args.speed is None:
        levels = load_yaml("levels.yaml") or {}
        args.speed = float(levels.get(args.level, {}).get("tts_speed", 1.0))

    from tutor.tts import KokoroTTS

    tts = KokoroTTS(
        engine=cfg.get("engine", "kokoro"),
        device=cfg.get("device", "cuda"),
        accent=cfg.get("accent", "american"),
        voice=cfg.get("voice", "af_heart"),
        sample_rate=int(cfg.get("sample_rate", 24000)),
    )
    tts.load()

    sentences = _split_sentences(args.text)
    if len(sentences) < 2:
        tts.close()
        print(f"ERRO: o texto precisa de 2+ frases, achei {len(sentences)}")
        return 2

    try:
        whole = asyncio.run(_synthesize(tts, args.text, args.speed))
        raw_pieces = [asyncio.run(_synthesize(tts, s, args.speed)) for s in sentences]
    finally:
        tts.close()

    rate = tts.sample_rate
    pieces = list(raw_pieces)
    if not args.no_trim:
        from tutor.audio.silence import trim_silence

        # Exactly what the streaming pipeline does with each fragment.
        pieces = [trim_silence(p, rate) for p in pieces]
        whole = trim_silence(whole, rate)
    per_sentence = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    raw_audio = np.concatenate(raw_pieces) if raw_pieces else np.zeros(0, np.float32)
    if not whole.size or not per_sentence.size:
        print("ERRO: sintese devolveu audio vazio")
        return 1

    if args.save:
        import soundfile as sf

        base = Path(ROOT / args.save) if not Path(args.save).is_absolute() else Path(args.save)
        sf.write(base.with_name(base.stem + "_whole.wav"), whole, rate, subtype="PCM_16")
        sf.write(base.with_name(base.stem + "_sentences.wav"), per_sentence, rate, subtype="PCM_16")

    gaps_whole = _inner_gaps_ms(whole, rate, ratio=args.ratio)
    gaps_split = _inner_gaps_ms(per_sentence, rate, ratio=args.ratio)

    def _show(name: str, audio: np.ndarray, gaps: list[float]) -> None:
        tail = (
            f"fronteiras {len(gaps)}  silencio mediana {statistics.median(gaps):5.0f}ms"
            f"  maximo {max(gaps):5.0f}ms"
            if gaps
            else "nenhuma fronteira detectada"
        )
        print(f"{name:12s} duracao {audio.size / rate:5.2f}s  {tail}")

    _show("uma chamada", whole, gaps_whole)
    _show("por frase", per_sentence, gaps_split)
    print(f"frases ({len(sentences)}):")
    for i, sentence in enumerate(sentences, 1):
        print(f"  {i}. {sentence}")

    status = 0
    if args.verify:
        status = max(status, _verify_nothing_lost(raw_audio, per_sentence, rate, args))

    if not gaps_split:
        print("ERRO: nao detectei fronteiras na versao por frase")
        return 1

    if len(gaps_whole) == len(gaps_split):
        extras = [b - a for a, b in zip(gaps_whole, gaps_split)]
        detail = " ".join(f"{x:+.0f}" for x in extras)
        worst = max(extras)
        print(f"silencio extra por fronteira (ms): {detail}")
    else:
        # The detector can find a different number of pauses once the padding
        # is gone, so compare the worst pause of each version instead.
        worst = max(gaps_split) - max(gaps_whole)
        print(
            f"fronteiras diferentes ({len(gaps_whole)} vs {len(gaps_split)}); "
            f"pior {max(gaps_split):.0f}ms vs {max(gaps_whole):.0f}ms"
        )
    print(f"pior extra: {worst:.0f}ms  (limite {args.max_extra_gap:.0f}ms)")
    if worst > args.max_extra_gap:
        status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
