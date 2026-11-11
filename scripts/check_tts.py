"""Text-to-speech check: synthesize with Kokoro, measure time to first audio chunk.

Uses the real project classes (``KokoroTTS``), so a green run here validates the
same code the pipeline will use.

Usage: python scripts/check_tts.py [--text "..."] [--no-play] [--save out.wav]

`--lang pt-br` exercises the second Kokoro pipeline (the spoken correction),
which reuses the English pipeline's KModel.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import numpy as np

from _common import get_config


async def _synthesize(args: argparse.Namespace, tts: object) -> tuple[list[np.ndarray], float, float]:
    """Drain the stream once, recording when the first chunk and the last arrive."""
    from contextlib import aclosing

    chunks: list[np.ndarray] = []
    first_audio_at: float | None = None
    t0 = time.perf_counter()
    async with aclosing(
        tts.synthesize_stream(args.text, speed=args.speed, lang=args.lang, voice=args.voice)
    ) as stream:
        async for chunk in stream:
            if first_audio_at is None:
                first_audio_at = time.perf_counter()
            chunks.append(chunk)
    done = time.perf_counter() - t0
    if first_audio_at is None:
        return chunks, float("nan"), done
    return chunks, (first_audio_at - t0) * 1000, done


def main() -> int:
    cfg = get_config()
    tts_cfg = cfg.get("tts", {})

    parser = argparse.ArgumentParser(description="Verifica sintese de voz (Kokoro).")
    parser.add_argument(
        "--lang",
        choices=("en", "pt-br"),
        default="en",
        help="pipeline do Kokoro: en (tutor) ou pt-br (correcao falada)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="frase a sintetizar (padrao: ingles, ou uma frase em portugues "
        "quando --lang pt-br)",
    )
    parser.add_argument("--voice", default=None)
    parser.add_argument("--accent", default=tts_cfg.get("accent", "american"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default=tts_cfg.get("device", "cuda"))
    parser.add_argument("--no-play", action="store_true", help="nao reproduz, so mede")
    parser.add_argument("--save", type=str, default=None, help="salva o resultado em wav")
    parser.add_argument("--warmup", action="store_true", help="roda uma frase curta antes de medir")
    args = parser.parse_args()

    if args.text is None:
        args.text = (
            "Tudo bem! Esta numa boa para dizer a frase corrigida em voz alta."
            if args.lang == "pt-br"
            else "Good evening! Welcome in. Table for one, or are you waiting for someone?"
        )
    if args.voice is None:
        args.voice = (
            tts_cfg.get("feedback_voice", "pf_dora")
            if args.lang == "pt-br"
            else tts_cfg.get("voice", "af_heart")
        )

    from tutor.tts import KokoroTTS, clean_for_tts

    tts = KokoroTTS(
        engine=tts_cfg.get("engine", "kokoro"),
        device=args.device,
        accent=args.accent,
        voice=tts_cfg.get("voice", "af_heart"),
        sample_rate=int(tts_cfg.get("sample_rate", 24000)),
        feedback_voice=tts_cfg.get("feedback_voice", "pf_dora"),
        feedback_lang=tts_cfg.get("feedback_lang", "pt-br"),
    )

    print(
        f"carregando KPipeline lang={args.lang!r} voice={args.voice!r} "
        f"device={args.device} ..."
    )
    if args.lang == "en":
        tts.load()
        print(f"pipeline carregada em {tts.load_seconds:.2f}s")
    else:
        tts.load_feedback()
        print(f"pipeline pt-br carregada sobre o mesmo KModel ({tts.load_seconds:.2f}s do 1o)")

    if args.warmup:
        print("warmup: sintetizando frase curta...")
        tw = asyncio.run(tts.warmup(speed=args.speed, lang=args.lang))
        print(f"warmup feito em {tw:.2f}s")

    clean = clean_for_tts(args.text)
    if clean != args.text:
        print(f"texto limpo para TTS: {clean!r}")
    if not clean:
        print("[ERRO] texto vazio apos limpeza.", file=sys.stderr)
        return 1

    sr = tts.sample_rate
    chunks, first_ms, done = asyncio.run(_synthesize(args, tts))
    if not chunks:
        print("[ERRO] nenhum audio gerado.", file=sys.stderr)
        return 1

    print(f"\ntempo ate o 1o chunk de audio: {first_ms:.0f} ms")
    wave = np.concatenate(chunks)
    dur = len(wave) / sr
    peak = float(np.max(np.abs(wave)))
    print(f"chunks: {len(chunks)}  duracao do audio: {dur:.2f}s  geracao total: {done:.2f}s")
    print(f"1o chunk em {first_ms:.0f} ms | RTF total = {done / max(dur, 1e-6):.2f}")
    print(f"peak={peak:.3f}")
    if peak > 0.999:
        print("[AVISO] clipping no audio gerado.")

    if args.save:
        import soundfile as sf

        sf.write(args.save, wave, sr)
        print(f"salvo em {args.save}")

    if not args.no_play:
        import sounddevice as sd

        print("reproduzindo...")
        sd.play(wave, samplerate=sr, device=cfg.get("audio", {}).get("output_device"))
        sd.wait()
        print("OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
