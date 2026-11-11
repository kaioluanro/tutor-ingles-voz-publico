"""Speech-to-text check: transcribe a wav file (or record from the mic).

Usage:
  python scripts/check_stt.py --wav /path/to/audio.wav
  python scripts/check_stt.py --record 5
  python scripts/check_stt.py --wav x.wav --prompt "Learner: ..." --hotwords "table, menu"

Besides the text it prints the confidence numbers the Phase 12 gate reads
(avg_logprob / compression_ratio) and the verdict they produce.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from _common import get_config


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        # simple linear resample to the 16 kHz the config expects
        n16 = int(len(audio) * 16000 / sr)
        idx = np.linspace(0, len(audio) - 1, n16)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
        sr = 16000
    return audio, sr


def record(seconds: float, sr: int) -> np.ndarray:
    import sounddevice as sd

    print(f"Gravando {seconds:.1f}s do microfone... fale em ingles.")
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return rec.reshape(-1)


def main() -> int:
    cfg = get_config()
    stt = cfg.get("stt", {})
    audio_cfg = cfg.get("audio", {})
    sr = int(audio_cfg.get("capture_sample_rate", 16000))

    parser = argparse.ArgumentParser(description="Verifica transcricao (faster-whisper).")
    parser.add_argument("--wav", type=str, default=None, help="arquivo wav para transcrever")
    parser.add_argument("--record", type=float, default=None, help="gravar N segundos do microfone")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="initial_prompt passado ao Whisper (contexto da conversa)",
    )
    parser.add_argument(
        "--hotwords",
        type=str,
        default=stt.get("hotwords") or None,
        help="palavras-chave sugeridas ao Whisper",
    )
    args = parser.parse_args()

    if args.wav:
        audio, sr = load_wav(Path(args.wav))
        print(f"wav: {args.wav}  dur={len(audio)/sr:.2f}s")
    elif args.record:
        audio = record(args.record, sr)
    else:
        print("[ERRO] use --wav ARQ ou --record SEGUNDOS", file=sys.stderr)
        return 1

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f"peak={peak:.4f}")
    if peak < 1e-4:
        print("[ERRO] audio silencioso.", file=sys.stderr)
        return 1

    print(
        f"carregando whisper model={stt.get('model')} device={stt.get('device')} "
        f"compute_type={stt.get('compute_type')} ..."
    )
    from tutor.stt import WhisperSTT

    asr = WhisperSTT(
        model=stt.get("model", "small.en"),
        device=stt.get("device", "cuda"),
        compute_type=stt.get("compute_type", "int8_float16"),
        beam_size=int(stt.get("beam_size", 1)),
    )
    asr.load()
    print(f"modelo carregado em {asr.load_seconds:.2f}s")

    t0 = time.perf_counter()
    tr = asr.transcribe_sync(audio, prompt=args.prompt, hotwords=args.hotwords)
    dt = time.perf_counter() - t0

    print(f"\ntranscricao ({dt:.2f}s para {len(audio)/sr:.2f}s de audio):")
    print(f"  {tr.text!r}")
    print(f"  tempo de transcricao / duracao do audio = {dt / (len(audio)/sr):.2f}x")
    if dt / (len(audio) / sr) >= 1.0:
        print("[AVISO] transcricao mais lenta que o audio (meta: < 1x na GPU).")

    min_lp = float(stt.get("min_avg_logprob", -1.0))
    max_cr = float(stt.get("max_compression_ratio", 2.4))
    print("\nconfianca (gate da Fase 12):")
    print(f"  avg_logprob      = {tr.avg_logprob:.3f}   (minimo {min_lp:g})")
    print(f"  compression_ratio= {tr.compression_ratio:.3f}   (maximo {max_cr:g})")
    print(f"  no_speech_prob   = {tr.no_speech_prob:.3f}   (nao entra no gate)")
    ok = tr.avg_logprob >= min_lp and tr.compression_ratio <= max_cr
    print(f"  veredito: {'aceita' if ok else 'PEDIRIA REPETICAO'}")
    return 0 if tr.text.strip() else 1


if __name__ == "__main__":
    raise SystemExit(main())
