"""List audio devices, record 3 s from the input and play it back.

Usage: python scripts/check_audio.py [--input N] [--output N]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import sounddevice as sd

from _common import get_config

SR_REQUIRED = 16000


def main() -> int:
    cfg = get_config()
    audio_cfg = cfg.get("audio", {})
    sr = int(audio_cfg.get("capture_sample_rate", SR_REQUIRED))

    parser = argparse.ArgumentParser(description="Verifica captura e reproducao de audio.")
    parser.add_argument("--input", type=int, default=None, help="indice do dispositivo de entrada")
    parser.add_argument("--output", type=int, default=None, help="indice do dispositivo de saida")
    parser.add_argument("--seconds", type=float, default=3.0, help="duracao da gravacao")
    args = parser.parse_args()

    print("=== Dispositivos de audio ===")
    devices = sd.query_devices()
    default_in, default_out = sd.default.device
    for i, d in enumerate(devices):
        marker = ""
        if i == default_in:
            marker += " [PADRAO IN]"
        if i == default_out:
            marker += " [PADRAO OUT]"
        max_in = d["max_input_channels"]
        max_out = d["max_output_channels"]
        kind = "IN " if max_in else "OUT"
        if not max_in and not max_out:
            continue
        print(f"{i:3d} {kind} {d['name']!r} sr={d['default_samplerate']:.0f}{marker}")

    dev_in = args.input if args.input is not None else audio_cfg.get("input_device")
    dev_out = args.output if args.output is not None else audio_cfg.get("output_device")

    print(f"\n=== Gravando {args.seconds:.1f}s de {dev_in or 'dispositivo padrao'} ===")
    try:
        recording = sd.rec(
            int(args.seconds * sr),
            samplerate=sr,
            channels=1,
            dtype="float32",
            device=dev_in,
        )
        sd.wait()
    except Exception as exc:  # noqa: BLE001 - report, do not swallow silently
        print(f"[ERRO] falha na gravacao: {exc}", file=sys.stderr)
        return 1

    peak = float(np.max(np.abs(recording))) if recording.size else 0.0
    rms = float(np.sqrt(np.mean(recording**2))) if recording.size else 0.0
    print(f"peak={peak:.4f}  rms={rms:.4f}")
    if peak < 1e-4:
        print("[AVISO] sinal praticamente silencioso: confira o microfone/mudo.")

    print("=== Reproduzindo a gravacao ===")
    t0 = time.perf_counter()
    try:
        sd.play(recording, samplerate=sr, device=dev_out)
        sd.wait()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERRO] falha na reproducao: {exc}", file=sys.stderr)
        return 1
    print(f"OK: gravado e reproduzido em {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
