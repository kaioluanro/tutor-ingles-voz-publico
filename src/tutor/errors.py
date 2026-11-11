"""Failures the learner can do something about, with a message that says what.

:class:`ConfigError` (see :mod:`tutor.config`) covers bad files. This module
covers hardware: a missing microphone or speaker is an environment problem, not
a bug, so it must not arrive as a traceback.
"""

from __future__ import annotations

from typing import Literal

DeviceKind = Literal["input", "output"]


class DeviceError(RuntimeError):
    """A PortAudio stream could not be opened."""


def available_devices(kind: DeviceKind) -> str:
    """Comma-separated ``index: name`` for every device of ``kind``."""
    try:
        import sounddevice as sd
    except Exception:  # noqa: BLE001 - listing is best effort
        return "(nao foi possivel listar os dispositivos)"

    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    names = [
        f"{index}: {device['name']}"
        for index, device in enumerate(sd.query_devices())
        if device[channel_key] > 0
    ]
    return ", ".join(names) if names else "(nenhum)"


def describe_device_failure(kind: DeviceKind, exc: BaseException) -> str:
    """Portuguese one-liner + the devices still available for ``kind``."""
    label = "microfone" if kind == "input" else "dispositivo de saida"
    return (
        f"sem {label}: {type(exc).__name__}: {exc}\n"
        f"  {kind}: {available_devices(kind)}\n"
        "  Confira `audio.input_device` / `audio.output_device` no config.yaml "
        "(null = padrao do sistema)."
    )
