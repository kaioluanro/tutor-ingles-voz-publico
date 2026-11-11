"""Tests for silence trimming (Phase 6).

The trim exists to remove Kokoro's per-call padding without touching speech,
so every test here is about the boundary: what stays, what goes, and what
happens when a fragment is too quiet to judge.
"""

from __future__ import annotations

import numpy as np
import pytest

from tutor.audio.silence import trim_silence

SR = 16000


def _tone(ms: float, *, amp: float = 1.0, freq: float = 440.0) -> np.ndarray:
    n = int(SR * ms / 1000.0)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(ms: float) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000.0), dtype=np.float32)


def test_trims_leading_and_trailing_padding() -> None:
    audio = np.concatenate([_silence(450), _tone(500), _silence(450)])
    out = trim_silence(audio, SR, keep_ms=60.0)
    # Padding is 450 ms, keep margin is 60 ms, so ~390 ms goes from each side.
    assert out.size < audio.size - int(SR * 0.7)
    assert out.size > int(SR * 0.5)


def test_speech_is_untouched_in_the_middle() -> None:
    tone = _tone(500)
    audio = np.concatenate([_silence(450), tone, _silence(450)])
    out = trim_silence(audio, SR, keep_ms=60.0)
    # The surviving signal must still be the tone, not a fragment of it.
    assert np.max(np.abs(out)) == pytest.approx(1.0, abs=0.01)
    # Nothing louder than the keep margin was removed from either end.
    assert out.size >= tone.size


def test_keep_margin_survives() -> None:
    audio = np.concatenate([_silence(300), _tone(300), _silence(300)])
    out = trim_silence(audio, SR, keep_ms=60.0)
    keep = int(SR * 0.060)
    assert out.size >= int(SR * 0.300) + keep * 2 - 10


def test_head_only() -> None:
    audio = np.concatenate([_silence(450), _tone(300), _silence(450)])
    out = trim_silence(audio, SR, head=True, tail=False, keep_ms=60.0)
    assert out.size > int(SR * 0.5)  # the trailing pad is still there
    assert out.size < audio.size


def test_tail_only() -> None:
    audio = np.concatenate([_silence(450), _tone(300), _silence(450)])
    out = trim_silence(audio, SR, head=False, tail=True, keep_ms=60.0)
    assert out.size < audio.size
    assert out.size > int(SR * 0.5)


def test_no_trimming_when_both_disabled() -> None:
    audio = np.concatenate([_silence(450), _tone(300)])
    assert trim_silence(audio, SR, head=False, tail=False).size == audio.size


def test_all_silence_is_returned_unchanged() -> None:
    audio = _silence(500)
    out = trim_silence(audio, SR)
    assert out.size == audio.size


def test_empty_audio_is_safe() -> None:
    assert trim_silence(np.zeros(0, np.float32), SR).size == 0


def test_audio_shorter_than_one_frame_is_untouched() -> None:
    audio = _tone(1)  # 16 samples, below the 10 ms frame
    assert trim_silence(audio, SR).size == audio.size


def test_soft_onset_is_not_clipped() -> None:
    """A quiet first syllable must survive the threshold."""
    ramp = np.linspace(0.0, 1.0, int(SR * 0.20), dtype=np.float32)
    audio = np.concatenate([_silence(300), ramp, _tone(200), _silence(300)])
    out = trim_silence(audio, SR, keep_ms=60.0)
    assert out.size > int(SR * 0.40)


def test_input_dtype_is_float32() -> None:
    audio = np.concatenate([_silence(400), _tone(300), _silence(400)]).astype(np.float64)
    assert trim_silence(audio, SR).dtype == np.float32


def test_result_does_not_alias_the_input() -> None:
    """Playback holds the trimmed array, so it must not pin the padded one."""
    audio = np.concatenate([_silence(400), _tone(300), _silence(400)])
    out = trim_silence(audio, SR)
    assert out.size < audio.size
    out[0] = -1.0
    assert audio[0] == 0.0
