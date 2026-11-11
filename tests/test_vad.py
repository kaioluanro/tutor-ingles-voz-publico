"""Tests for TurnDetector (Phase 3).

The detector is pure audio-in/audio-out, so every acceptance criterion of the
phase can be exercised with committed WAV fixtures instead of a live microphone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tutor.audio.vad import TurnDetector

FIXTURES = Path(__file__).parent / "fixtures"
SR = 16000
CHUNK = 512


@pytest.fixture(scope="module")
def silero_model():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def _load(name: str) -> np.ndarray:
    audio, rate = sf.read(FIXTURES / name, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SR:
        n = int(round(len(audio) * SR / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def _blocks(timeline: np.ndarray):
    for i in range(0, len(timeline), CHUNK):
        block = timeline[i : i + CHUNK]
        if block.size < CHUNK:
            block = np.pad(block, (0, CHUNK - block.size))
        yield block


def _run(detector: TurnDetector, timeline: np.ndarray) -> list[np.ndarray]:
    return [
        utterance
        for block in _blocks(timeline)
        if (utterance := detector.process(block)) is not None
    ]


def _onset_ms(audio: np.ndarray, thresh: float = 0.01) -> float:
    """Milliseconds of leading silence inside a returned utterance."""
    envelope = np.convolve(np.abs(audio), np.ones(320) / 320, mode="same")
    hits = np.where(envelope > thresh)[0]
    return float(hits[0]) / SR * 1000 if hits.size else -1.0


@pytest.fixture
def make_detector(silero_model):
    def factory(**kwargs) -> TurnDetector:
        defaults = dict(
            sample_rate=SR,
            threshold=0.5,
            chunk_samples=CHUNK,
            min_utterance_ms=400,
            max_utterance_s=30.0,
            pre_roll_ms=300,
            end_silence_ms=1100,
            model=silero_model,
        )
        defaults.update(kwargs)
        return TurnDetector(**defaults)

    return factory


# --------------------------------------------------------------------------- #
# acceptance: speak, pause, speak -> two correct events
# --------------------------------------------------------------------------- #

def test_two_turns_detected(make_detector) -> None:
    first = _load("speech_a.wav")
    second = _load("speech_b.wav")
    detector = make_detector()

    timeline = np.concatenate(
        [_silence(1.0), first, _silence(2.5), second, _silence(2.5)]
    )
    turns = _run(detector, timeline)

    assert len(turns) == 2, f"esperava 2 turnos, veio {len(turns)}"
    assert detector.turn_count == 2

    # each utterance is the speech plus pre-roll/padding, never less than the speech
    for utterance, speech in zip(turns, (first, second), strict=True):
        assert utterance.size >= speech.size
        assert utterance.size <= speech.size + int(SR * 1.2)
        assert _onset_ms(utterance) > 0, "inicio da frase foi cortado"


def test_turns_keep_original_level(make_detector) -> None:
    speech = _load("speech_a.wav")
    detector = make_detector()
    turns = _run(detector, np.concatenate([_silence(1.0), speech, _silence(2.5)]))

    assert len(turns) == 1
    # peak must match the source: nothing amplified or clipped on the way through
    assert np.max(np.abs(turns[0])) == pytest.approx(np.max(np.abs(speech)), rel=0.05)


# --------------------------------------------------------------------------- #
# acceptance: keyboard/click noise does not become speech
# --------------------------------------------------------------------------- #

def test_white_noise_burst_is_not_speech(make_detector) -> None:
    rng = np.random.default_rng(0)
    for duration_s in (0.05, 0.15, 0.30):
        noise = (rng.standard_normal(int(SR * duration_s)) * 0.5).astype(np.float32)
        detector = make_detector()
        turns = _run(
            detector,
            np.concatenate([_silence(1.0), noise, _silence(2.5)]),
        )
        assert turns == [], f"ruido de {duration_s}s virou fala"
        assert detector.turn_count == 0


def test_impulse_click_is_not_speech(make_detector) -> None:
    rng = np.random.default_rng(1)
    click = np.zeros(int(SR * 0.4), dtype=np.float32)
    click[: int(SR * 0.01)] = rng.standard_normal(int(SR * 0.01)) * 0.9

    detector = make_detector()
    turns = _run(detector, np.concatenate([_silence(1.0), click, _silence(2.5)]))
    assert turns == []
    assert detector.turn_count == 0


def test_short_utterance_is_discarded(make_detector) -> None:
    """min_utterance_ms judges the speech, not the pre-roll/padding around it."""
    speech = _load("speech_short.wav")  # ~0.4 s
    assert speech.size < SR, "fixture de fala curta precisa ser < 1 s"

    detector = make_detector(min_utterance_ms=1000)
    turns = _run(detector, np.concatenate([_silence(1.0), speech, _silence(2.5)]))

    assert turns == []
    assert detector.turn_count == 0
    assert detector.discarded_short >= 1


# --------------------------------------------------------------------------- #
# acceptance: the start of a phrase is not cut
# --------------------------------------------------------------------------- #

def test_pre_roll_keeps_the_onset(make_detector) -> None:
    speech = _load("speech_a.wav")
    detector = make_detector(pre_roll_ms=300)

    # 1 s of silence ahead of the speech so the ring buffer is fully primed
    turns = _run(detector, np.concatenate([_silence(1.0), speech, _silence(2.5)]))

    assert len(turns) == 1
    lead_ms = _onset_ms(turns[0])
    assert lead_ms > 50, f"silencio inicial de {lead_ms:.0f} ms - inicio cortado"
    assert lead_ms <= 300 + 64, f"pre-roll de {lead_ms:.0f} ms excede o configurado"


def test_onset_survives_without_priming(make_detector) -> None:
    """Even with the ring buffer still filling, the speech onset must survive."""
    speech = _load("speech_a.wav")
    detector = make_detector(pre_roll_ms=300)

    turns = _run(detector, np.concatenate([_silence(0.05), speech, _silence(2.5)]))

    assert len(turns) == 1
    assert _onset_ms(turns[0]) >= 0


# --------------------------------------------------------------------------- #
# half-duplex
# --------------------------------------------------------------------------- #

def test_muted_discards_everything(make_detector) -> None:
    detector = make_detector()
    detector.muted = True
    speech = _load("speech_a.wav")

    turns = _run(detector, np.concatenate([_silence(1.0), speech, _silence(1.0)]))

    assert turns == []
    assert detector.turn_count == 0


def test_mute_mid_turn_clears_state(make_detector) -> None:
    detector = make_detector()
    speech = _load("speech_a.wav")
    timeline = np.concatenate([_silence(1.0), speech, _silence(3.0)])
    mute_at = int(SR * 1.5)  # inside the phrase, which spans 1.0 s .. 2.756 s
    unmute_at = int(SR * 3.0)

    turns: list[np.ndarray] = []
    for block in _blocks(timeline):
        start = detector.position
        if start >= mute_at:
            detector.muted = True
        if start >= unmute_at:
            detector.muted = False
        if (utterance := detector.process(block)) is not None:
            turns.append(utterance)

    assert turns == []
    assert detector.collecting is False
    assert detector.turn_count == 0


def test_unmuting_resumes_detection(make_detector) -> None:
    first = _load("speech_a.wav")
    second = _load("speech_b.wav")
    detector = make_detector()

    muted_timeline = np.concatenate([_silence(1.0), first, _silence(1.0)])
    _run(detector, muted_timeline)
    detector.muted = True
    _run(detector, np.concatenate([_silence(0.5), first, _silence(1.0)]))
    detector.muted = False

    turns = _run(detector, np.concatenate([_silence(1.0), second, _silence(2.5)]))
    assert len(turns) == 1, "deteccao nao voltou depois do unmute"


# --------------------------------------------------------------------------- #
# limits and state
# --------------------------------------------------------------------------- #

def test_long_speech_is_hard_cut(make_detector) -> None:
    speech = _load("speech_b.wav")  # ~2.8 s
    detector = make_detector(max_utterance_s=1.0)

    turns = _run(detector, np.concatenate([_silence(1.0), speech, _silence(2.5)]))

    assert len(turns) == 1
    assert turns[0].size == int(SR * 1.0)
    # the rest of the same phrase must not reappear as a second turn
    assert detector.turn_count == 1


def test_reset_drops_partial_turn(make_detector) -> None:
    detector = make_detector()
    speech = _load("speech_a.wav")
    timeline = np.concatenate([_silence(1.0), speech, _silence(2.5)])

    turns: list[np.ndarray] = []
    for index, block in enumerate(_blocks(timeline)):
        if index == 40:  # mid-phrase
            detector.reset()
        if (utterance := detector.process(block)) is not None:
            turns.append(utterance)

    # everything buffered before reset() is gone; no partial phrase leaks out
    assert detector.collecting is False
    assert all(utterance.size <= speech.size + int(SR * 1.2) for utterance in turns)


def test_empty_chunk_is_ignored(make_detector) -> None:
    detector = make_detector()
    assert detector.process(np.zeros(0, dtype=np.float32)) is None


def test_unsupported_sample_rate_rejected(silero_model) -> None:
    with pytest.raises(ValueError, match="8000 ou 16000"):
        TurnDetector(sample_rate=44100, model=silero_model)


def test_pre_roll_property(make_detector) -> None:
    detector = make_detector(pre_roll_ms=250)
    assert detector.pre_roll_samples == int(SR * 0.25)
