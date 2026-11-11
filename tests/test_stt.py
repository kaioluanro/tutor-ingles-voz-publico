"""Tests for the faster-whisper wrapper (Phase 4).

The model itself is never loaded here (it needs a GPU and 1.4 s); what is
covered is everything that decides *how* it is called: the config wiring, the
lazy single load and the thread hand-off that keeps the event loop free.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect

import numpy as np
import pytest

from tutor.config import load_config
from tutor.stt import Transcript, WhisperSTT, preload_cuda_libs


def test_from_config_wires_every_field() -> None:
    stt = load_config().stt
    asr = WhisperSTT.from_config(stt)
    assert asr.model_name == stt.model
    assert asr.device == stt.device
    assert asr.compute_type == stt.compute_type
    assert asr.beam_size == stt.beam_size
    assert asr.language == "en"  # English tutor, always


def test_model_loads_lazily() -> None:
    asr = WhisperSTT(model="small.en", device="cpu", compute_type="int8")
    assert asr.loaded is False
    assert asr.load_seconds is None


def test_transcribe_is_async_and_uses_the_executor() -> None:
    source = inspect.getsource(WhisperSTT.transcribe)
    assert "run_in_executor" in source
    assert "self._executor" in source


def test_empty_audio_short_circuits_without_loading() -> None:
    asr = WhisperSTT(device="cpu")
    # transcribe_sync returns before load() when there is nothing to decode,
    # so this must not try to build a WhisperModel.
    assert asr.transcribe_sync(np.zeros(0, dtype=np.float32)) == Transcript("")


def test_single_worker_executor_is_preserved() -> None:
    asr = WhisperSTT()
    assert asr._executor._max_workers == 1  # noqa: SLF001 - pins the contract
    asr.close()


def test_preload_is_idempotent() -> None:
    first = preload_cuda_libs()
    second = preload_cuda_libs()
    assert second == []  # already done on the second call
    assert isinstance(first, list)


def test_transcribe_sync_rejects_nothing_on_nonempty_audio() -> None:
    """The signature accepts float32 mono and reshapes it."""
    sig = inspect.signature(WhisperSTT.transcribe_sync)
    positional = [
        name
        for name, p in sig.parameters.items()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == ["self", "audio"]
    assert list(sig.parameters)[2:] == ["prompt", "hotwords"]
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for name, p in sig.parameters.items()
        if name in ("prompt", "hotwords")
    )


# -- context bias (Phase 12) ------------------------------------------------ #


class _FakeSegment:
    def __init__(
        self,
        text: str,
        *,
        avg_logprob: float = -0.5,
        compression_ratio: float = 1.0,
        no_speech_prob: float = 0.1,
        start: float = 0.0,
        end: float = 1.0,
    ) -> None:
        self.text = text
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio
        self.no_speech_prob = no_speech_prob
        self.start = start
        self.end = end


class _FakeModel:
    """Stands in for WhisperModel: records kwargs, returns fixed segments."""

    def __init__(self, segments: list[_FakeSegment] | None = None) -> None:
        self.segments = segments if segments is not None else [_FakeSegment("hello there")]
        self.kwargs: dict | None = None

    def transcribe(self, _samples, **kwargs):  # noqa: ANN001 - mirrors the real API
        self.kwargs = kwargs
        return iter(self.segments), object()


def _patched(segments: list[_FakeSegment] | None = None) -> tuple[WhisperSTT, _FakeModel]:
    asr = WhisperSTT(device="cpu")
    model = _FakeModel(segments)
    asr._model = model  # noqa: SLF001 - the point is to skip the 1.4 s load
    return asr, model


def test_context_prompt_and_hotwords_reach_the_decoder() -> None:
    asr, model = _patched()
    audio = np.zeros(16000, dtype=np.float32)
    result = asr.transcribe_sync(
        audio, prompt="waiter: table for two", hotwords="menu,allergy"
    )
    assert model.kwargs is not None
    assert model.kwargs["initial_prompt"] == "waiter: table for two"
    assert model.kwargs["hotwords"] == "menu,allergy"
    assert model.kwargs["condition_on_previous_text"] is False
    assert result.text == "hello there"


def test_no_hint_sends_no_hint_argument() -> None:
    """Omitting the kwargs keeps whisper's defaults instead of biasing it."""
    asr, model = _patched()
    asr.transcribe_sync(np.zeros(16000, dtype=np.float32))
    assert "initial_prompt" not in (model.kwargs or {})
    assert "hotwords" not in (model.kwargs or {})


def test_confidence_weights_by_duration_not_by_segment_count() -> None:
    """A long confident segment must not be drowned out by a short unsure one."""
    segments = [
        _FakeSegment("a", avg_logprob=-0.1, start=0.0, end=4.0),
        _FakeSegment("b", avg_logprob=-1.1, start=4.0, end=6.0),
    ]
    asr, _ = _patched(segments)
    got = asr.transcribe_sync(np.zeros(16000, dtype=np.float32))
    # (-0.1*4 + -1.1*2) / 6 = -6.6/6 / 6
    assert got.avg_logprob == pytest.approx(-0.4333, abs=1e-3)


def test_confidence_keeps_the_worst_segment_ratio() -> None:
    segments = [
        _FakeSegment("a", compression_ratio=1.2, no_speech_prob=0.2),
        _FakeSegment("b", compression_ratio=2.9, no_speech_prob=0.7),
    ]
    asr, _ = _patched(segments)
    got = asr.transcribe_sync(np.zeros(16000, dtype=np.float32))
    assert got.compression_ratio == pytest.approx(2.9)
    assert got.no_speech_prob == pytest.approx(0.7)


def test_transcription_without_text_is_empty() -> None:
    asr, _ = _patched([])
    got = asr.transcribe_sync(np.zeros(16000, dtype=np.float32))
    assert got.empty
    assert got.text == ""


def test_transcript_is_frozen_and_empty_only_looks_at_text() -> None:
    got = Transcript(text="   ")
    assert got.empty
    assert not Transcript(text="hi").empty
    with pytest.raises(dataclasses.FrozenInstanceError):
        got.text = "x"  # type: ignore[misc]
