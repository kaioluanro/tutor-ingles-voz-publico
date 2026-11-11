"""Tests for the Kokoro wrapper (Phase 4).

``clean_for_tts`` is pure string handling and is covered exhaustively. The
streaming plumbing (worker thread -> ``asyncio.Queue`` -> async generator) is
exercised with a stub pipeline so the suite never loads a neural model.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from tutor.tts import KokoroTTS, clean_for_tts


# -- clean_for_tts ---------------------------------------------------------- #


def test_strips_markdown() -> None:
    assert clean_for_tts("**Hello** there") == "Hello there"
    assert clean_for_tts("# Welcome") == "Welcome"
    assert clean_for_tts("`code` and more") == "and more"
    assert clean_for_tts("a *b* _c_ d") == "a b c d"


def test_strips_urls() -> None:
    assert clean_for_tts("see https://example.com/x?y=1 now") == "see now"
    assert clean_for_tts("http://a.b") == ""


def test_strips_stage_directions() -> None:
    assert clean_for_tts("Really? (laughs) Yes.") == "Really? Yes."
    assert clean_for_tts("Wait (audio: beeps) ok") == "Wait ok"
    assert clean_for_tts("(sighs softly) fine") == "fine"
    # Ordinary parentheses are content, not stage directions.
    assert clean_for_tts("table for two (please)") == "table for two please"


def test_strips_emoji_and_symbols() -> None:
    assert clean_for_tts("Hello world 😀!") == "Hello world!"
    assert clean_for_tts("a — b | c") == "a b c"
    assert clean_for_tts("50% off") == "50 off"


def test_keeps_normal_sentence_intact() -> None:
    sentence = "Good evening! Table for one, or are you waiting for someone?"
    assert clean_for_tts(sentence) == sentence


def test_collapses_whitespace_and_handles_empty() -> None:
    assert clean_for_tts("  a \n\t b  ") == "a b"
    assert clean_for_tts("") == ""
    assert clean_for_tts("   ") == ""
    assert clean_for_tts("((()))") == ""


# -- construction ----------------------------------------------------------- #


def test_rejects_unknown_accent() -> None:
    with pytest.raises(ValueError, match="accent desconhecido"):
        KokoroTTS(accent="french")


def test_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="engine desconhecido"):
        KokoroTTS(engine="piper")


def test_loads_lazily() -> None:
    tts = KokoroTTS()
    assert tts.loaded is False
    assert tts.load_seconds is None


# -- streaming plumbing ----------------------------------------------------- #


class _FakeResult:
    def __init__(self, audio: np.ndarray | None) -> None:
        self.audio = audio


class _FakePipeline:
    def __init__(self, values: tuple[float, ...] = (0.1, 0.2, 0.3)) -> None:
        self.values = values
        self.calls: list[tuple[str, str, float]] = []

    def __call__(self, text: str, *, voice: str, speed: float = 1.0):
        self.calls.append((text, voice, speed))
        for value in self.values:
            yield _FakeResult(np.full(8, value, dtype=np.float32))
        yield _FakeResult(None)  # Kokoro emits a None audio now and then


def _stubbed(values: tuple[float, ...] = (0.1, 0.2, 0.3)) -> KokoroTTS:
    tts = KokoroTTS(voice="af_heart", sample_rate=24000)
    tts._pipeline = _FakePipeline(values)  # bypass model loading entirely
    tts._load_seconds = 0.0
    return tts


async def _drain(tts: KokoroTTS, text: str, **kwargs: object) -> list[np.ndarray]:
    from contextlib import aclosing

    async with aclosing(tts.synthesize_stream(text, **kwargs)) as stream:  # type: ignore[arg-type]
        return [chunk async for chunk in stream]


def test_stream_yields_chunks_in_order() -> None:
    async def scenario() -> list[np.ndarray]:
        tts = _stubbed()
        return await _drain(tts, "Hello there")

    chunks = asyncio.run(scenario())
    assert len(chunks) == 3
    assert np.allclose([float(c[0]) for c in chunks], [0.1, 0.2, 0.3])
    assert all(c.dtype == np.float32 and c.shape == (8,) for c in chunks)


def test_stream_cleans_text_before_synthesis() -> None:
    async def scenario() -> tuple[str, float, list[np.ndarray]]:
        tts = _stubbed()
        chunks = await _drain(tts, "**Hello** world", speed=0.88)
        pipeline = tts._pipeline
        text, _voice, speed = pipeline.calls[0]
        return text, speed, chunks

    text, speed, chunks = asyncio.run(scenario())
    assert text == "Hello world"
    assert speed == 0.88
    assert len(chunks) == 3


def test_stream_of_empty_text_produces_nothing() -> None:
    async def scenario() -> tuple[list[np.ndarray], int]:
        tts = _stubbed()
        chunks = await _drain(tts, "   ")
        pipeline = tts._pipeline
        return chunks, len(pipeline.calls)

    chunks, calls = asyncio.run(scenario())
    assert chunks == []
    assert calls == 0


def test_stream_propagates_pipeline_errors() -> None:
    class _Boom:
        def __call__(self, text: str, *, voice: str, speed: float = 1.0):
            yield _FakeResult(np.zeros(4, dtype=np.float32))
            raise RuntimeError("synthesis failed")
            yield _FakeResult(np.zeros(4, dtype=np.float32))  # pragma: no cover

    async def scenario() -> list[np.ndarray]:
        tts = KokoroTTS()
        tts._pipeline = _Boom()
        return await _drain(tts, "Hello")

    with pytest.raises(RuntimeError, match="synthesis failed"):
        asyncio.run(scenario())


# -- two languages (Phase 12) ---------------------------------------------- #

_SENTINEL = object()


class _RecordingKPipeline:
    """Stands in for ``kokoro.KPipeline`` and remembers how it was built."""

    made: list[dict] = []

    def __init__(self, *, lang_code, model=None, repo_id=None, device=None, **kw):
        self.lang_code = lang_code
        self.model = _SENTINEL if model is None else model
        self.repo_id = repo_id or "hexgrad/Kokoro-82M"
        self.device = device
        type(self).made.append(
            {"lang_code": lang_code, "model": self.model, "repo_id": self.repo_id}
        )

    def __call__(self, *args, **kwargs):
        return iter(())


@pytest.fixture()
def fake_kokoro(monkeypatch):
    _RecordingKPipeline.made = []
    monkeypatch.setattr("kokoro.KPipeline", _RecordingKPipeline)
    return _RecordingKPipeline


def test_the_two_pipelines_share_one_model(fake_kokoro) -> None:
    tts = KokoroTTS()
    tts.load()
    tts.load_feedback()

    made = fake_kokoro.made
    assert [m["lang_code"] for m in made] == ["a", "pt-br"]
    assert made[1]["model"] is made[0]["model"], "the weights must be reused"
    assert made[1]["repo_id"] == made[0]["repo_id"]


def test_each_language_gets_its_own_voice(fake_kokoro) -> None:
    tts = KokoroTTS()
    english, en_voice = tts._resolve("en")
    portuguese, pt_voice = tts._resolve("pt-br")

    assert en_voice == "af_heart"
    assert pt_voice == "pf_dora"
    assert english is not portuguese
    assert tts.loaded and tts.feedback_loaded


def test_the_feedback_aliases_reach_the_same_pipeline(fake_kokoro) -> None:
    tts = KokoroTTS()
    first, voice = tts._resolve("pt-br")
    for alias in ("pt", "p", "PT-BR", "portuguese"):
        pipeline, alias_voice = tts._resolve(alias)
        assert pipeline is first
        assert alias_voice == voice


def test_an_unknown_language_is_rejected_not_phonemised_wrong() -> None:
    tts = KokoroTTS()
    tts._pipeline = _FakePipeline()
    with pytest.raises(ValueError, match="lang desconhecido"):
        tts._resolve("fr")


def test_a_portuguese_stream_never_touches_the_accent_pipeline() -> None:
    async def scenario() -> tuple[list[np.ndarray], list, list]:
        tts = _stubbed((0.1,))
        feedback = _FakePipeline((0.7,))
        tts._feedback_pipeline = feedback
        chunks = await _drain(tts, "Tudo bem", lang="pt-br")
        return chunks, tts._pipeline.calls, feedback.calls

    chunks, english_calls, pt_calls = asyncio.run(scenario())
    assert len(chunks) == 1
    # Kokoro phonemises with one language's rules; mixing them here would
    # produce audio that sounds like neither.
    assert english_calls == []
    assert pt_calls[0][0] == "Tudo bem"
    assert pt_calls[0][1] == "pf_dora"


def test_warmup_can_touch_the_portuguese_pipeline() -> None:
    async def scenario() -> float:
        tts = _stubbed((0.1,))
        tts._feedback_pipeline = _FakePipeline((0.7,))
        return await tts.warmup(lang="pt-br")

    assert asyncio.run(scenario()) >= 0.0


def test_a_quoted_phrase_keeps_its_space() -> None:
    # The Portuguese correction says "O correto e 'went'"; squeezing the space
    # before the quote produced "e'went'", which espeak reads differently.
    assert clean_for_tts("O correto e 'went'.") == "O correto e 'went'."
    assert clean_for_tts("world !") == "world!"
