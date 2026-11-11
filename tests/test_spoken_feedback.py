"""Spoken correction and the low-confidence gate (Phase 12).

Two things are being protected here:

* a transcript Whisper is not sure about must never reach the LLM — the drift
  it causes lives on in the dialogue history for the rest of the session;
* a correction is spoken only when there is something to correct, in English
  first and Portuguese after, and never at the cost of failing the turn.

Everything runs without a microphone, a GPU or llama-server: the session is
built directly and its devices are replaced by fakes, exactly as in
``test_failures``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from tutor.config import load_all
from tutor.core.feedback import Feedback, FeedbackRequest, FeedbackResult
from tutor.core.prompts import PromptError, load_clarify
from tutor.core.session import Session
from tutor.stt import Transcript

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
LEVELS = ROOT / "levels.yaml"

ANSWER = "I went to the market yesterday."


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #

class _FakePlayback:
    def __init__(self) -> None:
        self.started_at = None
        self.fed = 0

    def stop(self) -> None:
        return None

    def end_of_stream(self) -> None:
        return None

    def feed(self, chunk: object) -> None:
        self.fed += 1

    async def wait_until_done(self) -> None:
        return None


class _FakeTTS:
    """Records every phrase with the pipeline it was asked for."""

    sample_rate = 24000

    def __init__(self) -> None:
        self.spoken: list[tuple[str, str]] = []
        self.fail_on: set[str] = set()
        #: Raise on every call after this one, so the answer can play and the
        #: correction fail (the reverse of a device that is dead from the start).
        self.fail_after: int | None = None
        self.calls = 0

    async def synthesize_stream(
        self, text: str, *, speed: float = 1.0, lang: str = "en", voice: str | None = None
    ):
        self.calls += 1
        if text in self.fail_on or (
            self.fail_after is not None and self.calls > self.fail_after
        ):
            raise RuntimeError("kokoro explodiu")
        self.spoken.append((text, lang))
        if False:  # pragma: no cover - makes this an async generator
            yield np.zeros(0, dtype=np.float32)


class _FakeASR:
    def __init__(self, transcript: Transcript | None = None) -> None:
        self.transcript = transcript or Transcript("hello", avg_logprob=-0.4)
        self.prompts: list[str | None] = []
        self.hotwords: list[str | None] = []

    async def transcribe(
        self, audio: object, *, prompt: str | None = None, hotwords: str | None = None
    ) -> Transcript:
        self.prompts.append(prompt)
        self.hotwords.append(hotwords)
        return self.transcript


class _FakeLLM:
    """One turn, one answer, no network."""

    async def stream(self, messages, *, max_tokens, temperature=None):
        if False:  # pragma: no cover
            yield ""
        yield ANSWER

    async def complete(self, messages, *, max_tokens, json_mode=False) -> str:
        return ANSWER


class _FakeFeedback:
    """Hands the correction over the instant the turn submits it (or never)."""

    def __init__(
        self,
        session: Session,
        feedback: Feedback | None = None,
        *,
        delay_s: float = 0.0,
        never: bool = False,
    ) -> None:
        self.session = session
        self.feedback = feedback
        self.delay_s = delay_s
        self.never = never
        self.requests: list[FeedbackRequest] = []

    def submit(self, request: FeedbackRequest) -> None:
        self.requests.append(request)
        if self.never or self.feedback is None:
            return
        if self.delay_s:
            self._run_later(request)
            return
        self.session._on_feedback(FeedbackResult(request=request, feedback=self.feedback))

    def _run_later(self, request: FeedbackRequest) -> None:
        async def later() -> None:
            await asyncio.sleep(self.delay_s)
            assert self.feedback is not None
            self.session._on_feedback(
                FeedbackResult(request=request, feedback=self.feedback)
            )

        asyncio.get_running_loop().create_task(later())


def _correction(**overrides: object) -> Feedback:
    payload: dict = {
        "corrected": ANSWER,
        "errors": [
            {
                "original": "I go to the market",
                "fix": ANSWER,
                "type": "grammar",
                "explanation_pt": "Passado simples: 'went', nao 'go'.",
            }
        ],
        "better_phrasing": None,
        "spoken_pt": "Boa! O correto e 'went', passado de 'go'.",
        "score": 6,
    }
    payload.update(overrides)
    return Feedback.model_validate_json(json.dumps(payload, ensure_ascii=False))


def _session(
    tmp_path: Path,
    *,
    level: str = "beginner",
    transcript: Transcript | None = None,
    feedback: Feedback | None = None,
    never_feedback: bool = False,
    speak: bool = True,
    wait_s: float = 0.05,
) -> tuple[Session, _FakeTTS, _FakeASR]:
    cfg, levels = load_all(CONFIG, LEVELS)
    cfg.storage.db_path = str(tmp_path / "tutor.db")
    cfg.pipeline.speak_feedback = speak
    cfg.pipeline.speak_feedback_wait_s = wait_s
    session = Session(
        cfg,
        level_name=level,
        level=levels[level],
        scenario="restaurant",
        listen=False,
        feedback=True,
    )
    session.playback = _FakePlayback()  # type: ignore[assignment]
    tts = _FakeTTS()
    session.tts = tts  # type: ignore[assignment]
    asr = _FakeASR(transcript)
    session.asr = asr  # type: ignore[assignment]
    session.llm = _FakeLLM()  # type: ignore[assignment]
    session.feedback = _FakeFeedback(  # type: ignore[assignment]
        session, feedback, never=never_feedback
    )
    return session, tts, asr


def _close(session: Session) -> None:
    if session.db is not None:
        session.db.close()
        session.db = None


AUDIO = b"\x01" * 1600


# --------------------------------------------------------------------------- #
# low-confidence gate
# --------------------------------------------------------------------------- #

def test_a_guessed_transcript_never_reaches_the_llm(tmp_path: Path) -> None:
    guessed = Transcript("i want eat", avg_logprob=-2.4, compression_ratio=3.1)
    session, tts, _ = _session(tmp_path, transcript=guessed)

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        result = asyncio.run(scenario())
    finally:
        _close(session)

    assert result is None
    assert session.dialogue.turns == 0
    assert session.dialogue.history == []
    # The pre-written line, not the model's guess.
    assert tts.spoken[0][0] == load_clarify().en
    assert tts.spoken[0][1] == "en"


def test_a_confident_transcript_is_left_alone(tmp_path: Path) -> None:
    session, tts, asr = _session(tmp_path, transcript=Transcript("i want eat", avg_logprob=-0.4))

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        result = asyncio.run(scenario())
    finally:
        _close(session)

    assert result is not None
    assert session.dialogue.turns == 1
    assert [text for text, _ in tts.spoken] == [ANSWER]
    assert asr.prompts == [None]  # no dialogue yet, nothing to hint with


def test_repetition_is_english_only_when_portuguese_help_is_off(tmp_path: Path) -> None:
    guessed = Transcript("nada", avg_logprob=-3.0)
    session, tts, _ = _session(tmp_path, level="intermediate", transcript=guessed)

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        asyncio.run(scenario())
    finally:
        _close(session)

    assert [lang for _, lang in tts.spoken] == ["en"]
    assert tts.spoken[0][0] == load_clarify().en


def test_repetition_adds_portuguese_when_the_level_allows_it(tmp_path: Path) -> None:
    guessed = Transcript("nada", avg_logprob=-3.0)
    session, tts, _ = _session(tmp_path, transcript=guessed)

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        asyncio.run(scenario())
    finally:
        _close(session)

    assert [lang for _, lang in tts.spoken] == ["en", "pt-br"]
    assert tts.spoken[1][0] == load_clarify().pt


def test_a_tts_failure_while_repeating_does_not_raise(tmp_path: Path) -> None:
    guessed = Transcript("nada", avg_logprob=-3.0)
    session, tts, _ = _session(tmp_path, transcript=guessed)
    tts.fail_on.add(load_clarify().en)

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        assert asyncio.run(scenario()) is None
    finally:
        _close(session)


# --------------------------------------------------------------------------- #
# the context hint
# --------------------------------------------------------------------------- #

def test_the_asr_prompt_carries_the_recent_dialogue(tmp_path: Path) -> None:
    session, _, asr = _session(tmp_path)

    async def scenario() -> None:
        await session.turn(AUDIO)

    try:
        assert session._asr_prompt() is None  # nothing has been said yet
        asyncio.run(scenario())
        assert session._asr_prompt() is not None  # turn 1 is on record now
        asyncio.run(scenario())
    finally:
        _close(session)

    hints = [p for p in asr.prompts if p]
    assert len(hints) == 1  # only the second turn had anything to hint with
    assert "hello" in hints[0]
    assert "I went to the market" in hints[0]
    assert len(hints[0]) <= 400


def test_the_context_hint_can_be_turned_off(tmp_path: Path) -> None:
    session, _, asr = _session(tmp_path)
    session.cfg.stt.use_dialogue_context = False

    async def scenario() -> None:
        await session.turn(AUDIO)
        await session.turn(AUDIO)

    try:
        asyncio.run(scenario())
    finally:
        _close(session)

    assert asr.prompts == [None, None]


def test_hotwords_from_the_config_reach_the_recogniser(tmp_path: Path) -> None:
    session, _, asr = _session(tmp_path)
    session.cfg.stt.hotwords = "waiter, table, menu"

    async def scenario() -> None:
        await session.turn(AUDIO)

    try:
        asyncio.run(scenario())
    finally:
        _close(session)

    assert asr.hotwords == ["waiter, table, menu"]


# --------------------------------------------------------------------------- #
# spoken correction
# --------------------------------------------------------------------------- #

def test_a_mistake_is_read_in_english_then_portuguese(tmp_path: Path) -> None:
    session, tts, _ = _session(tmp_path, feedback=_correction())

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        result = asyncio.run(scenario())
    finally:
        _close(session)

    assert result is not None
    # [0] is the answer itself; the correction follows it.
    assert tts.spoken[0] == (ANSWER, "en")
    assert tts.spoken[1:] == [
        (ANSWER, "en"),
        ("Boa! O correto e 'went', passado de 'go'.", "pt-br"),
    ]


def test_a_clean_sentence_is_not_spoken_back(tmp_path: Path) -> None:
    session, tts, _ = _session(tmp_path, feedback=_correction(errors=[]))

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        asyncio.run(scenario())
    finally:
        _close(session)

    assert tts.spoken == [(ANSWER, "en")]


def test_the_spoken_line_falls_back_to_the_written_explanation(tmp_path: Path) -> None:
    fb = _correction(spoken_pt=None)
    assert fb.spoken_correction_pt == "Passado simples: 'went', nao 'go'."
    session, tts, _ = _session(tmp_path, feedback=fb)

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        asyncio.run(scenario())
    finally:
        _close(session)

    assert tts.spoken[1:] == [
        (ANSWER, "en"),
        ("Passado simples: 'went', nao 'go'.", "pt-br"),
    ]


def test_a_correction_that_arrives_while_waiting_is_still_spoken(tmp_path: Path) -> None:
    """Regression: the waiter used to take the slot out before the worker could
    fill it, so every real (async) correction timed out and stayed silent."""
    session, tts, _ = _session(tmp_path, feedback=_correction(), wait_s=2.0)
    session.feedback.delay_s = 0.05  # type: ignore[union-attr]

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        assert asyncio.run(scenario()) is not None
    finally:
        _close(session)

    assert tts.spoken[1:] == [
        (ANSWER, "en"),
        ("Boa! O correto e 'went', passado de 'go'.", "pt-br"),
    ]


def test_a_correction_that_never_arrives_does_not_block_the_turn(tmp_path: Path) -> None:
    session, tts, _ = _session(tmp_path, feedback=_correction(), never_feedback=True)

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        result = asyncio.run(scenario())
    finally:
        _close(session)

    assert result is not None
    assert tts.spoken == [(ANSWER, "en")]


def test_a_tts_failure_while_correcting_is_a_warning_not_a_crash(tmp_path: Path) -> None:
    session, tts, _ = _session(tmp_path, feedback=_correction())
    tts.fail_after = 1  # the answer plays, the correction does not

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        assert asyncio.run(scenario()) is not None
    finally:
        _close(session)

    assert tts.spoken == [(ANSWER, "en")]


def test_the_flag_turns_the_spoken_correction_off(tmp_path: Path) -> None:
    session, tts, _ = _session(tmp_path, feedback=_correction(), speak=False)

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        assert asyncio.run(scenario()) is not None
    finally:
        _close(session)

    assert session.speak_feedback is False
    assert tts.spoken == [(ANSWER, "en")]


def test_the_correction_is_computed_whether_or_not_it_is_spoken(
    tmp_path: Path,
) -> None:
    session, tts, _ = _session(tmp_path, feedback=_correction())

    async def scenario() -> object:
        return await session.turn(AUDIO)

    try:
        asyncio.run(scenario())
    finally:
        _close(session)

    assert session.feedback is not None
    assert session.feedback.requests  # computed either way
    assert len(tts.spoken) == 3  # answer + english + portuguese


# --------------------------------------------------------------------------- #
# prompt asset
# --------------------------------------------------------------------------- #

def test_clarify_file_exists_with_both_lines() -> None:
    phrase = load_clarify()
    assert phrase.en and phrase.pt
    assert phrase.en != phrase.pt


def test_a_missing_clarify_label_is_an_error_not_a_silence(tmp_path, monkeypatch) -> None:
    broken = tmp_path / "clarify.md"
    broken.write_text("EN: sorry\nsem rotulo PT\n", encoding="utf-8")
    import tutor.core.prompts as prompts_mod

    monkeypatch.setattr(prompts_mod, "CLARIFY_PATH", broken)
    with pytest.raises(PromptError, match="PT"):
        prompts_mod.load_clarify()
