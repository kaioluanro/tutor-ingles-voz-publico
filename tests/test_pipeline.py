"""Tests for the two turn pipelines (Phase 6).

The fakes are deliberately timing-aware: the LLM yields tokens on a schedule
and the fake playback stamps ``started_at`` the way PortAudio's callback would,
so these tests can assert the one property the whole phase exists for —
**streaming starts talking before the model has finished writing**.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

import pytest

from tutor.core.pipeline import sequential_turn, streaming_turn
from tutor.llm.base import Message
from tutor.timeline import TurnTimeline

ANSWER = "Hello there my friend. How are you today?"
# Chosen so the first sentence is long enough to leave the chunker at min_words=3.
FIRST_SENTENCE = "Hello there my friend."
SECOND_SENTENCE = "How are you today?"


class FakeLLM:
    """Yields ``ANSWER`` word by word, pausing ``token_delay_s`` per word."""

    def __init__(
        self,
        text: str = ANSWER,
        *,
        token_delay_s: float = 0.004,
        fail_after: int | None = None,
    ) -> None:
        self.text = text
        self.token_delay_s = token_delay_s
        self.fail_after = fail_after
        self.streamed = 0

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float | None = None,
    ):
        words = self.text.split(" ")
        for i, word in enumerate(words):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("llm explodiu")
            if self.token_delay_s:
                await asyncio.sleep(self.token_delay_s)
            self.streamed += 1
            # The last token carries no trailing space, exactly like a real
            # answer, so it only leaves the chunker on flush().
            yield word if i == len(words) - 1 else word + " "

    async def complete(self, messages: list[Message], *, max_tokens: int, json_mode: bool = False) -> str:
        return self.text


class FakeTTS:
    """Records what it was asked to synthesise and yields one chunk per call."""

    sample_rate = 24000

    def __init__(self, *, delay_s: float = 0.01, fail_on: int | None = None) -> None:
        self.requested: list[str] = []
        self.delay_s = delay_s
        self.fail_on = fail_on

    async def synthesize_stream(self, text: str, *, speed: float = 1.0, lang: str = "en"):
        index = len(self.requested)
        self.requested.append(text)
        if self.fail_on is not None and index >= self.fail_on:
            raise RuntimeError("tts explodiu")
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        # Shorter than one RMS frame, so the pipeline's silence trim leaves it
        # exactly as it is; these tests are about ordering, not DSP.
        yield np.full(24, 0.5, dtype=np.float32)


class FakePlayback:
    """Mimics Playback's contract without touching a sound device."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.started_at: float | None = None
        self.stopped = False
        self.ended = False
        self._done = asyncio.Event()

    def feed(self, chunk) -> None:
        self.chunks.append(chunk)
        if self.started_at is None:
            # The real callback stamps with the audio clock; perf_counter is
            # the same clock TurnTimeline uses.
            self.started_at = time.perf_counter()

    def end_of_stream(self) -> None:
        self.ended = True
        self._done.set()

    def stop(self) -> None:
        self.stopped = True
        self.chunks.clear()
        self.ended = False
        self._done.set()

    async def wait_until_done(self) -> None:
        # The device drains after the last chunk lands.
        await self._done.wait()
        await asyncio.sleep(0.005)


def _timeline() -> TurnTimeline:
    tl = TurnTimeline(turn=1)
    tl.mark("t_speech_end")
    return tl


def _run(coro) -> TurnTimeline:
    return asyncio.run(coro)


def _messages() -> list[Message]:
    return [{"role": "user", "content": "hi"}]


# -- streaming: the point of the phase -------------------------------------- #


def test_streaming_starts_playback_before_the_llm_finishes() -> None:
    async def scenario() -> tuple[TurnTimeline, FakePlayback]:
        tl, pb = _timeline(), FakePlayback()
        answer = await streaming_turn(
            llm=FakeLLM(),
            messages=_messages(),
            max_tokens=80,
            tts=FakeTTS(),
            speed=1.0,
            playback=pb,
            tl=tl,
            sentence_queue_max=2,
            min_words=3,
        )
        return tl, answer, pb

    tl, answer, pb = _run(scenario())
    assert answer == ANSWER
    assert tl.marks["t_playback_start"] < tl.marks["t_llm_done"]
    assert len(pb.chunks) == 2  # one per sentence


def test_streaming_emits_the_first_sentence_while_tokens_still_arrive() -> None:
    async def scenario() -> tuple[TurnTimeline, FakeTTS]:
        tl, tts = _timeline(), FakeTTS()
        await streaming_turn(
            llm=FakeLLM(),
            messages=_messages(),
            max_tokens=80,
            tts=tts,
            speed=1.0,
            playback=FakePlayback(),
            tl=tl,
            sentence_queue_max=2,
            min_words=3,
        )
        return tl, tts

    tl, tts = _run(scenario())
    # The first sentence leaves the chunker before the model is done, so TTS
    # sees it as its own request instead of the whole answer.
    assert tts.requested == [FIRST_SENTENCE, SECOND_SENTENCE]
    assert tl.marks["t_first_sentence"] < tl.marks["t_llm_done"]


def test_streaming_marks_every_stage_of_section_9() -> None:
    async def scenario() -> TurnTimeline:
        tl = _timeline()
        await streaming_turn(
            llm=FakeLLM(),
            messages=_messages(),
            max_tokens=80,
            tts=FakeTTS(),
            speed=1.0,
            playback=FakePlayback(),
            tl=tl,
            sentence_queue_max=2,
            min_words=3,
        )
        return tl

    tl = _run(scenario())
    for name in (
        "t_llm_first_token",
        "t_first_sentence",
        "t_tts_first_audio",
        "t_playback_start",
        "t_llm_done",
    ):
        assert name in tl.marks, name
        assert tl.offset_ms(name) is not None


def test_streaming_empty_answer_still_closes_playback() -> None:
    async def scenario() -> FakePlayback:
        pb = FakePlayback()
        answer = await streaming_turn(
            llm=FakeLLM(text=""),
            messages=_messages(),
            max_tokens=80,
            tts=FakeTTS(),
            speed=1.0,
            playback=pb,
            tl=_timeline(),
            sentence_queue_max=2,
            min_words=3,
        )
        assert answer == ""
        return pb

    pb = _run(scenario())
    assert pb.ended is True
    assert pb.started_at is None  # nothing to play, so playback never started


def test_streaming_queue_size_is_respected() -> None:
    """A tiny queue must not deadlock: the worker keeps draining it."""

    async def scenario() -> int:
        pb = FakePlayback()
        await streaming_turn(
            llm=FakeLLM(text=" ".join(f"Sentence number {i} is here." for i in range(6))),
            messages=_messages(),
            max_tokens=80,
            tts=FakeTTS(delay_s=0.001),
            speed=1.0,
            playback=pb,
            tl=_timeline(),
            sentence_queue_max=1,
            min_words=3,
        )
        return len(pb.chunks)

    assert _run(scenario()) == 6


# -- failure handling ------------------------------------------------------- #


def test_llm_failure_cancels_the_turn_and_stops_playback() -> None:
    async def scenario() -> tuple[FakePlayback, list[str]]:
        pb = FakePlayback()
        with pytest.raises(RuntimeError, match="llm explodiu"):
            await streaming_turn(
                llm=FakeLLM(fail_after=2),
                messages=_messages(),
                max_tokens=80,
                tts=FakeTTS(),
                speed=1.0,
                playback=pb,
                tl=_timeline(),
                sentence_queue_max=1,
                min_words=3,
            )
        return pb, []

    pb, _ = _run(scenario())
    assert pb.stopped is True


def test_tts_failure_skips_the_sentence_and_keeps_the_turn(capsys) -> None:
    """A sentence that will not synthesise is dropped, never escalated.

    The answer is already in the dialogue history and on the timeline; losing
    a turn (and the correction queued with it) over a Kokoro hiccup would be a
    worse trade than one silent sentence.
    """

    async def scenario() -> tuple[TurnTimeline, str]:
        tl = _timeline()
        answer = await streaming_turn(
            llm=FakeLLM(),
            messages=_messages(),
            max_tokens=80,
            tts=FakeTTS(fail_on=0),
            speed=1.0,
            playback=FakePlayback(),
            tl=tl,
            sentence_queue_max=1,
            min_words=3,
        )
        return tl, answer

    tl, answer = _run(scenario())
    assert "t_llm_done" in tl.marks
    assert answer
    assert "tts explodiu" in capsys.readouterr().err


def test_cancelled_turn_stops_playback_and_leaves_it_clean() -> None:
    """Cancelling mid-stream must not leave audio queued for the next turn."""

    async def scenario() -> FakePlayback:
        pb = FakePlayback()

        class HangingLLM:
            async def stream(self, messages, *, max_tokens, temperature=None):
                for _ in range(500):
                    await asyncio.sleep(0.002)
                    yield "word "

            async def complete(self, messages, *, max_tokens, json_mode=False):
                return ""

        task = asyncio.create_task(
            streaming_turn(
                llm=HangingLLM(),
                messages=_messages(),
                max_tokens=80,
                tts=FakeTTS(delay_s=10.0),
                speed=1.0,
                playback=pb,
                tl=_timeline(),
                sentence_queue_max=1,
                min_words=1,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pb

    pb = _run(scenario())
    assert pb.stopped is True
    assert pb.chunks == []


# -- sequential baseline ---------------------------------------------------- #


def test_sequential_waits_for_the_whole_answer() -> None:
    async def scenario() -> tuple[TurnTimeline, FakeTTS]:
        tl, tts = _timeline(), FakeTTS()
        answer = await sequential_turn(
            llm=FakeLLM(),
            messages=_messages(),
            max_tokens=80,
            tts=tts,
            speed=1.0,
            playback=FakePlayback(),
            tl=tl,
        )
        return tl, tts

    tl, tts = _run(scenario())
    assert tl.marks["t_playback_start"] > tl.marks["t_llm_done"]
    # The baseline hands the whole answer over in one piece.
    assert tts.requested == [ANSWER]


def test_sequential_marks_every_stage_of_section_9() -> None:
    async def scenario() -> TurnTimeline:
        tl = _timeline()
        await sequential_turn(
            llm=FakeLLM(),
            messages=_messages(),
            max_tokens=80,
            tts=FakeTTS(),
            speed=1.0,
            playback=FakePlayback(),
            tl=tl,
        )
        return tl

    tl = _run(scenario())
    for name in (
        "t_llm_first_token",
        "t_llm_done",
        "t_first_sentence",
        "t_tts_first_audio",
        "t_playback_start",
    ):
        assert name in tl.marks, name


def test_sequential_keeps_the_answer_when_the_tts_raises(capsys) -> None:
    """Baseline parity: silence, a logged line and the text — not an error."""

    async def scenario() -> tuple[FakePlayback, str]:
        pb = FakePlayback()
        answer = await sequential_turn(
            llm=FakeLLM(),
            messages=_messages(),
            max_tokens=80,
            tts=FakeTTS(fail_on=0),
            speed=1.0,
            playback=pb,
            tl=_timeline(),
        )
        return pb, answer

    pb, answer = _run(scenario())
    assert pb.ended is True
    assert answer
    assert "tts explodiu" in capsys.readouterr().err
