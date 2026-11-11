"""The two ways a turn can run, side by side.

``sequential_turn`` is the Phase 5 baseline: it waits for the whole LLM answer
before synthesising anything. ``streaming_turn`` is the product: three
``asyncio`` tasks run at once — the producer turns tokens into sentences, the
TTS worker turns sentences into audio, and the playback task waits for the
device — so the first sentence is already leaving the speakers while the model
is still writing the third.

Both fill the same :class:`~tutor.timeline.TurnTimeline`, so the benchmark
prints directly comparable numbers.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable

import numpy as np

from tutor.audio.playback import Playback
from tutor.audio.silence import trim_silence
from tutor.core.chunker import SentenceChunker
from tutor.llm.base import LLMClient, Message
from tutor.timeline import TurnTimeline
from tutor.tts import KokoroTTS


async def sequential_turn(
    *,
    llm: LLMClient,
    messages: list[Message],
    max_tokens: int,
    tts: KokoroTTS,
    speed: float,
    playback: Playback,
    tl: TurnTimeline,
    temperature: float | None = None,
    on_llm_done: Callable[[str], None] | None = None,
    on_state: Callable[[str], None] | None = None,
) -> str:
    """Collect the whole answer, synthesise it, then play it."""
    parts: list[str] = []
    seen_token = False
    async for token in llm.stream(
        messages, max_tokens=max_tokens, temperature=temperature
    ):
        if not seen_token:
            tl.mark("t_llm_first_token")
            seen_token = True
        parts.append(token)
    tl.mark("t_llm_done")
    answer = "".join(parts)
    if on_llm_done is not None:
        on_llm_done(answer)

    tl.mark("t_first_sentence")
    seen_audio = False
    try:
        async for chunk in tts.synthesize_stream(answer, speed=speed):
            if not seen_audio:
                tl.mark("t_tts_first_audio")
                seen_audio = True
                if on_state is not None:
                    on_state("speaking")
            playback.feed(chunk)
    except Exception as exc:  # noqa: BLE001 - the text survived, the voice did not
        # The answer is already committed to the timeline and to the dialogue
        # history; staying silent beats failing the turn over a TTS hiccup.
        print(f"[AVISO] resposta nao sintetizada: {exc}", file=sys.stderr)
    finally:
        playback.end_of_stream()

    await playback.wait_until_done()
    _stamp_playback(tl, playback)
    return answer


async def streaming_turn(
    *,
    llm: LLMClient,
    messages: list[Message],
    max_tokens: int,
    tts: KokoroTTS,
    speed: float,
    playback: Playback,
    tl: TurnTimeline,
    sentence_queue_max: int,
    min_words: int,
    temperature: float | None = None,
    on_llm_done: Callable[[str], None] | None = None,
    on_state: Callable[[str], None] | None = None,
) -> str:
    """Run producer, TTS and playback concurrently for one turn."""
    sentences: asyncio.Queue[str | None] = asyncio.Queue(maxsize=sentence_queue_max)
    chunker = SentenceChunker(min_words=min_words)
    parts: list[str] = []
    playback_done = asyncio.Event()

    async def producer() -> None:
        """LLM tokens -> sentences on the queue."""
        seen_token = False
        try:
            async for token in llm.stream(
                messages, max_tokens=max_tokens, temperature=temperature
            ):
                if not seen_token:
                    tl.mark("t_llm_first_token")
                    seen_token = True
                parts.append(token)
                for sentence in chunker.feed(token):
                    tl.mark("t_first_sentence")
                    await sentences.put(sentence)
            for sentence in chunker.flush():
                tl.mark("t_first_sentence")
                await sentences.put(sentence)
        finally:
            # No await here: this runs while the task may be being cancelled.
            tl.mark("t_llm_done")
        if on_llm_done is not None:
            # Success path only. The single llama-server slot is free from now
            # on, even though the audio is still leaving the speakers: handing
            # it straight to the background correction is what buys that worker
            # a whole playback of head start instead of the sliver of time
            # between turns, which is never enough for it to finish.
            on_llm_done("".join(parts))
        await sentences.put(None)  # end marker, success path only

    async def tts_worker() -> None:
        """Sentences -> audio chunks -> playback queue."""
        seen_audio = False
        try:
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    break
                try:
                    audio = await render_text(tts, sentence, speed=speed)
                except Exception as exc:  # noqa: BLE001 - one sentence, not a turn
                    print(
                        f"[AVISO] frase nao sintetizada "
                        f"({sentence.strip()[:48]!r}): {exc}",
                        file=sys.stderr,
                    )
                    continue
                if audio is None:
                    continue
                if not seen_audio:
                    tl.mark("t_tts_first_audio")
                    seen_audio = True
                    if on_state is not None:
                        on_state("speaking")
                # Kokoro pads every call; without this the boundary between two
                # sentences grows by up to 890 ms of audible quiet.
                playback.feed(trim_silence(audio, tts.sample_rate))
            playback.end_of_stream()
        except BaseException:
            playback.end_of_stream()
            raise
        finally:
            playback_done.set()

    async def playback_task() -> None:
        """Wait for the device to drain and record when sound really came out."""
        await playback_done.wait()
        await playback.wait_until_done()
        _stamp_playback(tl, playback)

    tasks = [
        asyncio.create_task(producer(), name="producer"),
        asyncio.create_task(tts_worker(), name="tts_worker"),
        asyncio.create_task(playback_task(), name="playback"),
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        _drain(sentences)
        chunker.reset()
        playback.stop()
        raise
    return "".join(parts)


def _stamp_playback(tl: TurnTimeline, playback: Playback) -> None:
    """Record the PortAudio callback's timestamp, not this coroutine's."""
    if playback.started_at is not None:
        tl.mark("t_playback_start", playback.started_at)


async def render_text(
    tts: KokoroTTS, text: str, *, speed: float, lang: str = "en"
) -> np.ndarray | None:
    """Synthesise one sentence into a single array, or ``None`` if it was empty.

    Buffering a sentence costs one round of its own synthesis (RTF ~0.02) and
    buys the silence trim, which has to see the whole fragment to know where
    its noise floor is. ``lang`` picks the Kokoro pipeline: English answers and
    the Portuguese correction come out of different pipelines, because Kokoro
    phonemises text with the rules of one language at a time.
    """
    pieces = []
    async for chunk in tts.synthesize_stream(text, speed=speed, lang=lang):
        pieces.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    if not pieces:
        return None
    return np.concatenate(pieces)


def _drain(queue: asyncio.Queue) -> None:
    """Empty a queue that a cancelled turn left behind."""
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
