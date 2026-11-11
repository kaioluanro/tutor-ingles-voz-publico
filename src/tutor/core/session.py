"""A conversation session: devices, models and one timeline per turn.

The same session runs the Phase 5 baseline or the Phase 6 streaming pipeline —
``streaming=False`` collects the whole answer before speaking, ``streaming=True``
runs the three concurrent tasks of :mod:`tutor.core.pipeline`. Everything else
(load once, listen with VAD, record, save, report) is identical, so the two
tables the benchmark prints stay comparable.

It also owns the opening turn: the AI speaks first, using the scenario's
``Opening line``, before any user speech exists.

The system prompt is assembled by :mod:`tutor.core.prompts` from
``prompts/base.md`` + ``levels/<nível>.md`` + ``scenarios/<cenário>.md``, and
``level.temperature`` rides along on every LLM call.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
from rich.console import Console

from tutor.audio.capture import AudioCapture
from tutor.audio.playback import Playback
from tutor.audio.silence import trim_silence
from tutor.audio.vad import TurnDetector
from tutor.config import AppConfig, LevelConfig, SttConfig
from tutor.core.dialogue import DialogueManager
from tutor.core.feedback import (
    Feedback,
    FeedbackRequest,
    FeedbackResult,
    FeedbackWorker,
)
from tutor.core.pipeline import render_text, sequential_turn, streaming_turn
from tutor.core.prompts import (
    ClarifyLines,
    PromptBundle,
    load_clarify,
    load_prompts,
    parse_new_expressions,
)
from tutor.errors import DeviceError, describe_device_failure
from tutor.llm.base import LLMClient, Message
from tutor.llm.errors import describe as describe_llm, is_llm_unavailable
from tutor.llm.health import wait_for_server
from tutor.llm.router import get_conversation_llm, get_feedback_llm
from tutor.stt import Transcript, WhisperSTT
from tutor.storage import Database
from tutor.timeline import TurnTimeline
from tutor.tts import KokoroTTS
from tutor.ui.terminal import TerminalUI

# A mic block can be silent for a long time without anything being wrong, but
# not hearing a single block at all means the device is gone.
SILENT_BLOCK_TIMEOUT_S = 10.0

#: How much of the dialogue goes back to Whisper as ``initial_prompt``. Whisper
#: only honours the prompt on its first decoding window, and a long one starts
#: bleeding into the transcript, so this is deliberately small.
ASR_PROMPT_CHARS = 400
#: Messages kept for that hint (the dialogue itself holds more).
ASR_PROMPT_MESSAGES = 4
#: ``turn -> Event`` entries kept when the caller never waits for them.
FEEDBACK_WAIT_SLOTS = 8


def _open_db(cfg: AppConfig) -> Database | None:
    """Open ``cfg.storage.db_path``, or ``None`` if the file cannot be created.

    Losing the progress log must never cost the learner a conversation, so a
    bad path degrades to "nothing is saved" instead of aborting the session.
    """
    try:
        return Database(cfg.storage.db_path)
    except Exception as exc:  # noqa: BLE001 - persistence is best effort
        print(
            f"[AVISO] banco indisponivel ({cfg.storage.db_path}): {exc}",
            file=sys.stderr,
        )
        return None


def _rollup_transcript(evicted: list[Message], *, previous: str | None = None) -> str:
    """Evicted pairs as plain lines, prefixed with the digest they extend.

    Shaped exactly like :meth:`Session._transcript`, so `prompts/rollup.md` can
    describe one format for both.
    """
    lines: list[str] = []
    if previous:
        lines.append(f"Earlier summary: {previous}")
    for message in evicted:
        who = "Learner" if message["role"] == "user" else "Tutor"
        lines.append(f"{who}: {message['content']}")
    return "\n".join(lines)


class Session:
    """Run a conversation and record a timeline per turn."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        level_name: str,
        level: LevelConfig,
        scenario: str | None = None,
        system: str | None = None,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
        console: Console | None = None,
        ui: TerminalUI | None = None,
        save_dir: Path | None = None,
        listen: bool = True,
        streaming: bool = False,
        feedback: bool = False,
        hide_transcript: bool = False,
    ) -> None:
        self.cfg = cfg
        self.level_name = level_name
        self.level = level
        self.save_dir = save_dir
        self.listen_to_mic = listen
        self.streaming = streaming
        # Prompt files are read every time, so editing one needs no restart.
        self.prompts: PromptBundle = load_prompts(level=level_name, scenario=scenario)
        # Every line the learner sees goes through the UI, so --hide-transcript
        # and the status panel have one place to live. The benchmark passes a
        # bare Console and never starts the panel.
        self.ui = ui or TerminalUI(
            console or Console(),
            level=level_name,
            scenario=self.prompts.scenario,
            accent=cfg.tts.accent,
            hide_transcript=hide_transcript,
        )
        self.console = self.ui.console
        # "Idle" = no turn in progress. Background feedback is gated on this
        # because llama-server handles one request at a time; see
        # tutor.core.feedback for why that matters.
        self.idle = asyncio.Event()
        self.idle.set()

        audio = cfg.audio
        self.detector = TurnDetector(
            sample_rate=audio.capture_sample_rate,
            threshold=audio.vad_threshold,
            chunk_samples=audio.vad_chunk,
            min_utterance_ms=audio.min_utterance_ms,
            max_utterance_s=audio.max_utterance_s,
            pre_roll_ms=audio.pre_roll_ms,
            end_silence_ms=level.end_silence_ms,
        )
        self.capture = AudioCapture(
            sample_rate=audio.capture_sample_rate,
            blocksize=audio.vad_chunk,
            device=audio.input_device if input_device is None else input_device,
        )
        self.playback = Playback(
            sample_rate=cfg.tts.sample_rate,
            device=audio.output_device if output_device is None else output_device,
            detector=self.detector,
            post_playback_mute_ms=audio.post_playback_mute_ms,
        )
        self.asr = WhisperSTT.from_config(cfg.stt)
        self.tts = KokoroTTS.from_config(cfg.tts)
        self.llm: LLMClient = get_conversation_llm(cfg)
        self.dialogue = DialogueManager(
            system=self.prompts.system if system is None else system,
            history_max_turns=cfg.pipeline.history_max_turns,
        )
        self._turn = 0
        # Every pair this session produced, untrimmed: the summary at the end
        # needs the whole conversation, while the model only sees a window.
        self._session_log: list[tuple[str, str]] = []
        # Phase 9: one row per session and per turn, for ``stats``/``review``.
        # ``_turn_row`` maps the timeline's turn number to its row id so a
        # correction arriving while the audio is still playing knows where to
        # write itself.
        self.db: Database | None = _open_db(cfg)
        self.session_id: int | None = None
        self._turn_row: dict[int, int] = {}
        self._summary_text: str | None = None
        self._feedback_llm: LLMClient | None = None
        self.feedback: FeedbackWorker | None = None
        # Rolling history (Phase 11): pairs evicted by ``history_max_turns``
        # wait here for the feedback LLM to condense them into ``set_rollup``.
        # One task at a time; a second eviction just extends the batch.
        self._rollup_batch: list[Message] = []
        self._rollup_task: asyncio.Task[None] | None = None
        # Phase 12: spoken correction. The result is computed while the answer
        # plays; the turn waits (bounded) for it and reads it out loud.
        self.speak_feedback: bool = bool(cfg.pipeline.speak_feedback and feedback)
        self._fb_events: dict[int, asyncio.Event] = {}
        self._fb_results: dict[int, Feedback | None] = {}
        # Clarified once per session: the two lines are static, the file is not.
        self._clarify: ClarifyLines | None = None
        if feedback:
            primary = get_feedback_llm(cfg)
            self._feedback_llm = primary
            # A hosted model never fights for the llama-server slot; the local
            # fallback only takes over while the system is idle.
            fallback = None if getattr(primary, "local_backend", True) else self.llm
            self.feedback = FeedbackWorker(
                llm=primary,
                idle=self.idle,
                timeout_s=cfg.pipeline.feedback_timeout_s,
                queue_size=cfg.pipeline.feedback_queue,
                fallback_llm=fallback,
                on_result=self._on_feedback,
                on_error=self._on_feedback_error,
            )

    # -- lifecycle --------------------------------------------------------- #

    async def __aenter__(self) -> "Session":
        loop = asyncio.get_running_loop()
        # Loading is measured once and reported, never paid inside a turn.
        await loop.run_in_executor(None, self.asr.load)
        await loop.run_in_executor(None, self.tts.load)
        await self.tts.warmup(speed=self.level.tts_speed)
        if self.speak_feedback:
            # Touching the pt-BR pipeline here also fetches its ~1 MB voice, so
            # a machine without network fails at startup instead of mid-turn.
            try:
                await self.tts.warmup(speed=self.level.tts_speed, lang="pt-br")
            except Exception as exc:  # noqa: BLE001 - spoken hint is optional
                self.speak_feedback = False
                self.ui.warn(f"correcao falada desativada: {exc}")
        # Open both devices up front so the first turn does not pay the cost of
        # creating the streams.
        self._open_devices()
        if self.feedback is not None:
            await self.feedback.start()
        if self.db is not None:
            self.session_id = self.db.create_session(
                level=self.level_name,
                scenario=self.prompts.scenario,
                accent=self.cfg.tts.accent,
            )
        self.ui.note(
            f"carregado: STT {self.asr.load_seconds:.2f}s  "
            f"TTS {self.tts.load_seconds:.2f}s"
        )
        return self

    def _open_devices(self) -> None:
        """Open both PortAudio streams, or name the one that is missing.

        A missing microphone is an environment problem, not a bug: it must
        reach the CLI as a sentence listing the devices, not as a traceback.
        """
        try:
            self.playback.open()
        except Exception as exc:
            raise DeviceError(describe_device_failure("output", exc)) from exc
        if not self.listen_to_mic:
            return
        try:
            self.capture.start()
        except Exception as exc:
            raise DeviceError(describe_device_failure("input", exc)) from exc

    async def __aexit__(self, *exc_info: object) -> None:
        try:
            await self._shutdown()
        finally:
            self._close_db()

    async def _shutdown(self) -> None:
        self._cancel_rollup()
        if self.feedback is not None:
            await self.feedback.close()
            if (
                self._feedback_llm is not None
                and self._feedback_llm is not self.llm
            ):
                aclose = getattr(self._feedback_llm, "aclose", None)
                if aclose is not None:
                    await aclose()
            self._feedback_llm = None
        await self.capture.stop()
        await self.playback.close()
        self.asr.close()
        self.tts.close()
        aclose = getattr(self.llm, "aclose", None)
        if aclose is not None:
            await aclose()

    def _close_db(self) -> None:
        """End the session row and hand the summary's phrases to the deck."""
        if self.db is None:
            return
        try:
            if self.session_id is not None:
                self.db.finish_session(self.session_id, summary=self._summary_text)
                if self._summary_text:
                    for phrase in parse_new_expressions(self._summary_text):
                        self.db.add_vocab(
                            expression=phrase, source_session_id=self.session_id
                        )
        except Exception as exc:  # noqa: BLE001 - persistence is best effort
            print(f"[AVISO] sessao nao gravada: {exc}", file=sys.stderr)
        finally:
            self.db.close()
            self.db = None

    # -- listening --------------------------------------------------------- #

    async def listen(self, *, timeout: float | None = None) -> np.ndarray:
        """Block until the VAD reports one complete utterance.

        ``timeout`` bounds the *whole* wait and raises :class:`TimeoutError`.
        It exists for replayed recordings: an utterance the VAD discards (too
        short, or never completed) would otherwise hang the benchmark forever.
        """
        self.ui.state("listening")
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while True:
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("a fala nao terminou dentro do prazo")
            wait = (
                SILENT_BLOCK_TIMEOUT_S
                if remaining is None
                else min(SILENT_BLOCK_TIMEOUT_S, remaining)
            )
            try:
                block = await asyncio.wait_for(self.capture.get(), timeout=wait)
            except TimeoutError:
                if remaining is not None:
                    raise TimeoutError("a fala nao terminou dentro do prazo") from None
                print(
                    "[AVISO] nenhum bloco de audio em "
                    f"{SILENT_BLOCK_TIMEOUT_S:.0f}s - microfone mudo?",
                    file=sys.stderr,
                )
                continue
            utterance = self.detector.process(block)
            if utterance is not None:
                self.ui.note("fala encerrada (VAD)")
                return utterance

    # -- opening turn ------------------------------------------------------ #

    async def greet(self, text: str | None = None) -> TurnTimeline:
        """Play turn 0: the AI opens the conversation with no user speech.

        The line is the scenario's ``Opening line``, so this turn never touches
        the LLM — the student hears the tutor before anything is generated.
        """
        text = text or self.prompts.greeting
        self.ui.state("speaking")
        tl = TurnTimeline(turn=0)
        # There is no user speech to end, so the conversation simply "starts"
        # now; latency for turn 0 therefore measures greeting playback.
        tl.mark("t_speech_end")
        tl.text_out = text
        seen_audio = False
        self.idle.clear()
        try:
            async for chunk in self.tts.synthesize_stream(
                text, speed=self.level.tts_speed
            ):
                if not seen_audio:
                    tl.mark("t_tts_first_audio")
                    seen_audio = True
                self.playback.feed(chunk)
            self.playback.end_of_stream()
            await self.playback.wait_until_done()
        except Exception as exc:  # noqa: BLE001 - opening line, not the session
            # A failed greeting would otherwise abort before turn 1; the learner
            # simply starts speaking instead.
            self.ui.warn(f"abertura nao pôde ser falada: {exc}")
        finally:
            self.playback.end_of_stream()
            self.idle.set()
        if self.playback.started_at is not None:
            tl.mark("t_playback_start", self.playback.started_at)
        self.dialogue.record_greeting(text)
        self.report(tl, title="latencia - abertura")
        return tl

    # -- one turn ---------------------------------------------------------- #

    async def turn(self, audio: np.ndarray) -> TurnTimeline | None:
        """Run one user turn, retrying once if the local backend is unreachable.

        A dead llama-server must not end the session. The first failure waits
        on ``/health`` (while the feedback gate stays closed, so the background
        worker cannot take the slot we are waiting for) and rebuilds the
        timeline, because a stream that died halfway may already have fed the
        playback. If the probe also fails, only **this** turn is dropped and
        the learner gets a sentence explaining it.
        """
        for attempt in range(2):
            tl = TurnTimeline(turn=self._turn + 1)
            tl.mark("t_speech_end")
            # Background feedback must not grab the single llama-server slot
            # while this turn is talking to the model, so the gate closes here
            # and only reopens from ``on_llm_done`` — one generation at a time.
            self.idle.clear()
            try:
                return await self._run_turn(audio, tl)
            except Exception as exc:
                if attempt or not is_llm_unavailable(exc):
                    raise
                # The playback may hold a fragment of the dead stream.
                self.playback.stop()
                self.playback.end_of_stream()
                self.ui.warn(
                    f"LLM indisponivel ({self.cfg.llm_local.base_url}): "
                    f"{describe_llm(exc)} - reconectando..."
                )
                waited = self.cfg.pipeline.reconnect_timeout_s
                if not await wait_for_server(
                    self.cfg.llm_local.base_url, timeout=waited
                ):
                    self.ui.warn(
                        f"LLM ainda fora do ar apos {waited:g}s - turno descartado."
                    )
                    return None
            finally:
                # No ``await`` between this and the next ``idle.clear()``, so
                # the feedback worker cannot slip into the slot during a retry.
                self.idle.set()
                self.ui.state("listening")
        return None

    async def _run_turn(self, audio: np.ndarray, tl: TurnTimeline) -> TurnTimeline | None:
        try:
            transcript = await self.asr.transcribe(
                audio,
                prompt=self._asr_prompt(),
                hotwords=self.cfg.stt.hotwords or None,
            )
        except Exception as exc:  # noqa: BLE001 - one bad buffer, one lost turn
            tl.mark("t_stt_done")
            self.ui.warn(f"transcricao falhou ({exc}) - turno descartado.")
            return None
        tl.mark("t_stt_done")
        text = transcript.text
        tl.text_in = text
        if not text.strip():
            self.ui.warn("transcricao vazia - turno descartado.")
            return None
        if not self._confident(transcript, self.cfg.stt):
            # A guess this weak would send the LLM a sentence the learner never
            # said, and the drift would live on in the dialogue history. One
            # cheap "sorry, could you repeat that" keeps both sides aligned.
            self.ui.warn(
                f"transcricao de baixa confianca "
                f"(logprob {transcript.avg_logprob:.2f}, "
                f"ratio {transcript.compression_ratio:.2f}) - pedindo repeticao."
            )
            await self._ask_to_repeat()
            return None
        self.ui.state("thinking")

        messages = self.dialogue.build(text)
        # The row exists *before* the model answers: ``on_llm_done`` queues the
        # correction while the audio is still playing, so it would otherwise
        # arrive looking for a turn that has not been written yet.
        turn_id = self._new_turn_row(tl.turn, text)
        # Queued from inside the pipeline, the instant the answer is complete,
        # so the correction owns the model for the whole playback instead of
        # the sliver between turns.
        def on_llm_done(answer: str) -> None:
            self.idle.set()
            if self.feedback is not None and answer.strip():
                # The slot must exist before the worker can finish, otherwise
                # a fast correction would find nobody waiting and be dropped.
                if self.speak_feedback:
                    self._arm_feedback(tl.turn)
                try:
                    self.feedback.submit(
                        FeedbackRequest(
                            user_text=text,
                            ai_text=answer,
                            level=self.level_name,
                            turn=tl.turn,
                        )
                    )
                except Exception:
                    self._complete_feedback(tl.turn, None)
                    raise

        if self.streaming:
            answer = await streaming_turn(
                llm=self.llm,
                messages=messages,
                max_tokens=self.level.max_tokens,
                tts=self.tts,
                speed=self.level.tts_speed,
                playback=self.playback,
                tl=tl,
                sentence_queue_max=self.cfg.pipeline.sentence_queue_max,
                min_words=self.cfg.pipeline.min_words_per_chunk,
                temperature=self.level.temperature,
                on_llm_done=on_llm_done,
                on_state=self.ui.state,
            )
        else:
            answer = await sequential_turn(
                llm=self.llm,
                messages=messages,
                max_tokens=self.level.max_tokens,
                tts=self.tts,
                speed=self.level.tts_speed,
                playback=self.playback,
                tl=tl,
                temperature=self.level.temperature,
                on_llm_done=on_llm_done,
                on_state=self.ui.state,
            )

        tl.text_out = answer
        evicted = self.dialogue.record(text, answer)
        if evicted:
            # Off the latency path: playback is already over and the learner
            # is reading the correction while this runs.
            self._submit_rollup(evicted)
        self._turn += 1
        self._session_log.append((text, answer))
        self._save(audio, tl.turn)
        if turn_id is not None:
            self.db.update_turn(turn_id, ai_text=answer, latency_ms=tl.latency_ms)
        if self.speak_feedback:
            await self._speak_correction(tl.turn)
        return tl

    # -- Phase 12: context guard and spoken correction --------------------- #

    def _asr_prompt(self) -> str | None:
        """A short hint of what the conversation is about, for Whisper.

        Whisper honours ``initial_prompt`` only on its first decoding window,
        so it is a *vocabular* hint, not memory: a few recent exchanges make it
        hear ``waiter`` instead of ``water``, and nothing more. Off when
        ``stt.use_dialogue_context`` is false, or when there is no history yet.
        """
        if not self.cfg.stt.use_dialogue_context:
            return None
        recent = list(self.dialogue.history[-ASR_PROMPT_MESSAGES:])
        if not recent:
            return None
        hint = " ".join(
            f"Learner: {m['content']}" if m["role"] == "user" else f"Tutor: {m['content']}"
            for m in recent
        )
        return hint[:ASR_PROMPT_CHARS] or None

    @staticmethod
    def _confident(transcript: Transcript, cfg: SttConfig) -> bool:
        """False when the transcript is a plausible guess rather than a hearing.

        A ``Transcript`` built without confidence numbers (defaults of ``0.0``)
        counts as confident: gate absence, not silence. ``no_speech_prob`` is
        deliberately not part of the test — it sits high on short clips like
        the fixtures, which would drop turns that were transcribed correctly.
        """
        if transcript.avg_logprob < cfg.min_avg_logprob:
            return False
        return transcript.compression_ratio <= cfg.max_compression_ratio

    def _clarify_lines(self) -> ClarifyLines:
        if self._clarify is None:
            self._clarify = load_clarify()
        return self._clarify

    async def _ask_to_repeat(self) -> None:
        """Say the pre-written 'sorry, please repeat' line, in the right mix.

        ``en`` alone when ``allow_portuguese_help`` is off: the Portuguese
        half would answer a beginner with the very thing the level forbids.
        """
        phrase = self._clarify_lines()
        pairs: list[tuple[str, str]] = [(phrase.en, "en")]
        if self.level.allow_portuguese_help and phrase.pt:
            pairs.append((phrase.pt, "pt-br"))
        await self._speak_lines(pairs)

    async def _speak_lines(self, lines: list[tuple[str, str]]) -> None:
        """Buffer, silence-trim and play each ``(text, lang)`` in order.

        The lines are queued back to back and the stream is closed once, so
        there is no dead air between the English and the Portuguese half. It
        never raises into a turn: a TTS or device failure logs a warning and
        stops the sequence where it is.
        """
        self.ui.state("speaking")
        played = False
        try:
            for text, lang in lines:
                if not text.strip():
                    continue
                audio = await render_text(
                    self.tts, text, speed=self.level.tts_speed, lang=lang
                )
                if audio is None:
                    continue
                # Kokoro pads every call; without this the gap between the two
                # halves of a correction grows by up to 890 ms of quiet.
                self.playback.feed(trim_silence(audio, self.tts.sample_rate))
                played = True
            if played:
                self.playback.end_of_stream()
                await self.playback.wait_until_done()
        except Exception as exc:  # noqa: BLE001 - spoken hint is best effort
            self.playback.end_of_stream()
            self.ui.warn(f"correcao falada falhou: {exc}")

    def _arm_feedback(self, turn: int) -> None:
        """Open the slot this turn will wait on, from inside the event loop."""
        while len(self._fb_events) >= FEEDBACK_WAIT_SLOTS:
            stale = next(iter(self._fb_events))
            self._fb_events.pop(stale, None)
            self._fb_results.pop(stale, None)
        self._fb_events.setdefault(turn, asyncio.Event())

    def _complete_feedback(self, turn: int, fb: Feedback | None) -> None:
        """Hand the result to a waiting turn; a late one is simply dropped.

        The event must still be in the dict: the waiter only removes it once it
        is done, so a correction that arrives while nobody waits is discarded
        instead of piling up in ``_fb_results``.
        """
        event = self._fb_events.get(turn)
        if event is None:
            return
        self._fb_results[turn] = fb
        event.set()

    async def _wait_feedback(self, turn: int) -> Feedback | None:
        """Wait ``speak_feedback_wait_s`` for this turn's correction, or give up.

        The slot is taken out only after the wait, because ``_complete_feedback``
        needs to find it: popping it first would leave the arriving correction
        with nobody to wake up, and the turn would always time out.
        """
        event = self._fb_events.get(turn)
        if event is None:
            return None
        result: Feedback | None = None
        try:
            await asyncio.wait_for(event.wait(), self.cfg.pipeline.speak_feedback_wait_s)
            result = self._fb_results.get(turn)
        except TimeoutError:
            result = None
        finally:
            self._fb_events.pop(turn, None)
            self._fb_results.pop(turn, None)
        return result

    async def _speak_correction(self, turn: int) -> None:
        """Read the correction aloud, once the answer has finished playing.

        Only mistakes are spoken. A clean sentence would mean hearing your own
        words back for nothing, and the corrected form is already on screen.
        """
        fb = await self._wait_feedback(turn)
        if fb is None or not fb.errors:
            return
        await self._speak_lines(
            [(fb.corrected.strip(), "en"), (fb.spoken_correction_pt, "pt-br")]
        )

    # -- persistence (Phase 9) ---------------------------------------------- #

    def _new_turn_row(self, turn: int, text: str) -> int | None:
        """Write the turn as soon as we know what was said. ``None`` if off."""
        if self.db is None or self.session_id is None:
            return None
        try:
            row_id = self.db.add_turn(
                session_id=self.session_id, idx=turn, user_text=text
            )
        except Exception as exc:  # noqa: BLE001 - persistence is best effort
            print(f"[AVISO] turno nao gravado: {exc}", file=sys.stderr)
            return None
        self._turn_row[turn] = row_id
        return row_id

    # -- driving ----------------------------------------------------------- #

    async def run(
        self, n_turns: int, out: list[TurnTimeline] | None = None
    ) -> list[TurnTimeline]:
        """Hold ``n_turns`` accepted conversations and return their timelines.

        Passing ``out`` lets a caller keep the turns completed so far when the
        session is interrupted by Ctrl+C.
        """
        timelines = out if out is not None else []
        while len(timelines) < n_turns:
            utterance = await self.listen()
            tl = await self.turn(utterance)
            if tl is None:
                continue
            timelines.append(tl)
            self.report(tl)
        return timelines

    def report(self, tl: TurnTimeline, *, title: str | None = None) -> None:
        """Print the transcript and this turn's row straight away."""
        self.ui.report(tl, title=title)

    # -- background correction --------------------------------------------- #

    def _on_feedback(self, result: FeedbackResult) -> None:
        """Show the correction and, if this turn asked for it, release it.

        The UI panel is the destination of record; the spoken line is an extra
        that only a turn waiting on :meth:`_speak_correction` will consume.
        """
        self.ui.feedback(result)
        self._store_feedback(result.request.turn, result.feedback)
        self._complete_feedback(result.request.turn, result.feedback)

    def _on_feedback_error(self, request: FeedbackRequest, reason: str) -> None:
        """Nothing to say — but the waiting turn must not spend its full wait."""
        self._complete_feedback(request.turn, None)

    def _store_feedback(self, turn: int, fb: Feedback) -> None:
        """Persist the score and each correction behind the panel."""
        if self.db is None:
            return
        row_id = self._turn_row.get(turn)
        if row_id is None:
            return
        try:
            self.db.set_turn_score(row_id, fb.score)
            for err in fb.errors:
                self.db.add_error(
                    turn_id=row_id,
                    type=err.type,
                    original=err.original,
                    fix=err.fix,
                    explanation_pt=err.explanation_pt,
                )
        except Exception as exc:  # noqa: BLE001 - persistence is best effort
            print(f"[AVISO] correcao nao gravada: {exc}", file=sys.stderr)

    # -- rolling history (Phase 11) --------------------------------------- #

    def _submit_rollup(self, evicted: list[Message]) -> None:
        """Ask the feedback LLM to condense pairs leaving the window.

        Never awaited by the turn: the conversation is already speaking/reading
        its correction, and a backend that is down only costs the old context
        (the pairs were dropped either way).
        """
        if self.feedback is None:
            return
        self._rollup_batch.extend(evicted)
        if self._rollup_task is None or self._rollup_task.done():
            self._rollup_task = asyncio.create_task(
                self._rollup_worker(), name="history-rollup"
            )

    async def _rollup_worker(self) -> None:
        """Drain every batch collected while the previous one was in flight."""
        assert self.feedback is not None
        while self._rollup_batch:
            batch, self._rollup_batch = self._rollup_batch, []
            summary = await self.feedback.rollup(
                _rollup_transcript(batch, previous=self.dialogue.rollup)
            )
            if summary:
                self.dialogue.set_rollup(summary)

    def _cancel_rollup(self) -> None:
        """Drop any pending digest; it only matters for future turns."""
        task, self._rollup_task = self._rollup_task, None
        self._rollup_batch.clear()
        if task is None or task.done():
            return
        task.cancel()

    async def summarize(self) -> str | None:
        """Close the session with `prompts/summary.md`. ``None`` when off/fails.

        The text is stored for SQLite and returned; **printing is the caller's
        job**, because the terminal wants it after the status panel has been
        taken down and the benchmark wants it inside its own layout.
        """
        if self.feedback is None:
            return None
        # The closing summary owns the backend; a rollup still in flight would
        # either queue behind it or steal the slot we are about to ask for.
        self._cancel_rollup()
        await self.feedback.wait_idle()
        text = await self.feedback.summarize(
            self._transcript(), level=self.level_name
        )
        if text:
            self._summary_text = text
        return text

    def _transcript(self) -> str:
        """Whole session as plain lines, for the summary prompt."""
        lines: list[str] = []
        greeting = self.dialogue.greeting
        if greeting:
            lines.append(f"Tutor: {greeting}")
        for index, (user_text, answer) in enumerate(self._session_log, start=1):
            lines.append(f"[{index}] Learner: {user_text}")
            lines.append(f"[{index}] Tutor: {answer}")
        return "\n".join(lines)

    def _save(self, audio: np.ndarray, turn: int) -> None:
        if self.save_dir is None:
            return
        try:
            import soundfile as sf

            self.save_dir.mkdir(parents=True, exist_ok=True)
            sf.write(
                self.save_dir / f"turn_{turn:02d}.wav",
                audio,
                self.cfg.audio.capture_sample_rate,
                subtype="PCM_16",
            )
        except Exception as exc:  # noqa: BLE001 - saving is best effort
            print(f"[AVISO] nao foi possivel salvar o wav: {exc}", file=sys.stderr)
