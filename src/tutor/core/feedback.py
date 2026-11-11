"""Background correction feedback (Phase 8).

Nothing here sits on the latency critical path: the session drops a request on
a queue, and a consumer task picks it up whenever the system is *idle*. That
word matters — ``llama-server`` runs with ``-np 1`` and serves one request at a
time, and it cannot be preempted: cancelling the client frees nothing, the
server finishes the abandoned generation anyway. So the rule is simply that a
correction and a conversation never ask for the slot at the same time. The
:class:`FeedbackWorker` waits on an ``asyncio.Event`` the session holds cleared
until the conversation's own answer has been generated, and the session queues
the correction at exactly that moment — the worker then owns the model for the
whole playback, which is far longer than one correction takes.

The remote model is preferred when it is enabled (it never touches that slot at
all); when it fails, the local model takes over — but only once the system is
idle again, which is the same rule applied to the local backend.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tutor.core.prompts import (
    build_feedback_system,
    build_rollup_system,
    build_summary_system,
)
from tutor.llm.base import LLMClient, Message

#: Exactly the five values ``prompts/feedback.md`` is allowed to emit.
ERROR_TYPES = ("grammar", "vocab", "word_order", "preposition", "other")

#: The correction is a short JSON blob; anything past this is wasted tokens.
FEEDBACK_MAX_TOKENS = 400
#: The session summary is prose in four labelled parts.
SUMMARY_MAX_TOKENS = 500
#: The rolling digest of evicted history; `prompts/rollup.md` caps it at 80 words.
ROLLUP_MAX_TOKENS = 160

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


class FeedbackParseError(ValueError):
    """The model's reply is not a usable feedback object."""


class FeedbackError(BaseModel):
    """One mistake inside the learner's sentence."""

    model_config = ConfigDict(extra="ignore")

    original: str
    fix: str
    type: str
    explanation_pt: str

    @field_validator("type", mode="before")
    @classmethod
    def _known_type(cls, value: object) -> str:
        """Relabel an unknown category instead of discarding a good correction.

        The prompt pins the vocabulary to :data:`ERROR_TYPES`; a model that
        invents ``"spelling"`` still did the useful part of the job, and Fase 9
        charts by category — so unknown labels land in ``"other"``.
        """
        text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        return text if text in ERROR_TYPES else "other"


class Feedback(BaseModel):
    """The shape ``prompts/feedback.md`` is asked to produce.

    ``corrected`` and ``errors`` are required — the prompt promises
    ``"errors": []`` when the sentence is clean, so their absence means the
    reply was truncated or malformed rather than "no mistakes found". ``score``
    stays optional on purpose: it is advisory, and a fabricated 0 would poison
    the average shown by Fase 9's ``stats``. ``spoken_pt`` is optional for the
    same reason: it is one more string the model can drop, and the first
    error's ``explanation_pt`` says the same thing less naturally.
    """

    model_config = ConfigDict(extra="ignore")

    corrected: str
    errors: list[FeedbackError]
    better_phrasing: str | None = None
    spoken_pt: str | None = None
    score: int | None = Field(default=None, ge=0, le=10)

    @property
    def spoken_correction_pt(self) -> str:
        """What the tutor says in Portuguese after reading ``corrected``.

        ``spoken_pt`` is written for the ear; ``explanation_pt`` is the written
        fallback when the model skipped the spoken field.
        """
        text = (self.spoken_pt or "").strip()
        if text:
            return text
        return self.errors[0].explanation_pt.strip() if self.errors else ""


def strip_code_fences(text: str) -> str:
    """Drop a surrounding ```` ```json ```` pair if the model added one."""
    match = _FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def parse_feedback(raw: str) -> Feedback:
    """Parse a model reply, or raise :class:`FeedbackParseError`."""
    if not raw or not raw.strip():
        raise FeedbackParseError("resposta vazia do LLM")
    text = strip_code_fences(raw)
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FeedbackParseError(f"JSON invalido: {exc.msg} (pos {exc.pos})") from exc
    if not isinstance(data, dict):
        raise FeedbackParseError(f"esperado um objeto JSON, veio {type(data).__name__}")
    try:
        return Feedback.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        raise FeedbackParseError(f"campos invalidos: {problems}") from exc


@dataclass
class FeedbackRequest:
    """One learner sentence waiting to be corrected."""

    user_text: str
    ai_text: str
    level: str
    turn: int = 0


@dataclass
class FeedbackResult:
    """What came back, plus enough context for the UI and (later) SQLite."""

    request: FeedbackRequest
    feedback: Feedback
    attempts: int = 1
    backend: str = "local"


ResultHook = Callable[[FeedbackResult], None]
ErrorHook = Callable[[FeedbackRequest, str], None]


def _is_local(llm: LLMClient) -> bool:
    return bool(getattr(llm, "local_backend", True))


def _backend_name(llm: LLMClient) -> str:
    return "local" if _is_local(llm) else "remoto"


class FeedbackWorker:
    """Owns a queue and one consumer task that only calls the LLM when idle."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        idle: asyncio.Event,
        timeout_s: float = 15.0,
        queue_size: int = 4,
        max_retries: int = 1,
        fallback_llm: LLMClient | None = None,
        on_result: ResultHook | None = None,
        on_error: ErrorHook | None = None,
        max_tokens: int = FEEDBACK_MAX_TOKENS,
    ) -> None:
        self._llm = llm
        self._fallback_llm = fallback_llm
        self._idle = idle
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._max_tokens = max_tokens
        self._on_result = on_result
        self._on_error = on_error
        self._queue: asyncio.Queue[FeedbackRequest] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task[None] | None = None
        self._current: asyncio.Task[None] | None = None
        self._inflight_llm: LLMClient | None = None
        self._closing = False
        self.results: list[FeedbackResult] = []
        self.dropped = 0

    # -- lifecycle --------------------------------------------------------- #

    async def start(self) -> None:
        if self._task is None:
            self._closing = False
            self._task = asyncio.create_task(self._consume(), name="feedback")

    async def close(self) -> None:
        if self._task is None:
            return
        self._closing = True
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        self._current = None
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    # -- submission -------------------------------------------------------- #

    def submit(self, request: FeedbackRequest) -> bool:
        """Queue one correction. Never blocks; ``False`` means the queue was full."""
        if not request.user_text.strip():
            return False
        try:
            self._queue.put_nowait(request)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            print(
                f"[AVISO] fila de feedback cheia ({self.dropped} descartados) - "
                "o turno seguinte tem prioridade",
                file=sys.stderr,
            )
            return False

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def wait_idle(self) -> None:
        """Resolve once nothing is queued and nothing is being generated.

        Used before the session summary, so the closing text never races a
        pending correction for the single llama-server slot.
        """
        while True:
            await self._idle.wait()
            if self.pending == 0 and self._current is None:
                return
            await asyncio.sleep(0.05)

    # -- consumption ------------------------------------------------------- #

    async def _consume(self) -> None:
        while True:
            request = await self._queue.get()
            # Each request runs in its own task so a failure inside one never
            # takes down the consumer that drains the rest of the queue.
            current = asyncio.create_task(self._handle(request), name="feedback-one")
            self._current = current
            try:
                await current
            except asyncio.CancelledError:
                if self._closing:
                    raise
            except Exception as exc:  # noqa: BLE001 - one bad turn must not kill the worker
                self._report_error(request, f"{type(exc).__name__}: {exc}")
            finally:
                if self._current is current:
                    self._current = None

    async def _handle(self, request: FeedbackRequest) -> None:
        system = build_feedback_system(request.level)
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": _user_content(request)},
        ]

        attempts = 0
        last_error = ""
        transport_failed = False
        while attempts <= self._max_retries:
            attempts += 1
            try:
                feedback = parse_feedback(await self._call(self._llm, messages))
            except asyncio.CancelledError:
                raise
            except FeedbackParseError as exc:
                last_error = str(exc)  # worth one more try: small models wobble
            except TimeoutError as exc:
                last_error = f"timeout depois de {self._timeout_s:g}s: {exc}"
            except Exception as exc:  # noqa: BLE001 - transport/provider failure
                last_error = f"{type(exc).__name__}: {exc}"
                transport_failed = True
                break
            else:
                self._finish(
                    FeedbackResult(
                        request=request,
                        feedback=feedback,
                        attempts=attempts,
                        backend=_backend_name(self._llm),
                    )
                )
                return

        if transport_failed and self._fallback_llm is not None:
            # Same idle rule, now against the single local slot.
            try:
                feedback = parse_feedback(await self._call(self._fallback_llm, messages))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                self._finish(
                    FeedbackResult(
                        request=request,
                        feedback=feedback,
                        attempts=attempts + 1,
                        backend=_backend_name(self._fallback_llm),
                    )
                )
                return

        self._report_error(request, last_error or "sem detalhe")

    async def _call(self, llm: LLMClient, messages: list[Message]) -> str:
        """Ask one backend, honouring the single-request server's idle rule."""
        if _is_local(llm):
            await self._idle.wait()
        # Hosted providers understand response_format; llama.cpp is asked for
        # raw JSON and the parser strips a fence if one appears anyway.
        json_mode = not _is_local(llm)
        self._inflight_llm = llm
        try:
            return await asyncio.wait_for(
                llm.complete(messages, max_tokens=self._max_tokens, json_mode=json_mode),
                timeout=self._timeout_s,
            )
        finally:
            self._inflight_llm = None

    def _finish(self, result: FeedbackResult) -> None:
        self.results.append(result)
        if self._on_result is not None:
            self._on_result(result)

    def _report_error(self, request: FeedbackRequest, reason: str) -> None:
        print(
            f"[AVISO] feedback descartado (turno {request.turn}): {reason}",
            file=sys.stderr,
        )
        if self._on_error is not None:
            self._on_error(request, reason)

    # -- session summary --------------------------------------------------- #

    async def summarize(self, transcript: str, *, level: str) -> str | None:
        """Close the session with `prompts/summary.md`. ``None`` on any failure."""
        if not transcript.strip():
            return None
        messages: list[Message] = [
            {"role": "system", "content": build_summary_system()},
            {"role": "user", "content": f"LEVEL: {level}\n\n{transcript}"},
        ]
        try:
            text = await self._call_summary(messages)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[AVISO] resumo da sessao falhou: {exc}", file=sys.stderr)
            return None
        return text.strip() or None

    async def _call_summary(
        self, messages: list[Message], *, max_tokens: int = SUMMARY_MAX_TOKENS
    ) -> str:
        llm = self._llm
        if _is_local(llm):
            await self._idle.wait()
        return await asyncio.wait_for(
            llm.complete(messages, max_tokens=max_tokens, json_mode=False),
            timeout=self._timeout_s,
        )

    async def rollup(self, transcript: str) -> str | None:
        """Condense turns that left the window into <=80 words. ``None`` on failure.

        Same contract as :meth:`summarize` — a lost digest costs context, never
        the session — but it runs on its own task so the conversation never
        waits for it.
        """
        if not transcript.strip():
            return None
        messages: list[Message] = [
            {"role": "system", "content": build_rollup_system()},
            {"role": "user", "content": transcript},
        ]
        try:
            text = await self._call_summary(messages, max_tokens=ROLLUP_MAX_TOKENS)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[AVISO] resumo do historico falhou: {exc}", file=sys.stderr)
            return None
        return text.strip() or None


def _user_content(request: FeedbackRequest) -> str:
    """The tutor's line gives context; only the learner's sentence is corrected."""
    return f"Tutor: {request.ai_text}\nLearner: {request.user_text}"
