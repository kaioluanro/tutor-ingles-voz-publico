"""Background correction feedback (Phase 8).

Two layers are covered, matching the plan's two demands:

* ``parse_feedback`` — the four cases the plan lists (valid, fenced, invalid,
  missing fields) plus the edge cases a 4B model actually produces.
* ``FeedbackWorker`` — the property the whole phase hangs on: **a local call is
  only ever made while the system is idle**, so a correction and a conversation
  never ask for llama-server's single slot at the same time.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tutor.core.feedback import (
    FeedbackParseError,
    FeedbackRequest,
    FeedbackWorker,
    parse_feedback,
    strip_code_fences,
)
from tutor.core.prompts import PromptError, build_feedback_system, build_summary_system
from tutor.llm.base import Message

VALID = {
    "corrected": "I went to the market yesterday.",
    "errors": [
        {
            "original": "I go to the market",
            "fix": "I went to the market",
            "type": "grammar",
            "explanation_pt": "Passado simples: 'went', nao 'go'.",
        }
    ],
    "better_phrasing": "I popped into the market yesterday.",
    "score": 7,
}


def _as_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False)


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condicao nao alcancada a tempo")
        await asyncio.sleep(0.01)


class FakeLLM:
    """A backend that answers ``reply`` after ``delay_s``."""

    def __init__(
        self,
        reply: str | None = None,
        *,
        delay_s: float = 0.0,
        error: Exception | None = None,
        local_backend: bool = True,
        replies: list[str] | None = None,
    ) -> None:
        self.reply = reply if reply is not None else _as_json(VALID)
        self.delay_s = delay_s
        self.error = error
        self.local_backend = local_backend
        self.replies = list(replies or [])
        self.calls = 0
        self.seen_messages: list[list[Message]] = []
        self.seen_json_mode: list[bool] = []

    async def stream(self, messages, *, max_tokens, temperature=None):  # pragma: no cover
        raise AssertionError("o feedback nunca usa stream()")

    async def complete(self, messages, *, max_tokens, json_mode: bool = False) -> str:
        self.calls += 1
        self.seen_messages.append(messages)
        self.seen_json_mode.append(json_mode)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        if self.replies:
            return self.replies.pop(0)
        return self.reply


# -- parse: the four cases the plan names ---------------------------------- #


def test_parse_valid_json():
    feedback = parse_feedback(_as_json(VALID))
    assert feedback.corrected == "I went to the market yesterday."
    assert feedback.score == 7
    assert feedback.errors[0].type == "grammar"
    assert feedback.better_phrasing.startswith("I popped")


def test_parse_json_with_fences():
    raw = "```json\n" + _as_json(VALID) + "\n```"
    assert parse_feedback(raw).score == 7


def test_parse_bare_fences_without_language():
    assert parse_feedback("```\n" + _as_json(VALID) + "\n```").errors


def test_parse_invalid_json():
    with pytest.raises(FeedbackParseError, match="JSON invalido"):
        parse_feedback("claro que sim, isso nao e json")


def test_parse_missing_corrected():
    payload = {"errors": [], "score": 8}
    with pytest.raises(FeedbackParseError, match="corrected"):
        parse_feedback(_as_json(payload))


def test_parse_missing_errors():
    # The prompt promises "errors": [] when the sentence is clean, so an absent
    # key means truncation, not "no mistakes found".
    payload = {"corrected": "I went home.", "score": 8}
    with pytest.raises(FeedbackParseError, match="errors"):
        parse_feedback(_as_json(payload))


def test_parse_missing_field_inside_an_error():
    payload = {
        "corrected": "x",
        "errors": [{"original": "I go", "fix": "I went", "type": "grammar"}],
    }
    with pytest.raises(FeedbackParseError, match="explanation_pt"):
        parse_feedback(_as_json(payload))


def test_parse_empty_reply():
    with pytest.raises(FeedbackParseError, match="vazia"):
        parse_feedback("   ")


def test_parse_reply_is_a_list():
    with pytest.raises(FeedbackParseError, match="objeto JSON"):
        parse_feedback("[1, 2, 3]")


def test_parse_score_out_of_range():
    payload = dict(VALID, score=99)
    with pytest.raises(FeedbackParseError, match="score"):
        parse_feedback(_as_json(payload))


def test_parse_unknown_error_type_is_relabelled():
    payload = dict(VALID, errors=[dict(VALID["errors"][0], type="spelling")])
    assert parse_feedback(_as_json(payload)).errors[0].type == "other"


def test_parse_empty_errors_is_accepted():
    payload = dict(VALID, errors=[], better_phrasing=None, score=10)
    feedback = parse_feedback(_as_json(payload))
    assert feedback.errors == []
    assert feedback.score == 10


def test_parse_ignores_unknown_extra_keys():
    payload = dict(VALID, note="oi", thinking="...")
    assert parse_feedback(_as_json(payload)).corrected == VALID["corrected"]


def test_strip_code_fences_leaves_plain_text_alone():
    assert strip_code_fences('{"a": 1}') == '{"a": 1}'


# -- the per-level rigor block --------------------------------------------- #


def test_feedback_system_keeps_only_this_levels_rigor():
    beginner = build_feedback_system("beginner")
    assert "### BEGINNER" in beginner
    assert "### ADVANCED" not in beginner
    assert "### INTERMEDIATE" not in beginner
    assert "Reply with ONLY JSON" in beginner


def test_feedback_system_for_every_level():
    for level in ("beginner", "intermediate", "advanced"):
        text = build_feedback_system(level)
        assert f"### {level.upper()}" in text
        assert len(text.split()) < 400, "prompt de feedback passou do orcamento"


def test_feedback_system_unknown_level_raises():
    with pytest.raises(PromptError, match="EXPERT"):
        build_feedback_system("expert")


def test_summary_system_has_the_four_parts():
    text = build_summary_system()
    for label in (
        "MOST FREQUENT ERRORS:",
        "NEW EXPRESSIONS:",
        "STRENGTHS:",
        "FOCUS FOR NEXT TIME:",
    ):
        assert label in text


# -- worker: the idle rule ------------------------------------------------- #


def test_worker_does_not_call_a_local_llm_while_busy():
    async def scenario():
        idle = asyncio.Event()  # cleared: a turn is in progress
        llm = FakeLLM()
        worker = FeedbackWorker(llm=llm, idle=idle, timeout_s=5.0)
        await worker.start()
        worker.submit(
            FeedbackRequest(user_text="I go", ai_text="Hello", level="beginner", turn=1)
        )
        await asyncio.sleep(0.05)
        assert llm.calls == 0, "LLM local chamado com o sistema ocupado"

        idle.set()
        await _wait_until(lambda: len(worker.results) == 1)
        assert llm.calls == 1
        assert worker.results[0].feedback.corrected == VALID["corrected"]
        assert worker.results[0].backend == "local"
        await worker.close()

    asyncio.run(scenario())


def test_worker_sends_the_level_and_both_utterance_lines():
    async def scenario():
        idle = asyncio.Event()
        idle.set()
        llm = FakeLLM()
        worker = FeedbackWorker(llm=llm, idle=idle, timeout_s=5.0)
        await worker.start()
        worker.submit(
            FeedbackRequest(
                user_text="I go to market", ai_text="Where did you go?", level="advanced", turn=2
            )
        )
        await _wait_until(lambda: llm.calls == 1)
        system = llm.seen_messages[0][0]["content"]
        assert "### ADVANCED" in system
        assert "### BEGINNER" not in system
        user = llm.seen_messages[0][1]["content"]
        assert user == "Tutor: Where did you go?\nLearner: I go to market"
        await worker.close()

    asyncio.run(scenario())


def test_worker_retries_once_then_discards():
    async def scenario():
        idle = asyncio.Event()
        idle.set()
        llm = FakeLLM(reply="isto nao e json")
        worker = FeedbackWorker(llm=llm, idle=idle, timeout_s=5.0, max_retries=1)
        await worker.start()
        worker.submit(
            FeedbackRequest(user_text="I go", ai_text="Hi", level="beginner", turn=1)
        )
        await _wait_until(lambda: llm.calls >= 2)
        await _wait_until(lambda: worker.pending == 0 and worker._current is None)
        assert llm.calls == 2, "esperava exatamente 1 nova tentativa"
        assert worker.results == []
        await worker.close()

    asyncio.run(scenario())


def test_worker_uses_json_mode_only_for_a_remote_backend():
    async def scenario():
        idle = asyncio.Event()
        idle.set()
        remote = FakeLLM(local_backend=False)
        worker = FeedbackWorker(llm=remote, idle=idle, timeout_s=5.0)
        await worker.start()
        worker.submit(
            FeedbackRequest(user_text="I go", ai_text="Hi", level="beginner", turn=1)
        )
        await _wait_until(lambda: len(worker.results) == 1)
        assert remote.seen_json_mode == [True]
        assert worker.results[0].backend == "remoto"
        await worker.close()

        local = FakeLLM()
        worker = FeedbackWorker(llm=local, idle=idle, timeout_s=5.0)
        await worker.start()
        worker.submit(
            FeedbackRequest(user_text="I go", ai_text="Hi", level="beginner", turn=2)
        )
        await _wait_until(lambda: len(worker.results) == 1)
        assert local.seen_json_mode == [False]
        assert worker.results[0].backend == "local"
        await worker.close()

    asyncio.run(scenario())


def test_worker_falls_back_to_local_when_the_remote_call_breaks():
    async def scenario():
        idle = asyncio.Event()
        idle.set()
        remote = FakeLLM(error=RuntimeError("sem internet"))
        local = FakeLLM()
        worker = FeedbackWorker(
            llm=remote,
            idle=idle,
            timeout_s=5.0,
            fallback_llm=local,
            max_retries=1,
        )
        await worker.start()
        worker.submit(
            FeedbackRequest(user_text="I go", ai_text="Hi", level="beginner", turn=1)
        )
        await _wait_until(lambda: len(worker.results) == 1)
        # Transport failures are not retried on the same backend.
        assert remote.calls == 1
        assert local.calls == 1
        assert worker.results[0].backend == "local"
        await worker.close()

    asyncio.run(scenario())


def test_worker_drops_when_the_queue_is_full():
    async def scenario():
        idle = asyncio.Event()  # stays busy: nothing drains the queue
        llm = FakeLLM()
        worker = FeedbackWorker(llm=llm, idle=idle, timeout_s=5.0, queue_size=1)
        await worker.start()
        first = FeedbackRequest(user_text="a", ai_text="b", level="beginner", turn=1)
        second = FeedbackRequest(user_text="c", ai_text="d", level="beginner", turn=2)
        third = FeedbackRequest(user_text="e", ai_text="f", level="beginner", turn=3)
        assert worker.submit(first) is True
        # Nothing drains it: the system is busy, so the second request cannot
        # be queued either and is reported rather than blocking the turn.
        assert worker.submit(second) is False
        assert worker.submit(third) is False
        assert worker.dropped == 2
        assert llm.calls == 0
        await worker.close()

    asyncio.run(scenario())


def test_worker_submit_ignores_blank_text():
    async def scenario():
        worker = FeedbackWorker(
            llm=FakeLLM(), idle=asyncio.Event(), timeout_s=5.0
        )
        assert worker.submit(FeedbackRequest(user_text="   ", ai_text="x",
                                             level="beginner", turn=1)) is False

    asyncio.run(scenario())


def test_worker_timeout_is_reported_and_not_retried_forever():
    async def scenario():
        idle = asyncio.Event()
        idle.set()
        llm = FakeLLM(delay_s=1.0)
        worker = FeedbackWorker(llm=llm, idle=idle, timeout_s=0.05, max_retries=1)
        await worker.start()
        worker.submit(
            FeedbackRequest(user_text="I go", ai_text="Hi", level="beginner", turn=1)
        )
        await _wait_until(lambda: worker.pending == 0 and worker._current is None)
        assert worker.results == []
        assert llm.calls >= 1
        await worker.close()

    asyncio.run(scenario())


def test_summarize_returns_the_models_text():
    async def scenario():
        idle = asyncio.Event()
        idle.set()
        llm = FakeLLM(reply="MOST FREQUENT ERRORS: none")
        worker = FeedbackWorker(llm=llm, idle=idle, timeout_s=5.0)
        await worker.start()
        text = await worker.summarize("Tutor: Hello\n[1] Learner: Hi", level="beginner")
        assert text.startswith("MOST FREQUENT ERRORS")
        assert "LEVEL: beginner" in llm.seen_messages[0][1]["content"]
        await worker.close()

    asyncio.run(scenario())


def test_summarize_is_none_for_an_empty_transcript():
    async def scenario():
        worker = FeedbackWorker(
            llm=FakeLLM(), idle=asyncio.Event(), timeout_s=5.0
        )
        assert await worker.summarize("   ", level="beginner") is None

    asyncio.run(scenario())


def test_summary_system_warns_against_translating_the_labels() -> None:
    """Regression: one session came back with all four labels in Portuguese
    (``NOVAS EXPRESSÕES:``), which left that session with no vocabulary at all,
    because ``parse_new_expressions`` matches the English name only."""
    text = build_summary_system().lower()
    assert "verbatim" in text
    assert "english" in text


# -- spoken_pt (Phase 12) -------------------------------------------------- #


def test_spoken_pt_is_parsed_and_preferred_for_speech() -> None:
    fb = parse_feedback(_as_json(dict(VALID, spoken_pt="Boa! O correto e 'went'.")))
    assert fb.spoken_correction_pt == "Boa! O correto e 'went'."


def test_without_spoken_pt_the_written_explanation_serves() -> None:
    fb = parse_feedback(_as_json(VALID))
    assert fb.spoken_correction_pt == VALID["errors"][0]["explanation_pt"]


def test_a_blank_spoken_pt_counts_as_missing() -> None:
    fb = parse_feedback(_as_json(dict(VALID, spoken_pt="   ")))
    assert fb.spoken_correction_pt == VALID["errors"][0]["explanation_pt"]


def test_nothing_is_spoken_when_there_is_nothing_to_correct() -> None:
    fb = parse_feedback(_as_json(dict(VALID, errors=[], spoken_pt=None)))
    assert fb.spoken_correction_pt == ""


def test_the_feedback_prompt_asks_for_the_spoken_line() -> None:
    text = build_feedback_system("beginner")
    assert "spoken_pt" in text
    assert "20 words" in text
    assert "null when" in text
