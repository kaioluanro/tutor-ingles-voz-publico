"""Rolling history (Phase 11): evicted turns are condensed, not forgotten."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tutor.config import load_all
from tutor.core.dialogue import DialogueManager
from tutor.core.prompts import build_rollup_system, build_summary_system
from tutor.core.session import Session, _rollup_transcript

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
LEVELS = ROOT / "levels.yaml"


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #

def test_rollup_prompt_exists_and_caps_the_length() -> None:
    text = build_rollup_system()
    assert "80 words" in text
    assert "no markdown" in text
    # Same prompt the session summary uses must not be the rollup.
    assert text != build_summary_system()


def test_rollup_transcript_labels_the_speakers() -> None:
    evicted = [
        {"role": "user", "content": "i want a table"},
        {"role": "assistant", "content": "A table for how many?"},
    ]
    text = _rollup_transcript(evicted)
    assert text == (
        "Learner: i want a table\nTutor: A table for how many?"
    )


def test_rollup_transcript_carries_the_previous_digest() -> None:
    evicted = [{"role": "user", "content": "thanks"}]
    text = _rollup_transcript(evicted, previous="They are ordering food.")
    assert text.splitlines() == [
        "Earlier summary: They are ordering food.",
        "Learner: thanks",
    ]


# --------------------------------------------------------------------------- #
# dialogue
# --------------------------------------------------------------------------- #

def test_record_returns_the_pairs_it_evicted() -> None:
    d = DialogueManager(history_max_turns=1)
    assert d.record("u0", "a0") == []
    evicted = d.record("u1", "a1")
    assert [m["role"] for m in evicted] == ["user", "assistant"]
    assert [m["content"] for m in evicted] == ["u0", "a0"]
    assert d.turns == 1


def test_rollup_rides_on_the_system_message_and_adds_nothing_else() -> None:
    d = DialogueManager(system="BASE", history_max_turns=1)
    d.record("u0", "a0")
    d.record("u1", "a1")
    before = len(d.build("u2"))
    d.set_rollup("They are ordering food.")
    messages = d.build("u2")
    assert len(messages) == before  # no extra message for the digest
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("BASE")
    assert "condensed" in messages[0]["content"]
    assert "They are ordering food." in messages[0]["content"]


def test_blank_rollup_is_ignored() -> None:
    d = DialogueManager(system="BASE")
    d.set_rollup("   \n")
    assert d.rollup is None
    assert d.build("hi")[0]["content"] == "BASE"


def test_reset_clears_the_rollup_too() -> None:
    d = DialogueManager()
    d.set_rollup("something")
    d.reset()
    assert d.rollup is None


def test_history_without_a_rollup_is_unchanged() -> None:
    """The default path (feedback off) must be byte-identical to before."""
    d = DialogueManager(system="BASE")
    d.record("hi", "hello")
    assert [m["content"] for m in d.build("bye")] == ["BASE", "hi", "hello", "bye"]


# --------------------------------------------------------------------------- #
# session wiring
# --------------------------------------------------------------------------- #

class _FakePlayback:
    def __init__(self) -> None:
        self.started_at = None

    def stop(self) -> None:
        return None

    def end_of_stream(self) -> None:
        return None

    def feed(self, chunk: object) -> None:
        return None

    async def wait_until_done(self) -> None:
        return None


class _RecordingRollup:
    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.seen: list[str] = []
        self.gate = gate

    async def rollup(self, transcript: str) -> str | None:
        self.seen.append(transcript)
        if self.gate is not None:
            await self.gate.wait()
        return f"digest {len(self.seen)}"


def _session(tmp_path: Path, **kwargs: object) -> Session:
    cfg, levels = load_all(CONFIG, LEVELS)
    cfg.storage.db_path = str(tmp_path / "tutor.db")
    session = Session(
        cfg,
        level_name="beginner",
        level=levels["beginner"],
        scenario="restaurant",
        listen=False,
        **kwargs,  # type: ignore[arg-type]
    )
    session.playback = _FakePlayback()  # type: ignore[assignment]
    return session


def _close(session: Session) -> None:
    if session.db is not None:
        session.db.close()
        session.db = None


def test_evicted_turns_are_handed_to_the_feedback_llm(tmp_path: Path) -> None:
    session = _session(tmp_path)
    stub = _RecordingRollup()
    session.feedback = stub  # type: ignore[assignment]
    session.dialogue.history_max_turns = 1
    session.dialogue.record("u0", "a0")
    evicted = session.dialogue.record("u1", "a1")

    async def scenario() -> None:
        session._submit_rollup(evicted)
        assert session._rollup_task is not None
        await session._rollup_task

    try:
        asyncio.run(scenario())
    finally:
        _close(session)

    assert stub.seen == ["Learner: u0\nTutor: a0"]
    assert session.dialogue.rollup == "digest 1"
    assert "digest 1" in session.dialogue.build("u2")[0]["content"]


def test_a_second_eviction_waits_for_the_first_and_includes_it(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    session = _session(tmp_path)
    stub = _RecordingRollup(gate)
    session.feedback = stub  # type: ignore[assignment]

    async def scenario() -> None:
        first = [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}]
        second = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
        session._submit_rollup(first)
        task = session._rollup_task
        assert task is not None
        await asyncio.sleep(0)  # the worker is now parked on the gate
        session._submit_rollup(second)  # task still running -> batch, no task
        assert session._rollup_task is task
        gate.set()
        await task

    try:
        asyncio.run(scenario())
    finally:
        _close(session)

    assert len(stub.seen) == 2
    assert stub.seen[0] == "Learner: u0\nTutor: a0"
    assert stub.seen[1].splitlines() == [
        "Earlier summary: digest 1",
        "Learner: u1",
        "Tutor: a1",
    ]


def test_without_feedback_nothing_is_queued(tmp_path: Path) -> None:
    session = _session(tmp_path)
    assert session.feedback is None
    session._submit_rollup([{"role": "user", "content": "u"}])
    assert session._rollup_task is None
    assert session._rollup_batch == []
    _close(session)


def test_cancelling_the_rollup_forgets_the_pending_batch(tmp_path: Path) -> None:
    gate = asyncio.Event()
    session = _session(tmp_path)
    session.feedback = _RecordingRollup(gate)  # type: ignore[assignment]

    async def scenario() -> object:
        session._submit_rollup([{"role": "user", "content": "u0"}])
        task = session._rollup_task
        assert task is not None
        await asyncio.sleep(0)
        session._cancel_rollup()
        assert session._rollup_task is None
        assert session._rollup_batch == []
        with pytest.raises(asyncio.CancelledError):
            await task
        return True

    try:
        assert asyncio.run(scenario()) is True
    finally:
        _close(session)


def test_failed_rollup_keeps_the_previous_digest(tmp_path: Path) -> None:
    """The backend going down costs the new context, never the old one."""

    class _Broken:
        async def rollup(self, transcript: str) -> str | None:
            return None

    session = _session(tmp_path)
    session.feedback = _Broken()  # type: ignore[assignment]
    session.dialogue.set_rollup("already condensed")

    async def scenario() -> None:
        session._submit_rollup([{"role": "user", "content": "u0"}])
        await session._rollup_task

    try:
        asyncio.run(scenario())
    finally:
        _close(session)

    assert session.dialogue.rollup == "already condensed"
