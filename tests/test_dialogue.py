"""Phase 5: message assembly for each turn."""

from __future__ import annotations

import pytest

from tutor.core.dialogue import DEFAULT_SYSTEM, DialogueManager


def test_first_turn_is_system_plus_the_utterance() -> None:
    d = DialogueManager()
    messages = d.build("hello")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == DEFAULT_SYSTEM
    assert messages[1]["content"] == "hello"
    assert d.turns == 0


def test_recorded_turn_reappears_on_the_next_build() -> None:
    d = DialogueManager()
    d.build("hi")
    d.record("hi", "Hello! How are you?")
    messages = d.build("fine")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2] == {"role": "assistant", "content": "Hello! How are you?"}
    assert messages[3]["content"] == "fine"


def test_build_does_not_mutate_the_history() -> None:
    d = DialogueManager()
    d.record("a", "b")
    before = d.history
    d.build("c")
    d.build("d")
    assert d.history == before


def test_history_keeps_only_the_newest_pairs() -> None:
    d = DialogueManager(history_max_turns=2)
    for i in range(5):
        d.record(f"u{i}", f"a{i}")
    assert d.turns == 2
    assert [m["content"] for m in d.history] == ["u3", "a3", "u4", "a4"]


def test_a_dangling_user_line_never_survives_trimming() -> None:
    d = DialogueManager(history_max_turns=1)
    d.record("u0", "a0")
    d.record("u1", "")  # no answer: not a usable pair
    assert d.turns == 1
    assert [m["role"] for m in d.history] == ["user", "assistant"]


def test_zero_history_turns_keeps_only_the_current_utterance() -> None:
    d = DialogueManager(history_max_turns=0)
    d.record("u0", "a0")
    assert d.history == []
    assert [m["role"] for m in d.build("hi")] == ["system", "user"]


def test_reset_clears_everything() -> None:
    d = DialogueManager()
    d.record("u", "a")
    d.reset()
    assert d.history == []
    assert d.turns == 0


def test_negative_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="history_max_turns"):
        DialogueManager(history_max_turns=-1)


def test_custom_system_replaces_the_default() -> None:
    d = DialogueManager(system="Custom rules.")
    assert d.build("hi")[0] == {"role": "system", "content": "Custom rules."}


# -- the AI opens the conversation (Phase 6) -------------------------------- #


def test_greeting_sits_between_the_system_prompt_and_the_history() -> None:
    d = DialogueManager()
    d.record_greeting("Hello! Welcome.")
    d.record("hi", "Hello there.")
    messages = d.build("fine")
    assert [m["role"] for m in messages] == [
        "system",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1] == {"role": "assistant", "content": "Hello! Welcome."}


def test_greeting_does_not_count_as_a_history_pair() -> None:
    d = DialogueManager(history_max_turns=2)
    d.record_greeting("Hello!")
    d.record("a", "b")
    d.record("c", "d")
    d.record("e", "f")  # over budget: the oldest pair goes, not the greeting
    assert d.turns == 2
    assert d.greeting == "Hello!"
    roles = [m["role"] for m in d.build("g")]
    assert roles == [
        "system",
        "assistant",  # greeting, outside the pair window
        "user",
        "assistant",
        "user",
        "assistant",
        "user",  # the utterance being answered
    ]


def test_greeting_is_replaced_not_appended() -> None:
    d = DialogueManager()
    d.record_greeting("first")
    d.record_greeting("second")
    assert d.greeting == "second"
    assert [m["role"] for m in d.build("x")].count("assistant") == 1


def test_blank_greeting_is_ignored() -> None:
    d = DialogueManager()
    d.record_greeting("   ")
    assert d.greeting is None
    assert [m["role"] for m in d.build("hi")] == ["system", "user"]


def test_reset_clears_the_greeting() -> None:
    d = DialogueManager()
    d.record_greeting("Hello!")
    d.reset()
    assert d.greeting is None
