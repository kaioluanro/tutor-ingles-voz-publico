"""Build the message list handed to the LLM on every turn.

The shape is fixed — ``system`` + the retained history + the utterance that
just arrived — and only the system text and the retention window are
configurable, so later phases can swap richer prompts in without touching the
pipeline.

History is stored as user/assistant pairs written by :meth:`record` *after* a
turn finishes, so :meth:`build` never sees the utterance it is about to answer
twice. ``history_max_turns`` counts pairs, not messages, matching
``pipeline.history_max_turns`` in the config.

When a pair is evicted, :meth:`record` hands it back instead of discarding it:
the session condenses it into :meth:`set_rollup` with the feedback LLM, so a
long conversation keeps its thread without keeping its tokens.
"""

from __future__ import annotations

from tutor.llm.base import Message

DEFAULT_SYSTEM = (
    "You are a friendly English tutor having a spoken conversation with a "
    "student. Answer in English only, in one or two short spoken sentences "
    "(max 45 words). Never use markdown, lists, emoji or stage directions, "
    "because your answer is read aloud by a text-to-speech engine. Ask one "
    "natural follow-up question to keep the conversation going."
)


class DialogueManager:
    """Accumulate the conversation and render the messages for the next turn."""

    def __init__(self, *, system: str = DEFAULT_SYSTEM, history_max_turns: int = 8) -> None:
        if history_max_turns < 0:
            raise ValueError(f"history_max_turns nao pode ser negativo: {history_max_turns}")
        self.system = system
        self.history_max_turns = history_max_turns
        self._history: list[Message] = []
        self._greeting: str | None = None
        # Digest of everything already evicted; None while nothing was lost.
        self._rollup: str | None = None

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    @property
    def greeting(self) -> str | None:
        """The AI's opening line, spoken before the student says anything."""
        return self._greeting

    @property
    def rollup(self) -> str | None:
        """Condensed memory of the turns that left the window."""
        return self._rollup

    def set_rollup(self, text: str) -> None:
        """Install the digest of the evicted turns.

        It rides on the **system** message rather than becoming an extra
        message: some OpenAI-compatible servers reject a second ``system`` in
        the middle of the list, and an ``assistant`` line the model never said
        would poison its own voice.
        """
        cleaned = text.strip()
        self._rollup = cleaned or None

    @property
    def turns(self) -> int:
        """Number of completed user/assistant pairs currently retained."""
        return len(self._history) // 2

    def reset(self) -> None:
        self._history.clear()
        self._greeting = None
        self._rollup = None

    def record_greeting(self, text: str) -> None:
        """Set the AI's opening line.

        It is kept outside the pair window on purpose: history is trimmed a
        pair at a time, and a lone leading assistant message would throw that
        alignment off.
        """
        if text.strip():
            self._greeting = text.strip()

    def record(self, user_text: str, assistant_text: str) -> list[Message]:
        """Append one completed turn; return the pairs evicted from the window.

        The assistant's *final* text is recorded even when only part of it was
        played back, so the model is never told about something it did not say.
        The caller gets the evicted pairs back because dropping them silently
        is what the rolling summary (Phase 11) exists to avoid.
        """
        if user_text:
            self._history.append({"role": "user", "content": user_text})
            if assistant_text:
                self._history.append({"role": "assistant", "content": assistant_text})
            else:
                # A user turn with no answer must not stay dangling at the head
                # of the window; it is not a usable pair.
                self._history.pop()
        return self._trim()

    def build(self, user_text: str) -> list[Message]:
        """Messages for the LLM: system (+ rollup), greeting, history, utterance."""
        system = self.system
        if self._rollup:
            system = (
                f"{system}\n\n"
                "Earlier in this conversation, condensed:\n"
                f"{self._rollup}"
            )
        messages: list[Message] = [{"role": "system", "content": system}]
        if self._greeting is not None:
            messages.append({"role": "assistant", "content": self._greeting})
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def _trim(self) -> list[Message]:
        """Keep only the newest ``history_max_turns`` pairs; return the rest."""
        keep = self.history_max_turns * 2
        drop = len(self._history) - keep
        if drop <= 0:
            return []
        if drop % 2:  # never cut a pair in half
            drop += 1
        evicted = self._history[:drop]
        del self._history[:drop]
        return evicted
