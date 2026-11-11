"""Split the LLM's token stream into units the TTS can speak.

The TTS cannot start until it has a sentence, and the LLM cannot wait for it —
so the producer task feeds every token here and gets back whole sentences as
soon as they are complete. Anything shorter than ``min_words`` is held back and
merged with the next sentence, because Kokoro reading "Yes." as a standalone
utterance sounds like a glitch, not like speech.
"""

from __future__ import annotations

import re

# Sentence enders, keeping any closing quotes/brackets with them so the
# boundary is consumed and not left behind for the next candidate.
SENT_END = re.compile(r'([.!?]+["\')\]]*)(\s+)')

# A period after one of these is an abbreviation, not a sentence boundary.
ABBREV = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
    }
)


def _speakable(text: str) -> bool:
    """True when the candidate holds at least one alphanumeric character."""
    return any(ch.isalnum() for ch in text)


class SentenceChunker:
    """Accumulate tokens; return finished, speakable sentences."""

    def __init__(self, min_words: int = 3) -> None:
        if min_words < 1:
            raise ValueError(f"min_words deve ser >= 1, recebido {min_words}")
        self.min_words = min_words
        self.buf = ""
        self.pending = ""

    @property
    def buffered(self) -> str:
        """Text held back but not yet emitted (useful for tests and flushing)."""
        return f"{self.pending} {self.buf}".strip()

    def feed(self, text: str) -> list[str]:
        """Append a token and return every sentence it completed."""
        self.buf += text
        out: list[str] = []
        start = 0
        for match in SENT_END.finditer(self.buf):
            candidate = self.buf[start : match.end(1)].strip()
            words = candidate.split()
            # No letters (stray "!!!") or an abbreviation ("Dr.") means this is
            # not a boundary: keep scanning from the same position.
            if not _speakable(candidate) or words[-1].lower() in ABBREV:
                continue
            self.pending = f"{self.pending} {candidate}".strip()
            start = match.end()
            if len(self.pending.split()) >= self.min_words:
                out.append(self.pending)
                self.pending = ""
        self.buf = self.buf[start:]
        return out

    def flush(self) -> list[str]:
        """Return whatever is left once the stream ends (often unpunctuated)."""
        rest = f"{self.pending} {self.buf}".strip()
        self.pending, self.buf = "", ""
        return [rest] if _speakable(rest) else []

    def reset(self) -> None:
        """Drop everything buffered (used when a turn is cancelled)."""
        self.pending, self.buf = "", ""
