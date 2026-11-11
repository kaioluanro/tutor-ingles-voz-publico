"""Tests for the sentence chunker (Phase 6).

The chunker decides when TTS can start, so the boundary cases matter: an
abbreviation must not split a sentence, a decimal point must not either, and
fragments must merge until they are worth speaking.

The contract the producer relies on is ``feed()`` during the stream plus one
``flush()`` at the end — the last sentence of an answer usually has no trailing
whitespace, so it only comes out on the flush.
"""

from __future__ import annotations

import pytest

from tutor.core.chunker import SentenceChunker


def _feed(chunker: SentenceChunker, text: str) -> list[str]:
    """Feed a whole string character by character, as the LLM producer would."""
    out: list[str] = []
    for char in text:
        out.extend(chunker.feed(char))
    return out


def _speak(chunker: SentenceChunker, text: str) -> list[str]:
    """Everything the TTS would be handed for one complete answer."""
    return _feed(chunker, text) + chunker.flush()


# -- the plan's minimum set ------------------------------------------------- #


def test_two_sentences_come_out_separately() -> None:
    assert _speak(SentenceChunker(min_words=1), "Hello there. How are you?") == [
        "Hello there.",
        "How are you?",
    ]


def test_abbreviations_do_not_split() -> None:
    assert _speak(SentenceChunker(min_words=1), "Dr. Smith is here.") == [
        "Dr. Smith is here."
    ]


def test_decimal_numbers_do_not_split() -> None:
    assert _speak(SentenceChunker(min_words=1), "It costs 3.5 dollars.") == [
        "It costs 3.5 dollars."
    ]


def test_short_sentences_are_merged_until_they_are_speakable() -> None:
    # "Yes." is one word, so with the configured min_words it waits.
    assert _speak(SentenceChunker(min_words=3), "Yes. I am.") == ["Yes. I am."]


def test_flush_returns_the_tail_without_final_punctuation() -> None:
    chunker = SentenceChunker()
    assert chunker.feed("The answer is") == []
    assert chunker.flush() == ["The answer is"]
    assert chunker.flush() == []  # nothing is ever returned twice


# -- streaming behaviour ---------------------------------------------------- #


def test_min_words_of_one_emits_each_sentence_immediately() -> None:
    chunker = SentenceChunker(min_words=1)
    assert chunker.feed("Hello there. ") == ["Hello there."]
    assert chunker.feed("How are you? ") == ["How are you?"]
    assert chunker.flush() == []


def test_min_words_of_three_holds_a_two_word_sentence() -> None:
    chunker = SentenceChunker(min_words=3)
    assert chunker.feed("Hello there. ") == []
    assert chunker.buffered == "Hello there."
    # The next sentence is long enough, so the pair goes out together.
    assert chunker.feed("How are you? ") == ["Hello there. How are you?"]


def test_tokens_arriving_across_calls_still_join() -> None:
    chunker = SentenceChunker(min_words=1)
    assert chunker.feed("Hel") == []
    assert chunker.feed("lo there. ") == ["Hello there."]
    assert chunker.feed("How are you?") == []
    assert chunker.flush() == ["How are you?"]


def test_closing_quote_stays_with_the_sentence() -> None:
    assert _speak(SentenceChunker(min_words=1), 'He said "stop." Then he left.') == [
        'He said "stop."',
        "Then he left.",
    ]


def test_flush_includes_a_short_sentence_held_back() -> None:
    chunker = SentenceChunker(min_words=3)
    assert chunker.feed("Yes.") == []
    assert chunker.flush() == ["Yes."]


def test_punctuation_only_text_is_never_handed_to_the_tts() -> None:
    chunker = SentenceChunker(min_words=1)
    assert chunker.feed("...") == []
    assert chunker.flush() == []


def test_reset_drops_everything_buffered() -> None:
    chunker = SentenceChunker()
    chunker.feed("Half a sentence without ending")
    chunker.reset()
    assert chunker.buffered == ""
    assert chunker.flush() == []


def test_min_words_must_be_positive() -> None:
    with pytest.raises(ValueError, match="min_words"):
        SentenceChunker(min_words=0)
