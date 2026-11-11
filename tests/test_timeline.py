"""Phase 5: latency marks (plan §9) and the table they are rendered into."""

from __future__ import annotations

import pytest

from tutor.timeline import (
    LATENCY_MARK,
    MARKS,
    ORIGIN,
    TurnTimeline,
    median_offsets_ms,
    median_latency_ms,
    mean_latency_ms,
    summary_table,
    turn_table,
)


def _turn(number: int, offsets_ms: dict[str, float]) -> TurnTimeline:
    """Build a timeline whose marks sit ``offsets_ms`` after ``t_speech_end``."""
    tl = TurnTimeline(turn=number)
    origin = 1000.0
    tl.mark(ORIGIN, origin)
    for name, ms in offsets_ms.items():
        tl.mark(name, origin + ms / 1000.0)
    return tl


def test_all_plan_marks_are_defined() -> None:
    names = [name for name, _ in MARKS]
    assert names == [
        "t_speech_end",
        "t_stt_done",
        "t_llm_first_token",
        "t_first_sentence",
        "t_tts_first_audio",
        "t_playback_start",
        "t_llm_done",
    ]
    assert ORIGIN == "t_speech_end"
    assert LATENCY_MARK == "t_playback_start"


def test_offset_is_measured_from_speech_end() -> None:
    tl = TurnTimeline(turn=1)
    tl.mark("t_speech_end", 10.0)
    tl.mark("t_stt_done", 10.5)
    assert tl.offset_ms("t_stt_done") == pytest.approx(500.0)
    assert tl.offset_ms("t_speech_end") == pytest.approx(0.0)


def test_unmarked_offset_is_none_not_zero() -> None:
    tl = TurnTimeline(turn=1)
    tl.mark("t_speech_end", 10.0)
    assert tl.offset_ms("t_tts_first_audio") is None
    assert tl.latency_ms is None


def test_first_mark_wins() -> None:
    tl = TurnTimeline(turn=1)
    tl.mark("t_speech_end", 1.0)
    tl.mark("t_speech_end", 2.0)
    assert tl.marks["t_speech_end"] == 1.0


def test_latency_follows_the_playback_mark() -> None:
    tl = TurnTimeline(turn=1)
    tl.mark("t_speech_end", 5.0)
    tl.mark("t_playback_start", 7.25)
    assert tl.latency_ms == pytest.approx(2250.0)


def test_complete_needs_every_mark() -> None:
    tl = TurnTimeline(turn=1)
    assert tl.complete is False
    tl.mark("t_speech_end", 1.0)
    for name, _ in MARKS[1:]:
        tl.mark(name, 1.0)
    assert tl.complete is True


def test_statistics_skip_turns_without_a_latency() -> None:
    good = [_turn(1, {LATENCY_MARK: 3000.0}), _turn(2, {LATENCY_MARK: 5000.0})]
    missing = _turn(3, {"t_stt_done": 400.0})  # playback never happened
    turns = [*good, missing]

    assert median_latency_ms(turns) == pytest.approx(4000.0)
    assert mean_latency_ms(turns) == pytest.approx(4000.0)
    # The unplayable turn must not drag the median down to its STT time.
    assert median_latency_ms([missing]) is None
    assert mean_latency_ms([]) is None


def test_median_offset_covers_every_mark() -> None:
    offsets = median_offsets_ms([_turn(1, {"t_stt_done": 400.0, LATENCY_MARK: 3000.0})])
    assert set(offsets) == {name for name, _ in MARKS}
    assert offsets["t_speech_end"] == pytest.approx(0.0)
    assert offsets["t_stt_done"] == pytest.approx(400.0)
    assert offsets["t_llm_done"] is None  # never marked on this turn


def test_turn_table_has_one_row_per_turn_and_a_latency_column() -> None:
    table = turn_table([_turn(1, {LATENCY_MARK: 3000.0}), _turn(2, {LATENCY_MARK: 4000.0})])
    from rich.console import Console
    import io

    out = io.StringIO()
    Console(file=out, width=200).print(table)
    text = out.getvalue()
    assert "latência" in text
    assert "+3000" in text
    assert "+4000" in text
    # Unmarked stages render as an em dash rather than a bogus zero.
    assert "—" in text


def test_summary_table_reports_the_median_latency() -> None:
    import io

    from rich.console import Console

    table = summary_table([_turn(1, {LATENCY_MARK: 3000.0})])
    out = io.StringIO()
    Console(file=out, width=200).print(table)
    text = out.getvalue()
    assert "latência percebida (mediana)" in text
    assert "+3000" in text


def test_render_is_silent_on_an_empty_list() -> None:
    import io

    from rich.console import Console

    out = io.StringIO()
    Console(file=out, width=200)
    from tutor.timeline import render

    render([], Console(file=out, width=200))
    assert out.getvalue() == ""
