"""Terminal front end (Phase 10a).

The two things the plan asks this phase to get right are pure rendering — the
status panel and ``--hide-transcript`` — plus the conversation loop that has to
survive Ctrl+C without losing the closing summary. None of it needs a
microphone or a model: the session is a stub and the console is a recorder.
"""

from __future__ import annotations

import asyncio

from rich.console import Console

from tutor.timeline import TurnTimeline
from tutor.ui.terminal import STATES, TerminalUI, converse


def make_console() -> Console:
    return Console(record=True, width=100, force_terminal=True)


def make_timeline(turn: int = 1) -> TurnTimeline:
    tl = TurnTimeline(turn=turn)
    tl.mark("t_speech_end")
    tl.mark("t_playback_start")
    tl.text_in = "My name is Kyle."
    tl.text_out = "Nice to meet you, Kyle."
    return tl


def output(ui: TerminalUI) -> str:
    return ui.console.export_text()


# -- the panel --------------------------------------------------------------- #


def test_states_are_portuguese_and_covers_the_three_the_plan_names() -> None:
    assert set(STATES) == {"listening", "thinking", "speaking", "closing"}
    assert STATES["listening"] == "escutando..."
    assert STATES["thinking"] == "pensando..."
    assert STATES["speaking"] == "falando..."


def test_panel_shows_level_scenario_state_and_latency() -> None:
    ui = TerminalUI(
        make_console(), level="beginner", scenario="restaurant", accent="american"
    )
    ui.state("thinking")
    ui.report(make_timeline())

    panel = ui._panel()
    text = panel.renderable
    body = text.plain
    assert "beginner" in body and "restaurant" in body
    assert STATES["thinking"] in body
    assert "ms" in body  # a latency was recorded by the timeline
    assert "american" in body


def test_panel_starts_listening_with_no_latency() -> None:
    ui = TerminalUI(make_console(), level="advanced", scenario="free_chat")
    assert ui.last_latency_ms is None
    body = ui._panel().renderable.plain
    assert STATES["listening"] in body
    assert "latencia" in body and "-" in body


def test_state_change_is_recorded_even_without_a_live_panel() -> None:
    """The benchmark never calls start(); state() must still be usable."""
    ui = TerminalUI(make_console())
    ui.state("speaking")
    assert ui.state_key == "speaking"
    assert ui.state_label == STATES["speaking"]
    ui.state("speaking")  # repeated calls are a no-op, no flicker
    assert ui.state_key == "speaking"


def test_unknown_state_key_falls_back_to_itself() -> None:
    ui = TerminalUI(make_console())
    ui.state("whatever")
    assert ui.state_label == "whatever"


# -- --hide-transcript ------------------------------------------------------- #


def test_report_prints_the_tutor_text_by_default() -> None:
    ui = TerminalUI(make_console(), level="beginner", scenario="restaurant")
    ui.report(make_timeline())
    text = output(ui)
    assert "My name is Kyle." in text
    assert "Nice to meet you, Kyle." in text


def test_hide_transcript_withholds_only_the_learners_listening_target() -> None:
    """Listening practice: the tutor's words are the exercise, so they go; the
    learner's own transcription stays as feedback on their speech."""
    ui = TerminalUI(make_console(), hide_transcript=True)
    ui.report(make_timeline())
    text = output(ui)
    assert "Nice to meet you, Kyle." not in text
    assert "My name is Kyle." in text
    assert "transcript oculto" in text
    assert "tutor" in text


def test_hide_transcript_also_hides_the_greeting() -> None:
    ui = TerminalUI(make_console(), hide_transcript=True)
    ui.report(make_timeline(turn=0))
    assert "Nice to meet you, Kyle." not in output(ui)


def test_hide_transcript_reaches_the_panel_too() -> None:
    ui = TerminalUI(make_console(), hide_transcript=True)
    ui.report(make_timeline())
    body = ui._panel().renderable.plain
    assert "Nice to meet you, Kyle." not in body
    assert "transcript oculto" in body


def test_latency_reaches_the_panel() -> None:
    ui = TerminalUI(make_console())
    ui.report(make_timeline())
    assert ui.last_latency_ms is not None and ui.last_latency_ms >= 0


# -- feedback and summary ---------------------------------------------------- #


def test_feedback_box_shows_the_error_the_fix_and_the_score() -> None:
    from tutor.core.feedback import Feedback, FeedbackError, FeedbackRequest, FeedbackResult

    ui = TerminalUI(make_console())
    result = FeedbackResult(
        request=FeedbackRequest(
            user_text="I goes", ai_text="Where did you go?", level="beginner", turn=1
        ),
        feedback=Feedback(
            corrected="I went to the market.",
            errors=[
                FeedbackError(
                    original="I goes",
                    fix="I went",
                    type="grammar",
                    explanation_pt="Passado simples.",
                )
            ],
            score=6,
        ),
        backend="local",
        attempts=1,
    )
    ui.feedback(result)
    text = output(ui)
    assert "I goes" in text and "I went" in text
    assert "Passado simples." in text
    assert "correcao - turno 1" in text and "nota 6/10" in text
    # The dashboard gets a one-line digest of the same box.
    assert ui.last_feedback.startswith("nota 6/10")
    assert "grammar" in ui.last_feedback


def test_summary_panel() -> None:
    ui = TerminalUI(make_console())
    ui.summary("MOST FREQUENT ERRORS: nada")
    assert "MOST FREQUENT ERRORS: nada" in output(ui)


# -- the conversation loop --------------------------------------------------- #


class StubSession:
    """Enough of Session for converse(): greet, listen, turn, report."""

    def __init__(self, *, answers: int = 2, interrupt_on: int | None = None) -> None:
        self.answers = answers
        self.interrupt_on = interrupt_on
        self.reported: list[TurnTimeline] = []
        self._calls = 0

    async def greet(self) -> None:
        return None

    async def listen(self) -> str:
        self._calls += 1
        if self.interrupt_on is not None and self._calls >= self.interrupt_on:
            raise asyncio.CancelledError
        return f"utterance {self._calls}"

    async def turn(self, audio: str) -> TurnTimeline | None:
        if self._calls > self.answers:
            return None
        return make_timeline(turn=self._calls)

    def report(self, tl: TurnTimeline, *, title: str | None = None) -> None:
        self.reported.append(tl)


def test_converse_runs_the_requested_number_of_turns() -> None:
    ui = TerminalUI(make_console())
    session = StubSession(answers=2)

    async def scenario() -> bool:
        return await converse(session, ui, max_turns=2)

    interrupted = asyncio.run(scenario())
    assert interrupted is False
    assert len(session.reported) == 2


def test_converse_stops_at_the_turn_limit() -> None:
    ui = TerminalUI(make_console())
    session = StubSession(answers=10)

    async def scenario() -> int:
        await converse(session, ui, max_turns=3)
        return session._calls

    assert asyncio.run(scenario()) == 3


def test_converse_reports_interruption_without_losing_the_session() -> None:
    """Ctrl+C lands as CancelledError; converse() must return True so the
    caller can still generate the summary while the session is open."""
    ui = TerminalUI(make_console())
    session = StubSession(answers=5, interrupt_on=2)

    async def scenario() -> bool:
        return await converse(session, ui)

    interrupted = asyncio.run(scenario())
    assert interrupted is True
    # The first turn did happen before the interrupt.
    assert len(session.reported) == 1


def test_discarded_turns_do_not_count_toward_the_limit() -> None:
    """``--turns`` counts accepted conversations; an empty transcription must
    not eat one of them."""
    ui = TerminalUI(make_console())

    class FlakySession(StubSession):
        async def turn(self, audio: str) -> TurnTimeline | None:
            # The first two utterances are transcribed as nothing.
            if self._calls < 3:
                return None
            return make_timeline(turn=self._calls)

    session = FlakySession(answers=10)

    async def scenario() -> tuple[int, int]:
        await converse(session, ui, max_turns=1)
        return session._calls, len(session.reported)

    calls, reported = asyncio.run(scenario())
    assert calls == 3
    assert reported == 1


def test_panel_is_optional_so_the_benchmark_prints_plain_tables() -> None:
    ui = TerminalUI(make_console())
    ui.state("thinking")  # no start(): must not touch a Live that is not there
    ui.stop()
    assert ui._live is None
