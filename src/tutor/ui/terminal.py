"""Terminal front end (Phase 10a): a status panel, the transcript and the close.

The dashboard is a :class:`rich.live.Live` panel pinned at the bottom of the
screen while the session runs; everything that happens over time (the greeting,
each turn's transcript, its latency table, the blue correction box) is printed
above it as history, so the screen keeps the whole conversation instead of
overwriting it.

Two rules shape the class:

* **Nothing here talks to audio or to the model.** It only renders what
  :class:`~tutor.core.session.Session` and the CLI hand it, so it can be tested
  with no microphone and no llama-server.
* **States are keys, not prose.** ``state("speaking")`` maps to ``falando...``
  here; the pipeline and the session stay free of Portuguese.
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from tutor.core.feedback import FeedbackResult
from tutor.timeline import TurnTimeline, turn_table

#: State keys understood by :meth:`TerminalUI.state`, in the order they occur.
STATES: dict[str, str] = {
    "listening": "escutando...",
    "thinking": "pensando...",
    "speaking": "falando...",
    "closing": "encerrando...",
}

DEFAULT_STATE = "listening"


class TerminalUI:
    """Renders one conversation. One instance per session."""

    def __init__(
        self,
        console: Console,
        *,
        level: str = "",
        scenario: str = "",
        accent: str = "",
        hide_transcript: bool = False,
    ) -> None:
        self.console = console
        self.level = level
        self.scenario = scenario
        self.accent = accent
        # --hide-transcript: listening practice. The tutor's words are the
        # thing being trained, so they are withheld; the learner's own
        # transcription stays, it is feedback on their speech.
        self.hide_transcript = hide_transcript

        self.state_key = DEFAULT_STATE
        self.last_latency_ms: float | None = None
        self.last_user = ""
        self.last_tutor = ""
        self.last_feedback = ""
        self._live: Live | None = None

    # -- dashboard ---------------------------------------------------------- #

    def start(self) -> None:
        """Pin the panel. Without it every :meth:`state` call is a no-op, which
        is what the benchmark wants: it prints tables, not a dashboard."""
        if self._live is not None:
            return
        self._live = Live(
            self._panel(),
            console=self.console,
            refresh_per_second=6,
            vertical_overflow="visible",
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is None:
            return
        self._live.stop()
        self._live = None

    @property
    def state_label(self) -> str:
        return STATES.get(self.state_key, self.state_key)

    def state(self, key: str) -> None:
        """Move the dashboard to ``listening``/``thinking``/``speaking``/..."""
        if key == self.state_key:
            return
        self.state_key = key
        self._refresh()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._panel())

    def _tutor_line(self, text: str) -> str:
        if self.hide_transcript:
            return ""
        return text

    def _panel(self) -> Panel:
        body = Text()
        body.append("nivel     ", style="dim")
        body.append(f"{self.level or '-'} · {self.scenario or '-'}")
        if self.accent:
            body.append(f"  ·  {self.accent}", style="dim")
        body.append("\n")

        body.append("estado    ")
        body.append(self.state_label, style="bold cyan")
        body.append("\n")

        body.append("latencia  ")
        if self.last_latency_ms is None:
            body.append("-", style="dim")
        else:
            body.append(f"{self.last_latency_ms:.0f} ms", style="green")

        body.append("\n")
        body.append("voce      ")
        body.append(self.last_user or "-", style="bold" if self.last_user else "dim")

        body.append("\n")
        body.append("tutor     ")
        tutor = self._tutor_line(self.last_tutor)
        body.append(tutor or "(transcript oculto)", style="" if tutor else "dim")

        if self.last_feedback:
            body.append("\n")
            body.append("feedback  ")
            body.append(self.last_feedback, style="yellow")

        return Panel(body, title="tutor", border_style="cyan", padding=(0, 1))

    # -- history ------------------------------------------------------------ #

    def note(self, message: str) -> None:
        self.console.print(f"[dim]{message}[/dim]")

    def warn(self, message: str) -> None:
        self.console.print(f"[yellow]{message}[/yellow]")

    def report(self, tl: TurnTimeline, *, title: str | None = None) -> None:
        """Print one turn and feed the dashboard with it."""
        tutor = self._tutor_line(tl.text_out)
        if tl.turn == 0:
            self.console.print("\n[bold]abertura[/bold]")
            self.console.print(f"  tutor: {tutor or '(transcript oculto)'}")
        else:
            self.console.print(f"\n[bold]turno {tl.turn}[/bold]")
            self.console.print(f"  voce: {tl.text_in}")
            self.console.print(f"  tutor: {tutor or '(transcript oculto)'}")
        self.console.print(
            turn_table([tl], title=title or f"latencia - turno {tl.turn}")
        )
        self.last_user = tl.text_in or ""
        self.last_tutor = tl.text_out or ""
        self.last_latency_ms = tl.latency_ms
        self._refresh()

    def feedback(self, result: FeedbackResult) -> None:
        """The blue correction box: errors in red, fixes in green, never spoken."""
        fb = result.feedback
        req = result.request
        score = fb.score if fb.score is not None else "-"
        title = f"correcao - turno {req.turn}  nota {score}/10"
        lines: list[str] = [f"  [bold]voce:[/bold] {req.user_text}"]
        if fb.errors:
            lines.append("")
            for err in fb.errors:
                lines.append(
                    f"  [red]{err.original}[/red]  ->  [green]{err.fix}[/green]"
                    f"   [dim]({err.type})[/dim]"
                )
                lines.append(f"  [dim]{err.explanation_pt}[/dim]")
        else:
            lines.append("  [green]nenhum erro nesta frase.[/green]")
        if fb.better_phrasing:
            lines.append("")
            lines.append(f"  mais natural: [cyan]{fb.better_phrasing}[/cyan]")
        lines.append(
            f"  [dim]{result.backend}"
            + (f" - {result.attempts} tentativa(s)" if result.attempts > 1 else "")
            + "[/dim]"
        )
        self.console.print(Panel("\n".join(lines), title=title, border_style="blue"))

        bits = [f"nota {score}/10"]
        if fb.errors:
            kinds = sorted({err.type for err in fb.errors})
            bits.append(f"{len(fb.errors)} erro(s): {', '.join(kinds)}")
        else:
            bits.append("sem erros")
        self.last_feedback = " · ".join(bits)
        self._refresh()

    def summary(self, text: str) -> None:
        self.console.print(Panel(text, title="resumo da sessao", border_style="magenta"))


# -- the conversation loop --------------------------------------------------- #


async def converse(session: object, ui: TerminalUI, *, max_turns: int | None = None) -> bool:
    """Greet, then listen and answer until Ctrl+C (or ``max_turns``).

    Returns True when the user interrupted: the caller still owns the session,
    so the summary is generated by whoever opened it.
    """
    finished = 0
    try:
        ui.state("listening")
        await session.greet()
        while max_turns is None or finished < max_turns:
            utterance = await session.listen()
            timeline = await session.turn(utterance)
            if timeline is None:
                continue
            session.report(timeline)
            finished += 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        # One Ctrl+C: asyncio cancels the task, which lands here as
        # CancelledError. Swallowing it lets the summary still be generated
        # while the session (and its SQLite row) are open.
        return True
    return False


async def run_tutor(
    cfg: object,
    *,
    level_name: str,
    level: object,
    scenario: str,
    console: Console | None = None,
    hide_transcript: bool = False,
    max_turns: int | None = None,
    feedback: bool = True,
    streaming: bool = True,
) -> int:
    """Open a session, hold the conversation and print the closing summary."""
    from tutor.core.session import Session  # local: session imports this module

    out = console or Console()
    ui = TerminalUI(
        out,
        level=level_name,
        scenario=scenario,
        accent=getattr(getattr(cfg, "tts", None), "accent", ""),
        hide_transcript=hide_transcript,
    )
    ui.start()
    interrupted = False
    summary: str | None = None
    try:
        async with Session(
            cfg,
            level_name=level_name,
            level=level,
            scenario=scenario,
            ui=ui,
            streaming=streaming,
            feedback=feedback,
        ) as session:
            interrupted = await converse(session, ui, max_turns=max_turns)
            ui.state("closing")
            summary = await session.summarize()
    finally:
        ui.stop()
    # Live's last frame leaves no newline of its own; without this the closing
    # summary is glued to the dashboard's bottom border.
    out.print()

    if interrupted:
        ui.note("interrompido pelo usuario.")
    if summary:
        ui.summary(summary)
    else:
        ui.note("sem resumo (feedback desligado ou sessao vazia).")
    return 0
