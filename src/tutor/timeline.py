"""Per-turn latency marks, exactly as the plan's §9 defines them.

Every mark is a ``time.perf_counter()`` timestamp taken by whichever stage
completes it. The turn's origin is ``t_speech_end`` — the instant the VAD
declared the user finished speaking — so the perceived latency is simply
``t_playback_start − t_speech_end``.

The marks are meaningful for both generations of the pipeline: in the Phase 5
sequential baseline ``t_llm_done`` precedes everything TTS-related, while in the
Phase 6 streaming pipeline ``t_playback_start`` comes *before* ``t_llm_done``.
That inversion is the whole point of the benchmark.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from rich.table import Table

# (mark, label) in the order given by plan §9. The first entry doubles as the
# turn origin, so it is not shown as a column of its own.
MARKS: tuple[tuple[str, str], ...] = (
    ("t_speech_end", "fim da fala"),
    ("t_stt_done", "texto pronto"),
    ("t_llm_first_token", "1o token"),
    ("t_first_sentence", "1a frase>TTS"),
    ("t_tts_first_audio", "1o audio"),
    ("t_playback_start", "playback"),
    ("t_llm_done", "fim LLM"),
)

ORIGIN: str = MARKS[0][0]
LATENCY_MARK: str = "t_playback_start"


def _fmt(ms: float | None) -> str:
    return "—" if ms is None else f"{ms:+7.0f}"


@dataclass
class TurnTimeline:
    """Marks and texts for one conversation turn."""

    turn: int = 0
    text_in: str = ""
    text_out: str = ""
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str, when: float | None = None) -> float:
        """Record a timestamp; a repeat for the same name is ignored.

        ``when`` lets a caller report a time captured elsewhere — playback's
        ``started_at`` comes from the PortAudio callback thread, which is more
        accurate than stamping it once the event loop gets round to it.
        """
        ts = time.perf_counter() if when is None else when
        self.marks.setdefault(name, ts)
        return ts

    @property
    def origin(self) -> float | None:
        return self.marks.get(ORIGIN)

    def offset_ms(self, name: str) -> float | None:
        """Milliseconds from ``t_speech_end``, or ``None`` if unmarked."""
        origin, ts = self.origin, self.marks.get(name)
        if origin is None or ts is None:
            return None
        return (ts - origin) * 1000.0

    @property
    def latency_ms(self) -> float | None:
        """Perceived latency: end of the user's speech to first audible audio."""
        return self.offset_ms(LATENCY_MARK)

    @property
    def complete(self) -> bool:
        """True when every mark of §9 was taken (usable for statistics)."""
        return all(name in self.marks for name, _ in MARKS)


def _usable(timelines: Sequence[TurnTimeline]) -> list[TurnTimeline]:
    return [tl for tl in timelines if tl.latency_ms is not None]


def median_latency_ms(timelines: Sequence[TurnTimeline]) -> float | None:
    values = [tl.latency_ms for tl in _usable(timelines)]
    return statistics.median(values) if values else None


def mean_latency_ms(timelines: Sequence[TurnTimeline]) -> float | None:
    values = [tl.latency_ms for tl in _usable(timelines)]
    return statistics.fmean(values) if values else None


def median_offsets_ms(timelines: Sequence[TurnTimeline]) -> dict[str, float | None]:
    """Median offset of every §9 mark, for the stage-by-stage breakdown."""
    usable = _usable(timelines)
    out: dict[str, float | None] = {}
    for name, _ in MARKS:
        values = [tl.offset_ms(name) for tl in usable]
        values = [v for v in values if v is not None]
        out[name] = statistics.median(values) if values else None
    return out


def turn_table(
    timelines: Sequence[TurnTimeline], *, title: str = "Latência por turno"
) -> Table:
    """One row per turn, every §9 mark in milliseconds from ``t_speech_end``."""
    table = Table(title=title, show_lines=False)
    table.add_column("turno", justify="right")
    for _, label in MARKS[1:]:
        table.add_column(label, justify="right")
    table.add_column("latência", justify="right", style="bold")

    for tl in timelines:
        row = [str(tl.turn)]
        row.extend(_fmt(tl.offset_ms(name)) for name, _ in MARKS[1:])
        row.append(_fmt(tl.latency_ms))
        table.add_row(*row)
    return table


def summary_table(timelines: Sequence[TurnTimeline]) -> Table:
    """Median of each mark across all usable turns."""
    offsets = median_offsets_ms(timelines)
    med = median_latency_ms(timelines)
    mean = mean_latency_ms(timelines)

    table = Table(title="Resumo (mediana por marca)", show_lines=False)
    table.add_column("marca")
    table.add_column("mediana", justify="right")
    for name, label in MARKS:
        table.add_row(label, _fmt(offsets.get(name)))
    table.add_section()
    table.add_row("latência percebida (mediana)", _fmt(med))
    table.add_row("latência percebida (média)", _fmt(mean))
    return table


def render(timelines: Sequence[TurnTimeline], console: object) -> None:
    """Print both tables. ``console`` is any object with a ``.print`` method."""
    printable = [tl for tl in timelines if tl.marks]
    if not printable:
        return
    console.print(turn_table(printable))
    if median_latency_ms(printable) is not None:
        console.print(summary_table(printable))
