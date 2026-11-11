"""Local persistence (Phase 9): sessions, turns, corrections and the review deck.

Everything is stdlib :mod:`sqlite3` — no ORM, no migrations, one ``CREATE TABLE IF
NOT EXISTS`` script run on connect. The database is opened by whoever needs it
(``Session`` while talking, ``python -m tutor stats``/``review`` after) and
closed again, so there is no long-lived connection to leak.

Deviation from the plan's schema, recorded in README §2: ``turns`` carries a
``score`` column. The plan asks ``stats`` for "evolução da nota média" and
"nota média das últimas 5 sessões", and the only place a score exists is
``Feedback.score`` — without it on the turn there is nothing to average.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

#: Leitner boxes are 1..5 (plan Fase 9).
MAX_BOX = 5
#: "nota média ≥ 8 sobe, ≤ 4 desce" (plan Fase 9).
UP_THRESHOLD = 8.0
DOWN_THRESHOLD = 4.0
#: How many sessions the level suggestion looks at (plan Fase 9).
SUGGESTION_WINDOW = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    level      TEXT NOT NULL,
    scenario   TEXT NOT NULL,
    accent     TEXT NOT NULL,
    summary    TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    user_text  TEXT NOT NULL,
    ai_text    TEXT NOT NULL,
    latency_ms INTEGER,
    score      INTEGER CHECK (score IS NULL OR (score BETWEEN 0 AND 10)),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id        INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    type           TEXT NOT NULL,
    original       TEXT NOT NULL,
    fix            TEXT NOT NULL,
    explanation_pt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vocab (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    expression       TEXT NOT NULL UNIQUE,
    source_session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    created_at       TEXT NOT NULL,
    last_review      TEXT,
    box              INTEGER NOT NULL DEFAULT 1 CHECK (box BETWEEN 1 AND 5)
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_errors_turn   ON errors(turn_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SessionRow:
    """One line of ``stats``."""

    id: int
    started_at: str
    level: str
    scenario: str
    turns: int
    avg_score: float | None
    errors: int


@dataclass(frozen=True)
class ErrorCount:
    type: str
    count: int


@dataclass(frozen=True)
class VocabCard:
    id: int
    expression: str
    box: int
    last_review: str | None
    is_new: bool


@dataclass(frozen=True)
class ErrorCard:
    original: str
    fix: str
    type: str
    explanation_pt: str
    count: int


class Database:
    """A single sqlite3 connection with the Phase 9 schema applied."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        # Without this, ON DELETE CASCADE is silently ignored.
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- lifecycle --------------------------------------------------------- #

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _exec(self, sql: str, params: Sequence[object] = ()) -> sqlite3.Cursor:
        cur = self._conn.execute(sql, tuple(params))
        self._conn.commit()
        return cur

    # -- writes ------------------------------------------------------------ #

    def create_session(self, *, level: str, scenario: str, accent: str) -> int:
        cur = self._exec(
            "INSERT INTO sessions (started_at, level, scenario, accent) "
            "VALUES (?, ?, ?, ?)",
            (_now(), level, scenario, accent),
        )
        return int(cur.lastrowid)

    def finish_session(self, session_id: int, *, summary: str | None = None) -> None:
        """Close the session. Safe to call twice; the summary is only overwritten
        when one is actually supplied."""
        if summary:
            self._exec(
                "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
                (_now(), summary, session_id),
            )
        else:
            self._exec(
                "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (_now(), session_id),
            )

    def add_turn(
        self,
        *,
        session_id: int,
        idx: int,
        user_text: str,
        ai_text: str = "",
        latency_ms: float | None = None,
    ) -> int:
        """Insert one turn.

        It is written as soon as the utterance is transcribed, *before* the model
        answers, because the background correction of that very turn can land
        while the audio is still playing and needs a row to attach to.
        """
        cur = self._exec(
            "INSERT INTO turns (session_id, idx, user_text, ai_text, latency_ms, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                idx,
                user_text,
                ai_text,
                None if latency_ms is None else round(latency_ms),
                _now(),
            ),
        )
        return int(cur.lastrowid)

    def update_turn(
        self, turn_id: int, *, ai_text: str, latency_ms: float | None
    ) -> None:
        self._exec(
            "UPDATE turns SET ai_text = ?, latency_ms = ? WHERE id = ?",
            (ai_text, None if latency_ms is None else round(latency_ms), turn_id),
        )

    def set_turn_score(self, turn_id: int, score: int | None) -> None:
        self._exec("UPDATE turns SET score = ? WHERE id = ?", (score, turn_id))

    def add_error(
        self,
        *,
        turn_id: int,
        type: str,
        original: str,
        fix: str,
        explanation_pt: str,
    ) -> int:
        cur = self._exec(
            "INSERT INTO errors (turn_id, type, original, fix, explanation_pt) "
            "VALUES (?, ?, ?, ?, ?)",
            (turn_id, type, original, fix, explanation_pt),
        )
        return int(cur.lastrowid)

    def add_vocab(self, *, expression: str, source_session_id: int | None) -> int | None:
        """Add a new expression to the deck.

        Returns ``None`` when it is already there: re-meeting a phrase must not
        reset the Leitner box the learner has climbed to.
        """
        expression = expression.strip()
        if not expression:
            return None
        cur = self._exec(
            "INSERT INTO vocab (expression, source_session_id, created_at, box) "
            "VALUES (?, ?, ?, 1) ON CONFLICT(expression) DO NOTHING",
            (expression, source_session_id, _now()),
        )
        return None if cur.rowcount == 0 else int(cur.lastrowid)

    def review_answer(self, vocab_id: int, *, correct: bool) -> int:
        """Leitner: right climbs one box (capped at 5), wrong falls back to 1."""
        row = self._conn.execute(
            "SELECT box FROM vocab WHERE id = ?", (vocab_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"vocab id {vocab_id} nao existe")
        box = row["box"] + 1 if correct else 1
        box = min(box, MAX_BOX)
        self._exec(
            "UPDATE vocab SET box = ?, last_review = ? WHERE id = ?",
            (box, _now(), vocab_id),
        )
        return box

    # -- reads ------------------------------------------------------------- #

    def recent_sessions(self, limit: int = 10) -> list[SessionRow]:
        rows = self._conn.execute(
            "SELECT s.id, s.started_at, s.level, s.scenario,"
            "       COUNT(DISTINCT t.id) AS turns,"
            "       AVG(t.score) AS avg_score,"
            "       COUNT(DISTINCT e.id) AS errors"
            "  FROM sessions s"
            "  LEFT JOIN turns  t ON t.session_id = s.id"
            "  LEFT JOIN errors e ON e.turn_id = t.id"
            " GROUP BY s.id"
            " ORDER BY s.id DESC"
            " LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            SessionRow(
                id=r["id"],
                started_at=r["started_at"],
                level=r["level"],
                scenario=r["scenario"],
                turns=r["turns"],
                avg_score=r["avg_score"],
                errors=r["errors"],
            )
            for r in rows
        ]

    def current_level(self) -> str | None:
        """The level of the newest session — what ``stats`` compares against."""
        row = self._conn.execute(
            "SELECT level FROM sessions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else str(row["level"])

    def error_counts(self, limit: int = 10) -> list[ErrorCount]:
        rows = self._conn.execute(
            "SELECT type, COUNT(*) AS n FROM errors"
            " GROUP BY type ORDER BY n DESC, type ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [ErrorCount(type=r["type"], count=r["n"]) for r in rows]

    def score_history(self, limit: int = SUGGESTION_WINDOW) -> list[tuple[int, float]]:
        """``(session_id, average)`` pairs, **oldest first**, scored sessions only.

        Sessions with no feedback at all are skipped: an unscored session is
        not a zero, it is a session nobody was graded on.
        """
        rows = self._conn.execute(
            "SELECT s.id, AVG(t.score) AS avg_score"
            "  FROM sessions s JOIN turns t ON t.session_id = s.id"
            " WHERE t.score IS NOT NULL"
            " GROUP BY s.id ORDER BY s.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(int(r["id"]), float(r["avg_score"])) for r in reversed(rows)]

    def session_scores(self, limit: int = SUGGESTION_WINDOW) -> list[float]:
        """Just the averages of :meth:`score_history`, oldest first."""
        return [average for _, average in self.score_history(limit)]

    def suggest_level(
        self, current: str, order: Sequence[str], *, window: int = SUGGESTION_WINDOW
    ) -> str | None:
        """Plan Fase 9: last ``window`` sessions' mean ≥ 8 → up, ≤ 4 → down.

        Only a suggestion — nothing here ever moves the learner on its own.
        """
        scores = self.session_scores(window)
        if not scores or current not in order:
            return None
        average = sum(scores) / len(scores)
        index = list(order).index(current)
        if average >= UP_THRESHOLD:
            return order[index + 1] if index + 1 < len(order) else None
        if average <= DOWN_THRESHOLD:
            return order[index - 1] if index > 0 else None
        return None

    def new_vocab(self, limit: int = 10) -> list[VocabCard]:
        """Cards never reviewed, oldest first."""
        return self._vocab(
            "WHERE last_review IS NULL ORDER BY id ASC LIMIT ?", limit
        )

    def due_vocab(self, limit: int = 10) -> list[VocabCard]:
        """Oldest-reviewed cards first (there are no box intervals: the plan's
        Leitner is deliberately simple)."""
        return self._vocab(
            "WHERE last_review IS NOT NULL ORDER BY last_review ASC, id ASC LIMIT ?",
            limit,
        )

    def _vocab(self, where: str, limit: int) -> list[VocabCard]:
        rows = self._conn.execute(
            f"SELECT id, expression, box, last_review FROM vocab {where}", (limit,)
        ).fetchall()
        return [
            VocabCard(
                id=r["id"],
                expression=r["expression"],
                box=r["box"],
                last_review=r["last_review"],
                is_new=r["last_review"] is None,
            )
            for r in rows
        ]

    def vocab_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM vocab").fetchone()[0])

    def recurring_errors(self, *, min_count: int = 2, limit: int = 10) -> list[ErrorCard]:
        """Mistakes the learner repeated — the deck half ``review`` shows."""
        rows = self._conn.execute(
            "SELECT original, fix, type, explanation_pt, COUNT(*) AS n"
            "   FROM errors"
            "  GROUP BY original, fix, type, explanation_pt"
            " HAVING COUNT(*) >= ?"
            "  ORDER BY n DESC, original ASC"
            " LIMIT ?",
            (min_count, limit),
        ).fetchall()
        return [
            ErrorCard(
                original=r["original"],
                fix=r["fix"],
                type=r["type"],
                explanation_pt=r["explanation_pt"],
                count=r["n"],
            )
            for r in rows
        ]
