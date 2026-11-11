"""Local persistence (Phase 9).

Everything runs against an in-memory database: no files, no fixtures on disk,
and the whole suite stays milliseconds long. The scenarios mirror what
``Session`` actually writes, so a schema change that breaks the session shows
up here instead of during a conversation.
"""

from __future__ import annotations

import pytest

from tutor.core.prompts import parse_new_expressions
from tutor.storage import Database
from tutor.storage.db import MAX_BOX, DOWN_THRESHOLD, UP_THRESHOLD

ORDER = ["beginner", "intermediate", "advanced"]


@pytest.fixture()
def db() -> Database:
    database = Database(":memory:")
    yield database
    database.close()


def open_session(
    db: Database,
    *,
    level: str = "beginner",
    scenario: str = "restaurant",
    accent: str = "american",
) -> int:
    return db.create_session(level=level, scenario=scenario, accent=accent)


def score_turn(
    db: Database,
    session_id: int,
    *,
    idx: int,
    score: int | None,
    latency_ms: float = 500.0,
    user_text: str = "I goes to the market",
    ai_text: str = "Where did you go?",
) -> tuple[int, int]:
    """Write one turn exactly the way ``Session`` does, plus its correction."""
    turn_id = db.add_turn(
        session_id=session_id, idx=idx, user_text=user_text, ai_text=""
    )
    db.update_turn(turn_id, ai_text=ai_text, latency_ms=latency_ms)
    if score is not None:
        db.set_turn_score(turn_id, score)
        db.add_error(
            turn_id=turn_id,
            type="grammar",
            original=user_text,
            fix="I went to the market",
            explanation_pt="Passado simples.",
        )
    return turn_id, db.recent_sessions(1)[0].errors


# -- schema ------------------------------------------------------------------ #


def test_schema_applies_on_connect(db: Database) -> None:
    tables = {
        row["name"]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {"sessions", "turns", "errors", "vocab"} <= tables


def test_session_roundtrip(db: Database) -> None:
    session_id = open_session(db, level="intermediate", accent="british")
    assert db.current_level() == "intermediate"

    db.finish_session(session_id, summary="MOST FREQUENT ERRORS: nada")
    row = db.recent_sessions(1)[0]
    assert row.id == session_id
    assert row.level == "intermediate"
    assert row.scenario == "restaurant"
    assert row.started_at
    raw = db._conn.execute(
        "SELECT ended_at, summary FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    assert raw["ended_at"], "finish_session nao gravou ended_at"
    assert "MOST FREQUENT" in raw["summary"]


def test_current_level_is_none_on_empty_db() -> None:
    with Database(":memory:") as empty:
        assert empty.current_level() is None
        assert empty.recent_sessions() == []


def test_turn_stores_latency_and_answer(db: Database) -> None:
    session_id = open_session(db)
    turn_id = db.add_turn(session_id=session_id, idx=1, user_text="hello")
    db.update_turn(turn_id, ai_text="Hi there!", latency_ms=527.4)

    row = db.recent_sessions(1)[0]
    assert row.turns == 1
    stored = db._conn.execute(
        "SELECT ai_text, latency_ms FROM turns WHERE id = ?", (turn_id,)
    ).fetchone()
    assert stored["ai_text"] == "Hi there!"
    assert stored["latency_ms"] == 527  # rounded, an INTEGER column


def test_negative_latency_is_stored_as_null(db: Database) -> None:
    """A greeting turn has no ``t_playback_start`` mark; NULL beats a lie."""
    session_id = open_session(db)
    turn_id = db.add_turn(session_id=session_id, idx=1, user_text="x")
    db.update_turn(turn_id, ai_text="y", latency_ms=None)
    assert (
        db._conn.execute(
            "SELECT latency_ms FROM turns WHERE id = ?", (turn_id,)
        ).fetchone()["latency_ms"]
        is None
    )


def test_deleting_a_session_cascades(db: Database) -> None:
    session_id = open_session(db)
    turn_id, _ = score_turn(db, session_id, idx=1, score=6)
    assert db.error_counts()

    db._exec("DELETE FROM sessions WHERE id = ?", (session_id,))
    assert db.recent_sessions() == []
    assert db.error_counts() == []
    assert db._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0
    assert db._conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0] == 0


def test_score_outside_range_is_rejected(db: Database) -> None:
    session_id = open_session(db)
    turn_id = db.add_turn(session_id=session_id, idx=1, user_text="x")
    with pytest.raises(Exception):
        db.set_turn_score(turn_id, 11)


# -- stats ------------------------------------------------------------------- #


def test_error_counts_are_ordered(db: Database) -> None:
    session_id = open_session(db)
    for idx, kind in enumerate(("grammar", "grammar", "articles", "vocab"), start=1):
        turn_id = db.add_turn(session_id=session_id, idx=idx, user_text="x")
        db.add_error(
            turn_id=turn_id,
            type=kind,
            original="x",
            fix="y",
            explanation_pt="porque sim",
        )
    counts = db.error_counts()
    assert counts[0].type == "grammar" and counts[0].count == 2
    assert [c.type for c in counts] == ["grammar", "articles", "vocab"]


def test_session_scores_are_oldest_first_and_ignore_unscored(db: Database) -> None:
    first = open_session(db)
    score_turn(db, first, idx=1, score=8)
    second = open_session(db)
    score_turn(db, second, idx=1, score=4)
    # A session with no feedback at all must not drag the average anywhere.
    open_session(db)

    assert db.session_scores() == [8.0, 4.0]


def test_score_history_keeps_the_session_ids_in_order(db: Database) -> None:
    """``stats`` labels the chart with the session it is showing, so the ids
    must survive the reversal to oldest-first."""
    first = open_session(db)
    score_turn(db, first, idx=1, score=5)
    second = open_session(db)
    score_turn(db, second, idx=1, score=9)
    assert db.score_history() == [(first, 5.0), (second, 9.0)]


def test_recent_sessions_reports_turns_score_and_errors(db: Database) -> None:
    session_id = open_session(db)
    score_turn(db, session_id, idx=1, score=9)
    score_turn(db, session_id, idx=2, score=7, user_text="fine")

    row = db.recent_sessions(1)[0]
    assert row.turns == 2
    assert row.avg_score == pytest.approx(8.0)
    assert row.errors == 2


# -- level suggestion -------------------------------------------------------- #


def test_suggest_level_climbs_at_or_above_threshold(db: Database) -> None:
    for _ in range(5):
        session_id = open_session(db, level="beginner")
        score_turn(db, session_id, idx=1, score=UP_THRESHOLD)
    assert db.suggest_level("beginner", ORDER) == "intermediate"


def test_suggest_level_descends_at_or_below_threshold(db: Database) -> None:
    for _ in range(5):
        session_id = open_session(db, level="intermediate")
        score_turn(db, session_id, idx=1, score=DOWN_THRESHOLD)
    assert db.suggest_level("intermediate", ORDER) == "beginner"


def test_suggest_level_silent_in_the_middle(db: Database) -> None:
    session_id = open_session(db, level="intermediate")
    score_turn(db, session_id, idx=1, score=6)
    assert db.suggest_level("intermediate", ORDER) is None


def test_suggest_level_never_climbs_past_advanced(db: Database) -> None:
    session_id = open_session(db, level="advanced")
    for idx in range(5):
        score_turn(db, session_id, idx=idx + 1, score=10)
    assert db.suggest_level("advanced", ORDER) is None


def test_suggest_level_only_looks_at_the_last_five(db: Database) -> None:
    """The window is the last 5 sessions, not the whole history."""
    for _ in range(5):
        low = open_session(db, level="intermediate")
        score_turn(db, low, idx=1, score=2)
    # One good session is inside the window and lifts the mean, but the four
    # bad ones before it still keep the average at or below the floor.
    good = open_session(db, level="intermediate")
    score_turn(db, good, idx=1, score=UP_THRESHOLD)
    assert len(db.session_scores()) == 5
    assert sum(db.session_scores()) / 5 <= DOWN_THRESHOLD
    assert db.suggest_level("intermediate", ORDER) == "beginner"


def test_suggest_level_empty_or_unknown(db: Database) -> None:
    assert db.suggest_level("beginner", ORDER) is None
    session_id = open_session(db)
    score_turn(db, session_id, idx=1, score=10)
    assert db.suggest_level("martian", ORDER) is None


# -- vocab / Leitner --------------------------------------------------------- #


def test_add_vocab_creates_box_one(db: Database) -> None:
    session_id = open_session(db)
    vocab_id = db.add_vocab(expression="I popped into the market", source_session_id=session_id)
    assert vocab_id is not None
    card = db.new_vocab()[0]
    assert card.expression == "I popped into the market"
    assert card.box == 1 and card.is_new


def test_add_vocab_is_idempotent_and_keeps_the_box(db: Database) -> None:
    session_id = open_session(db)
    vocab_id = db.add_vocab(expression="pop in", source_session_id=session_id)
    assert vocab_id is not None
    for _ in range(3):
        db.review_answer(vocab_id, correct=True)
    assert db.due_vocab()[0].box == 4

    # Meeting the phrase again in a later session must not reset the climb.
    assert db.add_vocab(expression="pop in", source_session_id=session_id) is None
    assert db.vocab_count() == 1
    assert db.due_vocab()[0].box == 4


def test_add_vocab_ignores_blank(db: Database) -> None:
    assert db.add_vocab(expression="   ", source_session_id=None) is None
    assert db.vocab_count() == 0


def test_leitner_climbs_to_five_then_stops(db: Database) -> None:
    vocab_id = db.add_vocab(expression="steady", source_session_id=None)
    assert vocab_id is not None
    boxes = [db.review_answer(vocab_id, correct=True) for _ in range(6)]
    assert boxes == [2, 3, 4, 5, 5, 5] and max(boxes) == MAX_BOX


def test_leitner_wrong_answer_returns_to_box_one(db: Database) -> None:
    vocab_id = db.add_vocab(expression="slip", source_session_id=None)
    assert vocab_id is not None
    for _ in range(3):
        db.review_answer(vocab_id, correct=True)
    assert db.review_answer(vocab_id, correct=False) == 1
    assert db.due_vocab()[0].box == 1


def test_review_answer_unknown_id(db: Database) -> None:
    with pytest.raises(KeyError):
        db.review_answer(999, correct=True)


def test_new_vocab_comes_before_reviewed(db: Database) -> None:
    fresh_id = db.add_vocab(expression="fresh", source_session_id=None)
    old_id = db.add_vocab(expression="old", source_session_id=None)
    assert old_id is not None
    db.review_answer(old_id, correct=True)
    assert [c.expression for c in db.new_vocab() + db.due_vocab()] == ["fresh", "old"]
    assert fresh_id is not None


# -- recurring errors -------------------------------------------------------- #


def test_recurring_errors_need_two_occurrences(db: Database) -> None:
    session_id = open_session(db)
    for idx in range(3):
        turn_id = db.add_turn(session_id=session_id, idx=idx + 1, user_text="I goes")
        db.add_error(
            turn_id=turn_id,
            type="grammar",
            original="I goes",
            fix="I go",
            explanation_pt="Concordancia sujeito-verbo.",
        )
    # A one-off mistake is noise, not a pattern.
    once = db.add_turn(session_id=session_id, idx=4, user_text="an apple")
    db.add_error(
        turn_id=once, type="articles", original="an apple", fix="a apple",
        explanation_pt="Artigo indefinido.",
    )

    cards = db.recurring_errors()
    assert len(cards) == 1
    assert cards[0].original == "I goes" and cards[0].count == 3
    assert cards[0].fix == "I go"


# -- summary parsing --------------------------------------------------------- #


def test_parse_new_expressions() -> None:
    summary = (
        "MOST FREQUENT ERRORS: erros de artigo.\n"
        "NEW EXPRESSIONS: I popped into the market; a table for two; "
        "I am happy to talk with you; how are you feeling; see you later.\n"
        "STRENGTHS: boa escuta.\n"
        "FOCUS FOR NEXT TIME: use o passado simples.\n"
    )
    assert parse_new_expressions(summary) == [
        "I popped into the market",
        "a table for two",
        "I am happy to talk with you",
        "how are you feeling",
        "see you later",
    ]


def test_parse_new_expressions_strips_trailing_periods() -> None:
    got = parse_new_expressions("NEW EXPRESSIONS: Nice to meet you.")
    assert got == ["Nice to meet you"]


def test_parse_new_expressions_missing_line_is_empty() -> None:
    assert parse_new_expressions("MOST FREQUENT ERRORS: nada repetido.") == []
    assert parse_new_expressions("") == []


# -- the acceptance scenario ------------------------------------------------- #


def test_stats_and_review_after_two_sessions(db: Database) -> None:
    """Plan Fase 9: "sessao salva corretamente; stats e review apos 2 sessoes"."""
    for round_ in range(2):
        session_id = open_session(db, level="beginner", scenario="restaurant")
        for idx in range(2):
            turn_id, _ = score_turn(db, session_id, idx=idx + 1, score=7)
            db.add_error(
                turn_id=turn_id,
                type="grammar",
                original="I goes",
                fix="I go",
                explanation_pt="Concordancia.",
            )
        db.finish_session(session_id, summary="MOST FREQUENT ERRORS: concordancia")
        for phrase in parse_new_expressions("NEW EXPRESSIONS: I popped in; a table for two"):
            db.add_vocab(expression=phrase, source_session_id=session_id)

    sessions = db.recent_sessions(10)
    assert len(sessions) == 2
    assert all(s.turns == 2 and s.avg_score == 7 for s in sessions)
    # 2 turns x (1 from score_turn + 1 added below) = 4 corrections each.
    assert all(s.errors == 4 for s in sessions)

    counts = db.error_counts()
    # ``error_counts`` groups by type: 2 sessions x 4 corrections, all grammar.
    assert counts and counts[0].type == "grammar" and counts[0].count == 8
    assert db.vocab_count() == 2
    assert len(db.new_vocab()) == 2
    # Both originals repeat 4x, so both belong in the review deck.
    assert {c.count for c in db.recurring_errors()} == {4}
    assert db.suggest_level("beginner", ORDER) is None  # 7 sits in the middle

    fresh = db.new_vocab()[0]
    assert db.review_answer(fresh.id, correct=True) == 2
