"""Assemble the system prompt from ``prompts/``.

Three files are concatenated, always in the same order:

``base.md`` + ``levels/<nível>.md`` + ``scenarios/<cenário>.md``

The scenario file also carries the tutor's **opening line**, which Phase 6
speaks as turn 0 before the student says anything. Everything is read from
disk on every call (a handful of small files) so editing a prompt needs no
restart, and a missing or empty file raises :class:`PromptError` naming the
path instead of silently producing a half prompt.

Kept deliberately small: a 4B model follows a short prompt with an example
much better than a long one full of rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tutor.config import ROOT

PROMPTS_DIR = ROOT / "prompts"
LEVELS_DIR = PROMPTS_DIR / "levels"
SCENARIOS_DIR = PROMPTS_DIR / "scenarios"

#: Used when the caller names no scenario; it is one of the six, not a fallback.
DEFAULT_SCENARIO = "free_chat"

_OPENING_RE = re.compile(r'^\s*Opening line:\s*["“](.+?)["”]\s*$', re.MULTILINE)

#: Rough tokens-per-word factor for the plan's ~600-token budget check.
TOKENS_PER_WORD = 1.4


class PromptError(RuntimeError):
    """A prompt file is missing, empty or malformed."""


@dataclass(frozen=True)
class PromptBundle:
    """The three files rendered into one system prompt, plus the greeting."""

    system: str
    greeting: str
    level: str
    scenario: str
    sources: tuple[Path, ...]

    @property
    def estimated_tokens(self) -> float:
        return len(self.system.split()) * TOKENS_PER_WORD


def _read(path: Path) -> str:
    if not path.exists():
        raise PromptError(f"prompt nao encontrado: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise PromptError(f"prompt vazio: {path}")
    return text


def _opening_line(scenario_path: Path, text: str) -> str:
    match = _OPENING_RE.search(text)
    if match is None:
        raise PromptError(
            f'{scenario_path.name}: falta uma linha `Opening line: "..."` '
            "(a IA precisa de uma frase de abertura para o turno 0)"
        )
    line = match.group(1).strip()
    if not line:
        raise PromptError(f"{scenario_path.name}: `Opening line` vazia")
    return line


def load_prompts(*, level: str, scenario: str | None = None) -> PromptBundle:
    """Read the three files for this level/scenario and join them."""
    scenario_name = scenario or DEFAULT_SCENARIO

    base_path = PROMPTS_DIR / "base.md"
    level_path = LEVELS_DIR / f"{level}.md"
    scenario_path = SCENARIOS_DIR / f"{scenario_name}.md"

    base = _read(base_path)
    level_text = _read(level_path)
    scenario_text = _read(scenario_path)
    greeting = _opening_line(scenario_path, scenario_text)

    system = "\n\n".join((base, level_text, scenario_text))
    return PromptBundle(
        system=system,
        greeting=greeting,
        level=level,
        scenario=scenario_name,
        sources=(base_path, level_path, scenario_path),
    )


def available_scenarios() -> list[str]:
    """Scenario names that actually have a file, sorted for stable output."""
    if not SCENARIOS_DIR.exists():
        return []
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.md"))


def available_levels() -> list[str]:
    if not LEVELS_DIR.exists():
        return []
    return sorted(p.stem for p in LEVELS_DIR.glob("*.md"))


# -- background prompts (Phase 8) ------------------------------------------- #

FEEDBACK_PATH = PROMPTS_DIR / "feedback.md"
SUMMARY_PATH = PROMPTS_DIR / "summary.md"
ROLLUP_PATH = PROMPTS_DIR / "rollup.md"

_LEVEL_HEADING = re.compile(r"(?m)^### ")


def build_feedback_system(level: str) -> str:
    """``feedback.md`` cut down to the common rules plus this level's rigor.

    All three rigor blocks live in one file so they stay in sync, but only the
    matching one is sent: a 4B model is distracted by rules for a level the
    learner is not at, and the plan caps the whole prompt at ~600 tokens.
    """
    text = _read(FEEDBACK_PATH)
    parts = _LEVEL_HEADING.split(text)
    head = parts[0].rstrip()
    wanted = f"### {level.upper()}"
    for part in parts[1:]:
        block = f"### {part.strip()}"
        if block.startswith(wanted) and (
            len(block) == len(wanted) or not block[len(wanted)].isalnum()
        ):
            return f"{head}\n\n{block}"
    raise PromptError(
        f"feedback.md: bloco de rigor '{wanted}' nao encontrado "
        "(esperado '### BEGINNER', '### INTERMEDIATE' ou '### ADVANCED')"
    )


def build_summary_system() -> str:
    return _read(SUMMARY_PATH)


def build_rollup_system() -> str:
    """``rollup.md`` — the digest of turns that just left the window (Phase 11)."""
    return _read(ROLLUP_PATH)


# -- spoken clarification (Phase 12) ---------------------------------------- #

CLARIFY_PATH = PROMPTS_DIR / "clarify.md"

_LABELED_LINE = re.compile(r"(?m)^(EN|PT):[ \t]*(.+?)[ \t]*$")


@dataclass(frozen=True)
class ClarifyLines:
    """The two lines the tutor says when a transcription cannot be trusted."""

    en: str
    pt: str


def load_clarify() -> ClarifyLines:
    """Read ``clarify.md``; a missing label is a broken asset, not a fallback.

    Both lines are user-facing speech, so they live in ``prompts/`` instead of
    being hardcoded in the session — the same rule the summary labels follow.
    """
    found = dict(_LABELED_LINE.findall(_read(CLARIFY_PATH)))
    missing = [label for label in ("EN", "PT") if not found.get(label)]
    if missing:
        raise PromptError(
            f"{CLARIFY_PATH.name}: falta a linha '{missing[0]}: ...'"
        )
    return ClarifyLines(en=found["EN"], pt=found["PT"])


#: ``NEW EXPRESSIONS: a; b; c`` — one line, semicolon separated, no list.
_NEW_EXPRESSIONS_RE = re.compile(r"(?im)^\s*NEW EXPRESSIONS:\s*(.+?)\s*$")


def parse_new_expressions(summary: str) -> list[str]:
    """Pull the deck half of a summary: the phrases worth keeping.

    Phase 9 stores them in ``vocab`` for ``python -m tutor review``. An absent
    line (the model ignored the format, or feedback was off) yields an empty
    list rather than an error: a missing vocabulary entry is never worth
    failing the session for.
    """
    match = _NEW_EXPRESSIONS_RE.search(summary)
    if match is None:
        return []
    found: list[str] = []
    for part in match.group(1).split(";"):
        # The model usually ends each phrase with a period; it is not part of
        # the expression and would break exact matching on review.
        phrase = part.strip().rstrip(".,").strip()
        if phrase and phrase.lower() != "none":
            found.append(phrase)
    return found
