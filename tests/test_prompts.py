"""Tests for prompt assembly (Phase 7).

Two things matter here: the pieces land in the right order, and a missing file
fails loudly instead of silently producing a half prompt. The size check is
part of the plan's acceptance — a 4B model follows a short prompt with an
example far better than a long one full of rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.config import LEVEL_NAMES, ROOT
from tutor.core import prompts as prompts_mod
from tutor.core.prompts import (
    DEFAULT_SCENARIO,
    PROMPTS_DIR,
    PromptError,
    TOKENS_PER_WORD,
    available_scenarios,
    load_prompts,
)

SCENARIOS = ("free_chat", "restaurant", "job_interview", "travel", "doctor", "small_talk")
BUDGET_TOKENS = 600.0


# -- every file the plan requires exists ------------------------------------ #


def test_prompt_files_exist_on_disk() -> None:
    assert (PROMPTS_DIR / "base.md").exists()
    for level in LEVEL_NAMES:
        assert (PROMPTS_DIR / "levels" / f"{level}.md").exists(), level
    for scenario in SCENARIOS:
        assert (PROMPTS_DIR / "scenarios" / f"{scenario}.md").exists(), scenario


def test_every_level_and_scenario_combination_loads() -> None:
    for level in LEVEL_NAMES:
        for scenario in SCENARIOS:
            bundle = load_prompts(level=level, scenario=scenario)
            assert bundle.system
            assert bundle.greeting


def test_available_scenarios_matches_the_directory() -> None:
    assert set(available_scenarios()) == set(SCENARIOS)


# -- shape of the assembled prompt ------------------------------------------ #


def test_parts_are_joined_base_then_level_then_scenario() -> None:
    base = (PROMPTS_DIR / "base.md").read_text(encoding="utf-8").strip()
    level = (PROMPTS_DIR / "levels" / "beginner.md").read_text(encoding="utf-8").strip()
    scenario = (PROMPTS_DIR / "scenarios" / "restaurant.md").read_text(encoding="utf-8").strip()

    system = load_prompts(level="beginner", scenario="restaurant").system
    assert system == f"{base}\n\n{level}\n\n{scenario}"
    # and each part shows up exactly once
    assert system.count(base) == 1
    assert system.count("LEVEL: BEGINNER") == 1
    assert system.count("SCENARIO: Restaurant.") == 1


def test_system_prompt_is_spoken_english_only() -> None:
    """The TTS reads whatever comes back, so no markdown may reach it."""
    for scenario in SCENARIOS:
        system = load_prompts(level="advanced", scenario=scenario).system
        assert "```" not in system
        assert not any(ch in system for ch in ("•", "❌", "✅", "####"))


# -- the budget the plan sets ----------------------------------------------- #


@pytest.mark.parametrize("level", list(LEVEL_NAMES))
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_combined_prompt_fits_the_token_budget(level: str, scenario: str) -> None:
    bundle = load_prompts(level=level, scenario=scenario)
    assert bundle.estimated_tokens <= BUDGET_TOKENS, (
        f"{level}/{scenario}: {bundle.estimated_tokens:.0f} tokens "
        f"(limite {BUDGET_TOKENS:.0f})"
    )


def test_token_estimate_uses_the_plan_formula() -> None:
    bundle = load_prompts(level="beginner", scenario="restaurant")
    assert bundle.estimated_tokens == pytest.approx(
        len(bundle.system.split()) * TOKENS_PER_WORD
    )


def test_every_level_prompt_carries_a_few_shot_example() -> None:
    """A 4B model follows an example better than a paragraph of rules."""
    for level in LEVEL_NAMES:
        text = (PROMPTS_DIR / "levels" / f"{level}.md").read_text(encoding="utf-8")
        assert "Learner:" in text and "Tutor:" in text, level


# -- the opening line (turn 0) ---------------------------------------------- #


def test_greeting_is_read_from_the_scenario() -> None:
    bundle = load_prompts(level="beginner", scenario="restaurant")
    assert bundle.greeting == (
        "Good evening! Welcome in. Table for one, or are you waiting for someone?"
    )


def test_greeting_never_keeps_the_quotes() -> None:
    for scenario in SCENARIOS:
        greeting = load_prompts(level="beginner", scenario=scenario).greeting
        assert not greeting.startswith(('"', "“")), scenario
        assert greeting.strip() == greeting


def test_missing_scenario_falls_back_to_free_chat() -> None:
    assert load_prompts(level="beginner").scenario == DEFAULT_SCENARIO
    assert DEFAULT_SCENARIO in available_scenarios()


# -- clear errors ------------------------------------------------------------ #


@pytest.fixture()
def empty_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A prompts/ tree with nothing in it, pointed at by the module."""
    (tmp_path / "levels").mkdir()
    (tmp_path / "scenarios").mkdir()
    monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", tmp_path)
    monkeypatch.setattr(prompts_mod, "LEVELS_DIR", tmp_path / "levels")
    monkeypatch.setattr(prompts_mod, "SCENARIOS_DIR", tmp_path / "scenarios")
    return tmp_path


def test_missing_base_names_the_path(empty_prompts: Path) -> None:
    with pytest.raises(PromptError) as exc:
        load_prompts(level="beginner")
    assert "base.md" in str(exc.value)
    assert str(empty_prompts) in str(exc.value)


def test_missing_level_names_the_level(empty_prompts: Path) -> None:
    (empty_prompts / "base.md").write_text("base", encoding="utf-8")
    with pytest.raises(PromptError) as exc:
        load_prompts(level="advanced")
    assert "advanced.md" in str(exc.value)


def test_missing_scenario_names_the_scenario(empty_prompts: Path) -> None:
    (empty_prompts / "base.md").write_text("base", encoding="utf-8")
    (empty_prompts / "levels" / "beginner.md").write_text("level", encoding="utf-8")
    with pytest.raises(PromptError) as exc:
        load_prompts(level="beginner", scenario="restaurant")
    assert "restaurant.md" in str(exc.value)


def test_empty_file_is_an_error_not_an_empty_prompt(empty_prompts: Path) -> None:
    (empty_prompts / "base.md").write_text("   \n", encoding="utf-8")
    with pytest.raises(PromptError, match="prompt vazio"):
        load_prompts(level="beginner")


def test_scenario_without_opening_line_is_rejected(empty_prompts: Path) -> None:
    (empty_prompts / "base.md").write_text("base", encoding="utf-8")
    (empty_prompts / "levels" / "beginner.md").write_text("level", encoding="utf-8")
    (empty_prompts / "scenarios" / "free_chat.md").write_text(
        "SCENARIO: Free chat.\nNo opening line here.\n", encoding="utf-8"
    )
    with pytest.raises(PromptError) as exc:
        load_prompts(level="beginner")
    assert "Opening line" in str(exc.value)
    assert "free_chat.md" in str(exc.value)


def test_error_is_a_runtime_error_so_the_cli_can_catch_it() -> None:
    assert issubclass(PromptError, RuntimeError)


def test_sources_are_reported_for_debugging() -> None:
    bundle = load_prompts(level="intermediate", scenario="travel")
    assert [p.name for p in bundle.sources] == [
        "base.md",
        "intermediate.md",
        "travel.md",
    ]
    assert all(p.is_absolute() or ROOT in p.parents for p in bundle.sources)
