"""Tests for the configuration loader and the LLM router (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tutor.__main__ import main
from tutor.config import (
    LEVEL_NAMES,
    AppConfig,
    ConfigError,
    load_all,
    load_config,
    load_levels,
)
from tutor.llm.openai_compat import filter_extra_body
from tutor.llm.router import get_conversation_llm, get_feedback_llm

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
LEVELS = ROOT / "levels.yaml"


# --------------------------------------------------------------------------- #
# config.yaml
# --------------------------------------------------------------------------- #

def test_default_config_loads() -> None:
    cfg = load_config(CONFIG)
    assert isinstance(cfg, AppConfig)
    assert cfg.audio.capture_sample_rate == 16000
    assert cfg.audio.vad_chunk == 512
    assert cfg.stt.model == "small.en"
    assert cfg.llm_local.extra_body["cache_prompt"] is True
    assert cfg.llm_local.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert cfg.storage.db_path.endswith(".db")


def test_config_missing_section_is_reported(tmp_path: Path) -> None:
    raw = CONFIG.read_text(encoding="utf-8")
    broken = tmp_path / "config.yaml"
    # drop the whole `stt:` block
    head, _, rest = raw.partition("\nstt:")
    _, _, tail = rest.partition("\nllm_local:")
    broken.write_text(f"{head}\nllm_local:{tail}", encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_config(broken)
    assert "stt" in str(exc.value)
    assert "secao" in str(exc.value)


def test_config_missing_field_is_reported(tmp_path: Path) -> None:
    # every section present, but the audio fields stripped
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["audio"] = {}
    broken = tmp_path / "config.yaml"
    broken.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_config(broken)
    message = str(exc.value)
    assert "config.yaml" in message
    assert "audio" in message
    # every missing field of the audio section is listed, not just the first
    assert "vad_threshold" in message and "capture_sample_rate" in message


def test_config_typo_field_is_rejected(tmp_path: Path) -> None:
    raw = CONFIG.read_text(encoding="utf-8").replace(
        "  vad_threshold: 0.5", "  vad_treshold: 0.5"
    )
    broken = tmp_path / "config.yaml"
    broken.write_text(raw, encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_config(broken)
    message = str(exc.value)
    assert "vad_treshold" in message or "vad_threshold" in message


def test_config_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "nope.yaml")
    assert "nao encontrado" in str(exc.value)


def test_config_invalid_yaml_is_reported(tmp_path: Path) -> None:
    broken = tmp_path / "config.yaml"
    broken.write_text("audio: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_config(broken)
    assert "YAML" in str(exc.value)


def test_config_rejects_unknown_top_level_section(tmp_path: Path) -> None:
    raw = CONFIG.read_text(encoding="utf-8") + "\nunknown_section:\n  a: 1\n"
    broken = tmp_path / "config.yaml"
    broken.write_text(raw, encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_config(broken)
    assert "unknown_section" in str(exc.value)


# --------------------------------------------------------------------------- #
# levels.yaml
# --------------------------------------------------------------------------- #

def test_levels_load() -> None:
    levels = load_levels(LEVELS)
    assert set(levels) >= set(LEVEL_NAMES)
    assert levels["beginner"].max_tokens == 80
    assert levels["advanced"].max_tokens == 220
    assert levels["beginner"].tts_speed < levels["intermediate"].tts_speed
    assert levels["beginner"].end_silence_ms > levels["advanced"].end_silence_ms
    assert levels["beginner"].allow_portuguese_help is True


def test_levels_missing_level_is_reported(tmp_path: Path) -> None:
    broken = tmp_path / "levels.yaml"
    broken.write_text("beginner:\n  cefr: A1\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_levels(broken)
    message = str(exc.value)
    assert "levels.yaml" in message
    assert "intermediate" in message and "advanced" in message


def test_levels_bad_value_is_reported(tmp_path: Path) -> None:
    broken = tmp_path / "levels.yaml"
    broken.write_text(
        LEVELS.read_text(encoding="utf-8").replace("max_tokens: 80", "max_tokens: nope"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as exc:
        load_levels(broken)
    assert "max_tokens" in str(exc.value)


def test_load_all_returns_both() -> None:
    cfg, levels = load_all(CONFIG, LEVELS)
    assert cfg.llm_local.model
    assert set(levels) >= set(LEVEL_NAMES)


# --------------------------------------------------------------------------- #
# router + request shape
# --------------------------------------------------------------------------- #

def test_conversation_always_local() -> None:
    cfg = load_config(CONFIG)
    client = get_conversation_llm(cfg)
    assert client.local_backend is True
    assert client.model == cfg.llm_local.model


def test_feedback_falls_back_to_local_when_remote_disabled() -> None:
    cfg = load_config(CONFIG)
    assert cfg.llm_remote.enabled is False
    client = get_feedback_llm(cfg)
    assert client.local_backend is True


def test_feedback_uses_remote_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(CONFIG)
    cfg.llm_remote.enabled = True
    cfg.llm_remote.model = "some/model:free"
    monkeypatch.setenv(cfg.llm_remote.api_key_env, "test-key")

    client = get_feedback_llm(cfg)
    assert client.local_backend is False
    assert client.model == "some/model:free"


def test_remote_without_api_key_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(CONFIG)
    cfg.llm_remote.enabled = True
    cfg.llm_remote.model = "some/model:free"
    monkeypatch.delenv(cfg.llm_remote.api_key_env, raising=False)

    with pytest.raises(ConfigError) as exc:
        get_feedback_llm(cfg)
    assert cfg.llm_remote.api_key_env in str(exc.value)


def test_remote_without_model_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(CONFIG)
    cfg.llm_remote.enabled = True
    monkeypatch.setenv(cfg.llm_remote.api_key_env, "test-key")

    with pytest.raises(ConfigError) as exc:
        get_feedback_llm(cfg)
    assert "model" in str(exc.value)


def test_local_request_carries_backend_specific_fields() -> None:
    cfg = load_config(CONFIG)
    kwargs = get_conversation_llm(cfg).build_request(
        [{"role": "user", "content": "hi"}], max_tokens=80
    )

    extra = kwargs["extra_body"]
    assert extra["cache_prompt"] is True
    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra["top_k"] == 20
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.8
    assert kwargs["max_tokens"] == 80
    assert kwargs["stream"] is False


def test_remote_request_never_carries_local_only_fields() -> None:
    # whatever a user puts in llm_remote.extra_body, local-only keys are dropped
    assert filter_extra_body(
        {"cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}},
        local_backend=False,
    ) == {}
    assert filter_extra_body(
        {"cache_prompt": True, "top_k": 5}, local_backend=False
    ) == {"top_k": 5}
    assert filter_extra_body(
        {"cache_prompt": True, "top_k": 5}, local_backend=True
    ) == {"cache_prompt": True, "top_k": 5}


def test_json_mode_adds_response_format() -> None:
    cfg = load_config(CONFIG)
    kwargs = get_conversation_llm(cfg).build_request(
        [{"role": "user", "content": "hi"}], max_tokens=80, json_mode=True
    )
    assert kwargs["response_format"] == {"type": "json_object"}


def test_temperature_override_wins() -> None:
    cfg = load_config(CONFIG)
    kwargs = get_conversation_llm(cfg).build_request(
        [{"role": "user", "content": "hi"}], max_tokens=80, temperature=0.1
    )
    assert kwargs["temperature"] == 0.1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_lists_levels(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--config", str(CONFIG), "--levels", str(LEVELS), "--list-levels"])
    assert code == 0
    assert "beginner" in capsys.readouterr().out


def test_cli_rejects_unknown_level() -> None:
    code = main(
        ["--config", str(CONFIG), "--levels", str(LEVELS), "--level", "expert"]
    )
    assert code == 2


def test_cli_rejects_unknown_scenario(tmp_path: Path) -> None:
    code = main(
        [
            "--config", str(CONFIG),
            "--levels", str(LEVELS),
            "--scenario", "definitely-not-a-scenario",
        ]
    )
    assert code == 2


def test_cli_accepts_scenario_path(tmp_path: Path) -> None:
    scenario = tmp_path / "restaurant.md"
    scenario.write_text("# Restaurant\n", encoding="utf-8")
    code = main(
        [
            "--config", str(CONFIG),
            "--levels", str(LEVELS),
            "--scenario", str(scenario),
            "--accent", "british",
            "--level", "advanced",
            "--check",
        ]
    )
    assert code == 0


def test_cli_reports_broken_config(tmp_path: Path) -> None:
    broken = tmp_path / "config.yaml"
    broken.write_text("audio: {}\n", encoding="utf-8")
    code = main(["--config", str(broken), "--levels", str(LEVELS)])
    assert code == 2
