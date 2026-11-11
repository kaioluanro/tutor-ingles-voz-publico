"""Smoke tests for the configuration files shipped with the project (Phase 1).

The pydantic models arrive in Phase 2; these tests already guard the contract
every later phase relies on: which sections and keys must exist.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

CONFIG_REQUIRED = {
    "audio": ["input_device", "output_device", "capture_sample_rate", "vad_chunk",
              "vad_threshold", "min_utterance_ms", "max_utterance_s",
              "pre_roll_ms", "post_playback_mute_ms"],
    "stt": ["model", "device", "compute_type", "beam_size"],
    "llm_local": ["base_url", "api_key", "model", "extra_body",
                  "temperature", "top_p", "top_k"],
    "llm_remote": ["enabled", "base_url", "api_key_env", "model",
                   "extra_body", "timeout_s"],
    "tts": ["engine", "device", "accent", "voice", "sample_rate"],
    "pipeline": ["sentence_queue_max", "min_words_per_chunk",
                 "history_max_turns", "context_tokens",
                 "feedback_queue", "feedback_timeout_s"],
    "storage": ["db_path"],
    "llama_server": ["binary", "model_path", "host", "port", "ctx_size",
                     "gpu_layers", "extra_args"],
}

LEVELS = ("beginner", "intermediate", "advanced")
LEVEL_REQUIRED = ("cefr", "max_sentences", "max_tokens", "temperature", "tts_speed",
                  "end_silence_ms", "allow_portuguese_help")


@pytest.fixture(scope="module")
def config() -> dict:
    with (ROOT / "config.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def levels() -> dict:
    with (ROOT / "levels.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_config_sections_exist(config: dict) -> None:
    missing = set(CONFIG_REQUIRED) - set(config)
    assert not missing, f"secoes faltando em config.yaml: {sorted(missing)}"


@pytest.mark.parametrize("section,keys", sorted(CONFIG_REQUIRED.items()))
def test_config_keys_exist(config: dict, section: str, keys: list[str]) -> None:
    missing = [k for k in keys if k not in config[section]]
    assert not missing, f"chaves faltando em config.yaml[{section}]: {missing}"


def test_levels_exist(levels: dict) -> None:
    missing = set(LEVELS) - set(levels)
    assert not missing, f"niveis faltando em levels.yaml: {sorted(missing)}"


@pytest.mark.parametrize("level", LEVELS)
def test_level_keys_exist(levels: dict, level: str) -> None:
    missing = [k for k in LEVEL_REQUIRED if k not in levels[level]]
    assert not missing, f"chaves faltando em levels.yaml[{level}]: {missing}"


def test_latency_budget_is_sane(config: dict) -> None:
    """The whole point of the project is low latency."""
    assert config["stt"]["model"] == "small.en"
    assert config["stt"]["device"] in {"cuda", "cpu"}
    assert config["tts"]["sample_rate"] == 24000
    assert config["pipeline"]["min_words_per_chunk"] >= 1
    assert config["audio"]["vad_chunk"] == 512


def test_secrets_are_not_hardcoded(config: dict) -> None:
    assert config["llm_local"]["api_key"] != ""
    assert config["llm_remote"]["api_key_env"] == "OPENROUTER_API_KEY"
    assert "api_key" not in config["llm_remote"]


def test_levels_speeds_are_ordered(config: dict) -> None:
    del config  # unused: speeds belong to levels
    with (ROOT / "levels.yaml").open(encoding="utf-8") as fh:
        levels = yaml.safe_load(fh)
    assert levels["beginner"]["tts_speed"] <= levels["intermediate"]["tts_speed"]
    assert levels["advanced"]["tts_speed"] >= levels["intermediate"]["tts_speed"]
    assert levels["beginner"]["end_silence_ms"] > levels["advanced"]["end_silence_ms"]
