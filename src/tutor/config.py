"""Typed configuration for the English tutor.

`config.yaml` holds machine/backend settings and `levels.yaml` holds the
per-level teaching settings. Both are strict: a missing section or a typo'd
field raises :class:`ConfigError` with a message that names the file, the
section and the field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

ROOT = Path(__file__).resolve().parents[2]

LevelName = Literal["beginner", "intermediate", "advanced"]
LEVEL_NAMES: tuple[LevelName, ...] = ("beginner", "intermediate", "advanced")

Device = Literal["cuda", "cpu"]
Accent = Literal["american", "british"]


class ConfigError(RuntimeError):
    """A config file is absent, incomplete or holds an invalid value."""


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_device: str | None
    output_device: str | None
    capture_sample_rate: int
    vad_chunk: int
    vad_threshold: float
    min_utterance_ms: int
    max_utterance_s: float
    pre_roll_ms: int
    post_playback_mute_ms: int


class SttConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    device: Device
    compute_type: str
    beam_size: int
    hotwords: str | None
    use_dialogue_context: bool
    min_avg_logprob: float
    max_compression_ratio: float


class LlmLocalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    api_key: str
    model: str
    extra_body: dict[str, Any]
    temperature: float
    top_p: float
    top_k: int


class LlmRemoteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    base_url: str
    api_key_env: str
    model: str
    extra_body: dict[str, Any]
    timeout_s: float


class TtsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str
    device: Device
    accent: Accent
    voice: str
    sample_rate: int
    feedback_voice: str
    feedback_lang: str


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence_queue_max: int
    min_words_per_chunk: int
    history_max_turns: int
    context_tokens: int
    feedback_queue: int
    feedback_timeout_s: float
    reconnect_timeout_s: float
    speak_feedback: bool
    speak_feedback_wait_s: float


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_path: str


class LlamaServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    binary: str
    model_path: str
    host: str
    port: int
    ctx_size: int
    gpu_layers: int
    extra_args: list[str]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: AudioConfig
    stt: SttConfig
    llm_local: LlmLocalConfig
    llm_remote: LlmRemoteConfig
    tts: TtsConfig
    pipeline: PipelineConfig
    storage: StorageConfig
    llama_server: LlamaServerConfig


class LevelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cefr: str
    max_sentences: int
    max_tokens: int
    temperature: float
    tts_speed: float
    end_silence_ms: int
    allow_portuguese_help: bool


def _default_path(filename: str) -> Path:
    """Prefer the working directory, fall back to the project root."""
    here = Path.cwd() / filename
    return here if here.exists() else ROOT / filename


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"arquivo de configuracao nao encontrado: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: YAML invalido\n{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: esperado um mapeamento no topo do arquivo")
    return data


def _render(path: Path, exc: ValidationError) -> str:
    lines = [f"{path.name}: configuracao invalida"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(raiz)"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate ``config.yaml``.

    Raises :class:`ConfigError` naming the file, the missing section and the
    missing field, so the user never has to read a pydantic traceback.
    """
    target = Path(path) if path is not None else _default_path("config.yaml")
    data = _read_yaml(target)

    missing = [name for name in AppConfig.model_fields if name not in data]
    if missing:
        listed = ", ".join(f"'{name}'" for name in missing)
        raise ConfigError(f"{target.name}: secao(oes) obrigatoria(s) ausente(s): {listed}")

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_render(target, exc)) from exc


def load_levels(path: str | Path | None = None) -> dict[str, LevelConfig]:
    """Load and validate ``levels.yaml`` keyed by level name."""
    target = Path(path) if path is not None else _default_path("levels.yaml")
    data = _read_yaml(target)

    missing = [name for name in LEVEL_NAMES if name not in data]
    if missing:
        listed = ", ".join(f"'{name}'" for name in missing)
        raise ConfigError(f"{target.name}: nivel(is) ausente(s): {listed}")

    levels: dict[str, LevelConfig] = {}
    for name, raw in data.items():
        if not isinstance(raw, dict):
            raise ConfigError(f"{target.name}: nivel '{name}' precisa ser um mapeamento")
        try:
            levels[name] = LevelConfig.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(_render(target, exc)) from exc
    return levels


def load_all(
    config_path: str | Path | None = None,
    levels_path: str | Path | None = None,
) -> tuple[AppConfig, dict[str, LevelConfig]]:
    """Convenience wrapper returning both files, used by the CLI entry point."""
    return load_config(config_path), load_levels(levels_path)
