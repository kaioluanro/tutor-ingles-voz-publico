"""Which backend serves which job.

* Conversation → **always local**: it sits on the latency critical path.
* Feedback/resume → **remote when enabled**, because it runs in the background
  and does not block the user. Falling back to the local model on a remote
  timeout only while the system is idle is wired up in Phase 7.
"""

from __future__ import annotations

import os

from tutor.config import AppConfig, ConfigError, LlmLocalConfig, LlmRemoteConfig
from tutor.llm.base import LLMClient
from tutor.llm.openai_compat import OpenAICompatClient

# A hosted provider is not required to be fast, but it must not be retried
# silently: the caller decides whether to fall back.
_REMOTE_MAX_RETRIES = 0


def _local_client(cfg: LlmLocalConfig) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        extra_body=cfg.extra_body,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        local_backend=True,
        max_retries=0,
    )


def _remote_client(cfg: LlmRemoteConfig) -> OpenAICompatClient:
    api_key = os.environ.get(cfg.api_key_env, "")
    if not api_key:
        raise ConfigError(
            f"llm_remote habilitado, mas a variavel de ambiente "
            f"'{cfg.api_key_env}' nao esta definida"
        )
    return OpenAICompatClient(
        base_url=cfg.base_url,
        api_key=api_key,
        model=cfg.model,
        extra_body=cfg.extra_body,
        timeout_s=cfg.timeout_s,
        local_backend=False,
        max_retries=_REMOTE_MAX_RETRIES,
    )


def get_conversation_llm(cfg: AppConfig) -> LLMClient:
    """Interactive dialogue always runs on the local model."""
    return _local_client(cfg.llm_local)


def get_feedback_llm(cfg: AppConfig) -> LLMClient:
    """Background feedback uses the remote model when it is turned on."""
    if cfg.llm_remote.enabled:
        if not cfg.llm_remote.model:
            raise ConfigError("llm_remote habilitado mas 'model' esta vazio")
        return _remote_client(cfg.llm_remote)
    return _local_client(cfg.llm_local)
