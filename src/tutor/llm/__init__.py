"""LLM client abstraction: local backend for chat, remote backend for feedback."""

from tutor.llm.base import LLMClient, Message
from tutor.llm.errors import describe, is_llm_unavailable
from tutor.llm.health import health_url, wait_for_server
from tutor.llm.openai_compat import OpenAICompatClient
from tutor.llm.router import get_conversation_llm, get_feedback_llm

__all__ = [
    "LLMClient",
    "Message",
    "OpenAICompatClient",
    "describe",
    "health_url",
    "is_llm_unavailable",
    "wait_for_server",
    "get_conversation_llm",
    "get_feedback_llm",
]
