"""Failure handling (Phase 11): the backend dies, the turn must not.

Everything here runs without a microphone, without a GPU and without
llama-server: the session is built directly, its devices are replaced by fakes,
and only the classification/probing logic is exercised against a real socket.
"""

from __future__ import annotations

import asyncio
import http.server
import threading
from pathlib import Path

import httpx
import openai
import pytest

from tutor.config import load_all
from tutor.core.session import Session
from tutor.errors import DeviceError, describe_device_failure
from tutor.llm.errors import describe, is_llm_unavailable
from tutor.llm.health import health_url, probe, wait_for_server
from tutor.stt import Transcript

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
LEVELS = ROOT / "levels.yaml"


# --------------------------------------------------------------------------- #
# liveness probe
# --------------------------------------------------------------------------- #

def test_health_url_strips_the_v1_prefix() -> None:
    assert health_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080/health"
    assert health_url("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080/health"


def test_health_url_leaves_other_paths_alone() -> None:
    assert health_url("https://openrouter.ai/api/v1") == (
        "https://openrouter.ai/api/health"
    )
    assert health_url("http://localhost") == "http://localhost/health"


class _Server:
    """``GET /health`` on an ephemeral port, with a switchable status code."""

    def __init__(self, status: int) -> None:
        self.status = status
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler
                body = b'{"status":"ok"}' if owner.status == 200 else b""
                self.send_response(owner.status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                return  # keep pytest's output clean

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/health"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return self.url[: -len("/health")]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def test_probe_accepts_a_live_server() -> None:
    server = _Server(200)
    try:
        assert probe(server.url) is True
    finally:
        server.close()


def test_probe_rejects_a_loading_server() -> None:
    """llama-server answers 503 while the model is still loading."""
    server = _Server(503)
    try:
        assert probe(server.url) is False
    finally:
        server.close()


def test_probe_is_false_when_nothing_is_listening() -> None:
    assert probe("http://127.0.0.1:1/health", timeout=0.5) is False


def test_wait_for_server_accepts_a_live_url() -> None:
    server = _Server(200)
    try:
        assert asyncio.run(
            wait_for_server(server.base_url, timeout=2.0, poll_s=0.05)
        ) is True
    finally:
        server.close()


def test_wait_for_server_gives_up_on_a_dead_url() -> None:
    assert (
        asyncio.run(wait_for_server("http://127.0.0.1:1/v1", timeout=0.3, poll_s=0.05))
        is False
    )


def test_wait_for_server_sees_a_server_that_comes_up() -> None:
    """The learner waited while llama-server loaded — and it came back."""
    server = _Server(503)
    threading.Timer(0.4, lambda: setattr(server, "status", 200)).start()
    try:
        assert asyncio.run(
            wait_for_server(server.base_url, timeout=3.0, poll_s=0.1)
        ) is True
    finally:
        server.close()


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #

def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(
        request=httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    )


def _status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    return openai.APIStatusError(
        f"http {status}", response=httpx.Response(status, request=request), body=None
    )


def test_connection_error_is_unavailable() -> None:
    assert is_llm_unavailable(_connection_error()) is True


def test_server_error_is_unavailable() -> None:
    assert is_llm_unavailable(_status_error(503)) is True
    assert "HTTP 503" in describe(_status_error(503))


def test_bad_request_is_not_unavailable() -> None:
    """A 4xx is our bug: retrying the same payload cannot fix it."""
    assert is_llm_unavailable(_status_error(400)) is False


def test_unrelated_exception_is_not_unavailable() -> None:
    assert is_llm_unavailable(ValueError("bad prompt")) is False
    assert is_llm_unavailable(KeyError("messages")) is False


def test_describe_is_portuguese_and_short() -> None:
    assert describe(_connection_error()) == "sem conexao"
    assert describe(TimeoutError()) == "tempo esgotado"
    assert describe(ValueError("anything")) == "ValueError"


# --------------------------------------------------------------------------- #
# devices
# --------------------------------------------------------------------------- #

def test_device_error_names_the_device_and_lists_the_alternatives() -> None:
    message = describe_device_failure("input", OSError("No such device"))
    error = DeviceError(message)
    assert "sem microfone" in str(error)
    assert "No such device" in str(error)
    assert "audio.input_device" in str(error)
    assert "input:" in str(error)


def test_device_error_for_the_speaker_says_saida() -> None:
    assert "sem dispositivo de saida" in describe_device_failure(
        "output", OSError("busy")
    )


# --------------------------------------------------------------------------- #
# session: one dead turn must not end the session
# --------------------------------------------------------------------------- #

class _FakePlayback:
    def __init__(self) -> None:
        self.started_at = None
        self.stopped = 0
        self.ended = 0
        self.fed = 0

    def stop(self) -> None:
        self.stopped += 1

    def end_of_stream(self) -> None:
        self.ended += 1

    def feed(self, chunk: object) -> None:
        self.fed += 1

    async def wait_until_done(self) -> None:
        return None


class _FakeTTS:
    sample_rate = 24000

    async def synthesize_stream(self, text: str, *, speed: float, lang: str = "en"):
        if False:  # pragma: no cover - makes this an async generator
            yield b""


class _FakeASR:
    """Stands in for :class:`~tutor.stt.WhisperSTT`, down to the new kwargs."""

    def __init__(self, text: str = "hello", exc: BaseException | None = None) -> None:
        self.text = text
        self.exc = exc
        self.prompts: list[str | None] = []

    async def transcribe(
        self, audio: object, *, prompt: str | None = None, hotwords: str | None = None
    ) -> Transcript:
        if self.exc is not None:
            raise self.exc
        self.prompts.append(prompt)
        return Transcript(self.text, avg_logprob=-0.4)


class _ScriptedLLM:
    """Dies on call 1, answers on call 2 — or always dies if ``always``."""

    def __init__(self, exc: BaseException, *, always: bool = False) -> None:
        self.exc = exc
        self.always = always
        self.calls = 0

    async def stream(self, messages, *, max_tokens, temperature=None):
        self.calls += 1
        if self.always or self.calls == 1:
            raise self.exc
        yield "Hello there."

    async def complete(self, messages, *, max_tokens, json_mode=False) -> str:
        raise self.exc


async def _never(*args: object, **kwargs: object) -> bool:
    """Stand-in for ``wait_for_server`` when the backend stays down."""
    return False


async def _healed(*args: object, **kwargs: object) -> bool:
    return True


def _session(tmp_path: Path) -> Session:
    cfg, levels = load_all(CONFIG, LEVELS)
    cfg.storage.db_path = str(tmp_path / "tutor.db")
    session = Session(
        cfg,
        level_name="beginner",
        level=levels["beginner"],
        scenario="restaurant",
        listen=False,
    )
    session.playback = _FakePlayback()  # type: ignore[assignment]
    session.tts = _FakeTTS()  # type: ignore[assignment]
    return session


def _close(session: Session) -> None:
    if session.db is not None:
        session.db.close()
        session.db = None


def test_dead_backend_drops_one_turn_and_keeps_the_session(
    tmp_path: Path, capsys
) -> None:
    session = _session(tmp_path)
    session.asr = _FakeASR()  # type: ignore[assignment]
    dead = _ScriptedLLM(_connection_error(), always=True)
    session.llm = dead  # type: ignore[assignment]

    async def scenario() -> object:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("tutor.core.session.wait_for_server", _never)
            return await session.turn(b"\x00" * 1600)

    try:
        result = asyncio.run(scenario())
    finally:
        _close(session)

    assert result is None
    assert dead.calls == 1  # gave up instead of looping forever
    assert session.idle.is_set()  # the feedback gate is open again
    out = capsys.readouterr().out
    assert "LLM indisponivel" in out
    assert "turno descartado" in out


def test_backend_that_recovers_is_retried_once(tmp_path: Path, capsys) -> None:
    session = _session(tmp_path)
    session.asr = _FakeASR()  # type: ignore[assignment]
    flaky = _ScriptedLLM(_connection_error())
    session.llm = flaky  # type: ignore[assignment]

    async def scenario() -> object:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("tutor.core.session.wait_for_server", _healed)
            return await session.turn(b"\x00" * 1600)

    try:
        timeline = asyncio.run(scenario())
    finally:
        _close(session)

    assert timeline is not None
    assert timeline.text_out == "Hello there."
    assert flaky.calls == 2
    assert "reconectando" in capsys.readouterr().out


def test_empty_transcription_still_just_discards_the_turn(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.asr = _FakeASR(text="   ")  # type: ignore[assignment]

    async def scenario() -> object:
        return await session.turn(b"\x00" * 1600)

    try:
        assert asyncio.run(scenario()) is None
    finally:
        _close(session)


def test_stt_failure_discards_the_turn_without_touching_the_llm(
    tmp_path: Path, capsys
) -> None:
    session = _session(tmp_path)
    session.asr = _FakeASR(exc=RuntimeError("wav corrompido"))  # type: ignore[assignment]
    dead = _ScriptedLLM(_connection_error(), always=True)
    session.llm = dead  # type: ignore[assignment]

    async def scenario() -> object:
        return await session.turn(b"\x00" * 1600)

    try:
        assert asyncio.run(scenario()) is None
    finally:
        _close(session)

    assert dead.calls == 0
    assert "transcricao falhou" in capsys.readouterr().out
