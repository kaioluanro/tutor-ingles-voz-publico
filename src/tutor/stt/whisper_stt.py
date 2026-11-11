"""Speech-to-text on top of ``faster-whisper``.

The model is loaded once per process and reused for every turn; loading costs
about 1.5 s and must never happen inside the interaction loop. Decoding runs in
a dedicated one-thread executor so two turns can never touch the same
``WhisperModel`` concurrently.

Decoding takes an optional ``prompt``/``hotwords`` so the decoder is biased
towards the words this conversation is actually about, and it returns a
:class:`Transcript` instead of a bare string because the confidence numbers
(``avg_logprob``, ``compression_ratio``) are what tell the session apart from a
sentence Whisper guessed at.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass
import ctypes
import pathlib
import sys

import numpy as np

Audio = np.ndarray

# ctranslate2 (faster-whisper) needs cuBLAS 12 + cuDNN 9 at runtime. On a system
# without a CUDA 12 toolkit those libs only exist inside the pip `nvidia-*`
# packages, which are not on the dynamic linker path. They must be loaded with
# RTLD_GLOBAL *before* the first `import faster_whisper`, otherwise the import
# itself fails with "libcublas.so.12 is not found".
_CUDNN_ORDER = (
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn.so.9",
)
_CUBLAS_ORDER = ("libcublasLt.so.12", "libcublas.so.12")
_preloaded = False


def preload_cuda_libs() -> list[str]:
    """Load the pip-provided cuDNN/cuBLAS into this process; returns what loaded."""
    global _preloaded
    if _preloaded:
        return []
    _preloaded = True

    site = (
        pathlib.Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info[0]}.{sys.version_info[1]}"
        / "site-packages"
    )
    loaded: list[str] = []
    for name in (*_CUDNN_ORDER, *_CUBLAS_ORDER):
        pkg = "cudnn" if name.startswith("libcudnn") else "cublas"
        path = site / "nvidia" / pkg / "lib" / name
        if not path.exists():
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            loaded.append(name)
        except OSError as exc:
            print(f"[AVISO] nao carregou {name}: {exc}", file=sys.stderr)
    return loaded


@dataclass(frozen=True)
class Transcript:
    """What Whisper heard, and how sure it was.

    ``avg_logprob`` is the duration-weighted mean over the segments (a 6 s
    segment should outweigh a 0.4 s one), ``compression_ratio`` and
    ``no_speech_prob`` keep the *worst* segment, because one garbage segment is
    enough to poison the turn. Whisper already uses these three numbers to
    decide whether to retry with a hotter temperature — they are simply not
    returned to the caller, and the session needs them to tell a confident
    sentence from a plausible guess.
    """

    text: str
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0

    @property
    def empty(self) -> bool:
        return not self.text.strip()


def _confidence(segments: list) -> tuple[float, float, float]:
    """Duration-weighted ``avg_logprob``; worst-case ratio and no-speech."""
    weighted = 0.0
    duration = 0.0
    plain: list[float] = []
    ratio = 0.0
    no_speech = 0.0
    for seg in segments:
        plain.append(float(seg.avg_logprob))
        span = max(float(seg.end) - float(seg.start), 0.0)
        weighted += float(seg.avg_logprob) * span
        duration += span
        ratio = max(ratio, float(seg.compression_ratio))
        no_speech = max(no_speech, float(seg.no_speech_prob))
    if duration > 0:
        return weighted / duration, ratio, no_speech
    if plain:
        return sum(plain) / len(plain), ratio, no_speech
    return 0.0, 0.0, 0.0


class WhisperSTT:
    """Transcribe 16 kHz mono float32 audio into English text."""

    def __init__(
        self,
        *,
        model: str = "small.en",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        beam_size: int = 1,
        language: str = "en",
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self._model: object | None = None
        self._load_seconds: float | None = None
        # One worker only: WhisperModel is not documented as re-entrant, and a
        # single queue also keeps consecutive turns in submission order.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="whisper-stt"
        )

    @classmethod
    def from_config(cls, cfg: object) -> "WhisperSTT":
        """Build from a ``SttConfig`` (or any object with the same attributes)."""
        return cls(
            model=cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            beam_size=cfg.beam_size,
        )

    # -- lifecycle --------------------------------------------------------- #

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

    def load(self) -> None:
        """Import faster-whisper and build the model (idempotent)."""
        if self._model is not None:
            return
        if self.device == "cuda":
            preload_cuda_libs()
        import time

        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        self._model = WhisperModel(
            self.model_name, device=self.device, compute_type=self.compute_type
        )
        self._load_seconds = time.perf_counter() - t0

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- decoding ---------------------------------------------------------- #

    def transcribe_sync(
        self,
        audio: Audio,
        *,
        prompt: str | None = None,
        hotwords: str | None = None,
    ) -> Transcript:
        """Blocking transcription; the audio must be 16 kHz mono float32.

        ``prompt`` seeds the decoder with the conversation's own vocabulary and
        ``hotwords`` biases single words; both are free (no extra decode) and
        both steer a heavily accented beginner away from fluent-but-wrong
        output. ``condition_on_previous_text`` stays off: a hallucination that
        feeds itself is worse than a missing hint.
        """
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return Transcript("")
        self.load()
        assert self._model is not None  # noqa: S101 - set by load()
        kwargs: dict[str, object] = {
            "language": self.language,
            "beam_size": self.beam_size,
            "condition_on_previous_text": False,
        }
        if prompt:
            kwargs["initial_prompt"] = prompt
        if hotwords:
            kwargs["hotwords"] = hotwords
        segments, _info = self._model.transcribe(samples, **kwargs)
        # The backend returns a generator; the confidence numbers need a pass
        # over every segment anyway.
        parts = list(segments)
        text = " ".join(seg.text.strip() for seg in parts).strip()
        if not text:
            return Transcript("")
        avg_logprob, compression_ratio, no_speech_prob = _confidence(parts)
        return Transcript(
            text=text,
            avg_logprob=avg_logprob,
            compression_ratio=compression_ratio,
            no_speech_prob=no_speech_prob,
        )

    async def transcribe(
        self,
        audio: Audio,
        *,
        prompt: str | None = None,
        hotwords: str | None = None,
    ) -> Transcript:
        """Run :meth:`transcribe_sync` without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.transcribe_sync(audio, prompt=prompt, hotwords=hotwords),
        )
