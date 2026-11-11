"""Text-to-speech on top of Kokoro, streamed chunk by chunk.

Kokoro yields one audio fragment per phoneme group as it synthesises, so the
player can start speaking before the whole sentence exists. The blocking
generator therefore runs on a worker thread that pushes 24 kHz ``float32``
chunks onto an ``asyncio.Queue`` consumed by :meth:`KokoroTTS.synthesize_stream`.

Two pipelines live here: the English one (the accent the tutor speaks) and a
second one for the spoken correction, whose explanation is Portuguese
Brazilian. Kokoro phonemises with the rules of **one language at a time** —
text in the wrong pipeline comes out with the wrong phonemes — so the two
languages can never share a pipeline. They do share the ``KModel``: the weights
are language-agnostic, which keeps the second pipeline at ~0 extra VRAM.
"""

from __future__ import annotations

import asyncio
import re
import sys
import threading
from collections.abc import AsyncIterator, Iterator

import numpy as np

Audio = np.ndarray

ACCENT_LANG = {"american": "a", "british": "b"}
#: ``lang=`` accepted by :meth:`KokoroTTS.synthesize_stream`.
ENGLISH_LANG = "en"
#: Kokoro's own default; passed explicitly so it does not print its warning and
#: so both pipelines provably resolve the same weights.
KOKORO_REPO = "hexgrad/Kokoro-82M"
#: Accepts what the config says (``pt-br``) and the short alias.
FEEDBACK_ALIASES = {"pt", "pt-br", "p", "portuguese"}


def clean_for_tts(text: str) -> str:
    """Strip markdown, URLs, emojis and stage directions so TTS reads plain words."""
    if not text:
        return ""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    # Stage directions must go before the symbol sweep below, which would
    # otherwise strip the parentheses that delimit them.
    text = re.sub(r"\((?:[^()]*?(?:stage|audio|sigh|laughs?)[^()]*?)\)", "", text, flags=re.I)
    text = re.sub(r"[*_#>~\[\](){}|]+", " ", text)
    # Keep letters, digits and basic punctuation; drop symbols and emoji.
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    # Removing an emoji leaves "world !" behind; TTS should not pause there.
    # The single quote is left out on purpose: squeezing "e 'went'" into
    # "e'went'" mangles the Portuguese correction the way espeak reads it.
    return re.sub(r"\s+([.,!?;:\")\]])", r"\1", text)


class KokoroTTS:
    """Synthesise speech, yielding audio as soon as it is produced.

    ``lang="en"`` (default) speaks with the accent voice; ``lang="pt-br"``
    speaks the correction with ``feedback_voice``. See the module docstring for
    why the two cannot share a pipeline.
    """

    def __init__(
        self,
        *,
        engine: str = "kokoro",
        device: str = "cuda",
        accent: str = "american",
        voice: str = "af_heart",
        sample_rate: int = 24000,
        queue_size: int = 64,
        feedback_voice: str = "pf_dora",
        feedback_lang: str = "pt-br",
    ) -> None:
        if engine != "kokoro":
            raise ValueError(f"engine desconhecido: {engine!r} (so 'kokoro' esta implementado)")
        if accent not in ACCENT_LANG:
            raise ValueError(
                f"accent desconhecido: {accent!r} (use {'|'.join(sorted(ACCENT_LANG))})"
            )
        self.engine = engine
        self.device = device
        self.accent = accent
        self.voice = voice
        self.sample_rate = sample_rate
        self.queue_size = queue_size
        self.feedback_voice = feedback_voice
        self.feedback_lang = feedback_lang
        self._pipeline: object | None = None
        self._feedback_pipeline: object | None = None
        self._load_seconds: float | None = None
        self._cancel: set[threading.Event] = set()

    @classmethod
    def from_config(cls, cfg: object, *, voice: str | None = None) -> "KokoroTTS":
        """Build from a ``TtsConfig`` (or any object with the same attributes)."""
        return cls(
            engine=cfg.engine,
            device=cfg.device,
            accent=cfg.accent,
            voice=voice or cfg.voice,
            sample_rate=cfg.sample_rate,
            feedback_voice=cfg.feedback_voice,
            feedback_lang=cfg.feedback_lang,
        )

    @classmethod
    def from_config(cls, cfg: object, *, voice: str | None = None) -> "KokoroTTS":
        """Build from a ``TtsConfig`` (or any object with the same attributes)."""
        return cls(
            engine=cfg.engine,
            device=cfg.device,
            accent=cfg.accent,
            voice=voice or cfg.voice,
            sample_rate=cfg.sample_rate,
        )

    # -- lifecycle --------------------------------------------------------- #

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def feedback_loaded(self) -> bool:
        return self._feedback_pipeline is not None

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

    def load(self) -> None:
        """Import Kokoro and build the pipeline (idempotent)."""
        if self._pipeline is not None:
            return
        import time

        device = self.device
        if device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    print("[AVISO] CUDA indisponivel no torch; usando cpu.", file=sys.stderr)
                    device = "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"

        from kokoro import KPipeline

        t0 = time.perf_counter()
        self._pipeline = KPipeline(
            lang_code=ACCENT_LANG[self.accent],
            repo_id=KOKORO_REPO,
            device=device,
        )
        self._load_seconds = time.perf_counter() - t0

    def load_feedback(self) -> None:
        """Build the Portuguese-Brazilian pipeline (idempotent).

        It reuses the English pipeline's ``KModel``: only the phonemiser is
        language-specific, the weights are not, so this costs no extra VRAM.
        The voice pack itself (~1 MB) is fetched from the Hub on the first
        synthesis, which is why :meth:`warmup` should touch this pipeline once
        before the conversation starts.
        """
        if self._feedback_pipeline is not None:
            return
        self.load()
        assert self._pipeline is not None  # noqa: S101 - set by load()
        from kokoro import KPipeline

        self._feedback_pipeline = KPipeline(
            lang_code=self.feedback_lang,
            repo_id=getattr(self._pipeline, "repo_id", None) or KOKORO_REPO,
            model=getattr(self._pipeline, "model", None),
        )

    def close(self) -> None:
        for event in list(self._cancel):
            event.set()

    # -- synthesis --------------------------------------------------------- #

    def _resolve(self, lang: str) -> tuple[object, str]:
        """The pipeline and the voice that speak ``lang``.

        English goes to the accent pipeline, everything the config calls
        ``feedback_lang`` to the Portuguese one; anything else is a bug, and
        silently phonemising English with Portuguese rules (or the reverse)
        produces audio that sounds like neither.
        """
        key = (lang or ENGLISH_LANG).strip().lower()
        if key in (ENGLISH_LANG, "en-us", "en-gb", "american", "british"):
            self.load()
            assert self._pipeline is not None  # noqa: S101 - set by load()
            return self._pipeline, self.voice
        if key in FEEDBACK_ALIASES or key == self.feedback_lang.strip().lower():
            self.load_feedback()
            assert self._feedback_pipeline is not None  # noqa: S101
            return self._feedback_pipeline, self.feedback_voice
        raise ValueError(
            f"lang desconhecido: {lang!r} (use 'en' ou {self.feedback_lang!r})"
        )

    def _generate(
        self,
        text: str,
        *,
        speed: float,
        lang: str = ENGLISH_LANG,
        voice: str | None = None,
    ) -> Iterator[Audio]:
        """Blocking generator over one ``KPipeline``; yields float32 24 kHz chunks."""
        pipeline, default_voice = self._resolve(lang)
        for result in pipeline(text, voice=voice or default_voice, speed=speed):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
            if chunk.size:
                yield chunk

    def synthesize(
        self, text: str, *, speed: float = 1.0, lang: str = ENGLISH_LANG
    ) -> Iterator[Audio]:
        """Blocking, fully synchronous synthesis."""
        return self._generate(text, speed=speed, lang=lang)

    async def synthesize_stream(
        self,
        text: str,
        *,
        speed: float = 1.0,
        voice: str | None = None,
        lang: str = ENGLISH_LANG,
    ) -> AsyncIterator[Audio]:
        """Yield chunks as Kokoro produces them, off the event loop.

        The generator is drained by a worker thread so a slow consumer (the
        audio device) never stalls the loop. Callers that may abandon the
        iteration early should wrap it in ``contextlib.aclosing`` so the thread
        is told to stop promptly. ``lang`` picks the pipeline before the thread
        starts, so a bad value fails on the caller instead of inside the worker.
        """
        clean = clean_for_tts(text)
        if not clean:
            return
        self._resolve(lang)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Audio | BaseException | None] = asyncio.Queue(maxsize=self.queue_size)
        stop = threading.Event()
        self._cancel.add(stop)

        def produce() -> None:
            try:
                for chunk in self._generate(clean, speed=speed, lang=lang, voice=voice):
                    if stop.is_set():
                        return
                    loop.call_soon_threadsafe(_put, chunk)
            except BaseException as exc:  # noqa: BLE001 - forwarded to consumer
                loop.call_soon_threadsafe(_put, exc)
            finally:
                loop.call_soon_threadsafe(_put, None)

        def _put(item: Audio | BaseException | None) -> None:
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:  # pragma: no cover - only if consumer stalls
                if item is not None:
                    stop.set()

        thread = threading.Thread(target=produce, daemon=True, name="kokoro-tts")
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            self._cancel.discard(stop)
            await loop.run_in_executor(None, thread.join, 1.0)

    async def warmup(self, *, speed: float = 1.0, lang: str = ENGLISH_LANG) -> float:
        """Synthesise a short phrase once so the first real turn pays no load cost.

        For the feedback pipeline this also fetches its voice from the Hub, so
        a machine without network fails here — at session start — instead of in
        the middle of a correction.
        """
        self._resolve(lang)
        phrase = "Warm up." if lang == ENGLISH_LANG else "Tudo bem."
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await loop.run_in_executor(
            None, lambda: list(self._generate(phrase, speed=speed, lang=lang))
        )
        return loop.time() - t0
