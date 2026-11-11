"""Microphone capture feeding an ``asyncio.Queue``.

The PortAudio callback runs on a native thread, so it must hand the buffer to
the event loop with ``loop.call_soon_threadsafe`` — touching an asyncio
structure directly from the callback thread is not safe.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd

Chunk = np.ndarray


class AudioCapture:
    """Continuously capture mono float32 audio and queue it for the VAD."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        blocksize: int = 512,
        device: str | int | None = None,
        channels: int = 1,
        maxsize: int = 0,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.device = device
        self.channels = channels
        self.queue: asyncio.Queue[Chunk] = asyncio.Queue(maxsize=maxsize)
        self.overflows = 0
        self._loop = loop
        self._stream: sd.InputStream | None = None

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        if self._stream is not None:
            return
        self._loop = self._loop or asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._on_audio,
        )
        self._stream.start()

    async def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[AVISO] falha ao fechar o stream de audio: {exc}", file=sys.stderr)
        # drain so a pending get() does not hang the shutdown
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - race, harmless
                break

    async def __aenter__(self) -> "AudioCapture":
        self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    # -- consumption ------------------------------------------------------- #

    async def get(self) -> Chunk:
        """Await the next captured block."""
        return await self.queue.get()

    def __aiter__(self) -> AsyncIterator[Chunk]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Chunk]:
        while True:
            yield await self.queue.get()

    # -- PortAudio side ---------------------------------------------------- #

    def _on_audio(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        if status:
            print(f"[AVISO] stream de audio: {status}", file=sys.stderr)
        if self._loop is None:
            return
        block = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
        if block.size == 0:
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue, block)
        except RuntimeError:
            # event loop already closed; nothing left to feed
            pass

    def _enqueue(self, block: Chunk) -> None:
        try:
            self.queue.put_nowait(block)
            return
        except asyncio.QueueFull:
            pass
        # Bounded queue: drop the oldest block so latency stays capped instead
        # of the backlog growing without limit.
        try:
            self.queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - race, harmless
            pass
        try:
            self.queue.put_nowait(block)
            self.overflows += 1
        except asyncio.QueueFull:  # pragma: no cover - race, harmless
            pass
