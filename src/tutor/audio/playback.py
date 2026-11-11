"""Audio playback feeding a ``sounddevice.OutputStream`` from a chunk queue.

The PortAudio callback runs on a native thread, so it only ever pulls from a
thread-safe ``deque`` and reports "drained" back to the event loop with
``loop.call_soon_threadsafe``. Every state transition (mute, done event) is
applied on the loop thread, where they are serialised against :meth:`feed`.

Half-duplex: while the tutor speaks the :class:`~tutor.audio.vad.TurnDetector`
is muted so the speaker output cannot re-trigger the VAD; it is unmuted
``post_playback_mute_ms`` after the buffer drains, letting the room settle.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

Chunk = np.ndarray


class Playback:
    """Queue 24 kHz float32 chunks and play them as soon as they arrive."""

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        blocksize: int = 1024,
        device: str | int | None = None,
        channels: int = 1,
        detector: object | None = None,
        post_playback_mute_ms: float = 0,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.device = device
        self.channels = channels
        self.post_playback_mute_ms = post_playback_mute_ms
        self.underruns = 0

        self._detector = detector
        self._loop = loop
        self._stream: sd.OutputStream | None = None
        self._pending: deque[Chunk] = deque()
        self._residual: np.ndarray | None = None
        self._playing = False
        self._drain_pending = False
        self._ended = False
        self._awaiting_start = False
        self._started_at: float | None = None
        self._done: asyncio.Event | None = None
        self._unmute_handle: asyncio.TimerHandle | None = None
        self._lock = threading.Lock()

    # -- state ------------------------------------------------------------- #

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def started_at(self) -> float | None:
        """``time.perf_counter()`` when the device consumed the first sample.

        This is the honest ``t_playback_start``: feeding a chunk only means the
        audio is queued, and opening the stream does not make sound come out.
        Recorded on the callback thread so no event-loop scheduling delay is
        added to the measurement.
        """
        return self._started_at

    @property
    def pending_samples(self) -> int:
        with self._lock:
            total = sum(chunk.size for chunk in self._pending)
            return total + (self._residual.size if self._residual is not None else 0)

    # -- lifecycle --------------------------------------------------------- #

    def open(self) -> None:
        """Open the output stream (idempotent)."""
        if self._stream is not None:
            return
        self._loop = self._loop or asyncio.get_running_loop()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._on_audio,
        )
        self._stream.start()

    async def close(self) -> None:
        """Stop playback and close the device."""
        self.stop()
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[AVISO] falha ao fechar o stream de saida: {exc}", file=sys.stderr)

    async def __aenter__(self) -> "Playback":
        self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- feeding ----------------------------------------------------------- #

    def feed(self, chunk: Chunk) -> None:
        """Queue one chunk; opens the stream and mutes the VAD on first use.

        Safe to call from the event loop only (that is where the mute timer and
        the done event live).
        """
        block = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        self._loop = self._loop or asyncio.get_running_loop()
        self.open()
        # Order matters: append first so a concurrently scheduled _on_drained()
        # re-check fails, then clear the done event, then re-arm. _mute() is
        # called unconditionally because it also cancels a pending unmute timer.
        restart = not self._playing
        with self._lock:
            self._pending.append(block)
        if self._done is not None:
            self._done.clear()
        self._drain_pending = False
        self._ended = False
        self._playing = True
        if restart:
            self._awaiting_start = True
            self._started_at = None
        self._mute()

    def end_of_stream(self) -> None:
        """Tell playback no more audio will arrive.

        Without this the final partial device block (which is padded with
        silence by nature) is indistinguishable from a genuine starvation, so
        :attr:`underruns` would report one "cut" at the end of every utterance.
        """
        self._ended = True

    def stop(self) -> None:
        """Drop everything queued so playback ends within one device block."""
        with self._lock:
            self._pending.clear()
            self._residual = None
        self._playing = False
        self._drain_pending = False
        self._ended = False
        self._awaiting_start = False
        if self._done is not None:
            self._done.set()
        self._schedule_unmute()

    async def wait_until_done(self) -> None:
        """Wait until every queued sample has been handed to the device."""
        while self._playing:
            if self._done is None:
                self._done = asyncio.Event()
            if not self._playing:
                break
            await self._done.wait()
            # The device still holds up to one block: allow it to flush before
            # declaring the turn over, and re-check in case audio arrived.
            await asyncio.sleep(self.blocksize / self.sample_rate)

    # -- half-duplex ------------------------------------------------------- #

    def _mute(self) -> None:
        if self._unmute_handle is not None:
            self._unmute_handle.cancel()
            self._unmute_handle = None
        if self._detector is not None:
            self._detector.muted = True

    def _schedule_unmute(self) -> None:
        if self._detector is None:
            return
        delay = self.post_playback_mute_ms / 1000.0
        if delay <= 0 or self._loop is None:
            self._unmute()
            return
        if self._unmute_handle is not None:
            self._unmute_handle.cancel()
        self._unmute_handle = self._loop.call_later(delay, self._unmute)

    def _unmute(self) -> None:
        self._unmute_handle = None
        if self._detector is not None:
            self._detector.muted = False

    # -- callback side ----------------------------------------------------- #

    def _on_audio(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        if status:
            print(f"[AVISO] stream de saida: {status}", file=sys.stderr)
        written = 0
        while written < frames:
            residual = self._residual
            if residual is None or residual.size == 0:
                with self._lock:
                    residual = self._pending.popleft() if self._pending else None
                self._residual = residual
                if residual is None:
                    break
            take = min(frames - written, residual.size)
            if self.channels == 1:
                outdata[written : written + take, 0] = residual[:take]
            else:  # pragma: no cover - the tutor plays mono
                outdata[written : written + take] = residual[:take, None]
            rest = residual[take:]
            self._residual = rest if rest.size else None
            written += take

        if written and self._awaiting_start:
            # First sample handed to the hardware for this utterance.
            self._awaiting_start = False
            self._started_at = time.perf_counter()

        if written < frames:
            # Pad with silence. While _playing is set this means the producer
            # starved the device (counted as an underrun); on the very last
            # block of an utterance it is simply the natural end.
            outdata[written:] = 0.0
            if self._playing and not self._ended:
                # Silence while an utterance is still being produced: the
                # consumer fell behind and this block is a real gap. At the
                # natural end end_of_stream() has been called, so the trailing
                # partial block is not counted; an idle device is not playing.
                self.underruns += 1
            # One-shot: the callback must not re-schedule a report every block,
            # and _on_drained re-checks the queue so a feed() that won the race
            # keeps us playing (and the mic muted).
            if self._playing and not self._drain_pending and self._is_drained():
                self._drain_pending = True
                loop = self._loop
                if loop is not None:
                    try:
                        loop.call_soon_threadsafe(self._on_drained)
                    except RuntimeError:
                        self._drain_pending = False  # loop already closed

    def _is_drained(self) -> bool:
        with self._lock:
            return not self._pending and self._residual is None

    def _on_drained(self) -> None:
        self._drain_pending = False
        # Re-check on the loop thread: feed() is serialised here, so audio that
        # arrived while the callback was reporting keeps us playing (and muted).
        if not self._is_drained():
            return
        self._playing = False
        if self._done is not None:
            self._done.set()
        self._schedule_unmute()
