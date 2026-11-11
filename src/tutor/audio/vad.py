"""Turn detection: split the incoming audio into user utterances.

Responsibilities beyond the raw Silero events:

* keep a **pre-roll ring buffer** so the beginning of a phrase is never cut off
  (the Silero ``start`` event is reported retroactively, but we still hold the
  samples that precede it);
* discard noise shorter than ``min_utterance_ms`` (keyboard clicks);
* hard-cut at ``max_utterance_s`` so a stuck VAD cannot buffer forever;
* honour half-duplex through :attr:`TurnDetector.muted`: while muted every
  chunk is dropped and the detector is reset, so the AI's own voice never
  re-triggers the VAD.

Index bookkeeping: Silero reports positions relative to its own counter, which
is reset by ``reset_states()``. We keep a monotonic ``position`` for the real
stream and translate every reported index through the difference with the VAD
counter, so a reset mid-stream cannot desynchronise the two.
"""

from __future__ import annotations

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

Chunk = np.ndarray


class TurnDetector:
    """Feed 16 kHz mono chunks, get complete utterances back."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        chunk_samples: int = 512,
        min_utterance_ms: float = 400,
        max_utterance_s: float = 30.0,
        pre_roll_ms: float = 300,
        speech_pad_ms: float | None = None,
        end_silence_ms: int = 1100,
        model: object | None = None,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError(
                f"silero-vad so aceita 8000 ou 16000 Hz, recebido {sample_rate}"
            )
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.muted = False
        self.discarded_short = 0
        self.turn_count = 0

        self._min_samples = int(sample_rate * min_utterance_ms / 1000)
        self._max_samples = int(sample_rate * max_utterance_s)
        self._pre_roll_samples = int(sample_rate * pre_roll_ms / 1000)
        # The reported start must never fall behind the ring buffer, otherwise
        # we would claim audio we did not keep.
        window_ms = chunk_samples / sample_rate * 1000
        pad = pre_roll_ms if speech_pad_ms is None else speech_pad_ms
        self._speech_pad_ms = max(0.0, min(pad, pre_roll_ms - window_ms))
        self._speech_pad_samples = int(sample_rate * self._speech_pad_ms / 1000)

        self._model = model if model is not None else load_silero_vad()
        self._vad = VADIterator(
            self._model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=int(end_silence_ms),
            speech_pad_ms=int(self._speech_pad_ms),
        )

        self.position = 0  # absolute samples fed since construction
        self._ring: list[Chunk] = []
        self._ring_len = 0
        self._parts: list[tuple[int, Chunk]] = []
        self._turn_start = 0
        self._speech_onset: int | None = None
        self._collecting = False
        self._suppress_until_end = False

    # -- public API -------------------------------------------------------- #

    @property
    def pre_roll_samples(self) -> int:
        return self._pre_roll_samples

    @property
    def collecting(self) -> bool:
        return self._collecting

    def reset(self) -> None:
        """Drop everything buffered and restart detection from scratch."""
        self._vad.reset_states()
        self._ring = []
        self._ring_len = 0
        self._parts = []
        self._collecting = False
        self._suppress_until_end = False
        self._turn_start = 0
        self._speech_onset = None

    def process(self, chunk: Chunk) -> Chunk | None:
        """Feed one block; return the utterance when the user finishes a turn."""
        block = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return None

        if self.muted:
            # Half-duplex: the AI is talking, the mic must not listen.
            if self._collecting or self._ring or self._vad.triggered:
                self.reset()
            return None

        chunk_start = self.position
        self.position += block.size

        with torch.no_grad():
            event = self._vad(block)

        offset = self.position - self._vad.current_sample

        self._ring.append(block)
        self._ring_len += block.size
        self._trim_ring()

        if event and "start" in event and not self._collecting:
            reported = int(event["start"]) + offset
            ring_start = self.position - self._ring_len
            begin = max(reported, ring_start, 0)
            self._collecting = True
            self._suppress_until_end = False
            self._turn_start = begin
            self._parts = self._slice_ring(begin)
            # Silero pads the reported boundaries by `speech_pad`; undo that so
            # min_utterance_ms judges the speech itself, not the padding.
            self._speech_onset = reported + self._speech_pad_samples
        elif self._collecting:
            self._parts.append((chunk_start, block))

        if not self._collecting:
            return None

        if self._suppress_until_end:
            if event and "end" in event:
                self._suppress_until_end = False
                self._clear_turn()
            return None

        collected = sum(len(payload) for _, payload in self._parts)
        if collected >= self._max_samples:
            # Hard cut: report what we have and stop growing the buffer.
            self._suppress_until_end = True
            return self._finish(None, self.position - (self._speech_onset or 0))

        if event and "end" in event:
            # reported_end = temp_end + speech_pad - window
            end = int(event["end"]) + offset - self._speech_pad_samples + block.size
            speech_len = end - (self._speech_onset or end)
            return self._finish(int(event["end"]) + offset, speech_len)

        return None

    # -- internals --------------------------------------------------------- #

    def _trim_ring(self) -> None:
        while self._ring and self._ring_len - self._ring[0].size >= self._pre_roll_samples:
            dropped = self._ring.pop(0)
            self._ring_len -= dropped.size

    def _slice_ring(self, begin: int) -> list[tuple[int, Chunk]]:
        """Every ring sample from absolute index ``begin`` to the present."""
        out: list[tuple[int, Chunk]] = []
        cursor = self.position - self._ring_len
        for payload in self._ring:
            start = cursor
            cursor += payload.size
            if cursor <= begin:
                continue
            take_from = max(0, begin - start)
            piece = payload[take_from:]
            if piece.size:
                out.append((start + take_from, piece))
        return out

    @staticmethod
    def _slice_parts(
        parts: list[tuple[int, Chunk]], begin: int, keep: int
    ) -> list[tuple[int, Chunk]]:
        out: list[tuple[int, Chunk]] = []
        remaining = keep
        for start, payload in parts:
            if remaining <= 0:
                break
            skip = max(0, begin - start)
            piece = payload[skip:]
            if piece.size <= remaining:
                out.append((start + skip, piece))
                remaining -= piece.size
            else:
                out.append((start + skip, piece[:remaining]))
                remaining = 0
        return out

    def _clear_turn(self) -> None:
        self._collecting = False
        self._parts = []
        self._turn_start = 0
        self._speech_onset = None

    def _finish(self, trim_at: int | None, speech_len: int) -> Chunk | None:
        """Assemble the utterance; ``speech_len`` is the padded-out speech span."""
        parts = self._parts
        if trim_at is not None:
            parts = self._slice_parts(parts, self._turn_start, trim_at - self._turn_start)

        payload = np.concatenate([chunk for _, chunk in parts]) if parts else np.zeros(0, np.float32)
        self._clear_turn()
        self._ring = []
        self._ring_len = 0

        # Judged on the speech itself: pre-roll and trailing pad would make any
        # click look longer than min_utterance_ms.
        if speech_len < self._min_samples:
            self.discarded_short += 1
            return None

        self.turn_count += 1
        if payload.size > self._max_samples:
            payload = payload[: self._max_samples]
        return payload
