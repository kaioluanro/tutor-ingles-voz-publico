"""Tests for the PortAudio -> asyncio bridge (Phase 3).

The capture thread cannot be exercised headlessly, but the hand-off logic can:
the callback only schedules work on the loop, and the queue bounds are enforced
in plain Python.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from tutor.audio.capture import AudioCapture


def _block(value: float = 0.5, size: int = 512) -> np.ndarray:
    return np.full((size, 1), value, dtype=np.float32)


def test_callback_schedules_onto_the_loop() -> None:
    async def scenario() -> list[np.ndarray]:
        capture = AudioCapture()
        capture._loop = asyncio.get_running_loop()
        capture._on_audio(_block(), 512, None, None)
        await asyncio.sleep(0)  # let call_soon_threadsafe run
        items = []
        while not capture.queue.empty():
            items.append(capture.queue.get_nowait())
        return items

    items = asyncio.run(scenario())
    assert len(items) == 1
    assert items[0].shape == (512,)
    assert items[0].dtype == np.float32


def test_callback_ignores_empty_block() -> None:
    async def scenario() -> int:
        capture = AudioCapture()
        capture._loop = asyncio.get_running_loop()
        capture._on_audio(np.zeros((0, 1), dtype=np.float32), 0, None, None)
        await asyncio.sleep(0)
        return capture.queue.qsize()

    assert asyncio.run(scenario()) == 0


def test_callback_without_loop_is_a_noop() -> None:
    capture = AudioCapture()
    assert capture._loop is None
    capture._on_audio(_block(), 512, None, None)
    assert capture.queue.empty()


def test_bounded_queue_drops_oldest() -> None:
    async def scenario() -> tuple[int, int, float]:
        capture = AudioCapture(maxsize=2)
        for value in (0.1, 0.2, 0.3):
            capture._enqueue(np.full(512, value, dtype=np.float32))
        size = capture.queue.qsize()
        first = capture.queue.get_nowait()
        return size, capture.overflows, float(first[0])

    size, overflows, oldest = asyncio.run(scenario())
    assert size == 2
    assert overflows == 1
    assert oldest == pytest.approx(0.2), "o bloco mais antigo deveria ter sido descartado"


def test_unbounded_queue_keeps_everything() -> None:
    async def scenario() -> int:
        capture = AudioCapture()
        for _ in range(50):
            capture._enqueue(np.full(512, 0.5, dtype=np.float32))
        return capture.queue.qsize()

    assert asyncio.run(scenario()) == 50
