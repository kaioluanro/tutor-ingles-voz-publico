"""Tests for the PortAudio playback side (Phase 4).

The callback cannot run headlessly, but everything it touches can: the block
filler, the drain handshake with the event loop and the half-duplex mute
schedule are plain Python. ``Playback.open`` is stubbed so no device is needed.
"""

from __future__ import annotations

import asyncio
from collections import deque

import numpy as np

from tutor.audio.playback import Playback


class _Detector:
    def __init__(self) -> None:
        self.muted = False


def _block(value: float = 0.5, size: int = 8) -> np.ndarray:
    return np.full(size, value, dtype=np.float32)


# -- callback filler -------------------------------------------------------- #


def test_callback_fills_block_and_pads_the_rest() -> None:
    async def scenario() -> tuple[np.ndarray, int, bool]:
        playback = Playback(blocksize=8, channels=1)
        playback._loop = asyncio.get_running_loop()
        playback._playing = True
        playback._pending.append(_block(0.5, 4))
        out = np.zeros((8, 1), dtype=np.float32)
        playback._on_audio(out, 8, None, None)
        await asyncio.sleep(0)
        return out.reshape(-1), playback.underruns, playback.is_playing

    out, underruns, playing = asyncio.run(scenario())
    assert np.allclose(out[:4], 0.5)
    assert np.allclose(out[4:], 0.0)
    assert underruns == 1
    # Draining hands control back to the loop exactly once.
    assert playing is False


def test_callback_spans_two_blocks_without_underrun() -> None:
    async def scenario() -> tuple[list[np.ndarray], int]:
        playback = Playback(blocksize=8, channels=1)
        playback._loop = asyncio.get_running_loop()
        playback._playing = True
        playback._pending.append(np.arange(16, dtype=np.float32))
        outs = []
        for _ in range(2):
            out = np.zeros((8, 1), dtype=np.float32)
            playback._on_audio(out, 8, None, None)
            outs.append(out.reshape(-1).copy())
        await asyncio.sleep(0)
        return outs, playback.underruns

    outs, underruns = asyncio.run(scenario())
    assert np.allclose(outs[0], np.arange(8))
    assert np.allclose(outs[1], np.arange(8, 16))
    assert underruns == 0


def test_callback_on_empty_queue_is_silent() -> None:
    async def scenario() -> tuple[np.ndarray, int]:
        playback = Playback(blocksize=4, channels=1)
        playback._loop = asyncio.get_running_loop()
        out = np.full((4, 1), 9.0, dtype=np.float32)
        playback._on_audio(out, 4, None, None)
        await asyncio.sleep(0)
        return out.reshape(-1), playback.underruns

    out, underruns = asyncio.run(scenario())
    assert np.allclose(out, 0.0)
    assert underruns == 0  # an idle device is not an underrun


# -- drain handshake -------------------------------------------------------- #


def test_drained_report_is_ignored_when_audio_arrived() -> None:
    async def scenario() -> tuple[bool, bool]:
        detector = _Detector()
        playback = Playback(detector=detector, post_playback_mute_ms=0)
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        playback.feed(_block(0.25, 16))
        assert detector.muted is True
        # Simulate the race: the callback saw an empty queue, but feed() won.
        playback._pending.append(_block(0.1, 4))
        playback._playing = True
        playback._on_drained()
        return playback.is_playing, detector.muted

    playing, muted = asyncio.run(scenario())
    assert playing is True  # still playing, so the report was rejected
    assert muted is True  # ...and the mic stayed muted


def test_drained_schedules_unmute_after_post_playback_mute() -> None:
    async def scenario() -> tuple[bool, bool, bool]:
        detector = _Detector()
        playback = Playback(
            detector=detector, post_playback_mute_ms=60, blocksize=64, sample_rate=64000
        )
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        playback.feed(_block(0.25, 16))
        assert detector.muted is True
        playback._pending.clear()
        playback._playing = True
        playback._on_drained()
        still_muted = detector.muted
        playing_after = playback.is_playing
        await asyncio.sleep(0.12)
        return still_muted, playing_after, detector.muted

    still_muted, playing_after, released = asyncio.run(scenario())
    assert playing_after is False
    assert still_muted is True  # held during the settle window
    assert released is False  # released once the window elapsed


def test_feed_rearms_after_a_scheduled_unmute() -> None:
    async def scenario() -> tuple[bool, bool]:
        detector = _Detector()
        playback = Playback(detector=detector, post_playback_mute_ms=40)
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        playback.feed(_block(0.2, 8))
        playback._pending.clear()
        playback._playing = True
        playback._on_drained()
        playback.feed(_block(0.2, 8))  # interrupted before the timer fired
        await asyncio.sleep(0.10)
        return playback.is_playing, detector.muted

    playing, muted = asyncio.run(scenario())
    assert playing is True
    assert muted is True  # the pending unmute was cancelled by the new feed


# -- stop / wait ------------------------------------------------------------ #


def test_stop_empties_the_queue_and_reports_idle() -> None:
    async def scenario() -> tuple[int, bool, bool]:
        detector = _Detector()
        playback = Playback(detector=detector, post_playback_mute_ms=0)
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        for _ in range(3):
            playback.feed(_block(0.3, 8))
        playback.stop()
        return playback.pending_samples, playback.is_playing, detector.muted

    pending, playing, muted = asyncio.run(scenario())
    assert pending == 0
    assert playing is False
    assert muted is False  # post_playback_mute_ms == 0 -> released at once


def test_wait_until_done_returns_once_audio_drains() -> None:
    async def scenario() -> float:
        playback = Playback(blocksize=64, sample_rate=64000, post_playback_mute_ms=0)
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        playback.feed(_block(0.4, 64))

        async def drain_later() -> None:
            await asyncio.sleep(0.02)
            playback._pending.clear()
            playback._playing = False
            playback._on_drained()

        asyncio.ensure_future(drain_later())
        await asyncio.wait_for(playback.wait_until_done(), timeout=2.0)
        return 0.0

    asyncio.run(scenario())


def test_wait_until_done_is_immediate_when_idle() -> None:
    async def scenario() -> None:
        playback = Playback()
        playback._loop = asyncio.get_running_loop()
        await asyncio.wait_for(playback.wait_until_done(), timeout=0.5)

    asyncio.run(scenario())


def test_stop_clears_a_pending_drain() -> None:
    async def scenario() -> tuple[deque, bool]:
        playback = Playback(post_playback_mute_ms=0)
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        playback.feed(_block(0.5, 8))
        playback.stop()
        return playback._pending, playback.is_playing

    pending, playing = asyncio.run(scenario())
    assert len(pending) == 0
    assert playing is False


# -- underrun accounting ---------------------------------------------------- #


def test_natural_tail_is_not_counted_as_an_underrun() -> None:
    async def scenario() -> int:
        playback = Playback(blocksize=8, channels=1)
        playback._loop = asyncio.get_running_loop()
        playback._playing = True
        playback._ended = True  # end_of_stream() was called
        playback._pending.append(_block(0.5, 4))
        out = np.zeros((8, 1), dtype=np.float32)
        playback._on_audio(out, 8, None, None)
        await asyncio.sleep(0)
        return playback.underruns

    assert asyncio.run(scenario()) == 0


def test_starved_block_while_producing_counts_an_underrun() -> None:
    async def scenario() -> int:
        playback = Playback(blocksize=8, channels=1)
        playback._loop = asyncio.get_running_loop()
        playback._playing = True  # utterance in progress, queue empty
        out = np.full((8, 1), 9.0, dtype=np.float32)
        playback._on_audio(out, 8, None, None)
        await asyncio.sleep(0)
        return playback.underruns

    assert asyncio.run(scenario()) == 1


def test_feed_after_end_of_stream_clears_the_flag() -> None:
    async def scenario() -> tuple[bool, bool]:
        playback = Playback(post_playback_mute_ms=0)
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        playback.feed(_block(0.5, 8))
        playback.end_of_stream()
        ended_at_end = playback._ended
        playback.feed(_block(0.5, 8))
        return ended_at_end, playback._ended

    ended, ended_after_feed = asyncio.run(scenario())
    assert ended is True
    assert ended_after_feed is False  # a new utterance is under way


# -- t_playback_start ------------------------------------------------------- #


def test_started_at_is_unset_until_the_device_consumes_audio() -> None:
    async def scenario() -> tuple[bool, float | None]:
        playback = Playback(blocksize=8, channels=1)
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        before = playback.started_at
        playback.feed(_block(0.5, 4))
        after_feed = playback.started_at  # queued, not yet heard
        return (before is None and after_feed is None), after_feed

    clean, after_feed = asyncio.run(scenario())
    assert clean
    assert after_feed is None


def test_started_at_is_stamped_by_the_callback() -> None:
    async def scenario() -> float | None:
        playback = Playback(blocksize=8, channels=1)
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        playback.feed(_block(0.5, 4))
        out = np.zeros((8, 1), dtype=np.float32)
        playback._on_audio(out, 8, None, None)
        return playback.started_at

    started = asyncio.run(scenario())
    assert started is not None
    import time

    # A perf_counter timestamp taken moments ago, not some placeholder.
    assert abs(time.perf_counter() - started) < 5.0


def test_started_at_survives_later_feeds_of_the_same_utterance() -> None:
    async def scenario() -> tuple[float | None, float | None]:
        playback = Playback(blocksize=8, channels=1)
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        playback.feed(_block(0.5, 4))
        out = np.zeros((8, 1), dtype=np.float32)
        playback._on_audio(out, 8, None, None)
        first = playback.started_at
        playback.feed(_block(0.5, 4))  # still the same utterance
        return first, playback.started_at

    first, second = asyncio.run(scenario())
    assert first is not None
    assert second == first  # a streaming follow-up must not re-stamp


def test_a_new_utterance_restarts_the_measurement() -> None:
    async def scenario() -> tuple[float | None, float | None]:
        playback = Playback(blocksize=8, channels=1)
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        playback.feed(_block(0.5, 4))
        playback._on_audio(np.zeros((8, 1), np.float32), 8, None, None)
        await asyncio.sleep(0)  # _on_drained -> idle
        assert playback.is_playing is False
        playback.feed(_block(0.5, 4))
        return None, playback.started_at

    _, restarted = asyncio.run(scenario())
    assert restarted is None  # re-armed, waiting for the callback again


def test_stop_prevents_a_late_callback_from_stamping() -> None:
    async def scenario() -> float | None:
        playback = Playback(blocksize=8, channels=1)
        playback._loop = asyncio.get_running_loop()
        playback.open = lambda: None  # type: ignore[method-assign]
        playback.feed(_block(0.5, 4))
        playback.stop()
        out = np.zeros((8, 1), dtype=np.float32)
        playback._on_audio(out, 8, None, None)
        return playback.started_at

    assert asyncio.run(scenario()) is None
