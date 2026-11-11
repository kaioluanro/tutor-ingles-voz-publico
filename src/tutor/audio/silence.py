"""Silence trimming for streamed TTS fragments.

Kokoro pads every synthesis call with a little quiet at both ends. Phase 4
measured roughly 300 ms of lead and 450 ms of trail per call, which is
invisible when a whole answer is generated in one go but becomes a visible
stutter the moment Phase 6 generates it sentence by sentence — the same answer
gained up to 890 ms of silence at a boundary.

Trimming only what is below the fragment's own noise floor restores the
timing of a single call. A margin (``keep_ms``) is kept on both sides so the
first consonant and the natural decay of the last word are never clipped.
"""

from __future__ import annotations

import numpy as np

FRAME_MS = 10


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    head: bool = True,
    tail: bool = True,
    threshold_ratio: float = 0.05,
    keep_ms: float = 60.0,
) -> np.ndarray:
    """Remove leading/trailing quiet, keeping ``keep_ms`` of margin.

    ``threshold_ratio`` is relative to this fragment's own peak, so a soft
    sentence is trimmed against its own noise floor instead of a loud one's.
    A fragment that never crosses the threshold is returned unchanged — there
    is nothing to gain and something to lose by guessing.
    """
    wave = np.asarray(audio, dtype=np.float32).reshape(-1)
    if wave.size == 0 or (not head and not tail):
        return wave

    frame = max(1, int(sample_rate * FRAME_MS / 1000))
    usable = (wave.size // frame) * frame
    if usable == 0:
        return wave

    rms = np.sqrt(np.mean(wave[:usable].reshape(-1, frame) ** 2, axis=1))
    peak = float(rms.max()) if rms.size else 0.0
    threshold = max(peak * threshold_ratio, 1e-4)
    loud = np.flatnonzero(rms > threshold)
    if loud.size == 0:
        return wave

    keep = int(sample_rate * keep_ms / 1000.0)
    start = int(loud[0]) * frame - keep if head else 0
    end = (int(loud[-1]) + 1) * frame + keep if tail else wave.size
    start = max(0, min(start, wave.size))
    end = max(start, min(end, wave.size))
    # Copy: the caller hands the result straight to the playback queue, which
    # keeps it until the device drains, and a view would pin the original
    # (fully padded) array in memory for as long as that lasts.
    return wave[start:end].copy()
