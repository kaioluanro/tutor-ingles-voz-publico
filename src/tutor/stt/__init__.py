"""Speech-to-text."""

from tutor.stt.whisper_stt import Transcript, WhisperSTT, preload_cuda_libs

__all__ = ["Transcript", "WhisperSTT", "preload_cuda_libs"]
