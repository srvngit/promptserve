"""Host audio helpers: TTS synthesis, playback, and microphone capture."""

from tts_utils.synthesize import SynthesisResult, synthesize_pcm
from tts_utils.system_audio import (
    DEFAULT_CHANNELS,
    DEFAULT_RATE_HZ,
    pcm_duration_s,
    play_pcm,
    record_pcm,
    speak,
)

__all__ = [
    "DEFAULT_CHANNELS",
    "DEFAULT_RATE_HZ",
    "SynthesisResult",
    "pcm_duration_s",
    "play_pcm",
    "record_pcm",
    "speak",
    "synthesize_pcm",
]
