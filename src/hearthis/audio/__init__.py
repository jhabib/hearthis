"""Audio ingestion and representation primitives."""

from .ingest import CanonicalAudioConfig, ingest_audio
from .spectrogram import SpectrogramConfig, generate_spectrograms

__all__ = [
    "CanonicalAudioConfig",
    "SpectrogramConfig",
    "generate_spectrograms",
    "ingest_audio",
]

