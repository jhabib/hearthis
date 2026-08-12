"""Audio ingestion and representation primitives."""

from .ingest import CanonicalAudioConfig, ingest_audio
from .demucs_backend import DemucsBackend
from .separation import (
    SeparationBackend,
    SeparationConfig,
    separate_sources,
)
from .spectrogram import SpectrogramConfig, generate_spectrograms

__all__ = [
    "CanonicalAudioConfig",
    "DemucsBackend",
    "SeparationBackend",
    "SeparationConfig",
    "SpectrogramConfig",
    "generate_spectrograms",
    "ingest_audio",
    "separate_sources",
]

