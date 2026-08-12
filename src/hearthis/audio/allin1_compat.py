"""Madmom-compatible preprocessing for the All-In-One structure model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
from scipy.io import wavfile

SAMPLE_RATE = 44_100
FRAME_SIZE = 2_048
HOP_SIZE = 441
STEM_ORDER = ("bass", "drums", "other", "vocals")


def logarithmic_filterbank() -> np.ndarray:
    """Construct Madmom's normalized semitone-spaced triangular filterbank."""
    frequencies = 440.0 * 2.0 ** (np.arange(-47, 64) / 12.0)
    frequencies = frequencies[(frequencies >= 30.0) & (frequencies <= 17_000.0)]
    bin_frequencies = np.fft.fftfreq(FRAME_SIZE, 1.0 / SAMPLE_RATE)[: FRAME_SIZE // 2]
    indices = np.searchsorted(bin_frequencies, frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices -= frequencies - left < right - frequencies
    bins = np.unique(indices)
    matrix = np.zeros((FRAME_SIZE // 2, len(bins) - 2), dtype=np.float32)
    for column, (start, center, stop) in enumerate(zip(bins[:-2], bins[1:-1], bins[2:])):
        if stop - start < 2:
            center, stop = start, start + 1
        values = np.zeros(stop - start, dtype=np.float32)
        middle = center - start
        values[:middle] = np.linspace(0, 1, middle, endpoint=False, dtype=np.float32)
        values[middle:] = np.linspace(1, 0, stop - center, endpoint=False, dtype=np.float32)
        values /= values.sum()
        matrix[start:stop, column] = values
    return matrix


def _mono_float(path: Path) -> np.ndarray:
    sample_rate, audio = wavfile.read(path)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, received {sample_rate} Hz")
    if audio.ndim == 2:
        audio = audio.astype(np.float32).mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"expected mono or stereo audio, received shape {audio.shape}")
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
        audio = audio.astype(np.float32) / scale
    return np.asarray(audio, dtype=np.float32)


def stem_spectrogram(path: Path, *, device: str = "cpu") -> np.ndarray:
    """Return one All-In-One input tensor with shape ``frames x 81``."""
    audio = torch.from_numpy(_mono_float(path)).to(device)
    frame_count = int(np.ceil(len(audio) / HOP_SIZE))
    padded = torch_functional.pad(audio, (FRAME_SIZE // 2, FRAME_SIZE // 2 - 1))
    frames = padded.unfold(0, FRAME_SIZE, HOP_SIZE)[:frame_count]
    window = torch.from_numpy(np.hanning(FRAME_SIZE).astype(np.float32)).to(device)
    spectrum = torch.fft.fft(frames * window, n=FRAME_SIZE, dim=1)
    magnitude = spectrum[:, : FRAME_SIZE // 2].abs()
    filters = torch.from_numpy(logarithmic_filterbank()).to(device)
    return torch.log10(1.0 + magnitude @ filters).cpu().numpy().astype(np.float32)


def stacked_stem_spectrogram(stem_directory: Path, *, device: str = "cpu") -> np.ndarray:
    """Return All-In-One input ordered as bass, drums, other, and vocals."""
    stem_directory = stem_directory.expanduser().resolve()
    tensors = [stem_spectrogram(stem_directory / f"{stem}.wav", device=device) for stem in STEM_ORDER]
    shapes = {tensor.shape for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError(f"stem spectrogram shapes disagree: {sorted(shapes)}")
    return np.stack(tensors)
