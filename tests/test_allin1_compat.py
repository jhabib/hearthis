from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hearthis.audio.allin1_compat import logarithmic_filterbank, stem_spectrogram


def test_filterbank_matches_allin1_geometry_and_normalization() -> None:
    filters = logarithmic_filterbank()
    assert filters.shape == (1024, 81)
    np.testing.assert_allclose(filters.sum(axis=0), 1.0, atol=1e-6)


def test_spectrogram_has_expected_frames_and_bands(tmp_path: Path) -> None:
    sample_rate = 44_100
    times = np.arange(sample_rate) / sample_rate
    audio = np.round(np.sin(2 * np.pi * 440 * times) * 16_000).astype("<i2")
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio.tobytes())
    spectrogram = stem_spectrogram(path)
    assert spectrogram.shape == (100, 81)
    assert np.isfinite(spectrogram).all()
    assert spectrogram.min() >= 0


@pytest.mark.parametrize("dtype,scale", [(np.int16, 16000), (np.int32, 1000000000), (np.float32, 0.5)])
def test_identical_mono_and_stereo_have_identical_features(tmp_path: Path, dtype, scale) -> None:
    times = np.arange(4410) / 44_100
    mono = (np.sin(2 * np.pi * 440 * times) * scale).astype(dtype)
    wavfile.write(tmp_path / "mono.wav", 44_100, mono)
    wavfile.write(tmp_path / "stereo.wav", 44_100, np.column_stack([mono, mono]))
    np.testing.assert_allclose(
        stem_spectrogram(tmp_path / "mono.wav"),
        stem_spectrogram(tmp_path / "stereo.wav"), rtol=0, atol=1e-6,
    )
