from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np

from hearthis.audio.ingest import ingest_audio
from hearthis.audio.spectrogram import SpectrogramConfig, generate_spectrograms


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sine(path: Path, sample_rate: int = 22_050, seconds: float = 1.0) -> None:
    time = np.arange(round(sample_rate * seconds)) / sample_rate
    mono = 0.4 * np.sin(2 * np.pi * 440 * time)
    pcm = np.round(mono * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def test_ingestion_is_canonical_and_repeatable(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_sine(source)

    manifest_path = ingest_audio(
        source, tmp_path / "artifacts", recording_id="sine-fixture"
    )
    first = json.loads(manifest_path.read_text())
    first_audio_hash = first["canonical_audio"]["sha256"]

    repeated_manifest = ingest_audio(
        source, tmp_path / "artifacts", recording_id="sine-fixture"
    )
    second = json.loads(repeated_manifest.read_text())

    assert first["schema"] == "hearthis.canonical-audio/v1"
    assert first["source"]["sha256"] == _sha256(source)
    assert first["canonical_audio"]["channels"] == 2
    assert first["canonical_audio"]["sample_rate_hz"] == 44_100
    assert first["canonical_audio"]["sample_width_bytes"] == 3
    assert first["canonical_audio"]["compression_type"] == "NONE"
    assert abs(first["canonical_audio"]["duration_seconds"] - 1.0) < 0.001
    assert second["canonical_audio"]["sha256"] == first_audio_hash


def test_spectrogram_timestamps_and_frequency_bins(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_sine(source, seconds=2.0)
    ingestion_path = ingest_audio(
        source, tmp_path / "artifacts", recording_id="sine-fixture"
    )
    ingestion = json.loads(ingestion_path.read_text())
    canonical = Path(ingestion["canonical_audio"]["path"])
    config = SpectrogramConfig(name="test", n_fft=1024, hop_length=256)

    [spectrogram_manifest] = generate_spectrograms(
        canonical, ingestion_path.parent / "spectrograms", [config]
    )
    metadata = json.loads(spectrogram_manifest.read_text())
    tensor = np.load(metadata["tensor"]["path"])

    frequency_hz = tensor["frequency_hz"]
    time_seconds = tensor["time_seconds"]
    magnitude_dbfs = tensor["magnitude_dbfs"]
    dominant_frequency = frequency_hz[np.argmax(magnitude_dbfs.mean(axis=1))]

    assert magnitude_dbfs.shape == (frequency_hz.size, time_seconds.size)
    assert frequency_hz.size == config.n_fft // 2 + 1
    assert np.all(np.diff(time_seconds) > 0)
    assert abs(time_seconds[0] - config.n_fft / (2 * 44_100)) < 1e-12
    assert abs(float(dominant_frequency) - 440.0) < 44_100 / config.n_fft
    assert metadata["canonical_wave"]["sha256"] == _sha256(canonical)
