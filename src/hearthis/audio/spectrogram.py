"""Generate time-aligned numerical spectrogram artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .ingest import sha256_file


SPECTROGRAM_SCHEMA = "hearthis.spectrogram/v1"


@dataclass(frozen=True)
class SpectrogramConfig:
    name: str
    n_fft: int
    hop_length: int
    window: str = "hann"
    minimum_db: float = -120.0

    def validate(self) -> None:
        if not self.name or any(character in self.name for character in "/\\"):
            raise ValueError("spectrogram name must be a nonempty path-safe identifier")
        if self.n_fft <= 0 or self.n_fft & (self.n_fft - 1):
            raise ValueError("n_fft must be a positive power of two")
        if not 0 < self.hop_length <= self.n_fft:
            raise ValueError("hop_length must lie in [1, n_fft]")
        if self.minimum_db >= 0:
            raise ValueError("minimum_db must be negative")


DEFAULT_CONFIGS = (
    SpectrogramConfig(name="transient", n_fft=1024, hop_length=256),
    SpectrogramConfig(name="harmonic", n_fft=4096, hop_length=1024),
)


def _pcm_to_float32(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float32, copy=False)
    if not np.issubdtype(samples.dtype, np.integer):
        raise TypeError(f"unsupported waveform dtype: {samples.dtype}")
    scale = float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max))
    return samples.astype(np.float32) / scale


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _config_digest(config: SpectrogramConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_spectrograms(
    canonical_wave: Path,
    output_dir: Path,
    configs: Iterable[SpectrogramConfig] = DEFAULT_CONFIGS,
) -> list[Path]:
    """Create compressed STFT artifacts whose timestamps denote frame centers."""

    canonical_wave = canonical_wave.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not canonical_wave.is_file():
        raise ValueError(f"canonical waveform does not exist: {canonical_wave}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate, samples = wavfile.read(canonical_wave)
    waveform = _pcm_to_float32(samples)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("canonical waveform must contain one or two nonempty channels")

    waveform_sha256 = sha256_file(canonical_wave)
    manifests = []
    for config in configs:
        config.validate()
        frequencies, times, complex_stft = signal.stft(
            waveform,
            fs=sample_rate,
            window=config.window,
            nperseg=config.n_fft,
            noverlap=config.n_fft - config.hop_length,
            nfft=config.n_fft,
            boundary=None,
            padded=False,
        )
        magnitude = np.abs(complex_stft).astype(np.float32, copy=False)
        floor = np.float32(10 ** (config.minimum_db / 20))
        magnitude_dbfs = 20 * np.log10(np.maximum(magnitude, floor))
        magnitude_dbfs = np.maximum(magnitude_dbfs, config.minimum_db).astype(
            np.float32, copy=False
        )

        tensor_path = output_dir / f"{config.name}.npz"
        temporary_tensor = output_dir / f"{config.name}.tmp.npz"
        with temporary_tensor.open("wb") as handle:
            np.savez_compressed(
                handle,
                magnitude_dbfs=magnitude_dbfs,
                frequency_hz=frequencies.astype(np.float32),
                time_seconds=times.astype(np.float64),
            )
        temporary_tensor.replace(tensor_path)

        metadata_path = output_dir / f"{config.name}.json"
        metadata = {
            "schema": SPECTROGRAM_SCHEMA,
            "name": config.name,
            "canonical_wave": {
                "path": str(canonical_wave),
                "sha256": waveform_sha256,
            },
            "config": asdict(config),
            "config_sha256": _config_digest(config),
            "sample_rate_hz": int(sample_rate),
            "waveform_samples": int(waveform.size),
            "frequency_bins": int(frequencies.size),
            "time_frames": int(times.size),
            "time_coordinate": "center of analysis window in seconds",
            "first_frame_center_seconds": float(times[0]) if times.size else None,
            "last_frame_center_seconds": float(times[-1]) if times.size else None,
            "tensor": {
                "path": str(tensor_path),
                "bytes": tensor_path.stat().st_size,
                "sha256": sha256_file(tensor_path),
                "arrays": {
                    "magnitude_dbfs": "float32[frequency_bin,time_frame]",
                    "frequency_hz": "float32[frequency_bin]",
                    "time_seconds": "float64[time_frame]",
                },
            },
        }
        _atomic_json(metadata_path, metadata)
        manifests.append(metadata_path)
    return manifests

