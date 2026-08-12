"""Model-independent source separation with alignment and reconstruction checks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
from scipy.io import wavfile

from .ingest import sha256_file


SEPARATION_SCHEMA = "hearthis.source-separation/v1"
REQUIRED_STEMS = ("vocals", "drums", "bass", "other")


class SeparationBackend(Protocol):
    """Backend contract for producing time-aligned stem waveforms."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def separate(self, canonical_wave: Path, output_dir: Path) -> Mapping[str, Path]: ...


@dataclass(frozen=True)
class SeparationConfig:
    required_stems: tuple[str, ...] = REQUIRED_STEMS
    maximum_normalized_reconstruction_error: float = 0.15

    def validate(self) -> None:
        if not self.required_stems or len(set(self.required_stems)) != len(
            self.required_stems
        ):
            raise ValueError("required_stems must be unique and nonempty")
        if self.maximum_normalized_reconstruction_error <= 0:
            raise ValueError("maximum reconstruction error must be positive")



def _to_float32(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float32, copy=False)
    if not np.issubdtype(samples.dtype, np.integer):
        raise TypeError(f"unsupported waveform dtype: {samples.dtype}")
    info = np.iinfo(samples.dtype)
    scale = float(max(abs(info.min), info.max))
    return samples.astype(np.float32) / scale


def _audio_geometry(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, samples = wavfile.read(path)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError(f"invalid stem waveform shape for {path}: {samples.shape}")
    return int(sample_rate), _to_float32(samples)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def separate_sources(
    canonical_wave: Path,
    output_dir: Path,
    backend: SeparationBackend,
    config: SeparationConfig = SeparationConfig(),
) -> Path:
    """Run a backend, validate its stems, and return a separation manifest."""

    canonical_wave = canonical_wave.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    config.validate()
    if not canonical_wave.is_file():
        raise ValueError(f"canonical waveform does not exist: {canonical_wave}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mixture_rate, mixture = _audio_geometry(canonical_wave)
    produced = dict(backend.separate(canonical_wave, output_dir))
    missing = set(config.required_stems) - produced.keys()
    unexpected = produced.keys() - set(config.required_stems)
    if missing or unexpected:
        raise RuntimeError(
            f"stem contract mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    reconstruction = np.zeros_like(mixture, dtype=np.float32)
    stem_records: dict[str, dict[str, Any]] = {}
    for stem in config.required_stems:
        stem_path = Path(produced[stem]).resolve()
        stem_rate, stem_audio = _audio_geometry(stem_path)
        if stem_rate != mixture_rate:
            raise RuntimeError(
                f"sample-rate mismatch for {stem}: {stem_rate} != {mixture_rate}"
            )
        if stem_audio.shape != mixture.shape:
            raise RuntimeError(
                f"timeline mismatch for {stem}: {stem_audio.shape} != {mixture.shape}"
            )
        reconstruction += stem_audio
        stem_records[stem] = {
            "path": str(stem_path),
            "bytes": stem_path.stat().st_size,
            "sha256": sha256_file(stem_path),
            "sample_rate_hz": stem_rate,
            "frame_count": int(stem_audio.shape[0]),
            "channels": int(stem_audio.shape[1]),
            "duration_seconds": round(stem_audio.shape[0] / stem_rate, 9),
            "peak_absolute_amplitude": float(np.max(np.abs(stem_audio))),
        }

    residual = mixture - reconstruction
    mixture_rms = float(np.sqrt(np.mean(np.square(mixture, dtype=np.float64))))
    residual_rms = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))
    normalized_error = residual_rms / max(mixture_rms, np.finfo(float).eps)
    snr_db = (
        math.inf
        if residual_rms == 0
        else 20 * math.log10(max(mixture_rms, np.finfo(float).eps) / residual_rms)
    )
    metrics = {
        "mixture_rms": mixture_rms,
        "residual_rms": residual_rms,
        "normalized_reconstruction_error": normalized_error,
        "reconstruction_snr_db": snr_db,
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "threshold": config.maximum_normalized_reconstruction_error,
        "within_threshold": bool(
            normalized_error <= config.maximum_normalized_reconstruction_error
        ),
    }

    manifest = {
        "schema": SEPARATION_SCHEMA,
        "canonical_wave": {
            "path": str(canonical_wave),
            "sha256": sha256_file(canonical_wave),
            "sample_rate_hz": mixture_rate,
            "frame_count": int(mixture.shape[0]),
            "channels": int(mixture.shape[1]),
        },
        "backend": dict(backend.identity),
        "config": asdict(config),
        "stems": stem_records,
        "reconstruction": metrics,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path
