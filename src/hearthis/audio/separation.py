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
        if (not math.isfinite(self.maximum_normalized_reconstruction_error)
                or self.maximum_normalized_reconstruction_error <= 0):
            raise ValueError("maximum reconstruction error must be finite and positive")



def _to_float32(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float32, copy=False)
    if not np.issubdtype(samples.dtype, np.integer):
        raise TypeError(f"unsupported waveform dtype: {samples.dtype}")
    info = np.iinfo(samples.dtype)
    if info.min == 0:
        midpoint = float(info.max + 1) / 2
        return (samples.astype(np.float32) - midpoint) / midpoint
    scale = float(max(abs(info.min), info.max))
    return samples.astype(np.float32) / scale


def _audio_geometry(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, samples = wavfile.read(path)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] not in (1, 2):
        raise ValueError(f"invalid stem waveform shape for {path}: {samples.shape}")
    if sample_rate <= 0 or not np.isfinite(samples).all():
        raise ValueError(f"waveform must have a positive sample rate and finite samples: {path}")
    return int(sample_rate), _to_float32(samples)


def _reconstruction_metrics(
    mixture: np.ndarray, reconstruction: np.ndarray, threshold: float
) -> dict[str, Any]:
    residual = mixture - reconstruction
    mixture_rms = float(np.sqrt(np.mean(np.square(mixture, dtype=np.float64))))
    residual_rms = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))
    normalized_error = residual_rms / max(mixture_rms, np.finfo(float).eps)
    snr_db = (
        math.inf
        if residual_rms == 0
        else 20 * math.log10(max(mixture_rms, np.finfo(float).eps) / residual_rms)
    )
    return {
        "mixture_rms": mixture_rms,
        "residual_rms": residual_rms,
        "normalized_reconstruction_error": normalized_error,
        "reconstruction_snr_db": snr_db,
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "threshold": threshold,
        "within_threshold": bool(math.isfinite(normalized_error) and normalized_error <= threshold),
    }


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
    # The backend writes into this directory. Invalidate its prior manifest
    # before a rerun can replace any stems, including runs that fail early.
    (output_dir / "manifest.json").unlink(missing_ok=True)
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

    metrics = _reconstruction_metrics(
        mixture, reconstruction, config.maximum_normalized_reconstruction_error
    )
    if not metrics["within_threshold"]:
        raise RuntimeError(
            "stem reconstruction exceeds threshold: "
            f"{metrics['normalized_reconstruction_error']:.6g} > {metrics['threshold']:.6g}"
        )

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
