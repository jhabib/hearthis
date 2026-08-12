"""Demucs backend using the model API and project-owned WAV I/O."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from scipy.io import wavfile

from .separation import REQUIRED_STEMS


def _to_float32(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float32, copy=False)
    if not np.issubdtype(samples.dtype, np.integer):
        raise TypeError(f"unsupported waveform dtype: {samples.dtype}")
    info = np.iinfo(samples.dtype)
    scale = float(max(abs(info.min), info.max))
    return samples.astype(np.float32) / scale


@dataclass(frozen=True)
class DemucsBackend:
    model: str = "htdemucs"
    device: str = "cpu"
    shifts: int = 1
    overlap: float = 0.25

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "name": "demucs",
            "package_version": importlib.metadata.version("demucs"),
            "model": self.model,
            "device": self.device,
            "shifts": self.shifts,
            "overlap": self.overlap,
            "output_encoding": "float32 WAV",
            "clip_mode": "clamp at 0.99",
            "io_backend": "scipy.io.wavfile",
        }

    def separate(self, canonical_wave: Path, output_dir: Path) -> Mapping[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_rate, samples = wavfile.read(canonical_wave)
        waveform = _to_float32(samples)
        if waveform.ndim == 1:
            waveform = waveform[:, None]

        model = get_model(self.model)
        if sample_rate != model.samplerate:
            raise RuntimeError(
                f"Demucs expects {model.samplerate} Hz, received {sample_rate} Hz"
            )
        if waveform.shape[1] != model.audio_channels:
            raise RuntimeError(
                f"Demucs expects {model.audio_channels} channels, "
                f"received {waveform.shape[1]}"
            )
        if set(model.sources) != set(REQUIRED_STEMS):
            raise RuntimeError(f"unexpected Demucs stem set: {model.sources}")

        audio = torch.from_numpy(np.ascontiguousarray(waveform.T))
        reference = audio.mean(0)
        reference_mean = reference.mean()
        reference_std = reference.std()
        normalized = (audio - reference_mean) / reference_std

        model.eval()
        with torch.inference_mode():
            sources = apply_model(
                model,
                normalized[None],
                device=self.device,
                shifts=self.shifts,
                split=True,
                overlap=self.overlap,
                progress=True,
                num_workers=0,
            )[0]
        sources = sources * reference_std + reference_mean
        sources = sources.clamp(-0.99, 0.99).cpu().numpy()

        outputs = {}
        for source, stem in zip(sources, model.sources):
            destination = output_dir / f"{stem}.wav"
            temporary = output_dir / f"{stem}.tmp.wav"
            wavfile.write(temporary, sample_rate, np.ascontiguousarray(source.T))
            temporary.replace(destination)
            outputs[stem] = destination
        return outputs
