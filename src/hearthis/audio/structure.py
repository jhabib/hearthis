"""Beat and structural-section analysis aligned to existing audio artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import librosa
import numpy as np
from scipy.io import wavfile
from sklearn.cluster import AgglomerativeClustering

from .ingest import sha256_file


STRUCTURE_SCHEMA = "hearthis.song-structure/v1"


class StructureBackend(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def analyze(self, canonical_wave: Path) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class LibrosaStructureBackend:
    hop_length: int = 512
    beats_per_bar: int = 4
    target_section_seconds: float = 20.0

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "name": "librosa-baseline",
            "librosa_version": librosa.__version__,
            "hop_length": self.hop_length,
            "beats_per_bar": self.beats_per_bar,
            "section_method": "agglomerative chroma segmentation",
            "section_labels": "unsupervised repeated-section groups",
            "downbeats": "four-beat phase estimate from onset strength",
        }

    def analyze(self, canonical_wave: Path) -> Mapping[str, Any]:
        waveform, sample_rate = librosa.load(canonical_wave, sr=None, mono=True)
        duration = waveform.size / sample_rate
        onset = librosa.onset.onset_strength(
            y=waveform, sr=sample_rate, hop_length=self.hop_length
        )
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset,
            sr=sample_rate,
            hop_length=self.hop_length,
            units="frames",
        )
        beat_frames = np.asarray(beat_frames, dtype=int)
        beat_seconds = librosa.frames_to_time(
            beat_frames, sr=sample_rate, hop_length=self.hop_length
        )

        downbeat_seconds = np.array([], dtype=float)
        downbeat_phase = None
        if beat_frames.size >= self.beats_per_bar:
            phase_scores = []
            for phase in range(self.beats_per_bar):
                frames = beat_frames[phase :: self.beats_per_bar]
                phase_scores.append(float(onset[np.clip(frames, 0, onset.size - 1)].mean()))
            downbeat_phase = int(np.argmax(phase_scores))
            downbeat_seconds = beat_seconds[downbeat_phase :: self.beats_per_bar]

        chroma = librosa.feature.chroma_cqt(
            y=waveform, sr=sample_rate, hop_length=self.hop_length
        )
        section_count = max(2, min(12, round(duration / self.target_section_seconds)))
        section_count = min(section_count, chroma.shape[1])
        boundary_frames = librosa.segment.agglomerative(chroma, section_count)
        boundary_seconds = librosa.frames_to_time(
            boundary_frames, sr=sample_rate, hop_length=self.hop_length
        )
        boundaries = np.unique(np.clip(np.r_[0.0, boundary_seconds, duration], 0, duration))

        embeddings = []
        intervals = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            start_frame = int(librosa.time_to_frames(start, sr=sample_rate, hop_length=self.hop_length))
            end_frame = int(librosa.time_to_frames(end, sr=sample_rate, hop_length=self.hop_length))
            end_frame = max(start_frame + 1, min(end_frame, chroma.shape[1]))
            embeddings.append(chroma[:, start_frame:end_frame].mean(axis=1))
            intervals.append((float(start), float(end)))

        group_count = min(4, max(1, round(math.sqrt(len(intervals)))))
        if len(intervals) == 1:
            groups = np.zeros(1, dtype=int)
        else:
            groups = AgglomerativeClustering(n_clusters=group_count).fit_predict(
                np.asarray(embeddings)
            )
        sections = [
            {
                "index": index,
                "start_seconds": start,
                "end_seconds": end,
                "label": f"section_{chr(65 + int(group))}",
                "label_kind": "unsupervised_similarity_group",
            }
            for index, ((start, end), group) in enumerate(zip(intervals, groups))
        ]
        scalar_tempo = float(np.asarray(tempo).reshape(-1)[0])
        return {
            "tempo_bpm": scalar_tempo,
            "beats_seconds": beat_seconds.tolist(),
            "downbeats_seconds": downbeat_seconds.tolist(),
            "downbeat_phase": downbeat_phase,
            "sections": sections,
        }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _spectrogram_axes(manifests: list[Path]) -> dict[str, np.ndarray]:
    axes = {}
    for path in manifests:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        tensor = np.load(metadata["tensor"]["path"])
        axes[metadata["name"]] = tensor["time_seconds"]
    return axes


def analyze_structure(
    canonical_wave: Path,
    output_dir: Path,
    backend: StructureBackend,
    spectrogram_manifests: list[Path] | None = None,
) -> Path:
    canonical_wave = canonical_wave.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate, samples = wavfile.read(canonical_wave)
    frame_count = int(samples.shape[0])
    duration = frame_count / sample_rate
    result = dict(backend.analyze(canonical_wave))

    beats = np.asarray(result["beats_seconds"], dtype=float)
    downbeats = np.asarray(result["downbeats_seconds"], dtype=float)
    for name, values in (("beats", beats), ("downbeats", downbeats)):
        if values.size and (np.any(np.diff(values) <= 0) or values[0] < 0 or values[-1] > duration):
            raise RuntimeError(f"invalid {name} timeline")

    axes = _spectrogram_axes(spectrogram_manifests or [])
    sections = result["sections"]
    previous_end = 0.0
    for section in sections:
        start = float(section["start_seconds"])
        end = float(section["end_seconds"])
        if abs(start - previous_end) > 1e-6 or end <= start or end > duration + 1e-6:
            raise RuntimeError("sections must form a contiguous, positive timeline")
        section["start_sample"] = round(start * sample_rate)
        section["end_sample"] = min(frame_count, round(end * sample_rate))
        section["spectrogram_frames"] = {
            name: {
                "start": int(np.searchsorted(times, start, side="left")),
                "end": int(np.searchsorted(times, end, side="left")),
            }
            for name, times in axes.items()
        }
        previous_end = end
    if not sections or abs(previous_end - duration) > 1e-6:
        raise RuntimeError("sections must cover the complete recording")

    manifest = {
        "schema": STRUCTURE_SCHEMA,
        "canonical_wave": {
            "path": str(canonical_wave),
            "sha256": sha256_file(canonical_wave),
            "sample_rate_hz": int(sample_rate),
            "frame_count": frame_count,
            "duration_seconds": duration,
        },
        "backend": dict(backend.identity),
        "tempo_bpm": result["tempo_bpm"],
        "beats_seconds": result["beats_seconds"],
        "downbeats_seconds": result["downbeats_seconds"],
        "downbeat_phase": result["downbeat_phase"],
        "sections": sections,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path
