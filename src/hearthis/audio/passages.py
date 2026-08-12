"""Create short beat-aligned passages nested within structural sections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .ingest import sha256_file


PASSAGE_SCHEMA = "hearthis.passages/v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _axes(spectrogram_manifests: list[Path]) -> dict[str, np.ndarray]:
    result = {}
    for path in spectrogram_manifests:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        with np.load(metadata["tensor"]["path"]) as tensor:
            result[metadata["name"]] = tensor["time_seconds"].copy()
    return result


def _passage_boundaries(
    section_start: float,
    section_end: float,
    beats: np.ndarray,
    beats_per_passage: int,
    minimum_seconds: float,
) -> list[float]:
    inside = beats[(beats > section_start) & (beats < section_end)]
    candidates = inside[beats_per_passage - 1 :: beats_per_passage]
    boundaries = [section_start]
    for candidate in candidates:
        if candidate - boundaries[-1] >= minimum_seconds:
            boundaries.append(float(candidate))
    if section_end - boundaries[-1] < minimum_seconds and len(boundaries) > 1:
        boundaries.pop()
    boundaries.append(section_end)
    return boundaries


def build_passages(
    structure_manifest: Path,
    output_path: Path,
    *,
    beats_per_passage: int = 8,
    minimum_seconds: float = 4.0,
    spectrogram_manifests: list[Path] | None = None,
    separation_manifest: Path | None = None,
) -> Path:
    if beats_per_passage <= 0:
        raise ValueError("beats_per_passage must be positive")
    if minimum_seconds <= 0:
        raise ValueError("minimum_seconds must be positive")
    structure_manifest = structure_manifest.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    structure = json.loads(structure_manifest.read_text(encoding="utf-8"))
    sample_rate = int(structure["canonical_wave"]["sample_rate_hz"])
    frame_count = int(structure["canonical_wave"]["frame_count"])
    beats = np.asarray(structure["beats_seconds"], dtype=float)
    axes = _axes(spectrogram_manifests or [])

    aligned_stems = {}
    if separation_manifest is not None:
        separation_manifest = separation_manifest.expanduser().resolve()
        separation = json.loads(separation_manifest.read_text(encoding="utf-8"))
        aligned_stems = {
            stem: {"path": record["path"], "sha256": record["sha256"]}
            for stem, record in separation["stems"].items()
        }

    passages = []
    for section in structure["sections"]:
        boundaries = _passage_boundaries(
            float(section["start_seconds"]),
            float(section["end_seconds"]),
            beats,
            beats_per_passage,
            minimum_seconds,
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            passages.append(
                {
                    "index": len(passages),
                    "section_index": int(section["index"]),
                    "section_label": section["label"],
                    "start_seconds": start,
                    "end_seconds": end,
                    "start_sample": round(start * sample_rate),
                    "end_sample": min(frame_count, round(end * sample_rate)),
                    "spectrogram_frames": {
                        name: {
                            "start": int(np.searchsorted(times, start, side="left")),
                            "end": int(np.searchsorted(times, end, side="left")),
                        }
                        for name, times in axes.items()
                    },
                }
            )

    for previous, current in zip(passages, passages[1:]):
        if abs(previous["end_seconds"] - current["start_seconds"]) > 1e-6:
            raise RuntimeError("passages must form a contiguous timeline")
    duration = float(structure["canonical_wave"]["duration_seconds"])
    if not passages or passages[0]["start_seconds"] != 0 or abs(
        passages[-1]["end_seconds"] - duration
    ) > 1e-6:
        raise RuntimeError("passages must cover the complete recording")

    manifest = {
        "schema": PASSAGE_SCHEMA,
        "structure_manifest": {
            "path": str(structure_manifest),
            "sha256": sha256_file(structure_manifest),
        },
        "canonical_wave": structure["canonical_wave"],
        "aligned_stems": aligned_stems,
        "config": {
            "beats_per_passage": beats_per_passage,
            "minimum_seconds": minimum_seconds,
        },
        "passages": passages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, manifest)
    return output_path

