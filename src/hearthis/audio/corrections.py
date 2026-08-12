"""Apply traceable human corrections to a structural-analysis hypothesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .ingest import sha256_file
from .structure import STRUCTURE_SCHEMA

CORRECTION_SCHEMA = "hearthis.structure-correction/v1"


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
        with np.load(metadata["tensor"]["path"]) as tensor:
            axes[metadata["name"]] = tensor["time_seconds"].copy()
    return axes


def apply_structure_correction(
    base_manifest_path: Path,
    correction_path: Path,
    output_path: Path,
    *,
    spectrogram_manifests: list[Path] | None = None,
) -> Path:
    """Replace one section hypothesis while preserving metrical observations."""
    base_manifest_path = base_manifest_path.expanduser().resolve()
    correction_path = correction_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    if correction.get("schema") != CORRECTION_SCHEMA:
        raise ValueError(f"expected correction schema {CORRECTION_SCHEMA}")
    expected_hash = correction.get("base_manifest_sha256")
    actual_hash = sha256_file(base_manifest_path)
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError("correction targets a different base manifest")

    canonical = base["canonical_wave"]
    sample_rate = int(canonical["sample_rate_hz"])
    frame_count = int(canonical["frame_count"])
    duration = float(canonical["duration_seconds"])
    axes = _spectrogram_axes(spectrogram_manifests or [])
    sections = []
    previous_end = 0.0
    for index, supplied in enumerate(correction["sections"]):
        start = float(supplied["start_seconds"])
        end = float(supplied["end_seconds"])
        if abs(start - previous_end) > 1e-6 or end <= start or end > duration + 1e-6:
            raise ValueError("corrected sections must form a contiguous, positive timeline")
        section = {
            "index": index,
            "start_seconds": start,
            "end_seconds": end,
            "label": supplied["label"],
            "label_kind": "human_annotation",
            "confidence": float(supplied.get("confidence", 1.0)),
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
        if "notes" in supplied:
            section["notes"] = supplied["notes"]
        sections.append(section)
        previous_end = end
    if not sections or abs(previous_end - duration) > 1e-6:
        raise ValueError("corrected sections must cover the complete recording")

    manifest = {
        **base,
        "schema": STRUCTURE_SCHEMA,
        "backend": {
            "name": "human-corrected-structure",
            "base_backend": base["backend"],
        },
        "sections": sections,
        "correction": {
            "schema": CORRECTION_SCHEMA,
            "path": str(correction_path),
            "sha256": sha256_file(correction_path),
            "base_manifest_path": str(base_manifest_path),
            "base_manifest_sha256": actual_hash,
            "annotator": correction["annotator"],
            "created_at": correction["created_at"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, manifest)
    return output_path
