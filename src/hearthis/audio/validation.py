"""Cross-artifact validation for one processed recording."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .ingest import sha256_file

VALIDATION_SCHEMA = "hearthis.artifact-validation/v1"


class ArtifactValidationError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contiguous(intervals: list[dict[str, Any]], duration: float) -> bool:
    if not intervals or abs(float(intervals[0]["start_seconds"])) > 1e-6:
        return False
    for previous, current in zip(intervals, intervals[1:]):
        if abs(float(previous["end_seconds"]) - float(current["start_seconds"])) > 1e-6:
            return False
    return abs(float(intervals[-1]["end_seconds"]) - duration) <= 1e-6


def validate_recording(
    recording_dir: Path,
    *,
    structure_manifest: Path | None = None,
    passages_manifest: Path | None = None,
    learned_input: Path | None = None,
) -> dict[str, Any]:
    recording_dir = recording_dir.expanduser().resolve()
    ingestion = _read(recording_dir / "manifest.json")
    canonical = ingestion["canonical_audio"]
    canonical_hash = canonical["sha256"]
    frame_count = int(canonical["frame_count"])
    sample_rate = int(canonical["sample_rate_hz"])
    duration = frame_count / sample_rate
    checks: dict[str, Any] = {"canonical_hash": canonical_hash}
    errors: list[str] = []

    if sha256_file(Path(canonical["path"])) != canonical_hash:
        errors.append("canonical waveform hash mismatch")

    spectrogram_paths = sorted((recording_dir / "spectrograms").glob("*.json"))
    checks["spectrogram_count"] = len(spectrogram_paths)
    for path in spectrogram_paths:
        manifest = _read(path)
        if manifest["canonical_wave"]["sha256"] != canonical_hash:
            errors.append(f"spectrogram lineage mismatch: {manifest['name']}")
        with np.load(manifest["tensor"]["path"]) as tensor:
            times = tensor["time_seconds"]
            if times.size and (np.any(np.diff(times) <= 0) or times[-1] > duration + 1e-6):
                errors.append(f"invalid spectrogram timeline: {manifest['name']}")

    separation = _read(recording_dir / "stems" / "manifest.json")
    if separation["canonical_wave"]["sha256"] != canonical_hash:
        errors.append("separation lineage mismatch")
    for name, stem in separation["stems"].items():
        if int(stem["frame_count"]) != frame_count:
            errors.append(f"stem frame mismatch: {name}")
        if int(stem["sample_rate_hz"]) != sample_rate:
            errors.append(f"stem sample-rate mismatch: {name}")
        if sha256_file(Path(stem["path"])) != stem["sha256"]:
            errors.append(f"stem hash mismatch: {name}")
    checks["stem_count"] = len(separation["stems"])

    structure_manifest = structure_manifest or recording_dir / "structure" / "manifest.json"
    structure = _read(structure_manifest)
    if structure["canonical_wave"]["sha256"] != canonical_hash:
        errors.append("structure lineage mismatch")
    if not _contiguous(structure["sections"], duration):
        errors.append("structure timeline is incomplete or discontinuous")
    checks["section_count"] = len(structure["sections"])

    passages_manifest = passages_manifest or recording_dir / "passages" / "manifest.json"
    passages = _read(passages_manifest)
    if passages["structure_manifest"]["sha256"] != sha256_file(structure_manifest):
        errors.append("passage structure lineage mismatch")
    if not _contiguous(passages["passages"], duration):
        errors.append("passage timeline is incomplete or discontinuous")
    for name, stem in passages["aligned_stems"].items():
        if name not in separation["stems"] or stem["sha256"] != separation["stems"][name]["sha256"]:
            errors.append(f"passage stem lineage mismatch: {name}")
    checks["passage_count"] = len(passages["passages"])

    if learned_input is not None:
        learned = np.load(learned_input, mmap_mode="r")
        expected_shape = (4, int(np.ceil(frame_count / 441)), 81)
        checks["learned_input_shape"] = list(learned.shape)
        if learned.shape != expected_shape:
            errors.append(f"learned input shape mismatch: {learned.shape} != {expected_shape}")

    report = {
        "schema": VALIDATION_SCHEMA,
        "recording_dir": str(recording_dir),
        "valid": not errors,
        "checks": checks,
        "errors": errors,
    }
    if errors:
        raise ArtifactValidationError("; ".join(errors))
    return report
