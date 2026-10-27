"""Apply traceable human or LLM annotations to a structure hypothesis."""

from __future__ import annotations

import json
import math
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
    annotation_source = correction.get("annotation_source", "human")
    if annotation_source not in ("human", "llm"):
        raise ValueError("annotation_source must be human or llm")
    if annotation_source == "llm":
        provenance = correction.get("provenance", {})
        if (not provenance.get("model") or not provenance.get("method")
                or type(provenance.get("audio_listened")) is not bool
                or type(provenance.get("independent_reference")) is not bool
                or (provenance["independent_reference"] and not provenance["audio_listened"])):
            raise ValueError("LLM annotations require model, method, audio_listened and valid independent_reference")
        evidence = Path(provenance["evidence_path"])
        if not evidence.is_absolute():
            evidence = correction_path.parent / evidence
        if sha256_file(evidence) != provenance["evidence_sha256"]:
            raise ValueError("LLM annotation evidence hash mismatch")
        if "base_manifest_sha256" not in correction:
            raise ValueError("LLM annotations require base_manifest_sha256")
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
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0
                or abs(start - previous_end) > 1e-6 or end <= start or end > duration + 1e-6):
            raise ValueError("corrected sections must form a contiguous, positive timeline")
        confidence = float(supplied.get("confidence", 1.0))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("annotation confidence must be finite and between zero and one")
        if not isinstance(supplied["label"], str) or not supplied["label"].strip():
            raise ValueError("annotation label must be a nonempty string")
        section = {
            "index": index,
            "start_seconds": start,
            "end_seconds": end,
            "label": supplied["label"],
            "label_kind": f"{annotation_source}_annotation",
            "confidence": confidence,
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
            "name": f"{annotation_source}-corrected-structure",
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
            "annotation_source": annotation_source,
            **({"provenance": correction["provenance"]} if "provenance" in correction else {}),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, manifest)
    return output_path
