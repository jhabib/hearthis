from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hearthis.audio.corrections import CORRECTION_SCHEMA, apply_structure_correction


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    base = _write_json(
        tmp_path / "base.json",
        {
            "schema": "hearthis.song-structure/v1",
            "canonical_wave": {
                "path": "/recording.wav",
                "sha256": "a" * 64,
                "sample_rate_hz": 100,
                "frame_count": 1000,
                "duration_seconds": 10.0,
            },
            "backend": {"name": "test"},
            "tempo_bpm": 120,
            "beats_seconds": [1, 2, 3],
            "downbeats_seconds": [1],
            "downbeat_phase": 0,
            "sections": [],
        },
    )
    correction = _write_json(
        tmp_path / "correction.json",
        {
            "schema": CORRECTION_SCHEMA,
            "annotator": "reviewer-1",
            "created_at": "2026-08-12T15:00:00Z",
            "sections": [
                {"start_seconds": 0, "end_seconds": 4, "label": "intro"},
                {"start_seconds": 4, "end_seconds": 10, "label": "verse"},
            ],
        },
    )
    return base, correction


def test_correction_realigns_sections_and_preserves_beats(tmp_path: Path) -> None:
    base, correction = _fixtures(tmp_path)
    tensor_path = tmp_path / "spec.npz"
    np.savez(tensor_path, time_seconds=np.arange(0, 10, 0.25))
    spec_manifest = _write_json(
        tmp_path / "spec.json",
        {"name": "test", "tensor": {"path": str(tensor_path)}},
    )
    output = apply_structure_correction(
        base,
        correction,
        tmp_path / "corrected.json",
        spectrogram_manifests=[spec_manifest],
    )
    result = json.loads(output.read_text())
    assert result["beats_seconds"] == [1, 2, 3]
    assert result["sections"][0]["label"] == "intro"
    assert result["sections"][0]["end_sample"] == 400
    assert result["sections"][1]["spectrogram_frames"]["test"]["start"] == 16
    assert result["correction"]["annotator"] == "reviewer-1"


def test_correction_rejects_a_timeline_gap(tmp_path: Path) -> None:
    base, correction = _fixtures(tmp_path)
    value = json.loads(correction.read_text())
    value["sections"][1]["start_seconds"] = 5
    correction.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        apply_structure_correction(base, correction, tmp_path / "corrected.json")
