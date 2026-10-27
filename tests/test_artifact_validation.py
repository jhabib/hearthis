from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hearthis.audio.ingest import ingest_audio, sha256_file
from hearthis.audio.corrections import CORRECTION_SCHEMA, apply_structure_correction
from hearthis.audio.passages import build_passages
from hearthis.audio.separation import separate_sources
from hearthis.audio.spectrogram import generate_spectrograms
from hearthis.audio.structure import analyze_structure
from hearthis.audio.validation import ArtifactValidationError, _contiguous, validate_recording
from test_end_to_end import ExactFourStemBackend, FixedFourSecondStructureBackend, _source_recording


def _rewrite(path: Path, change) -> None:
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    source = tmp_path / "source.wav"
    _source_recording(source)
    ingestion = ingest_audio(source, tmp_path / "artifacts")
    root = ingestion.parent
    canonical = Path(json.loads(ingestion.read_text())["canonical_audio"]["path"])
    specs = generate_spectrograms(canonical, root / "spectrograms")
    separation = separate_sources(canonical, root / "stems", ExactFourStemBackend())
    for name, record in json.loads(separation.read_text())["stems"].items():
        generate_spectrograms(Path(record["path"]), root / "stems" / "spectrograms" / name)
    structure = analyze_structure(canonical, root / "structure", FixedFourSecondStructureBackend(), specs)
    build_passages(structure, root / "passages" / "manifest.json",
                   spectrogram_manifests=specs, separation_manifest=separation)
    return root


def test_valid_recording_checks_all_stem_spectrograms(recording: Path) -> None:
    report = validate_recording(recording)
    assert report["valid"] is True
    assert report["checks"]["stem_spectrogram_count"] == 8
    assert report["checks"]["normalized_reconstruction_error"] == 0


@pytest.mark.parametrize("directory", ["spectrograms", "stems/spectrograms/vocals"])
@pytest.mark.parametrize("corruption,message", [
    ("hash", "hash mismatch"),
    ("nan", "nonfinite spectrogram"),
    ("dimensions", "dimensions mismatch"),
    ("time", "invalid spectrogram timeline"),
    ("frequency", "frequency axis"),
])
def test_corrupt_spectrogram_is_rejected(recording: Path, directory, corruption, message) -> None:
    manifest_path = recording / directory / "transient.json"
    manifest = json.loads(manifest_path.read_text())
    tensor_path = Path(manifest["tensor"]["path"])
    with np.load(tensor_path) as tensor:
        arrays = {name: tensor[name].copy() for name in tensor.files}
    if corruption in ("hash", "nan"):
        arrays["magnitude_dbfs"][0, 0] = np.nan
    elif corruption == "dimensions":
        arrays["magnitude_dbfs"] = arrays["magnitude_dbfs"][:, :-1]
    elif corruption == "time":
        arrays["time_seconds"] += 0.01
    else:
        arrays["frequency_hz"][1] += 100
    np.savez_compressed(tensor_path, **arrays)
    if corruption != "hash":
        manifest["tensor"]["sha256"] = sha256_file(tensor_path)
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactValidationError, match=message):
        validate_recording(recording)


@pytest.mark.parametrize("directory", ["spectrograms", "stems/spectrograms/bass"])
def test_missing_spectrograms_are_rejected(recording: Path, directory) -> None:
    for path in (recording / directory).glob("*.json"):
        path.unlink()
    with pytest.raises(ArtifactValidationError, match="missing spectrogram"):
        validate_recording(recording)


@pytest.mark.parametrize("intervals", [
    [(0, 3), (3, 2), (2, 4)], [(0, 0), (0, 4)],
    [(0, 1), (2, 4)], [(0, 3), (2, 4)],
    [(0, float("nan")), (float("nan"), 4)], [(0, float("inf"))],
])
def test_invalid_interval_sequence_is_rejected(intervals) -> None:
    assert not _contiguous([{"start_seconds": start, "end_seconds": end}
                            for start, end in intervals], 4)


@pytest.mark.parametrize("field,value,message", [
    ("start_sample", 1, "sample alignment"),
    ("index", 7, "index mismatch"),
    ("confidence", float("nan"), "confidence"),
])
def test_invalid_section_geometry_is_rejected(recording: Path, field, value, message) -> None:
    structure = recording / "structure" / "manifest.json"
    _rewrite(structure, lambda d: d["sections"][0].update({field: value}))
    with pytest.raises(ArtifactValidationError, match=message):
        validate_recording(recording)


def test_invalid_section_frame_range_is_rejected(recording: Path) -> None:
    structure = recording / "structure" / "manifest.json"
    _rewrite(structure, lambda d: d["sections"][0]["spectrogram_frames"]["transient"].update(start=1))
    with pytest.raises(ArtifactValidationError, match="frame alignment"):
        validate_recording(recording)


@pytest.mark.parametrize("field,value", [("section_index", 1), ("section_label", "wrong")])
def test_passage_must_match_its_parent_section(recording: Path, field, value) -> None:
    _rewrite(recording / "passages" / "manifest.json", lambda d: d["passages"][0].update({field: value}))
    with pytest.raises(ArtifactValidationError, match="containment or label"):
        validate_recording(recording)


def test_missing_passage_stem_is_rejected(recording: Path) -> None:
    _rewrite(recording / "passages" / "manifest.json", lambda d: d["aligned_stems"].pop("vocals"))
    with pytest.raises(ArtifactValidationError, match="stem coverage"):
        validate_recording(recording)


def test_failed_quality_flag_is_rejected(recording: Path) -> None:
    _rewrite(recording / "stems" / "manifest.json", lambda d: d["reconstruction"].update(within_threshold=False))
    with pytest.raises(ArtifactValidationError, match="reconstruction"):
        validate_recording(recording)
    with pytest.raises(ValueError, match="failed separation"):
        build_passages(recording / "structure" / "manifest.json", recording / "failed-passages.json",
                       separation_manifest=recording / "stems" / "manifest.json")
    assert not (recording / "failed-passages.json").exists()


def test_reconstruction_is_recomputed_from_audio(recording: Path) -> None:
    path = recording / "stems" / "manifest.json"
    separation = json.loads(path.read_text())
    for record in separation["stems"].values():
        audio_path = Path(record["path"])
        rate, audio = wavfile.read(audio_path)
        wavfile.write(audio_path, rate, np.zeros_like(audio))
        record["sha256"] = sha256_file(audio_path)
    path.write_text(json.dumps(separation))
    # Stem spectrograms are optional. Remove their directory from this check
    # so the recomputed quality gate is the first failing invariant.
    shutil.rmtree(recording / "stems" / "spectrograms")
    with pytest.raises(ArtifactValidationError, match="reconstruction"):
        validate_recording(recording)


@pytest.mark.parametrize("contents", ["{", "[]", "{}"])
def test_malformed_manifest_has_actionable_validation_error(recording: Path, contents) -> None:
    (recording / "manifest.json").write_text(contents)
    with pytest.raises(ArtifactValidationError, match="invalid or missing artifact"):
        validate_recording(recording)


def test_learned_input_values_are_validated(recording: Path) -> None:
    learned = recording / "learned.npy"
    np.save(learned, np.full((4, 400, 81), np.nan, dtype=np.float32))
    with pytest.raises(ArtifactValidationError, match="learned input shape or values"):
        validate_recording(recording, learned_input=learned)


def test_passage_builder_rejects_nonfinite_structure(recording: Path) -> None:
    structure = recording / "structure" / "manifest.json"
    _rewrite(structure, lambda d: d["sections"][0].update(end_seconds=float("nan")))
    with pytest.raises(ValueError, match="finite, positive"):
        build_passages(structure, recording / "invalid-passages.json")
    assert not (recording / "invalid-passages.json").exists()


def test_missing_required_resolution_is_rejected(recording: Path) -> None:
    (recording / "spectrograms" / "harmonic.json").unlink()
    with pytest.raises(ArtifactValidationError, match="missing required mixture"):
        validate_recording(recording)


@pytest.fixture
def learned_structure(recording: Path) -> Path:
    structure = recording / "structure" / "manifest.json"
    data = json.loads(structure.read_text())
    learned_input = recording / "input.npy"
    np.save(learned_input, np.zeros((4, 400, 81), dtype=np.float32))
    activations = recording / "activations.npz"
    np.savez_compressed(activations, beat_probability=np.zeros(400, dtype=np.float32),
                        downbeat_probability=np.zeros(400, dtype=np.float32),
                        boundary_probability=np.zeros(400, dtype=np.float32),
                        label_probability=np.full((2, 400), 0.5, dtype=np.float32))
    data.update(schema="hearthis.learned-song-structure/v1",
                learned_input={"path": str(learned_input), "sha256": sha256_file(learned_input),
                               "shape": [4, 400, 81]},
                activations={"path": str(activations), "sha256": sha256_file(activations),
                             "frames_per_second": 100})
    data["backend"]["label_vocabulary"] = ["section_A", "section_B"]
    structure.write_text(json.dumps(data))
    _rewrite(recording / "passages" / "manifest.json",
             lambda d: d["structure_manifest"].update(sha256=sha256_file(structure)))
    assert validate_recording(recording)["valid"]
    return structure


@pytest.mark.parametrize("corruption,message", [
    ("hash", "activations hash"), ("nan", "invalid learned activations"),
    ("sum", "not normalized"), ("shape", "invalid learned activations"),
])
def test_learned_activations_are_validated(recording: Path, learned_structure: Path, corruption, message) -> None:
    data = json.loads(learned_structure.read_text())
    path = Path(data["activations"]["path"])
    with np.load(path) as tensor:
        arrays = {name: tensor[name].copy() for name in tensor.files}
    if corruption in ("hash", "nan"):
        arrays["beat_probability"][0] = np.nan
    elif corruption == "sum":
        arrays["label_probability"][:] = 0.25
    else:
        arrays["beat_probability"] = arrays["beat_probability"][:-1]
    np.savez_compressed(path, **arrays)
    if corruption != "hash":
        data["activations"]["sha256"] = sha256_file(path)
        learned_structure.write_text(json.dumps(data))
        _rewrite(recording / "passages" / "manifest.json",
                 lambda d: d["structure_manifest"].update(sha256=sha256_file(learned_structure)))
    with pytest.raises(ArtifactValidationError, match=message):
        validate_recording(recording)


def test_learned_input_lineage_is_validated(recording: Path, learned_structure: Path) -> None:
    different_input = recording / "other-input.npy"
    np.save(different_input, np.ones((4, 400, 81), dtype=np.float32))
    with pytest.raises(ArtifactValidationError, match="learned input lineage"):
        validate_recording(recording, learned_input=different_input)


def test_llm_overlay_keeps_learned_artifacts_and_provenance(recording: Path, learned_structure: Path) -> None:
    evidence = recording / "evidence.json"
    evidence.write_text(json.dumps({"synthetic": True}))
    base = json.loads(learned_structure.read_text())
    correction = recording / "llm-correction.json"
    correction.write_text(json.dumps({
        "schema": CORRECTION_SCHEMA, "annotation_source": "llm", "annotator": "test-llm",
        "created_at": "2026-10-27T00:00:00Z", "base_manifest_sha256": sha256_file(learned_structure),
        "provenance": {"model": "test-llm", "method": "synthetic fixture",
                       "audio_listened": False, "independent_reference": False,
                       "evidence_path": evidence.name, "evidence_sha256": sha256_file(evidence)},
        "sections": [{"start_seconds": s["start_seconds"], "end_seconds": s["end_seconds"],
                      "label": "draft", "confidence": 0.3} for s in base["sections"]],
    }))
    specs = sorted((recording / "spectrograms").glob("*.json"))
    corrected = apply_structure_correction(learned_structure, correction, recording / "llm-structure.json",
                                          spectrogram_manifests=specs)
    passages = build_passages(corrected, recording / "llm-passages.json",
                              spectrogram_manifests=specs,
                              separation_manifest=recording / "stems" / "manifest.json")
    assert validate_recording(recording, structure_manifest=corrected, passages_manifest=passages)["valid"]
    data = json.loads(corrected.read_text())
    assert data["activations"] == base["activations"]
    assert data["correction"]["annotation_source"] == "llm"
    assert data["correction"]["provenance"]["independent_reference"] is False
    assert {s["label_kind"] for s in data["sections"]} == {"llm_annotation"}
    evidence.write_text("changed after applying the annotation")
    with pytest.raises(ArtifactValidationError, match="LLM annotation evidence hash"):
        validate_recording(recording, structure_manifest=corrected, passages_manifest=passages)


@pytest.mark.parametrize("field,value", [("sample_rate_hz", 0), ("frame_count", -1)])
def test_invalid_canonical_geometry_has_validation_error(recording: Path, field, value) -> None:
    _rewrite(recording / "manifest.json", lambda d: d["canonical_audio"].update({field: value}))
    with pytest.raises(ArtifactValidationError, match="positive integers"):
        validate_recording(recording)
