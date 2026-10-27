"""Cross-artifact validation for one processed recording."""

from __future__ import annotations

import json
import math
import wave
from zipfile import BadZipFile
from pathlib import Path
from typing import Any

import numpy as np

from .ingest import sha256_file
from .separation import REQUIRED_STEMS, SeparationConfig, _audio_geometry, _reconstruction_metrics
from .spectrogram import SpectrogramConfig, _config_digest

VALIDATION_SCHEMA = "hearthis.artifact-validation/v1"


class ArtifactValidationError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contiguous(intervals: list[dict[str, Any]], duration: float) -> bool:
    if not intervals or not math.isfinite(duration) or duration <= 0:
        return False
    previous_end = 0.0
    for interval in intervals:
        start, end = float(interval["start_seconds"]), float(interval["end_seconds"])
        if (not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start or end > duration + 1e-6
                or abs(start - previous_end) > 1e-6):
            return False
        previous_end = end
    return abs(previous_end - duration) <= 1e-6


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactValidationError(message)


def _check_hash(record: dict[str, Any], name: str) -> Path:
    path = Path(record["path"])
    _require(sha256_file(path) == record["sha256"], f"{name} hash mismatch: {path}")
    return path


def _spectrogram_axes(
    directory: Path, waveform_hash: str, sample_rate: int, frame_count: int,
) -> dict[str, np.ndarray]:
    axes = {}
    paths = sorted(directory.glob("*.json"))
    _require(bool(paths), f"missing spectrogram manifests: {directory}")
    for path in paths:
        metadata = _read(path)
        name = metadata["name"]
        _require(metadata["schema"] == "hearthis.spectrogram/v1", f"invalid spectrogram schema: {path}")
        _require(name not in axes, f"duplicate spectrogram name: {directory}/{name}")
        _require(metadata["canonical_wave"]["sha256"] == waveform_hash,
                 f"spectrogram lineage mismatch: {path}")
        config = SpectrogramConfig(**metadata["config"])
        config.validate()
        _require(metadata["config_sha256"] == _config_digest(config),
                 f"spectrogram config hash mismatch: {path}")
        _require(config.name == name and metadata["sample_rate_hz"] == sample_rate
                 and metadata["waveform_samples"] == frame_count,
                 f"spectrogram configuration or waveform geometry mismatch: {path}")
        tensor_path = _check_hash(metadata["tensor"], "spectrogram tensor")
        with np.load(tensor_path, allow_pickle=False) as tensor:
            times, frequencies, magnitude = (
                tensor["time_seconds"], tensor["frequency_hz"], tensor["magnitude_dbfs"]
            )
            frames = max(0, (frame_count - config.n_fft) // config.hop_length + 1)
            bins = config.n_fft // 2 + 1
            _require(times.shape == (frames,) and frames > 0
                     and frequencies.shape == (bins,) and magnitude.shape == (bins, frames)
                     and metadata["time_frames"] == frames and metadata["frequency_bins"] == bins,
                     f"spectrogram dimensions mismatch: {path}")
            _require(all(np.isfinite(a).all() for a in (times, frequencies, magnitude)),
                     f"nonfinite spectrogram values: {path}")
            expected_times = (config.n_fft / 2 + np.arange(frames) * config.hop_length) / sample_rate
            _require(np.allclose(times, expected_times, rtol=0, atol=1e-9),
                     f"invalid spectrogram timeline: {path}")
            _require(np.allclose(frequencies, np.fft.rfftfreq(config.n_fft, 1 / sample_rate),
                                 rtol=1e-6, atol=1e-5), f"invalid spectrogram frequency axis: {path}")
            axes[name] = times.copy()
    return axes


def _check_lineage(record: dict[str, Any], canonical: dict[str, Any], name: str) -> None:
    _require(record["sha256"] == canonical["sha256"], f"{name} lineage mismatch")
    for key in ("frame_count", "sample_rate_hz"):
        _require(record[key] == canonical[key], f"{name} {key} mismatch")
    if "duration_seconds" in record:
        duration = canonical["frame_count"] / canonical["sample_rate_hz"]
        _require(math.isclose(float(record["duration_seconds"]), duration, abs_tol=1e-6),
                 f"{name} duration mismatch")


def _check_intervals(
    intervals: list[dict[str, Any]], canonical: dict[str, Any],
    axes: dict[str, np.ndarray], name: str,
) -> None:
    sample_rate, frame_count = canonical["sample_rate_hz"], canonical["frame_count"]
    _require(_contiguous(intervals, frame_count / sample_rate),
             f"{name} timeline must be finite, positive, complete and contiguous")
    for index, interval in enumerate(intervals):
        _require(type(interval["index"]) is int and interval["index"] == index,
                 f"{name} index mismatch: {index}")
        start, end = float(interval["start_seconds"]), float(interval["end_seconds"])
        expected = (round(start * sample_rate), min(frame_count, round(end * sample_rate)))
        actual = (interval["start_sample"], interval["end_sample"])
        _require(all(type(v) is int for v in actual) and actual == expected
                 and 0 <= actual[0] < actual[1] <= frame_count,
                 f"{name} sample alignment mismatch: {index}")
        ranges = interval["spectrogram_frames"]
        _require(ranges.keys() == axes.keys(), f"{name} spectrogram coverage mismatch: {index}")
        for representation, times in axes.items():
            expected_frames = {"start": int(np.searchsorted(times, start, side="left")),
                               "end": int(np.searchsorted(times, end, side="left"))}
            _require(ranges[representation] == expected_frames
                     and all(type(v) is int for v in ranges[representation].values()),
                     f"{name} frame alignment mismatch: {index}/{representation}")
        for key in ("confidence", "boundary_confidence"):
            if key in interval:
                value = float(interval[key])
                _require(math.isfinite(value) and 0 <= value <= 1,
                         f"invalid {name} {key}: {index}")


def _validate_recording(
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
    frame_count = canonical["frame_count"]
    sample_rate = canonical["sample_rate_hz"]
    _require(type(frame_count) is int and frame_count > 0
             and type(sample_rate) is int and sample_rate > 0,
             "canonical frame count and sample rate must be positive integers")
    duration = frame_count / sample_rate
    checks: dict[str, Any] = {"canonical_hash": canonical_hash}

    _require(ingestion["schema"] == "hearthis.canonical-audio/v1", "invalid ingestion schema")
    canonical_path = _check_hash(canonical, "canonical waveform")
    with wave.open(str(canonical_path), "rb") as audio:
        _require((audio.getnframes(), audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
                 == (frame_count, sample_rate, canonical["channels"], canonical["sample_width_bytes"]),
                 "canonical waveform geometry mismatch")
    actual_rate, mixture = _audio_geometry(canonical_path)
    _require(actual_rate == sample_rate and mixture.shape[0] == frame_count,
             "canonical decoded geometry mismatch")
    axes = _spectrogram_axes(recording_dir / "spectrograms", canonical_hash, sample_rate, frame_count)
    _require({"transient", "harmonic"} <= axes.keys(), "missing required mixture spectrogram")
    checks["spectrogram_count"] = len(axes)

    separation = _read(recording_dir / "stems" / "manifest.json")
    _require(separation["schema"] == "hearthis.source-separation/v1", "invalid separation schema")
    _check_lineage(separation["canonical_wave"], canonical, "separation")
    config = SeparationConfig(**{
        **separation["config"], "required_stems": tuple(separation["config"]["required_stems"])
    })
    config.validate()
    _require(set(separation["stems"]) == set(config.required_stems) == set(REQUIRED_STEMS),
             "stem contract mismatch")
    reconstruction = np.zeros_like(mixture)
    stem_spectrogram_count = 0
    for name in config.required_stems:
        stem = separation["stems"][name]
        stem_path = _check_hash(stem, f"stem {name}")
        stem_rate, samples = _audio_geometry(stem_path)
        _require(stem_rate == sample_rate and samples.shape == mixture.shape,
                 f"stem audio geometry mismatch: {name}")
        _require(stem["frame_count"] == frame_count and stem["sample_rate_hz"] == sample_rate
                 and stem["channels"] == mixture.shape[1], f"stem metadata geometry mismatch: {name}")
        reconstruction += samples
        stem_root = recording_dir / "stems" / "spectrograms"
        if stem_root.exists():
            stem_axes = _spectrogram_axes(stem_root / name, stem["sha256"], sample_rate, frame_count)
            _require(stem_axes.keys() == axes.keys(), f"stem spectrogram coverage mismatch: {name}")
            _require(all(np.array_equal(stem_axes[key], axes[key]) for key in axes),
                     f"stem spectrogram timeline mismatch: {name}")
            stem_spectrogram_count += len(stem_axes)
    checks["stem_count"] = len(separation["stems"])
    checks["stem_spectrogram_count"] = stem_spectrogram_count
    actual_metrics = _reconstruction_metrics(
        mixture, reconstruction, config.maximum_normalized_reconstruction_error
    )
    metrics = separation["reconstruction"]
    _require(metrics["within_threshold"] is True and actual_metrics["within_threshold"],
             "stem reconstruction exceeds threshold or is flagged as failed")
    for key, actual in actual_metrics.items():
        if key != "within_threshold":
            _require(math.isclose(float(metrics[key]), actual, rel_tol=1e-5, abs_tol=1e-7),
                     f"stem reconstruction metric mismatch: {key}")
    checks["normalized_reconstruction_error"] = actual_metrics["normalized_reconstruction_error"]

    structure_manifest = structure_manifest or recording_dir / "structure" / "manifest.json"
    structure = _read(structure_manifest)
    _require(structure["schema"] in ("hearthis.song-structure/v1", "hearthis.learned-song-structure/v1"),
             "invalid structure schema")
    _check_lineage(structure["canonical_wave"], canonical, "structure")
    _check_intervals(structure["sections"], canonical, axes, "structure")
    if "correction" in structure:
        correction = structure["correction"]
        correction_path = _check_hash(correction, "structure correction")
        _check_hash({"path": correction["base_manifest_path"],
                     "sha256": correction["base_manifest_sha256"]}, "correction base manifest")
        annotation = _read(correction_path)
        source = annotation.get("annotation_source", "human")
        _require(annotation["schema"] == "hearthis.structure-correction/v1"
                 and source in ("human", "llm")
                 and correction.get("annotation_source", "human") == source
                 and all(s["label_kind"] == f"{source}_annotation" for s in structure["sections"]),
                 "structure annotation source mismatch")
        if source == "llm":
            provenance = annotation["provenance"]
            _require(correction["provenance"] == provenance
                     and type(provenance["independent_reference"]) is bool
                     and (not provenance["independent_reference"] or provenance["audio_listened"] is True),
                     "LLM annotation provenance mismatch")
            evidence_path = Path(provenance["evidence_path"])
            if not evidence_path.is_absolute():
                evidence_path = correction_path.parent / evidence_path
            _check_hash({"path": str(evidence_path), "sha256": provenance["evidence_sha256"]},
                        "LLM annotation evidence")
    for name in ("beats_seconds", "downbeats_seconds"):
        times = np.asarray(structure[name], dtype=float)
        _require(times.ndim == 1 and np.isfinite(times).all()
                 and (not times.size or (np.all(np.diff(times) > 0) and times[0] >= 0 and times[-1] <= duration)),
                 f"invalid {name} timeline")
    checks["section_count"] = len(structure["sections"])

    passages_manifest = passages_manifest or recording_dir / "passages" / "manifest.json"
    passages = _read(passages_manifest)
    _require(passages["schema"] == "hearthis.passages/v1", "invalid passage schema")
    _check_lineage(passages["canonical_wave"], canonical, "passage")
    _require(passages["structure_manifest"]["sha256"] == sha256_file(structure_manifest),
             "passage structure lineage mismatch")
    _check_intervals(passages["passages"], canonical, axes, "passage")
    _require(passages["aligned_stems"].keys() == separation["stems"].keys(), "passage stem coverage mismatch")
    for name, stem in passages["aligned_stems"].items():
        _require(stem["sha256"] == separation["stems"][name]["sha256"], f"passage stem lineage mismatch: {name}")
        _check_hash(stem, f"passage stem {name}")
    for passage in passages["passages"]:
        index = passage["section_index"]
        _require(type(index) is int and 0 <= index < len(structure["sections"]), "invalid passage section index")
        section = structure["sections"][index]
        _require(section["start_seconds"] <= passage["start_seconds"] < passage["end_seconds"] <= section["end_seconds"]
                 and passage["section_label"] == section["label"],
                 f"passage section containment or label mismatch: {passage['index']}")
    checks["passage_count"] = len(passages["passages"])

    learned_record = structure.get("learned_input")
    if learned_record is not None:
        stored_input = _check_hash(learned_record, "learned input")
        if learned_input is not None:
            _require(sha256_file(learned_input) == learned_record["sha256"], "learned input lineage mismatch")
        learned_input = stored_input
    if learned_input is not None:
        _require(sample_rate == 44_100, "learned input requires 44100 Hz canonical audio")
        learned = np.load(learned_input, mmap_mode="r", allow_pickle=False)
        expected_shape = (4, (frame_count + 440) // 441, 81)
        checks["learned_input_shape"] = list(learned.shape)
        _require(learned.shape == expected_shape and np.isfinite(learned).all(), "learned input shape or values mismatch")
        if learned_record is not None:
            _require(learned_record["shape"] == list(expected_shape), "learned input metadata shape mismatch")
    if structure["schema"] == "hearthis.learned-song-structure/v1":
        _require(learned_record is not None and "activations" in structure, "missing learned model artifacts")
    if "activations" in structure:
        activations_path = _check_hash(structure["activations"], "learned activations")
        fps = float(structure["activations"]["frames_per_second"])
        _require(math.isfinite(fps) and fps > 0, "invalid learned activation frame rate")
        frames = math.ceil(duration * fps)
        backend = structure["backend"]
        while "base_backend" in backend:
            backend = backend["base_backend"]
        vocabulary = backend["label_vocabulary"]
        with np.load(activations_path, allow_pickle=False) as activations:
            for name in ("beat_probability", "downbeat_probability", "boundary_probability", "label_probability"):
                values = activations[name]
                shape = (len(vocabulary), frames) if name == "label_probability" else (frames,)
                _require(values.shape == shape and np.isfinite(values).all()
                         and np.all((values >= 0) & (values <= 1)), f"invalid learned activations: {name}")
            _require(np.allclose(activations["label_probability"].sum(axis=0), 1, atol=1e-5),
                     "learned label probabilities are not normalized")

    report = {
        "schema": VALIDATION_SCHEMA,
        "recording_dir": str(recording_dir),
        "valid": True,
        "checks": checks,
        "errors": [],
    }
    return report


def validate_recording(
    recording_dir: Path, *, structure_manifest: Path | None = None,
    passages_manifest: Path | None = None, learned_input: Path | None = None,
) -> dict[str, Any]:
    """Validate contents and lineage; reject incomplete or failed artifacts."""
    try:
        return _validate_recording(
            recording_dir, structure_manifest=structure_manifest,
            passages_manifest=passages_manifest, learned_input=learned_input,
        )
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, wave.Error, BadZipFile) as error:
        raise ArtifactValidationError(f"invalid or missing artifact in {recording_dir}: {error}") from error
