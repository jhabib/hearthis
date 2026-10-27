"""Check persisted model runs independently of their production path."""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import numpy as np

from ..audio.ingest import sha256_file
from .response import SharedResponsePrior, Song
from .taste import Intent, Listener
from scipy.io import wavfile


def validate_model_run(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest["schema"] != "hearthis.taste-program/v1":
        raise ValueError("invalid taste-program schema")
    artifacts = {}
    for name, record in manifest["artifacts"].items():
        target = Path(record["path"])
        if sha256_file(target) != record["sha256"]:
            raise ValueError(f"model artifact hash mismatch: {name}")
        artifacts[name] = json.loads(target.read_text())
    listener = Listener.from_dict(artifacts["listener-session.json"])
    Listener.from_dict(artifacts["listener-prior.json"])
    Listener.from_dict(artifacts["listener-posterior.json"])
    intent = Intent(**artifacts["intent.json"])
    intent.validate(listener.feature_names)
    catalog = artifacts["catalog.json"]
    prior = SharedResponsePrior(**catalog["shared_prior"])
    songs = [Song.from_dict(song) for song in catalog["songs"]]
    if len(songs) != manifest["song_count"] or listener.feature_names != prior.music_names + prior.response_names:
        raise ValueError("model schema dimensions or song count mismatch")
    for recording in catalog["provenance"]:
        for reference in recording["references"].values():
            if sha256_file(Path(reference["path"])) != reference["sha256"]:
                raise ValueError("model source provenance mismatch")
    if "joint-posterior.json" in artifacts:
        from .joint import JointPosterior
        JointPosterior.from_dict(artifacts["joint-posterior.json"], songs, prior, listener)
    if "generated-audio.json" in artifacts:
        from .waveform import WaveformModel
        generated = artifacts["generated-audio.json"]
        reference = generated["model"]
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("generated audio model provenance mismatch")
        decoder = WaveformModel.load(Path(reference["path"]))
        if len(generated["waveforms"]) != len(artifacts["ideal-songs.json"]["samples"]):
            raise ValueError("generated waveforms do not cover ideal samples")
        for index, record in enumerate(generated["waveforms"]):
            wave_path = Path(record["path"])
            if sha256_file(wave_path) != record["sha256"]:
                raise ValueError("generated waveform hash mismatch")
            rate, audio = wavfile.read(wave_path)
            if (rate != decoder.config["sample_rate_hz"] or audio.ndim != 1 or len(audio) != record["frame_count"]
                    or not np.isfinite(audio).all() or not np.isclose(len(audio) / rate, record["duration_seconds"])
                    or record["ideal_sample_index"] != index
                    or not np.allclose(record["music"], artifacts["ideal-songs.json"]["samples"][index]["music"])):
                raise ValueError("generated waveform geometry or ideal-state lineage mismatch")
    ranking = artifacts["ranking.json"]
    if {r["song_id"] for r in ranking} != {s.song_id for s in songs} or len(ranking) != len(songs):
        raise ValueError("ranking does not cover the catalog")
    for name in ("ranking_probability", "catalog_target_probability"):
        weights = np.array([r[name] for r in ranking])
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            raise ValueError(f"invalid model probability normalization: {name}")
    for row in ranking:
        if row["score"] is not None:
            expected = intent.beta * row["expected_utility"] - intent.risk * intent.beta * row["utility_sd"]
            if not np.isclose(row["score"], expected):
                raise ValueError("risk score does not match posterior utility moments")
        if row["artist"].casefold() in {a.casefold() for a in intent.excluded_artists} and row["score"] is not None:
            raise ValueError("excluded artist appears in eligible ranking")
    playlist = artifacts["playlist.json"]
    allowed = {r["song_id"] for r in ranking if r["score"] is not None}
    if (len(playlist["songs"]) != playlist["item_count"] or not set(playlist["songs"]) <= allowed
            or playlist["item_count"] != ceil(intent.horizon_seconds / playlist["nominal_mean_duration_seconds"])):
        raise ValueError("playlist violates horizon or eligibility")
    for sample in artifacts["ideal-songs.json"]["samples"]:
        features = np.array(sample["augmented_features"])
        if (features.shape != (len(listener.feature_names),) or not np.isfinite(features).all()
                or np.any((features < 0) | (features > 1)) or not intent.feasible(features, listener.feature_names)):
            raise ValueError("ideal song violates compactness or feasibility")
    for song_id, explanation in artifacts["explanations.json"].items():
        contributions = sum(v["conditional_shapley"] for v in explanation["features"].values())
        expected_utility = next(r["expected_utility"] for r in ranking if r["song_id"] == song_id)
        if (not np.isclose(contributions, explanation["utility_difference"], atol=1e-9)
                or not np.isclose(explanation["candidate_expected_utility"], expected_utility)):
            raise ValueError("explanation fails efficiency or posterior consistency")
    return {"valid": True, "song_count": len(songs), "artifact_count": len(artifacts),
            "playlist_items": playlist["item_count"], "explanation_count": len(artifacts["explanations.json"])}


def validate_end_to_end(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest["schema"] != "hearthis.end-to-end/v1":
        raise ValueError("invalid end-to-end schema")
    for key in ("configuration", "model", "shared_learning"):
        if key not in manifest:
            if key == "shared_learning":
                continue  # Existing v1 runs did not hash this optional report.
            raise ValueError("end-to-end manifest lacks required lineage")
        reference = manifest[key]
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError(f"end-to-end provenance mismatch: {key}")
    for reference in manifest.get("edits", []):
        from .editing import validate_edit
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("end-to-end edit provenance mismatch")
        validate_edit(reference["path"])
    if manifest.get("evaluation"):
        from .evaluation import validate_evaluation
        reference = manifest["evaluation"]
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("end-to-end evaluation provenance mismatch")
        validate_evaluation(reference["path"])
    from ..audio.validation import validate_recording
    for record in manifest["recordings"]:
        validate_recording(Path(record["recording_dir"]),
                           structure_manifest=Path(record["structure_manifest"]) if record.get("structure_manifest") else None,
                           passages_manifest=Path(record["passages_manifest"]) if record.get("passages_manifest") else None)
    return validate_model_run(Path(manifest["model"]["path"]))
