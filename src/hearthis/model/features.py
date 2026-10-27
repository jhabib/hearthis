"""Reproducible global, section and passage features from validated audio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..audio.ingest import sha256_file
from ..audio.separation import _audio_geometry
from ..audio.validation import validate_recording
from .response import Region, SharedResponsePrior, Song

MUSIC_NAMES = ("rms_level", "spectral_brightness", "spectral_flux", "vocal_share", "bass_share", "tempo")
BROAD_MUSIC_NAMES = MUSIC_NAMES + (
    "rhythm_regularity", "harmonic_concentration", "percussive_share", "instrumental_share",
    "stereo_width", "dynamic_range", "harmonic_complexity", "vocal_expression",
    "arrangement_density", "production_distortion", "lyric_valence", "lyric_narrativity")
RESPONSE_NAMES = ("arousal", "intimacy", "tension")
FEATURE_SCHEMA = "hearthis.feature-catalog/v1"


def extract_song(recording_dir: Path, prior: SharedResponsePrior, *, structure_manifest: Path | None = None,
                 passages_manifest: Path | None = None, metadata: dict | None = None,
                 listening_evidence: Path | None = None) -> tuple[Song, dict]:
    """The reference q(X,H|W,F) is a point representation of anchored measurements.

    Perceptual coordinates remain distributions. They are never filled from
    loudness or spectral proxies under the guise of measured listener response.
    """
    if prior.music_names not in (MUSIC_NAMES, BROAD_MUSIC_NAMES) or prior.response_names != RESPONSE_NAMES:
        raise ValueError("audio analyzer requires its declared music/response coordinate names")
    recording_dir = recording_dir.resolve()
    structure_manifest = structure_manifest or recording_dir / "structure" / "manifest.json"
    passages_manifest = passages_manifest or recording_dir / "passages" / "manifest.json"
    validation = validate_recording(recording_dir, structure_manifest=structure_manifest,
                                    passages_manifest=passages_manifest)
    ingestion = json.loads((recording_dir / "manifest.json").read_text())
    structure = json.loads(structure_manifest.read_text())
    passages = json.loads(passages_manifest.read_text())
    separation = json.loads((recording_dir / "stems" / "manifest.json").read_text())
    sample_rate, mixture = _audio_geometry(Path(ingestion["canonical_audio"]["path"]))
    stems = {name: _audio_geometry(Path(record["path"]))[1]
             for name, record in separation["stems"].items()}
    spectrum = json.loads((recording_dir / "spectrograms" / "transient.json").read_text())
    with np.load(spectrum["tensor"]["path"]) as tensor:
        times = tensor["time_seconds"].copy()
        frequencies = tensor["frequency_hz"].copy()
        magnitude = 10 ** (tensor["magnitude_dbfs"].astype(np.float64) / 20)
    duration = mixture.shape[0] / sample_rate
    recording_id = ingestion["recording_id"]
    intervals = [("global", 0, 0.0, duration)]
    intervals += [("section", s["index"], s["start_seconds"], s["end_seconds"]) for s in structure["sections"]]
    intervals += [("passage", p["index"], p["start_seconds"], p["end_seconds"]) for p in passages["passages"]]
    evidence = None
    if listening_evidence is not None:
        evidence = json.loads(listening_evidence.read_text())
        if (evidence.get("execution") != "live_provider"
                or evidence["canonical_wave_sha256"] != ingestion["canonical_audio"]["sha256"]):
            raise ValueError("response supervision requires matching evidence from a live audio provider")
        review_audio = evidence["input_audio"]
        audio_path = listening_evidence.parent / review_audio["path"]
        if sha256_file(audio_path) != review_audio["sha256"]:
            raise ValueError("listening input audio hash mismatch")
    regions = []
    for kind, index, start, end in intervals:
        first, last = round(start * sample_rate), min(mixture.shape[0], round(end * sample_rate))
        audio = mixture[first:last]
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        level = np.clip((20 * np.log10(max(rms, 1e-12)) + 60) / 60, 0, 1)
        start_frame, end_frame = np.searchsorted(times, [start, end], side="left")
        selected = magnitude[:, start_frame:end_frame]
        mass = selected.sum()
        brightness = float((frequencies[:, None] * selected).sum() / max(mass, 1e-12) / (sample_rate / 2))
        if selected.shape[1] > 1:
            flux = np.maximum(np.diff(selected, axis=1), 0).sum(axis=0)
            flux = float(np.mean(flux / np.maximum(selected[:, 1:].sum(axis=0), 1e-12)))
        else:
            flux = 0.0
        energies = {name: float(np.mean(samples[first:last].astype(np.float64) ** 2))
                    for name, samples in stems.items()}
        total = max(sum(energies.values()), 1e-12)
        beats = np.asarray(structure["beats_seconds"])
        inside = beats[(beats >= start) & (beats < end)]
        tempo = 60 / np.median(np.diff(inside)) if inside.size >= 2 else float(structure["tempo_bpm"])
        music = np.clip([level, brightness, flux, energies["vocals"] / total,
                         energies["bass"] / total, tempo / 240], 0, 1)
        variance = [0.04] * len(music)
        if prior.music_names == BROAD_MUSIC_NAMES:
            mono = audio.mean(axis=1) if audio.ndim == 2 else audio
            steps = np.diff(inside)
            regularity = float(np.exp(-np.std(steps) / max(np.mean(steps), 1e-12))) if len(steps) > 1 else 0.5
            frame_energy = (selected ** 2).sum(axis=0)
            concentration = float(np.mean(np.max(selected ** 2, axis=0) / np.maximum(frame_energy, 1e-12))) if selected.shape[1] else 0.5
            width = float(np.mean((audio[:, 0] - audio[:, 1]) ** 2) / max(4 * np.mean(audio ** 2), 1e-12)) if audio.ndim == 2 and audio.shape[1] == 2 else 0.0
            blocks = [np.sqrt(np.mean(mono[i:i + max(1, sample_rate // 10)] ** 2)) for i in range(0, len(mono), max(1, sample_rate // 10))]
            spread = float(np.clip(20 * np.log10(max(np.percentile(blocks, 90), 1e-12) / max(np.percentile(blocks, 10), 1e-12)) / 60, 0, 1))
            music = np.r_[music, np.clip([regularity, concentration, energies["drums"] / total,
                                        energies["other"] / total, width, spread], 0, 1), [0.5] * 6]
            variance += [0.16] * 6 + [2.0] * 6
        profile = prior.profile(music)
        if evidence is not None:
            if kind == "global":
                for mode in evidence["review"].get("descriptor_modes", []):
                    if mode["weight"] > 0 and mode["confidence"] > 0:
                        profile.update({i: mode["descriptors"][name] for i, name in enumerate(RESPONSE_NAMES)},
                                       confidence=mode["confidence"], strength=4 * mode["weight"])
            for section in evidence["review"]["sections"]:
                overlap = max(0, min(end, section["end_seconds"]) - max(start, section["start_seconds"]))
                for mode in section.get("descriptor_modes", []):
                    if overlap > 0 and mode["weight"] > 0 and mode["confidence"] > 0:
                        profile.update({i: mode["descriptors"][name] for i, name in enumerate(RESPONSE_NAMES)},
                                       confidence=mode["confidence"], strength=4 * mode["weight"] * overlap / (end - start))
            profile.provenance = "weak subjective audio-LLM supervision; listener calibration unverified"
        regions.append(Region(f"{recording_id}/{kind}/{index}", kind, start, end, music, profile,
                              variance if prior.music_names == BROAD_MUSIC_NAMES else None))
    metadata = metadata or {}
    song = Song(recording_id, str(metadata.get("artist", "unknown")),
                str(metadata.get("title", recording_id)), duration, regions)
    if evidence is not None and prior.music_names == BROAD_MUSIC_NAMES:
        reports = []
        for report in evidence["review"].get("semantic_reports", []):
            reports.append({**report, "region_id": song.global_region.region_id})
        for section in evidence["review"]["sections"]:
            candidates = [r for r in song.regions if r.kind == "section"]
            region = max(candidates, key=lambda r: max(0, min(r.end_seconds, section["end_seconds"]) - max(r.start_seconds, section["start_seconds"])))
            reports.extend({**report, "region_id": region.region_id} for report in section.get("semantic_reports", []))
        if reports:
            song.observations = {"schema": "hearthis.song-observations/v1", "song_id": song.song_id,
                                 "provenance": "weak live audio-LLM semantic reports; calibration unverified",
                                 "semantic_reports": [{"feature": r["feature"], "value": r["value"], "region_id": r["region_id"],
                                                       "source": "audio_llm", "noise_sd": 0.05 + 0.45 * (1 - r["confidence"])} for r in reports]}
            from .modalities import validate_observations
            validate_observations(song, song.observations, prior.music_names)
    references = {}
    for name, path in (("ingestion", recording_dir / "manifest.json"), ("structure", structure_manifest),
                       ("passages", passages_manifest), ("separation", recording_dir / "stems" / "manifest.json")):
        references[name] = {"path": str(path), "sha256": sha256_file(path)}
    if listening_evidence is not None:
        references["listening_evidence"] = {"path": str(listening_evidence), "sha256": sha256_file(listening_evidence)}
    return song, {"recording_id": recording_id, "references": references, "validation": validation,
                  "analyzer": {"rms_level": "clipped (20log10(rms)+60)/60; not LUFS",
                               "spectral_brightness": "magnitude-weighted spectral centroid divided by Nyquist",
                               "spectral_flux": "mean positive spectral difference divided by frame magnitude mass",
                               "vocal_share": "vocal stem mean-square energy / total stem energy; bleed possible",
                               "bass_share": "bass stem mean-square energy / total stem energy; bleed possible",
                               "tempo": "median inferred beat tempo / 240, clipped; estimate not ground truth"},
                  "response_status": "audio-LLM supervised" if evidence is not None else "uninformed prior",
                  "lyrics_status": "not supplied or inferred"}
