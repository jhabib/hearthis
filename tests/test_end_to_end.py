from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from hearthis.audio.allin1_compat import stacked_stem_spectrogram
from hearthis.audio.corrections import CORRECTION_SCHEMA, apply_structure_correction
from hearthis.audio.ingest import ingest_audio, sha256_file
from hearthis.audio.passages import build_passages
from hearthis.audio.separation import separate_sources
from hearthis.audio.spectrogram import generate_spectrograms
from hearthis.audio.structure import analyze_structure
from hearthis.audio.validation import validate_recording


class ExactFourStemBackend:
    @property
    def identity(self):
        return {"name": "exact-four-stem-e2e", "version": "1"}

    def separate(self, canonical_wave: Path, output_dir: Path):
        sample_rate, mixture = wavfile.read(canonical_wave)
        mixture = mixture.astype(np.float32) / float(2**31)
        paths = {}
        for stem in ("vocals", "drums", "bass", "other"):
            path = output_dir / f"{stem}.wav"
            wavfile.write(path, sample_rate, mixture / 4.0)
            paths[stem] = path
        return paths


class FixedFourSecondStructureBackend:
    @property
    def identity(self):
        return {"name": "fixed-e2e", "version": "1"}

    def analyze(self, canonical_wave: Path):
        return {
            "tempo_bpm": 120.0,
            "beats_seconds": [0.5 * index for index in range(1, 8)],
            "downbeats_seconds": [0.5, 2.5],
            "downbeat_phase": 0,
            "sections": [
                {
                    "index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "label": "section_A",
                    "label_kind": "test_hypothesis",
                },
                {
                    "index": 1,
                    "start_seconds": 2.0,
                    "end_seconds": 4.0,
                    "label": "section_B",
                    "label_kind": "test_hypothesis",
                },
            ],
        }


def _source_recording(path: Path) -> None:
    sample_rate = 22_050
    time = np.arange(sample_rate * 4, dtype=np.float64) / sample_rate
    signal = 0.35 * np.sin(2 * np.pi * 220 * time)
    signal += 0.15 * np.sin(2 * np.pi * 440 * time)
    pcm = np.round(signal * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def test_complete_song_representation_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _source_recording(source)

    ingestion_path = ingest_audio(source, tmp_path / "artifacts", recording_id="e2e-song")
    ingestion = json.loads(ingestion_path.read_text())
    recording_dir = ingestion_path.parent
    canonical = Path(ingestion["canonical_audio"]["path"])
    canonical_hash = ingestion["canonical_audio"]["sha256"]
    assert ingestion["canonical_audio"]["sample_rate_hz"] == 44_100
    assert ingestion["canonical_audio"]["channels"] == 2

    spectrogram_manifests = generate_spectrograms(canonical, recording_dir / "spectrograms")
    for path in spectrogram_manifests:
        manifest = json.loads(path.read_text())
        assert manifest["canonical_wave"]["sha256"] == canonical_hash

    separation_path = separate_sources(
        canonical,
        recording_dir / "stems",
        ExactFourStemBackend(),
    )
    separation = json.loads(separation_path.read_text())
    assert separation["canonical_wave"]["sha256"] == canonical_hash
    assert separation["reconstruction"]["within_threshold"] is True
    assert {record["frame_count"] for record in separation["stems"].values()} == {176_400}

    learned_input = stacked_stem_spectrogram(recording_dir / "stems")
    assert learned_input.shape == (4, 400, 81)
    learned_path = recording_dir / "allin1-input.npy"
    np.save(learned_path, learned_input)

    structure_path = analyze_structure(
        canonical,
        recording_dir / "structure",
        FixedFourSecondStructureBackend(),
        spectrogram_manifests,
    )
    correction_path = recording_dir / "human-correction.json"
    correction_path.write_text(
        json.dumps(
            {
                "schema": CORRECTION_SCHEMA,
                "base_manifest_sha256": sha256_file(structure_path),
                "annotator": "e2e-test",
                "created_at": "2026-08-12T00:00:00Z",
                "sections": [
                    {"start_seconds": 0, "end_seconds": 1, "label": "intro"},
                    {"start_seconds": 1, "end_seconds": 3, "label": "verse"},
                    {"start_seconds": 3, "end_seconds": 4, "label": "outro"},
                ],
            }
        ),
        encoding="utf-8",
    )
    corrected_path = apply_structure_correction(
        structure_path,
        correction_path,
        recording_dir / "structure-human" / "manifest.json",
        spectrogram_manifests=spectrogram_manifests,
    )
    corrected = json.loads(corrected_path.read_text())
    assert [section["label"] for section in corrected["sections"]] == [
        "intro",
        "verse",
        "outro",
    ]
    assert corrected["canonical_wave"]["sha256"] == canonical_hash

    passages_path = build_passages(
        corrected_path,
        recording_dir / "passages-human" / "manifest.json",
        beats_per_passage=2,
        minimum_seconds=0.25,
        spectrogram_manifests=spectrogram_manifests,
        separation_manifest=separation_path,
    )
    passages = json.loads(passages_path.read_text())
    assert passages["structure_manifest"]["sha256"] == sha256_file(corrected_path)
    assert passages["passages"][0]["start_seconds"] == 0
    assert passages["passages"][-1]["end_seconds"] == 4
    assert {passage["section_label"] for passage in passages["passages"]} == {
        "intro",
        "verse",
        "outro",
    }
    for previous, current in zip(passages["passages"], passages["passages"][1:]):
        assert previous["end_seconds"] == current["start_seconds"]
    assert set(passages["aligned_stems"]) == {"vocals", "drums", "bass", "other"}

    report = validate_recording(
        recording_dir,
        structure_manifest=corrected_path,
        passages_manifest=passages_path,
        learned_input=learned_path,
    )
    assert report["valid"] is True
    assert report["checks"]["learned_input_shape"] == [4, 400, 81]
