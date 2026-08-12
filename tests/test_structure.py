from __future__ import annotations

import json
import wave
from pathlib import Path

from hearthis.audio.ingest import ingest_audio
from hearthis.audio.spectrogram import SpectrogramConfig, generate_spectrograms
from hearthis.audio.structure import analyze_structure


class FixedStructureBackend:
    @property
    def identity(self):
        return {"name": "fixed-test-backend", "version": "1"}

    def analyze(self, canonical_wave: Path):
        return {
            "tempo_bpm": 120.0,
            "beats_seconds": [0.5, 1.0, 1.5],
            "downbeats_seconds": [0.5],
            "downbeat_phase": 0,
            "sections": [
                {
                    "index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "label": "section_A",
                    "label_kind": "test",
                },
                {
                    "index": 1,
                    "start_seconds": 1.0,
                    "end_seconds": 2.0,
                    "label": "section_B",
                    "label_kind": "test",
                },
            ],
        }


def test_structure_aligns_sections_to_samples_and_spectrograms(tmp_path: Path) -> None:
    source = tmp_path / "silence.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\x00\x00" * 16_000)
    ingestion_path = ingest_audio(source, tmp_path / "artifacts", recording_id="song")
    ingestion = json.loads(ingestion_path.read_text())
    canonical = Path(ingestion["canonical_audio"]["path"])
    spectrograms = generate_spectrograms(
        canonical,
        ingestion_path.parent / "spectrograms",
        [SpectrogramConfig(name="test", n_fft=1024, hop_length=256)],
    )

    structure_path = analyze_structure(
        canonical,
        ingestion_path.parent / "structure",
        FixedStructureBackend(),
        spectrograms,
    )
    structure = json.loads(structure_path.read_text())

    assert structure["schema"] == "hearthis.song-structure/v1"
    assert structure["sections"][0]["start_sample"] == 0
    assert structure["sections"][0]["end_sample"] == 44_100
    assert structure["sections"][1]["end_sample"] == 88_200
    assert structure["sections"][0]["spectrogram_frames"]["test"]["start"] == 0
    assert structure["sections"][1]["spectrogram_frames"]["test"]["end"] > 0
