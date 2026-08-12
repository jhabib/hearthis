from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from hearthis.audio.ingest import ingest_audio
from hearthis.audio.separation import REQUIRED_STEMS, separate_sources


def _write_sine(path: Path, sample_rate: int = 44_100) -> None:
    time = np.arange(sample_rate) / sample_rate
    samples = np.round(
        0.25 * np.sin(2 * np.pi * 220 * time) * np.iinfo(np.int16).max
    ).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


class ExactFakeBackend:
    @property
    def identity(self):
        return {"name": "exact-test-backend", "version": "1"}

    def separate(self, canonical_wave: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        vocals = output_dir / "vocals.wav"
        shutil.copyfile(canonical_wave, vocals)
        with wave.open(str(canonical_wave), "rb") as source:
            parameters = source.getparams()
            silence = b"\x00" * (
                parameters.nframes * parameters.nchannels * parameters.sampwidth
            )
        stems = {"vocals": vocals}
        for stem in ("drums", "bass", "other"):
            destination = output_dir / f"{stem}.wav"
            with wave.open(str(destination), "wb") as output:
                output.setparams(parameters)
                output.writeframes(silence)
            stems[stem] = destination
        return stems


class IncompleteFakeBackend(ExactFakeBackend):
    def separate(self, canonical_wave: Path, output_dir: Path):
        stems = dict(super().separate(canonical_wave, output_dir))
        del stems["bass"]
        return stems


def test_separation_validates_alignment_and_reconstruction(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_sine(source)
    ingestion_path = ingest_audio(source, tmp_path / "artifacts", recording_id="sine")
    ingestion = json.loads(ingestion_path.read_text())
    canonical = Path(ingestion["canonical_audio"]["path"])

    manifest_path = separate_sources(
        canonical, ingestion_path.parent / "stems", ExactFakeBackend()
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema"] == "hearthis.source-separation/v1"
    assert set(manifest["stems"]) == set(REQUIRED_STEMS)
    assert manifest["reconstruction"]["normalized_reconstruction_error"] == 0
    assert manifest["reconstruction"]["within_threshold"] is True
    for stem in manifest["stems"].values():
        assert stem["frame_count"] == manifest["canonical_wave"]["frame_count"]
        assert stem["sample_rate_hz"] == manifest["canonical_wave"]["sample_rate_hz"]


def test_separation_rejects_incomplete_backend(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_sine(source)
    ingestion_path = ingest_audio(source, tmp_path / "artifacts", recording_id="sine")
    ingestion = json.loads(ingestion_path.read_text())

    with pytest.raises(RuntimeError, match="missing=.*bass"):
        separate_sources(
            Path(ingestion["canonical_audio"]["path"]),
            ingestion_path.parent / "stems",
            IncompleteFakeBackend(),
        )
