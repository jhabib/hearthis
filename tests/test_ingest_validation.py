from __future__ import annotations

import wave
from pathlib import Path

import pytest

from hearthis.audio.ingest import ingest_audio


def test_ingestion_rejects_path_traversal_recording_id(tmp_path: Path) -> None:
    source = tmp_path / "silent.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\x00\x00" * 800)

    with pytest.raises(ValueError, match="recording_id"):
        ingest_audio(source, tmp_path / "artifacts", recording_id="../escape")
