"""Canonicalize source audio and record reproducible provenance."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg


MANIFEST_SCHEMA = "hearthis.canonical-audio/v1"


@dataclass(frozen=True)
class CanonicalAudioConfig:
    sample_rate_hz: int = 44_100
    channels: int = 2
    codec: str = "pcm_s24le"
    sample_width_bytes: int = 3

    def validate(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels not in (1, 2):
            raise ValueError("canonical audio currently supports one or two channels")
        if self.codec != "pcm_s24le" or self.sample_width_bytes != 3:
            raise ValueError("the v1 canonical format is 24-bit little-endian PCM")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()[0]


def _inspect_wave(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as audio:
        frame_count = audio.getnframes()
        sample_rate = audio.getframerate()
        return {
            "channels": audio.getnchannels(),
            "sample_rate_hz": sample_rate,
            "sample_width_bytes": audio.getsampwidth(),
            "frame_count": frame_count,
            "duration_seconds": round(frame_count / sample_rate, 9),
            "compression_type": audio.getcomptype(),
        }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def ingest_audio(
    source: Path,
    artifact_root: Path,
    *,
    recording_id: str | None = None,
    config: CanonicalAudioConfig = CanonicalAudioConfig(),
) -> Path:
    """Convert one source recording into canonical PCM and return its manifest."""

    source = source.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    config.validate()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"source is missing or empty: {source}")

    source_sha256 = sha256_file(source)
    stable_id = recording_id or source_sha256[:16]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", stable_id):
        raise ValueError("recording_id must contain only letters, digits, dot, dash, or underscore")

    recording_dir = artifact_root / stable_id
    recording_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = recording_dir / "mixture.wav"
    temporary_audio = recording_dir / "mixture.tmp.wav"
    manifest_path = recording_dir / "manifest.json"

    executable = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        executable,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        "-ac",
        str(config.channels),
        "-ar",
        str(config.sample_rate_hz),
        "-c:a",
        config.codec,
        str(temporary_audio),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        geometry = _inspect_wave(temporary_audio)
        expected = {
            "channels": config.channels,
            "sample_rate_hz": config.sample_rate_hz,
            "sample_width_bytes": config.sample_width_bytes,
            "compression_type": "NONE",
        }
        for key, expected_value in expected.items():
            if geometry[key] != expected_value:
                raise RuntimeError(
                    f"canonical validation failed for {key}: "
                    f"expected {expected_value!r}, got {geometry[key]!r}"
                )
        temporary_audio.replace(canonical_path)
    except Exception:
        temporary_audio.unlink(missing_ok=True)
        raise

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "recording_id": stable_id,
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": source_sha256,
        },
        "canonical_audio": {
            "path": str(canonical_path),
            "bytes": canonical_path.stat().st_size,
            "sha256": sha256_file(canonical_path),
            **geometry,
        },
        "conversion": {
            "config": asdict(config),
            "ffmpeg_version": _ffmpeg_version(executable),
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path

