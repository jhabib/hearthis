#!/usr/bin/env python3
"""Build beat-aligned passages for one processed recording."""

from __future__ import annotations

import argparse
from pathlib import Path

from hearthis.audio.passages import build_passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--beats", type=int, default=8)
    parser.add_argument("--minimum-seconds", type=float, default=4.0)
    args = parser.parse_args()
    recording_dir = args.recording_dir.expanduser().resolve()
    result = build_passages(
        recording_dir / "structure" / "manifest.json",
        recording_dir / "passages" / "manifest.json",
        beats_per_passage=args.beats,
        minimum_seconds=args.minimum_seconds,
        spectrogram_manifests=sorted((recording_dir / "spectrograms").glob("*.json")),
        separation_manifest=recording_dir / "stems" / "manifest.json",
    )
    print(result)


if __name__ == "__main__":
    main()
