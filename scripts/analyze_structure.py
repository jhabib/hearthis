#!/usr/bin/env python3
"""Run the baseline song-structure analyzer for one ingestion manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hearthis.audio.structure import LibrosaStructureBackend, analyze_structure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    ingestion = json.loads(args.manifest.read_text(encoding="utf-8"))
    canonical = Path(ingestion["canonical_audio"]["path"])
    spectrogram_dir = args.manifest.parent / "spectrograms"
    spectrograms = sorted(spectrogram_dir.glob("*.json"))
    result = analyze_structure(
        canonical,
        args.manifest.parent / "structure",
        LibrosaStructureBackend(),
        spectrograms,
    )
    print(result)


if __name__ == "__main__":
    main()
