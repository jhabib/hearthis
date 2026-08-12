#!/usr/bin/env python3
"""Apply a human section annotation to an existing structure manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from hearthis.audio.corrections import apply_structure_correction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("correction", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spectrogram-dir", type=Path)
    args = parser.parse_args()
    spectrograms = []
    if args.spectrogram_dir:
        spectrograms = sorted(args.spectrogram_dir.glob("*.json"))
    result = apply_structure_correction(
        args.base_manifest,
        args.correction,
        args.output,
        spectrogram_manifests=spectrograms,
    )
    print(result)


if __name__ == "__main__":
    main()
