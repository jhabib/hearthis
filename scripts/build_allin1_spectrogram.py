#!/usr/bin/env python3
"""Build Madmom-compatible All-In-One input without a Madmom dependency."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hearthis.audio.allin1_compat import stacked_stem_spectrogram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stem_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    tensor = stacked_stem_spectrogram(args.stem_directory, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, tensor)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
