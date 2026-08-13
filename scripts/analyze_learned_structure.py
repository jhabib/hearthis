#!/usr/bin/env python3
"""Run learned functional structure inference on prepared stem features."""

from __future__ import annotations

import argparse
from pathlib import Path

from hearthis.audio.learned_structure import analyze_learned_structure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="harmonix-fold0")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    recording_dir = args.recording_dir.expanduser().resolve()
    learned_input = args.input or recording_dir / "allin1" / "torch-compatible.npy"
    output_dir = args.output_dir or recording_dir / "structure-learned"
    result = analyze_learned_structure(
        recording_dir,
        learned_input,
        output_dir,
        model_name=args.model,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
