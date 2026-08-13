#!/usr/bin/env python3
"""Validate lineage and alignment across one processed recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hearthis.audio.validation import validate_recording


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--structure-manifest", type=Path)
    parser.add_argument("--passages-manifest", type=Path)
    parser.add_argument("--learned-input", type=Path)
    args = parser.parse_args()
    report = validate_recording(
        args.recording_dir,
        structure_manifest=args.structure_manifest,
        passages_manifest=args.passages_manifest,
        learned_input=args.learned_input,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
