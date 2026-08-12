"""Command-line entry point for the Hear This preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio.ingest import ingest_audio
from .audio.spectrogram import generate_spectrograms


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hearthis")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="canonicalize a source recording")
    ingest.add_argument("source", type=Path)
    ingest.add_argument("--artifact-root", type=Path, default=Path("artifacts/audio"))
    ingest.add_argument("--recording-id")

    spectrogram = commands.add_parser(
        "spectrogram", help="generate aligned spectrograms from an ingestion manifest"
    )
    spectrogram.add_argument("manifest", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "ingest":
        manifest = ingest_audio(
            args.source,
            args.artifact_root,
            recording_id=args.recording_id,
        )
        print(manifest)
        return

    ingestion = json.loads(args.manifest.read_text(encoding="utf-8"))
    canonical_wave = Path(ingestion["canonical_audio"]["path"])
    output_dir = args.manifest.parent / "spectrograms"
    for manifest in generate_spectrograms(canonical_wave, output_dir):
        print(manifest)


if __name__ == "__main__":
    main()
