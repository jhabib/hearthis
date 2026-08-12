"""Command-line entry point for the Hear This preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio.ingest import ingest_audio
from .audio.demucs_backend import DemucsBackend
from .audio.separation import separate_sources
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

    separate = commands.add_parser(
        "separate", help="separate an ingested mixture into aligned stems"
    )
    separate.add_argument("manifest", type=Path)
    separate.add_argument("--model", default="htdemucs")
    separate.add_argument("--device", default="cpu")
    separate.add_argument("--shifts", type=int, default=1)
    separate.add_argument("--overlap", type=float, default=0.25)
    separate.add_argument(
        "--spectrograms",
        action="store_true",
        help="generate both spectrogram resolutions for every validated stem",
    )
    return parser


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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

    ingestion = _read_manifest(args.manifest)
    canonical_wave = Path(ingestion["canonical_audio"]["path"])
    if args.command == "spectrogram":
        output_dir = args.manifest.parent / "spectrograms"
        for manifest in generate_spectrograms(canonical_wave, output_dir):
            print(manifest)
        return

    backend = DemucsBackend(
        model=args.model,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
    )
    separation_manifest = separate_sources(
        canonical_wave, args.manifest.parent / "stems", backend
    )
    print(separation_manifest)
    if args.spectrograms:
        separation = _read_manifest(separation_manifest)
        for stem, record in separation["stems"].items():
            output_dir = separation_manifest.parent / "spectrograms" / stem
            for manifest in generate_spectrograms(Path(record["path"]), output_dir):
                print(manifest)


if __name__ == "__main__":
    main()
