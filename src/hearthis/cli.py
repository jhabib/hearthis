"""Command-line entry point for the Hear This preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio.ingest import ingest_audio
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
    listen = commands.add_parser("listen", help="review the complete waveform with an audio-capable LLM")
    listen.add_argument("recording_dir", type=Path)
    listen.add_argument("--output-dir", type=Path, required=True)
    listen.add_argument("--model", required=True)
    listen.add_argument("--provider", choices=["https", "meta-ape"], default="https")
    listen.add_argument("--endpoint")
    listen.add_argument("--key-environment", default="HEARTHIS_AUDIO_API_KEY")
    run = commands.add_parser("run", help="run audio analysis, listening, posterior inference and recommendation")
    run.add_argument("configuration", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cpu")
    verify = commands.add_parser("verify-model", help="validate a persisted mathematical model run")
    verify.add_argument("manifest", type=Path)
    verify_run = commands.add_parser("verify-run", help="validate the full audio-to-recommendation run and its provenance")
    verify_run.add_argument("manifest", type=Path)
    commands.add_parser("demo", help="reproduce the paper's synthetic two-listener example")
    for name in ("generate", "edit", "evaluate"):
        command = commands.add_parser(name, help=f"execute a {name} configuration")
        command.add_argument("configuration", type=Path)
        command.add_argument("--output-dir", type=Path, required=True)
    verify_any = commands.add_parser("verify", help="validate any completed run, generation, edit or evaluation")
    verify_any.add_argument("manifest", type=Path)
    execute = commands.add_parser("execute-job", help="execute a queued job in an isolated worker process")
    execute.add_argument("kind", choices=["run", "generate", "edit", "evaluate"])
    execute.add_argument("configuration", type=Path)
    execute.add_argument("--output-dir", type=Path, required=True)
    bundle = commands.add_parser("bundle", help="export portable artifact dependencies for containers")
    bundle.add_argument("entry", type=Path)
    bundle.add_argument("--source-root", type=Path, required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)
    bundle.add_argument("--target-prefix", type=Path, required=True)
    return parser


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = _parser().parse_args()
    if args.command == "bundle":
        from .bundle import export_bundle
        print(export_bundle(args.entry, args.source_root, args.output_dir, args.target_prefix))
        return
    if args.command == "verify":
        from .service import validate_result
        print(json.dumps(validate_result(args.manifest), indent=2))
        return
    if args.command in ("generate", "edit", "evaluate", "execute-job"):
        from .service import execute_job
        try:
            print(execute_job(args.kind if args.command == "execute-job" else args.command, args.configuration, args.output_dir))
        except (ValueError, KeyError, TypeError) as error:
            if args.command != "execute-job":
                raise
            import sys
            print(str(error), file=sys.stderr)
            sys.exit(2)
        return
    if args.command == "demo":
        from .model.example import worked_example
        print(json.dumps(worked_example(), indent=2))
        return
    if args.command == "listen":
        from .audio.listening import review_audio
        print(review_audio(args.recording_dir, args.output_dir, model=args.model, provider=args.provider,
                           endpoint=args.endpoint, key_environment=args.key_environment))
        return
    if args.command == "run":
        from .pipeline import run_configuration
        print(run_configuration(args.configuration, args.output_dir, device=args.device))
        return
    if args.command == "verify-model":
        from .model.validation import validate_model_run
        print(json.dumps(validate_model_run(args.manifest), indent=2))
        return
    if args.command == "verify-run":
        from .model.validation import validate_end_to_end
        print(json.dumps(validate_end_to_end(args.manifest), indent=2))
        return
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

    from .audio.demucs_backend import DemucsBackend
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
