"""Export a hashed artifact graph with paths rebased for container mounts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from .audio.ingest import sha256_file


def export_bundle(entry, source_root, output_dir, target_prefix):
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    entry = Path(entry).resolve()
    target_prefix = Path(target_prefix)
    if output_dir.exists() or not entry.is_relative_to(source_root) or not target_prefix.is_absolute():
        raise ValueError("bundle requires an entry inside source root, a new output directory and absolute target prefix")
    if output_dir.is_relative_to(source_root) and entry.is_relative_to(output_dir):
        raise ValueError("bundle output cannot contain its source entry")
    done, active, original_hashes, json_data = {}, set(), {}, {}

    def reference(value, parent):
        path = Path(value)
        candidates = [path] if path.is_absolute() else [parent / path, parent.parent / path, source_root / path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists():
                if not resolved.is_relative_to(source_root):
                    raise ValueError("bundle dependency escapes source root")
                return resolved
        raise ValueError(f"bundle dependency is missing: {value}")

    def path_key(key):
        return key == "path" or key.endswith(("_path", "_manifest")) or key in ("recording_dir", "listening_correction", "source", "predicted", "reference", "waveforms", "spectrogram_manifests")

    def rewrite(value, parent, key=""):
        if isinstance(value, dict):
            result = {name: rewrite(item, parent, name) for name, item in value.items()}
            if isinstance(value.get("path"), str) and "sha256" in value:
                source = reference(value["path"], parent)
                if sha256_file(source) != value["sha256"]:
                    raise ValueError("bundle source hash mismatch")
            # Relative correction evidence references also carry a content hash.
            for name, item in value.items():
                if name.endswith("_path") and isinstance(item, str):
                    hash_key = name[:-5] + "_sha256"
                    if hash_key in value:
                        source = reference(item, parent)
                        if sha256_file(source) != value[hash_key]:
                            raise ValueError("bundle source hash mismatch")
            return result
        if isinstance(value, list):
            return [rewrite(item, parent, key) for item in value]
        if isinstance(value, str) and path_key(key):
            source = reference(value, parent)
            copy(source)
            return str(target_prefix / source.relative_to(source_root))
        return value

    def copy(source):
        if source in done:
            return
        if source in active:
            raise ValueError("bundle has cyclic artifact references")
        destination = output_dir / source.relative_to(source_root)
        if source.is_dir():
            if output_dir.is_relative_to(source):
                raise ValueError("bundle directory dependency contains the export directory")
            ingestion = source / "manifest.json"
            if ingestion.exists() and "canonical_audio" in json.loads(ingestion.read_text()):
                # Export the validated recording contract, excluding unrelated
                # historical draft directories and failed processing attempts.
                copy(ingestion)
                for name in ("spectrograms", "stems", "structure", "passages"):
                    copy(source / name)
                destination.mkdir(parents=True, exist_ok=True)
                done[source] = None
                return
            # Only traverse explicitly referenced artifact directories.
            for directory, _, files in os.walk(source):
                for filename in files:
                    copy((Path(directory) / filename).resolve())
            destination.mkdir(parents=True, exist_ok=True)
            done[source] = None
            return
        active.add(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".json":
            data = rewrite(json.loads(source.read_text()), source.parent)
            json_data[source] = data
            destination.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
        else:
            shutil.copyfile(source, destination)
        done[source] = sha256_file(destination)
        original_hashes[source] = sha256_file(source)
        active.remove(source)

    copy(entry)
    def replace_hashes(value, hashes):
        if isinstance(value, dict):
            return {key: replace_hashes(item, hashes) for key, item in value.items()}
        if isinstance(value, list):
            return [replace_hashes(item, hashes) for item in value]
        return hashes.get(value, value) if isinstance(value, str) else value
    # Propagate JSON content hashes through the graph, including detached
    # base_manifest_sha256 fields in correction annotations.
    for _ in range(len(json_data) + 2):
        hashes = {original_hashes[source]: digest for source, digest in done.items() if digest is not None}
        changed = False
        for source, data in json_data.items():
            destination = output_dir / source.relative_to(source_root)
            destination.write_text(json.dumps(replace_hashes(data, hashes), indent=2, sort_keys=True, allow_nan=False) + "\n")
            digest = sha256_file(destination)
            changed |= digest != done[source]
            done[source] = digest
        if not changed:
            break
    else:
        raise ValueError("bundle hash graph did not converge")
    result = output_dir / "bundle.json"
    result.write_text(json.dumps({"schema": "hearthis.bundle/v1", "entry": str(target_prefix / entry.relative_to(source_root)),
                                  "target_prefix": str(target_prefix), "files": [{"path": str(path.relative_to(source_root)), "sha256": digest}
                                    for path, digest in sorted(done.items()) if digest is not None]}, indent=2) + "\n")
    return result
