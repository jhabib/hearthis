#!/usr/bin/env python3
"""Download a deterministic, genre-diverse set of full-length CC0 FMA tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import defaultdict, deque
from pathlib import Path

from remotezip import RemoteZip


ARCHIVE_URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
DATASET_URL = "https://github.com/mdeff/fma"
LICENSE = "CC0 1.0 Universal"
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
DEFAULT_SEED = 20260812


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracks-csv",
        type=Path,
        default=Path("data/fma/metadata/fma_metadata/tracks.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/fma/audio"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/fma/starter_manifest.json")
    )
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-duration", type=int, default=120)
    parser.add_argument("--max-duration", type=int, default=600)
    return parser.parse_args()


def load_candidates(path: Path, min_duration: int, max_duration: int) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        level_one = next(reader)
        level_two = next(reader)
        next(reader)
        columns = [
            f"{group}.{field}" if group else field
            for group, field in zip(level_one, level_two)
        ]
        index = {name: position for position, name in enumerate(columns)}

        candidates = []
        for row in reader:
            duration = int(float(row[index["track.duration"]] or 0))
            genre = row[index["track.genre_top"]].strip()
            license_name = row[index["track.license"]].strip()
            if license_name != LICENSE or not genre:
                continue
            if not min_duration <= duration <= max_duration:
                continue

            track_id = int(row[0])
            identifier = f"{track_id:06d}"
            candidates.append(
                {
                    "track_id": track_id,
                    "title": row[index["track.title"]],
                    "artist_id": int(row[index["artist.id"]]),
                    "artist": row[index["artist.name"]],
                    "album_id": int(row[index["album.id"]]),
                    "album": row[index["album.title"]],
                    "genre_top": genre,
                    "metadata_duration_seconds": duration,
                    "license": license_name,
                    "license_url": LICENSE_URL,
                    "archive_member": f"fma_full/{identifier[:3]}/{identifier}.mp3",
                    "dataset_url": DATASET_URL,
                }
            )
    return candidates


def select_diverse(
    candidates: list[dict], archive_members: set[str], count: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, deque[dict]] = {}
    by_genre: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        if candidate["archive_member"] in archive_members:
            by_genre[candidate["genre_top"]].append(candidate)
    for genre, tracks in by_genre.items():
        rng.shuffle(tracks)
        grouped[genre] = deque(tracks)

    genres = sorted(grouped)
    selected: list[dict] = []
    used_artists: set[int] = set()
    while len(selected) < count:
        made_progress = False
        for genre in genres:
            queue = grouped[genre]
            deferred = []
            choice = None
            while queue:
                candidate = queue.popleft()
                if candidate["artist_id"] not in used_artists:
                    choice = candidate
                    break
                deferred.append(candidate)
            queue.extend(deferred)
            if choice is None and queue:
                choice = queue.popleft()
            if choice is not None:
                selected.append(choice)
                used_artists.add(choice["artist_id"])
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            break

    if len(selected) != count:
        raise RuntimeError(f"found only {len(selected)} suitable tracks, requested {count}")
    return selected


def ffprobe_duration(path: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        return round(float(subprocess.check_output(command, text=True).strip()), 3)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def download(zipped: RemoteZip, track: dict, output_dir: Path) -> dict:
    destination = output_dir / Path(track["archive_member"]).name
    digest = hashlib.sha256()
    if destination.exists():
        with destination.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        temporary = destination.with_suffix(".mp3.part")
        with zipped.open(track["archive_member"]) as source, temporary.open("wb") as sink:
            while chunk := source.read(1024 * 1024):
                sink.write(chunk)
                digest.update(chunk)
        temporary.replace(destination)

    result = dict(track)
    result.update(
        {
            "local_path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": digest.hexdigest(),
            "decoded_duration_seconds": ffprobe_duration(destination),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.tracks_csv, args.min_duration, args.max_duration)

    with RemoteZip(ARCHIVE_URL) as zipped:
        members = set(zipped.namelist())
        selected = select_diverse(candidates, members, args.count, args.seed)
        completed = []
        for position, track in enumerate(selected, start=1):
            print(
                f"[{position:02d}/{args.count}] {track['track_id']:06d} "
                f"{track['genre_top']}: {track['artist']} - {track['title']}",
                flush=True,
            )
            completed.append(download(zipped, track, args.output_dir))
            args.manifest.write_text(
                json.dumps(completed, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    print(f"Downloaded {len(completed)} tracks to {args.output_dir}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
