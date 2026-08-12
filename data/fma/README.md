# FMA starter corpus

This local starter corpus contains 25 full-length MP3 recordings selectively
downloaded from `fma_full.zip`. All selected recordings are labeled
`CC0 1.0 Universal` in the official FMA metadata.

## Local files

- Audio: `data/fma/audio/`
- Selection and integrity metadata: `data/fma/starter_manifest.json`
- Official metadata archive: `data/fma/fma_metadata.zip`
- Extracted metadata: `data/fma/metadata/`

The downloaded audio and bulk metadata are ignored by Git. The manifest and
this README are tracked so the corpus can be reproduced.

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_fma_starter.py
```

The downloader reads the remote ZIP directory using HTTP range requests and
downloads only the selected members. It records each file's SHA-256 hash,
metadata duration, artist, title, genre, archive member, and license.

## Current validation

- 25 MP3 files downloaded
- 197,132,288 bytes on disk
- All 25 decode successfully with macOS `afinfo`
- All durations exceed 120 seconds
- All SHA-256 hashes match the manifest
- Genres represented: Classical, Experimental, Instrumental, Old-Time /
  Historic, Rock, and Spoken

Dataset source: <https://github.com/mdeff/fma>

CC0 license: <https://creativecommons.org/publicdomain/zero/1.0/>
