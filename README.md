# Hear This

Initial implementation of a passage-level song-understanding pipeline for the
Hear This probabilistic taste model.

## Implemented

### 1. Canonical audio ingestion

The ingestion command decodes any FFmpeg-supported source into a deterministic
44.1 kHz, stereo, 24-bit PCM WAV. It records source and output hashes, audio
geometry, conversion configuration, and the FFmpeg version in an atomic JSON
manifest.

```sh
.venv/bin/hearthis ingest data/fma/audio/116320.mp3 \
  --artifact-root artifacts/audio \
  --recording-id fma-116320
```

### 2. Multiresolution spectrograms

The spectrogram command currently produces two compressed numerical STFT
representations:

- `transient`: 1,024-sample window and 256-sample hop
- `harmonic`: 4,096-sample window and 1,024-sample hop

Every tensor includes frequency bins and frame-center timestamps. Its manifest
records the canonical waveform hash, transform configuration, dimensions, time
range, tensor schema, and tensor hash.

```sh
.venv/bin/hearthis spectrogram \
  artifacts/audio/fma-116320/manifest.json
```

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e '.[dev,fma]'
```

## Tests

```sh
.venv/bin/pytest
```

Tests cover deterministic canonicalization, audio geometry, provenance hashes,
recording-ID path safety, spectrogram dimensions, frame-time alignment, and
frequency localization using a synthetic 440 Hz signal.

## Current real-song artifact

FMA track `116320` has been processed end to end locally:

- Source MP3: 5,710,474 bytes
- Canonical WAV: 178.416 seconds, 44.1 kHz, stereo, 24-bit PCM
- Transient spectrogram: 513 frequency bins and 30,732 time frames
- Harmonic spectrogram: 2,049 frequency bins and 7,680 time frames
- Total generated artifacts: approximately 128 MB

Generated audio representations live under `artifacts/` and are ignored by
Git because they can be reproduced from the tracked manifests and pipeline.

## Next implementation step

Integrate source separation behind a model-independent interface. The first
separator will produce time-aligned vocal, drum, bass, and accompaniment stems,
recombine them to measure reconstruction error, and run this same spectrogram
pipeline on every accepted stem.
