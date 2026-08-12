# Hear This

## All-In-One-compatible structural input

The production preprocessing path reproduces All-In-One's original Madmom
input using Torch. It creates 100-frame-per-second, 81-band log-frequency
spectrograms for bass, drums, other, and vocals. Madmom remains isolated in a
Python 3.11 reference environment for parity checks.

```sh
uv venv --python 3.11 .venv-madmom
uv pip install --python .venv-madmom/bin/python \
  -r requirements-madmom-reference.txt
.venv-madmom/bin/python scripts/export_madmom_reference.py \
  artifacts/audio/fma-116320/stems reference.npy
.venv/bin/python scripts/build_allin1_spectrogram.py \
  artifacts/audio/fma-116320/stems compatible.npy
```

On FMA track `116320`, both paths produced tensors shaped
`4 x 17,842 x 81`. Their mean absolute difference was `9.32e-9`, and their
maximum absolute difference was `1.13e-6`.

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

The spectrogram command produces two compressed numerical STFT
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

### 3. Source separation

The separation layer defines a model-independent backend contract. The first
backend uses Demucs `htdemucs` to produce float32 vocal, drum, bass, and other
stems. The pipeline requires exact agreement with the mixture timeline, sample
rate, frame count, and channel count. It records hashes and geometry for each
stem and measures mixture reconstruction error and SNR.

```sh
.venv/bin/pip install -r requirements-separation.txt
.venv/bin/hearthis separate \
  artifacts/audio/fma-116320/manifest.json \
  --device mps \
  --spectrograms
```

The `--spectrograms` option runs both STFT resolutions on every validated stem.

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
recording-ID path safety, spectrogram dimensions, frame-time alignment,
frequency localization, separation contracts, stem alignment, and mixture
reconstruction using a deterministic fake backend.

## Current real-song artifact

FMA track `116320` has been processed through ingestion, spectrogram generation,
and source separation:

- Source MP3: 5,710,474 bytes
- Canonical WAV: 178.416 seconds, 44.1 kHz, stereo, 24-bit PCM
- Mixture transient spectrogram: 513 frequency bins and 30,732 time frames
- Mixture harmonic spectrogram: 2,049 frequency bins and 7,680 time frames
- Four aligned float32 stems with exactly 7,868,160 frames each
- Normalized stem-reconstruction error: 0.0592
- Stem-reconstruction SNR: 24.55 dB
- All eight stem spectrogram tensors generated successfully
- Total generated artifacts: approximately 695 MB

Generated audio representations live under `artifacts/` and are ignored by
Git because they can be reproduced from the source corpus and pipeline.

## Structural-analysis baseline

The first structure backend uses Librosa onset, beat, chroma, and agglomerative
segmentation algorithms. It emits beat and estimated downbeat timelines,
contiguous section intervals, repeated-section similarity labels, sample ranges,
and frame ranges for every mixture spectrogram. Run it with:

```sh
.venv/bin/pip install -r requirements-structure.txt
.venv/bin/python scripts/analyze_structure.py \
  artifacts/audio/fma-116320/manifest.json
```

For FMA track `116320`, the baseline estimated 129.20 BPM, 349 beats, 87
downbeats, and nine contiguous sections aligned to waveform samples and both
spectrogram resolutions. These labels are unsupervised similarity groups, not
functional verse or chorus claims.

## Next implementation step

Evaluate structural boundaries against annotations, add shorter beat-aligned
passages, and integrate a learned backend for functional labels such as intro,
verse, chorus, bridge, breakdown, and outro.
