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
reconstruction using a deterministic fake backend. Regression tests also reject
corrupt tensors, nonfinite values, missing representations, invalid sample and
frame intervals, passages assigned to the wrong section, failed reconstruction,
stale separation manifests, and learned-artifact lineage mismatches. Identical
mono and duplicated-stereo PCM inputs must produce equivalent learned features.

Cross-artifact validation requires the transient and harmonic mixture
spectrograms and four aligned stems. Stem spectrogram generation remains
optional; when its directory exists, validation requires every stem and mixture
resolution. It recomputes reconstruction quality from the waveforms and checks
stored learned input and activation hashes, shapes, and probabilities. A failed
separation raises an error and leaves no success manifest for that run.

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

## Learned functional structure

The learned backend runs an open All-In-One Harmonix checkpoint and preserves
frame-level beat, downbeat, boundary, and functional-label probabilities. Its
manifest records the top three labels for every section, boundary confidence,
waveform samples, and spectrogram frames. A portable Torch adapter implements
the checkpoint's legacy NATTEN operators and records the instrument-attention
padding workaround in provenance.

```sh
.venv-madmom/bin/python scripts/analyze_learned_structure.py \
  artifacts/audio/fma-116320 \
  --input artifacts/audio/fma-116320/allin1/torch-compatible-unfold.npy \
  --device mps
```

For FMA track `116320`, one Harmonix fold estimated 128 BPM, 359 beats, 90
downbeats, and eight musical sections labeled intro, verse, chorus,
instrumental, solo, and outro. The learned timeline produced 25 aligned
passages and passed cross-artifact validation.

## End-to-end taste model

The model connects validated audio and LLM listening to joint inference over
music, multimodal region responses, shared parameters and evolving listener
tastes. It grounds natural-language requests and critiques, samples ideal
features, ranks the catalog, builds playlists, explains utility, and decodes
ideal states to waveform samples with a trained neural model.

```sh
.venv/bin/pip install -e '.[dev,analysis]'
.venv/bin/python -m hearthis.cli demo
.venv/bin/python -m hearthis.cli run examples/run-fma.json \
  --output-dir artifacts/end-to-end/new-run --device mps
```

The example requires the downloaded FMA starter corpus and the existing first
recording. It processes a second recording with Demucs and Librosa and submits
its complete waveform to Gemini through the authenticated Meta APE gateway.
Its listener profiles and feedback are explicitly synthetic demonstrations.
For the completed local audio artifacts, `examples/run-fma-processed.json`
reuses both reviewed recordings and runs all mathematical stages again.

```sh
.venv/bin/python -m hearthis.cli run examples/run-fma-processed.json \
  --output-dir artifacts/end-to-end/new-model-run
.venv/bin/python -m hearthis.cli verify-model \
  artifacts/end-to-end/fma-complete/taste/manifest.json
```

See [MODEL_IMPLEMENTATION.md](MODEL_IMPLEMENTATION.md) for the equation map,
configuration, feedback contracts, resume procedure, and approximation limits.

```sh
.venv/bin/python -m hearthis.cli run examples/run-fma-finished.json \
  --output-dir artifacts/end-to-end/new-full-run
.venv/bin/python -m hearthis.cli verify-run \
  artifacts/end-to-end/fma-final-v5/manifest.json
```

This full configuration trains the waveform VAE, performs LLM language
grounding and joint variational inference with hierarchical dynamics, and
generates WAV files. `examples/run-fma-trained.json` reuses the completed local
decoder and cached live language response. Existing feature-only examples
remain supported. Human preference calibration and generated musical quality
are **Unverified** pending independent evaluation.

## Complete model and distributed serving

`examples/run-complete.json` runs eighteen music coordinates and three response
descriptors through joint neural/kernel learning, autoregressive 24 kHz waveform
generation, localized editing and evaluation. Supplied lyric/metadata likelihoods
and uncertain semantic reports are supported. Missing labels remain uncertain.
Existing six-coordinate catalogs and decoder checkpoints remain compatible.

```sh
.venv/bin/python -m hearthis.cli run examples/run-complete.json \
  --output-dir artifacts/end-to-end/new-complete-run
.venv/bin/python -m hearthis.cli verify-run \
  artifacts/end-to-end/fma-all-stages/manifest.json
```

The HTTP API serves queued runs, generation, editing, evaluation, versioned
feedback and verified JSON/WAV artifacts. PostgreSQL coordinates independently
scaled workers. Docker Compose and Kubernetes deployments are included.
See [DEPLOYMENT.md](DEPLOYMENT.md) for startup and HTTP contracts and
[COMPLETION.md](COMPLETION.md) for current model capabilities and evidence.

## Direct LLM listening review

The Gemini reviews for FMA `116320` and `117057` received complete waveform
audio and a review prompt containing duration and output requirements. They
received no classifier outputs or draft annotations. The resulting five- and
seven-section timelines, subjective response descriptors, provider usage,
input hashes, and prompts are saved under `annotations/<recording>/` as
`listening-correction.json` and `listening-evidence.json`. Audio input files
remain in ignored artifact directories. The second review's endpoint was
rounded by 0.3 milliseconds; evidence records the correction to canonical
duration and the raw provider response remains in artifacts.

```sh
.venv/bin/python -m hearthis.cli listen artifacts/audio/fma-116320 \
  --output-dir artifacts/audio/fma-116320/new-review \
  --provider meta-ape --model gemini-2.5-pro
```

An OpenAI-compatible audio endpoint is also supported with `--provider https`,
`HEARTHIS_AUDIO_ENDPOINT` pointing to its HTTPS chat-completions URL, and
`HEARTHIS_AUDIO_API_KEY` configured locally. The endpoint must support WAV
`input_audio`. Failed or invalid responses never publish a completed annotation.

These files identify their author as an LLM. `independent_reference: true`
records that the listening request was blind to classifier annotations.
Boundary accuracy, instrument identification, and confidence calibration are
**Unverified** until an independent reviewer listens and checks them.

## LLM annotation draft

`annotations/fma-116320/llm-correction.json` contains eight provisional section
annotations written by an LLM in a music-reviewer role. Its evidence snapshot
records the learned posteriors and measured separated-stem RMS values. The LLM
did not listen to the recording. Boundaries are inherited from the learned
model, and confidence values are qualitative judgments rather than calibrated
probabilities. The draft uses broader labels for the uncertain chorus and solo.

The correction format supports `annotation_source: "llm"` and requires hashed
evidence and a base-manifest hash. Applying the draft preserves its provenance
and marks sections as `llm_annotation`. Existing human correction files remain
compatible and default to `human_annotation`.

```sh
.venv/bin/python scripts/apply_structure_correction.py \
  artifacts/audio/fma-116320/structure-learned/manifest.json \
  annotations/fma-116320/llm-correction.json \
  artifacts/audio/fma-116320/structure-llm/manifest.json \
  --spectrogram-dir artifacts/audio/fma-116320/spectrograms
.venv/bin/python scripts/build_passages.py artifacts/audio/fma-116320 \
  --structure-manifest artifacts/audio/fma-116320/structure-llm/manifest.json \
  --output artifacts/audio/fma-116320/passages-llm/manifest.json
.venv/bin/python scripts/validate_recording.py artifacts/audio/fma-116320 \
  --structure-manifest artifacts/audio/fma-116320/structure-llm/manifest.json \
  --passages-manifest artifacts/audio/fma-116320/passages-llm/manifest.json
```

Use this draft to exercise annotation and correction workflows. Agreement with
its source model cannot establish boundary or label accuracy. Independent
quality evaluation still requires reviewed annotations based on listening.
