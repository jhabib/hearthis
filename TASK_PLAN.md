# Hear This Task Plan

## Initial objective

Build a reproducible pipeline that converts a commercial song recording into:

1. Time-aligned waveform and spectrogram representations.
2. Separated vocal, drum, bass, and accompaniment waveforms.
3. A hierarchical timeline of musical parts, including sections and shorter passages.

This stage establishes the observable song representation used by later perceptual modeling, language grounding, recommendation, and controlled editing.

## Milestone 1: Audio ingestion and canonical waveform

- Define the supported input formats and audio licensing constraints.
- Decode inputs to a canonical lossless waveform using FFmpeg.
- Standardize sample rate, channel layout, amplitude convention, and duration metadata.
- Assign stable song and recording identifiers.
- Store processing configuration, tool versions, checksums, and provenance with every artifact.
- Detect corrupt, truncated, silent, clipped, or unsupported recordings.

### Deliverables

- A command that ingests one recording and produces a canonical waveform plus metadata.
- Deterministic tests using small licensed, public-domain, or synthetic fixtures.
- A machine-readable artifact manifest.

### Acceptance criteria

- Reprocessing the same input and configuration produces identical metadata and equivalent audio.
- Every derived artifact can be traced to its source recording and processing configuration.
- Failures return actionable, structured errors.

## Milestone 2: Multiresolution spectrograms

- Generate short-time Fourier transform spectrograms at multiple time and frequency resolutions.
- Generate mel spectrograms for learned audio models.
- Preserve mappings among waveform samples, spectrogram frames, and wall-clock timestamps.
- Decide which representations remain numerical tensors and which receive visual previews.
- Record transform parameters such as window, hop, FFT size, frequency range, magnitude scale, and normalization.

### Deliverables

- Numerical spectrogram tensors for the full mix and each available stem.
- Human-inspectable spectrogram images for debugging.
- A shared time-coordinate API for retrieving frames by song interval.

### Acceptance criteria

- A requested time interval resolves consistently across waveform and spectrogram representations.
- Spectrogram dimensions and frequency bins match their declared parameters.
- Reconstruction tests quantify information loss where inversion is supported.

## Milestone 3: Source separation

- Establish a baseline separator for vocals, drums, bass, and remaining accompaniment.
- Preserve sample-level alignment between the mixture and every separated stem.
- Measure mixture reconstruction error after recombining stems.
- Detect separation failures, bleed, missing vocals, and temporal drift.
- Compare separation quality on representative genres, production styles, and vocal arrangements.
- Keep the separation interface model-independent so the separator can be replaced later.

### Deliverables

- Time-aligned waveform files and spectrograms for each stem.
- Separation metadata containing model version, configuration, runtime, and quality signals.
- A small listening and quantitative evaluation set.

### Acceptance criteria

- Every stem has the same declared timeline as the canonical mixture.
- Stem recombination stays within a defined reconstruction-error threshold.
- Vocal passages can be located in both the vocal stem and original mixture.
- Quality failures are surfaced instead of silently entering downstream training data.

## Milestone 4: Song structure and passage segmentation

- Detect beats, downbeats, tempo changes, and candidate boundaries.
- Infer functional sections such as intro, verse, pre-chorus, chorus, bridge, instrumental, breakdown, and outro when supported by evidence.
- Create shorter passages within sections for localized vocal, instrumental, lyrical, and production analysis.
- Represent uncertain or overlapping boundaries probabilistically.
- Align section and passage boundaries with waveforms, stems, spectrogram frames, and lyrics when lyrics are available.
- Support multiple segmentation hypotheses when the recording has ambiguous structure.

### Deliverables

- A hierarchical timeline containing song, section, passage, beat, and frame intervals.
- Boundary confidence, section-label confidence, and alternate hypotheses.
- A visualization that overlays boundaries on the waveform and spectrogram.
- An annotation format for human corrections.

### Acceptance criteria

- All intervals are valid, ordered, time-aligned, and traceable to model outputs or corrections.
- The system preserves uncertainty instead of forcing unsupported labels.
- Human corrections can replace or supplement inferred boundaries without regenerating unrelated artifacts.
- Boundary and label quality are measured on an annotated evaluation set.

## Milestone 5: Unified song-artifact schema

- Define schemas for recordings, waveforms, stems, spectrograms, beats, sections, passages, lyrics, model outputs, and provenance.
- Separate immutable source artifacts from versioned derived artifacts.
- Allow several representations and segmentation hypotheses for the same recording.
- Provide APIs for retrieving any song interval across the mixture, stems, features, and structural hierarchy.

### Deliverables

- Versioned schema definitions and example records.
- End-to-end processing for a small evaluation catalog.
- A validation command that checks alignment, completeness, and provenance.

### Acceptance criteria

- One interval identifier retrieves aligned data from every available representation.
- New separation or segmentation models can add versioned outputs without overwriting prior results.
- Downstream models can consume artifacts without depending on a particular extraction implementation.

## Evaluation plan

- Use synthetic mixtures with known stems for alignment and reconstruction tests.
- Use licensed or public-domain multitrack recordings for separation evaluation.
- Use human-annotated songs for beat, boundary, and section-label evaluation.
- Include diverse genres, languages, vocal styles, recording eras, and production densities.
- Track quantitative metrics alongside structured listening reviews and annotator corrections.

## Initial execution order

1. Implement canonical ingestion and artifact manifests.
2. Implement multiresolution spectrogram generation and time alignment.
3. Integrate a baseline source separator and validate stem reconstruction.
4. Integrate beat, boundary, and section inference.
5. Add hierarchical passage generation and human correction support.
6. Run the complete pipeline on the evaluation catalog and document failure modes.

## Deferred work

- Listener taste inference and recommendation.
- Language grounding from critiques to passages.
- Perceptual and psychological response modeling.
- Personalized music editing and generation.
- Large-scale distributed processing and production serving infrastructure.
