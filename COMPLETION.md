# Current implementation and proof

The remaining software paths are connected through `examples/run-complete.json`
and the Docker/PostgreSQL service. This is a concrete finite probabilistic model
with explicit kernels and approximation boundaries. Implementing inference and
evaluation does not establish empirical preference accuracy or musical quality.

## Broad song representation and likelihoods

`broad_features: true` adds measured rhythm regularity, harmonic concentration,
percussive/instrumental shares, stereo width and dynamic range to the original
six music coordinates. Rhythm regularity is inferred beat-interval variation;
harmonic concentration is largest-bin spectral power share; stereo width is
side-channel energy relative to total energy; dynamic range uses 100 ms RMS
percentiles mapped to a 60 dB range. These are declared proxy coordinates with
uncertainty.

Semantic coordinates cover harmonic complexity, vocal expression, arrangement
density, production distortion, lyric valence and lyric narrativity. Missing
reports stay at a broad prior centered at 0.5. Supplied observations use
`recordings[].observations_path` and `schemas/song-observations.schema.json`.
Each report carries a source and observation noise. Listening can request
explicit `semantic_features` definitions; reports retain AI provenance.

Supplied lyrics may have aligned segments. The normalized lyric kernel uses
Poisson sequence length and categorical ordered tokens; metadata tags use
Bernoulli heads. Joint gradients update their neural parameters and music
states. Unknown words map to a declared unknown-token category. Lyrics are
never filled from invented transcripts. These kernels are trainable finite
models; linguistic quality requires independent data.
Ideal generation also samples model lyrics and metadata when a trained
observation kernel is present. They carry synthetic provenance. No unobserved
lyric transcript is attached to an input recording. Standalone generation can
use `model_run_path` with its matching decoder to include these modalities.

## Joint learning and temporal generation

Initial joint runs optimize the acoustic encoder/decoder, variational posterior,
shared response coefficients and taste kernels. Range precision stays positive;
interactions stay symmetric with zero diagonal; activation biases are learned.
Gaussian MAP priors regularize fitted kernel parameters. The selected neural
checkpoint is written to a new hashed `joint-waveform` manifest. Generation and
replay use that checkpoint. Old training artifacts remain unchanged.

New waveform training defaults to an autoregressive conditional VAE. Encoder
and decoder receive the previous observed window during fitting; generation
feeds back the previous sampled window. The Gaussian DCT likelihood and
whitening Jacobian define a normalized conditional density. Generation supports
up to one hour, configurable rate/capacity, and complete section trajectories.
The local full run generates 30-second, 24 kHz mono samples. Temporal dependence
is implemented; coherent musical structure and attribute adherence remain
**Unverified** until independently evaluated.

Segmentation supports all section counts from one to the supplied candidate
count, boundary perturbations, merged regions and global-only hypotheses.
Inactive nodes have zero duration and attention. The finite candidate union,
low-rank posterior and first-order temporal forecast remain explicit
computational approximations.

## Editing, feedback and evaluation

Editing infers posterior acoustic latents from the source, decodes changed
coordinates, and applies only the decoder difference inside requested intervals
with a local fade. Validation independently checks sample rate, duration,
channels, hashes and exact unchanged float PCM outside the intervals. Full runs
accept named feature targets through `edits`.

Listener state records optional observed event IDs and rejects duplicates.
HTTP feedback requires IDs and provenance, and creates an immutable state
version using the parent run's persistent catalog/listener/joint posterior.
Session requests remain separate from persistent personalization.

Evaluation implements exposed held-out Bernoulli log loss, Brier score,
calibration bins, optional inverse-propensity weighting, boundary/label metrics,
waveform measurements, repeated sampling diagnostics and independent-recording
waveform ELBO. It rejects known training event IDs and waveform source hashes.
Playlist diagnostics include split R-hat and exact enumerated cross-checks when
the finite domain is small enough. Independent repeated explanations report
Shapley standard errors; repeated ideal sampling reports normalizer variability.
Diagnostics are specific to the evaluated
configuration.

## Verified local executions

- `artifacts/end-to-end/fma-all-stages/manifest.json`: two actual LLM-reviewed
  recordings, eighteen music coordinates, three response descriptors, 1,931
  continuous variational coordinates and 9,807 complete waveform windows.
  Joint neural/kernel fitting, ranking, playlists, attribution, two generated
  samples, localized editing and evaluation completed. Independent validation
  checks all thirteen model artifacts and the outer edit/evaluation manifests.
- `artifacts/evaluation/held-out-waveform/manifest.json`: a third FMA recording,
  `020092`, excluded from decoder training and joint fitting, evaluated across
  4,033 complete windows. This is waveform likelihood evidence for one test
  recording, without a musical-quality claim.
- PostgreSQL integration tests exercise concurrent claims, idempotency,
  stale-worker fencing, heartbeat renewal, retries and exhaustion in an isolated
  database. Docker smoke evidence is written to `artifacts/service-smoke.json`.
- Kubernetes schema validation covers the eight resources in
  `deploy/kubernetes.yaml`. Actual local serving uses Docker Compose; a remote
  Kubernetes deployment requires the operator's cluster, image and RWX storage.

Real listener accuracy, population response calibration, independent listening
accuracy and perceived generated quality remain **Unverified**. Resolve these
with exposed real feedback and independently reviewed held-out recordings.
