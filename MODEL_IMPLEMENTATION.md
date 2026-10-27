# Executable mathematical model

Hearthis implements the supplied paper as an executable finite model, from
validated audio and LLM listening review through joint posterior inference,
recommendation, and learned waveform generation. The paper leaves neural
kernels unspecified; the implementation supplies explicit parametric kernels,
learned audio networks, and recorded approximation boundaries.

The completed capabilities and current verification are described in
`COMPLETION.md`. `examples/run-complete.json` enables broad coordinates, joint
neural/kernel learning, autoregressive 24 kHz generation, localized editing and
evaluation. Distributed serving is documented in `DEPLOYMENT.md`.

```sh
.venv/bin/pip install -e '.[dev,analysis]'
.venv/bin/python -m hearthis.cli run examples/run-fma-finished.json \
  --output-dir artifacts/end-to-end/new-full-run
.venv/bin/python -m hearthis.cli verify-run \
  artifacts/end-to-end/fma-final-v5/manifest.json
```

The example requires the locally processed FMA recordings listed in its
configuration and an authenticated Meta APE model gateway. It learns the
decoder, grounds a natural request and critique with Gemini, fits the joint
model with hierarchical dynamics, and generates WAVs. Profiles and feedback
are labeled synthetic. `examples/run-fma-trained.json` reuses the completed
decoder and cached live language response for local replay.

## Equation map

| Paper equations | Implementation | Contract |
| --- | --- | --- |
| 1 to 8 | `pipeline.py`, `model/features.py`, `model/waveform.py`, `model/joint.py` | Anchored uncertain music states; conditional neural waveform VAE and finite complete segmentation hypotheses |
| 9 | `model/response.py:ResponseProfile` | Positive Dirichlet mode weights and bounded Beta descriptor modes per aligned region |
| 10 to 12 | `ResponseProfile.predict`, `Song.predict` | Listener/intent-conditioned mode selection and bounded response; attention aggregates global, section, and passage responses |
| 17 | `model/features.py`, `Song.predict` | Six baseline or eighteen broad music coordinates plus three subjective descriptors, all in the unit cube |
| 18 to 20 | `model/taste.py:Listener`, `JointPosterior.forecast` | Hierarchical random transition matrices, Gaussian innovations and change points; joint temporal forecast and intent activation |
| 21 and 22 | `TasteComponent.energy` | Linear preference, half-factor range penalty, half-factor symmetric zero-diagonal interactions, softplus negative regions, normalized utility scale |
| 24 and 25 | `model/recommend.py:sample_ideal_songs` | Bounded mixture prior and conditional posterior importance sampling; zero feasible support fails explicitly |
| 13 to 15, 26 and 27 | `model/language.py`, `JointModel.log_observations` | LLM reverse semantic grounding, controlled text rendering, uncertain region likelihoods, Gaussian feature reports and critiques |
| 28 to 30 | `model/taste.py:fit_feedback` | Pairwise Bernoulli utility differences, exposure likelihood, conditional behavior likelihood with optional response coefficients and session bias |
| 31 to 35 | `model/joint.py:fit_joint`, `model/waveform.py:train_waveform`, `model/modalities.py` | Joint neural and variational learning with waveform, lyric/metadata likelihoods and taste kernels; Gaussian MAP regularization of fitted parameters |
| 36 to 40 | `model/probability.py:GaussianPosterior`, `Listener.condition_prompt` | Exact linear Gaussian update; session intent updates weights while unobserved prior differences persist |
| 41 | `rank_catalog` | Posterior utility mean and uncertainty, risk preference, external artist exclusions |
| 42 | `sample_playlist` | Finite Gibbs sequence sampler with all-pairs diversity, adjacent transitions, and horizon-derived item count |
| 49 | `explain_song` | Conditional replacement effects and all-coalition conditional Shapley contributions; predictive interpretation |

## Inputs, outputs, and state

`hearthis run CONFIG --output-dir NEW_DIRECTORY` accepts existing validated
recordings and/or raw audio sources. Raw sources pass through canonical PCM
conversion, two mixture spectrogram resolutions, four-stem separation, eight
stem spectrograms, structure analysis, passage alignment, and validation.
Optional listening submits the complete resampled waveform to an audio-capable
provider. Corrections preserve a source-manifest hash and hashed listening
evidence; they produce separate corrected structure and passage manifests.

The basic taste run writes eleven JSON artifacts and a manifest with content hashes:
catalog, listener prior, persistent listener posterior, session listener,
intent, feedback, inference diagnostics, ranking, ideal feature samples,
playlist, and explanations. `hearthis verify-model MANIFEST` independently
checks hashes, source lineage, model dimensions, probability normalization,
risk scores, playlist eligibility, compact feasibility, and explanation totals.
The full run adds `joint-posterior.json`, `generated-audio.json`, and WAV files.
`hearthis verify-run` also checks configuration/model lineage and source audio
alignment. Successful manifests are atomically published after verification.
Output directories must be new. Failed processing leaves inspectable partial
artifacts that can be referenced by a subsequent configuration.
Externally supplied corrections are read-only inputs. Each run writes corrected
timelines under its own `corrected/` directory. Optional edit and evaluation
manifests are hashed and independently verified by the outer run manifest.

The shared prior owns encoder coefficients and population descriptor modes.
Each song owns global, section, and passage response profiles. The joint
posterior preserves covariance across music, descriptor modes, mixture weights,
shared coefficients, listener weights, states, and transition matrices.
Each listener
owns component weight/state Gaussian posteriors and dynamics. Functions clone
caller-owned states before a model run. Historical feedback updates persistent
listener and song posteriors. Current request pseudo-observations update only
the session copy. Reusing the persistent posterior preserves personalization
without accidentally saving a temporary request as long-term preference.

## Coordinates and intent

Music coordinates are `rms_level`, `spectral_brightness`, `spectral_flux`,
`vocal_share`, `bass_share`, and `tempo`. Their exact analyzer definitions are
saved in catalog provenance. RMS is a level proxy, spectral brightness is a
normalized centroid, stem shares can contain separation bleed, and tempo is an
estimate. These coordinates contain no inferred lyrics or unsupported
composition/production labels.

Subjective coordinates are `arousal`, `intimacy`, and `tension`. They remain
random descriptor profiles, starting from uninformed priors or weak direct
audio-LLM supervision. Optional shared learning fits a regularized encoder from
declared descriptor observations. This is empirical Bayes conditioning on weak
supervision, with subsequent local song-profile updates; uncertainty does not
include uncertainty in the fitted shared coefficients.
That statement describes measured-feature compatibility fitting. The joint
path retains Gaussian uncertainty over the shared coefficient block.

Requests use explicit clauses, for example `arousal=0.6; vocal_share=0.65`, or
the structured `intent.targets` mapping. Unknown free-text requests fail with a
grounding error. Context offsets and multiplicative interactions are bounded
back into the unit cube. Minimum/maximum feature bounds restrict feasibility.
Artist exclusions affect catalog retrieval and playlists, while the ideal
feature density retains its original musical target. Positive risk penalizes
uncertainty; negative risk rewards uncertainty.

## Feedback and resume

Feedback records require `kind`, explicit `exposed`, and a recorded
`propensity` in (0,1]. An unexposed record has no observed `value`. An impossible
exposure under its declared policy fails. Pairwise events additionally require
two known song IDs and a Bernoulli preference. Behavior events require one
known song ID and a Bernoulli outcome. Propensities must come from the exposure
policy; they are not inferred from saves or skips. Optional inverse-propensity
weighting is labeled as a generalized ELBO, separately from likelihood fitting.

Critiques require an explicitly grounded feature, polarity, magnitude,
confidence, and region ID or normalized `region_probabilities`. Optional
`descriptor_reports` and `mode_probabilities` condition response profiles.
`step` records ordered dynamics, and an optional `intent` records the historical
context rather than fitting past feedback under the current request.

To resume a trained local state, create a configuration with `catalog_path`
pointing to the previous `taste/catalog.json`, `listener_path` pointing to
`taste/listener-posterior.json`, and only new `feedback` or `feedback_path`.
The saved catalog includes updated song response profiles and the shared
prior. For full joint replay, also set `joint_posterior_path` to the previous
`taste/joint-posterior.json`; Gaussian summaries alone discard cross-block
covariance. Supply only new feedback. A resumed posterior becomes the prior
for new observations, so previously observed waveform/feedback likelihoods
are not counted again. Ordered later steps forecast the joint state before
conditioning on new data. Resuming cannot simultaneously rebuild recordings
or refit the shared prior.
Replay whitens the prior's independent innovations and shared Gaussian
factors, uses a normalized auxiliary Gaussian ELBO, and optimizes with line
search in double precision. This avoids dense covariance construction and
preserves the prior's covariance directions. Unexposed records retain the
prior and add no preference observation.
Use a new output directory for every run. These paths are read as supplied;
run repository examples from the repository root.

## Verification and limits

`hearthis demo` reproduces the paper's four range energies and preferred odds:
listener A favors Diamonds by about 5.45, and listener B favors After Love by
about 12.37. These are the paper's synthetic profiles, not empirical song
measurements. Tests independently check these numbers, conjugate updates,
interaction scaling, mode permutation, exposure handling, feedback learning,
conditional imputation, Shapley efficiency, constraints, resume, audio bytes,
failure publication, and audio-to-model orchestration.

The full local run at `artifacts/end-to-end/fma-final-v5/manifest.json` uses two
actual Gemini waveform listening reviews, live natural-language grounding,
weak descriptor supervision, synthetic listener feedback, hierarchical
transitions, all complete audio windows, joint inference, ranking, playlists,
explanations, and three generated waveform samples. It demonstrates execution
and artifact consistency. Its recommendations provide no evidence of
preference accuracy for a real person.

## Neural waveform model and joint objective

The decoder observes resampled mono audio in nonoverlapping fixed-length
windows. New checkpoints condition on previous audio; legacy checkpoints retain
their independent-window kernel. An orthonormal DCT is invertible; recorded whitening scales contribute
their Jacobian to a normalized Gaussian waveform likelihood. A neural encoder
infers a Gaussian acoustic latent conditioned on audio and music coordinates;
a neural decoder maps sampled acoustic latents and those coordinates to
waveform-distribution means. Generation samples both acoustic latents and
observation noise and inverts the DCT. It uses no catalog-clip copying.

Decoder training uses a separate held-out window split and selects the best
paired validation ELBO among checkpoints improving training ELBO. The local
selected checkpoint is epoch 8; both paired objectives improved. The joint
likelihood then uses every complete source window, separately from the
pretraining subset. A final partial window shorter than the configured window
is excluded from the likelihood. Audio observation rate and window size are
explicit configuration, not full-fidelity mastered-audio reconstruction.

`fit_joint` evaluates transformed Beta/Dirichlet profile densities with their
Jacobians, Gaussian music/shared/listener priors, hierarchical transition
densities, exposure and feedback likelihoods, and the conditional waveform
ELBO. Finite segmentation and region-grounding alternatives are evaluated
explicitly; a listener's shared change-point indicator is integrated across
ordinary and enlarged-innovation hypotheses. Bernoulli confidence uses a
normalized reliability mixture. The Gaussian variational family uses a diagonal covariance plus
low-rank cross-block factors; the acoustic posterior is a conditional neural
Gaussian. Common random numbers reduce reparameterization noise and waveform
minibatches estimate the complete-data objective. Diagnostics retain paired
initial/final ELBO, iteration history, observation count and protocol.

`waveform`, `joint`, `dynamics`, `language`, and `generation` in the full
configuration enable these stages. Existing measured-feature runs remain
compatible. `waveform_path` reuses a hashed decoder; `language.response_cache_path`
reuses a live response only when its complete request hash matches. Invalid
language responses are retained for diagnosis and never become completed
groundings.

Remaining boundaries:

- Music coordinates use empirical anchored priors; lyrics and unannotated
  composition/production attributes are not invented. New segmentation supports
  different section counts, merged regions and global-only hypotheses on a
  finite candidate-node union. Its support remains computationally bounded.
- The neural acoustic posterior and low-rank continuous family approximate the
  joint posterior. Temporal bilinear dynamics use a first-order Gaussian
  forecast; covariance compression restores discarded marginal variance and
  a small diagonal floor stabilizes the covariance representation.
  Conjugate session conditioning retains covariance through conditional
  simulation. Saved Gaussian/Beta/Dirichlet summaries are compatibility views.
- Region attention and response kernels have declared model semantics.
  Population psychometric calibration is **Unverified**.
- New waveform checkpoints use autoregressive context and section trajectories;
  old independent-window checkpoints remain compatible.
  Generated audio fidelity, long-term musical structure, and adherence to every
  requested musical attribute are **Unverified** without listening evaluation
  and a larger independently evaluated training corpus.
- Ranking averages utility across feature draws and permits candidates with
  positive feasible posterior support. It reports feasible probability rather
  than guaranteeing every uncertain response draw meets a hard bound.
- Importance normalizers and conditional expectations are Monte Carlo
  approximations. Evaluation now reports repeated ranking estimates, importance
  effective sample sizes, split R-hat across playlist chains and exact finite
  playlist cross-checks. Finite diagnostics do not prove convergence for every
  configuration. Repeated explanation estimates report Shapley standard errors
  for the selected candidates.
- Listening judgments, boundaries, and confidence calibration are
  **Unverified** without independent listening review. Real preference accuracy
  is **Unverified** without actual exposed listener feedback and held-out
  evaluation. Synthetic demonstration observations are labeled in the inputs.
