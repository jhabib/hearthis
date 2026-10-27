"""Posterior ranking, ideal-song importance sampling, playlists and attribution."""

from __future__ import annotations

from math import ceil, comb

import numpy as np
from scipy.special import expit, logit, logsumexp

from .probability import FeaturePrior, probabilities
from .response import SharedResponsePrior, Song
from .taste import Intent, Listener


def rank_catalog(songs: list[Song], prior: SharedResponsePrior, listener: Listener, intent: Intent,
                 *, seed: int = 0, draws: int = 256, joint_posterior=None) -> tuple[list[dict], dict]:
    """Equation 41; catalog exclusions remain external to the ideal target."""
    if not songs or len({song.song_id for song in songs}) != len(songs):
        raise ValueError("catalog requires distinct songs")
    intent.validate(listener.feature_names)
    rng = np.random.default_rng(seed)
    coupled = joint_posterior.predict(songs, prior, listener, intent, seed, draws) if joint_posterior is not None else None
    centers, weights = (coupled["centers"], coupled["weights"]) if coupled is not None else listener.draws(rng, draws)
    rows, features_by_song, utility_samples = [], {}, []
    for song in songs:
        features, response_summary = ((coupled["features"][song.song_id], coupled["summaries"][song.song_id])
                                      if coupled is not None else song.predict(prior, listener, intent, centers, rng))
        values = listener.utility(features, intent, centers=centers, weights=weights)
        scores = intent.beta * values
        feasible = intent.feasible(features, listener.feature_names)
        eligible = song.artist.casefold() not in {a.casefold() for a in intent.excluded_artists}
        score = float(scores.mean() - intent.risk * scores.std())
        if not eligible or not feasible.any():
            score = -np.inf
        rows.append({"song_id": song.song_id, "title": song.title, "artist": song.artist,
                     "duration_seconds": song.duration_seconds, "eligible": eligible,
                     "feasible_probability": float(feasible.mean()),
                     "score": score if np.isfinite(score) else None,
                     "expected_utility": float(values.mean()), "utility_sd": float(values.std()),
                     "features_mean": features.mean(axis=0).tolist(),
                     "features_sd": features.std(axis=0).tolist(), **response_summary})
        utility_samples.append(np.where(feasible, scores, -np.inf))
        features_by_song[song.song_id] = features
    # Finite-catalog approximation of equation 24, with equal base mass.
    # Normalize each posterior draw separately before averaging predictions.
    target_logits = np.stack(utility_samples, axis=-1)
    valid_draws = np.isfinite(target_logits).any(axis=-1)
    if not valid_draws.any():
        raise ValueError("catalog has no feasible posterior support")
    target_probabilities = probabilities(target_logits[valid_draws]).mean(axis=0)
    rank_logits = np.array([r["score"] if r["score"] is not None else -np.inf for r in rows])
    ranking_probabilities = probabilities(rank_logits)
    for index, row in enumerate(rows):
        row["catalog_target_probability"] = float(target_probabilities[index])
        row["ranking_probability"] = float(ranking_probabilities[index])
    rows.sort(key=lambda r: -(r["score"] if r["score"] is not None else -np.inf))
    return rows, {"centers": centers, "weights": weights, "features": features_by_song,
                  "posterior_feasible_fraction": float(valid_draws.mean())}


def sample_ideal_songs(prior: SharedResponsePrior, music_base: FeaturePrior, listener: Listener,
                       intent: Intent, *, seed: int = 0, count: int = 16,
                       particles: int = 4096, response_draws: int = 16, joint_posterior=None) -> dict:
    """Equations 24 and 25 as a posterior mixture of importance samplers.

    Sample a taste posterior state, then normalize its conditional target over
    plausible music particles. Response profiles/assignments/noise are integrated
    under the generative predictive for hypothetical music, rather than copied
    from a retrieved song. Artist exclusions never alter this feature density.
    """
    intent.validate(listener.feature_names)
    if (music_base.dimension != len(prior.music_names) or particles <= 0
            or count <= 0 or response_draws <= 0):
        raise ValueError("invalid ideal-song sampling dimensions or counts")
    rng = np.random.default_rng(seed)
    coupled = None
    if joint_posterior is not None:
        model = joint_posterior.model
        with model.torch.no_grad():
            coupled = model.parameters(model.tensor(joint_posterior.sample(rng, count, intent.targets)))
        centers, weights = coupled[0].numpy(), coupled[1].numpy()
    else:
        centers, weights = listener.draws(rng, count)
    states, diagnostics = [], []
    m, q = len(prior.music_names), len(prior.response_names)
    target = np.array([intent.targets.get(name, 0.5) for name in prior.response_names])
    for draw in range(count):
        raw_music = music_base.sample(rng, particles)
        music = intent.map_music(raw_music, prior.music_names)
        designs = np.column_stack([np.ones(particles), raw_music])
        coefficients = coupled[2][draw].numpy() if coupled is not None else prior.coefficients
        eta_means = np.clip(expit(np.einsum("lqm,nm->nlq", coefficients, designs)), 1e-6, 1 - 1e-6)
        activation = listener.activation(intent, centers[draw])
        listener_center = np.sum(activation[:, None] * centers[draw], axis=0)[m:]
        response = np.zeros((particles, q))
        for _ in range(response_draws):
            eta = rng.beta(eta_means * prior.descriptor_strength,
                           (1 - eta_means) * prior.descriptor_strength)
            mixture = rng.dirichlet(prior.concentration, size=particles)
            compatibility = -8 * np.mean((eta - (listener_center + target) / 2) ** 2, axis=-1)
            assignment = probabilities(np.log(mixture) + compatibility)
            perception = expit(logit(np.clip(eta, 1e-8, 1 - 1e-8))
                                + listener_center - 0.5 + target - 0.5)
            response += np.sum(assignment[..., None] * perception, axis=1) / response_draws
        features = np.column_stack([music, response])
        utility = listener.utility(features, intent, centers=centers[draw], weights=weights[draw])
        feasible = intent.feasible(features, listener.feature_names)
        log_weights = np.where(feasible, intent.beta * utility, -np.inf)
        proposal_weights = probabilities(log_weights)
        selected = int(rng.choice(particles, p=proposal_weights))
        states.append({"music": raw_music[selected].tolist(), "augmented_features": features[selected].tolist(),
                       "utility": float(utility[selected]), "posterior_state_draw": draw})
        diagnostics.append({"log_normalizer_estimate": float(logsumexp(log_weights) - np.log(particles)),
                            "effective_sample_size": float(1 / np.sum(proposal_weights ** 2)),
                            "feasible_fraction": float(feasible.mean())})
    return {"method": "posterior conditional importance sampling on bounded music features",
            "particles_per_draw": particles, "response_draws": response_draws,
            "samples": states, "diagnostics": diagnostics,
            "representation": "Feature states; an attached waveform model decodes these states separately."}


def sample_playlist(ranking: list[dict], intent: Intent, *, seed: int = 0,
                    diversity_strength: float = 1.0, sequence_strength: float = 1.0,
                    sweeps: int = 200, record_every: int = 0) -> dict:
    """Gibbs sampling of equation 42 with all-pairs diversity and transitions.

    Repeats are supported by the paper's sequence domain and discouraged by the
    diversity potential. Gibbs convergence is not claimed by a finite run.
    """
    eligible = [r for r in ranking if r["score"] is not None and r["catalog_target_probability"] > 0]
    if not eligible or min(diversity_strength, sequence_strength) < 0 or sweeps <= 0 or record_every < 0:
        raise ValueError("playlist requires eligible candidates and valid sampling parameters")
    duration = np.mean([r["duration_seconds"] for r in eligible])
    size = ceil(intent.horizon_seconds / duration)
    if size > 1000:
        raise ValueError("playlist horizon exceeds the reference sampler's 1000-item limit")
    features = np.array([r["features_mean"] for r in eligible])
    squared_distance = np.mean((features[:, None, :] - features[None, :, :]) ** 2, axis=-1)
    diversity = -diversity_strength * np.exp(-squared_distance / 0.05)
    transition = -sequence_strength * squared_distance
    log_target = np.log([r["catalog_target_probability"] for r in eligible])
    rng = np.random.default_rng(seed)
    state = rng.choice(len(eligible), size=size, p=probabilities(log_target))
    trace = []
    for sweep in range(sweeps):
        for index in rng.permutation(size):
            log_weights = log_target.copy()
            for other in range(size):
                if other != index:
                    log_weights += diversity[:, state[other]]
            if index > 0:
                log_weights += transition[state[index - 1], :]
            if index + 1 < size:
                log_weights += transition[:, state[index + 1]]
            state[index] = rng.choice(len(eligible), p=probabilities(log_weights))
        if record_every and (sweep + 1) % record_every == 0:
            log_density = float(log_target[state].sum())
            log_density += sum(diversity[state[i], state[j]] for i in range(size) for j in range(i + 1, size))
            log_density += sum(transition[state[i], state[i + 1]] for i in range(size - 1))
            trace.append({"songs": [eligible[int(i)]["song_id"] for i in state], "log_density": float(log_density)})
    return {"method": "Gibbs sampler for all-pairs diversity and adjacent sequence factors",
            "horizon_seconds": intent.horizon_seconds, "item_count": size,
            "nominal_mean_duration_seconds": float(duration), "sweeps": sweeps,
            "songs": [eligible[int(i)]["song_id"] for i in state],
            "duration_seconds": float(sum(eligible[int(i)]["duration_seconds"] for i in state)),
            **({"trace": trace} if record_every else {}),
            "convergence": "Unverified; finite Gibbs run"}


def explain_song(features: np.ndarray, listener: Listener, intent: Intent, prior: FeaturePrior,
                 centers: np.ndarray, weights: np.ndarray, *, seed: int = 0, permutations: int = 128) -> dict:
    """Equation 49 and conditional Shapley values using cached coalitions.

    Conditional values integrate taste posterior draws and feature uncertainty.
    Every coalition is enumerated; conditional expectations are Monte Carlo
    estimates. Cached values guarantee efficiency independently of MC error.
    """
    dimension = len(listener.feature_names)
    if features.shape != (centers.shape[0], dimension) or prior.dimension != dimension or permutations <= 0:
        raise ValueError("coalition attribution requires matching dimensions and positive sampling")
    rng = np.random.default_rng(seed)
    full = (1 << dimension) - 1
    values = {}
    orders = [rng.permutation(dimension) for _ in range(permutations)] if dimension > 12 else None
    coalitions = set(range(full + 1)) if orders is None else {0, full, *(full ^ (1 << j) for j in range(dimension))}
    if orders is not None:
        for order in orders:
            mask = 0
            for j in order:
                mask |= 1 << int(j)
                coalitions.add(mask)
    for coalition in sorted(coalitions):
        retained = [i for i in range(dimension) if coalition & (1 << i)]
        samples = prior.sample_conditioned(rng, features, retained)
        values[coalition] = float(listener.utility(samples, intent, centers=centers, weights=weights).mean())
    shapley = np.zeros(dimension)
    counterfactual = np.zeros(dimension)
    for feature in range(dimension):
        bit = 1 << feature
        counterfactual[feature] = values[full] - values[full ^ bit]
        for coalition in (range(full + 1) if orders is None else []):
            if not coalition & bit:
                size = coalition.bit_count()
                weight = 1 / (dimension * comb(dimension - 1, size))
                shapley[feature] += weight * (values[coalition | bit] - values[coalition])
    if orders is not None:
        for order in orders:
            mask = 0
            for feature in order:
                next_mask = mask | (1 << int(feature))
                shapley[feature] += (values[next_mask] - values[mask]) / len(orders)
                mask = next_mask
    difference = values[full] - values[0]
    if not np.isclose(shapley.sum(), difference, atol=1e-9):
        raise RuntimeError("conditional Shapley efficiency failed")
    return {"candidate_expected_utility": values[full], "reference_expected_utility": values[0],
            "utility_difference": difference, "shapley_total": float(shapley.sum()),
            "features": {name: {"conditional_shapley": float(shapley[i]),
                                "conditional_replacement_effect": float(counterfactual[i])}
                         for i, name in enumerate(listener.feature_names)},
            "method": "all-coalition conditional Shapley with Monte Carlo conditional expectations" if orders is None else "permutation conditional Shapley with cached Monte Carlo expectations",
            "coalitions": len(values), "permutations": 0 if orders is None else len(orders),
            "interpretation": "predictive stated reasons; no causal intervention claim"}
