"""Persistent multimodal region profiles and listener-conditioned perceptions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit, logit

from .probability import array, probabilities
from .taste import Intent, Listener


@dataclass
class ResponseProfile:
    concentration: np.ndarray
    descriptor_alpha: np.ndarray
    descriptor_beta: np.ndarray
    provenance: str = "uninformed response prior"

    def __post_init__(self):
        self.concentration = array(self.concentration, ndim=1)
        self.descriptor_alpha = array(self.descriptor_alpha, ndim=2)
        self.descriptor_beta = array(self.descriptor_beta, ndim=2)
        if (self.descriptor_alpha.shape != self.descriptor_beta.shape
                or self.descriptor_alpha.shape[0] != self.concentration.size
                or self.concentration.size == 0 or self.descriptor_alpha.shape[1] == 0
                or np.any(self.concentration <= 0) or np.any(self.descriptor_alpha <= 0)
                or np.any(self.descriptor_beta <= 0)):
            raise ValueError("response profiles require positive compatible Dirichlet and Beta parameters")

    @property
    def mode_means(self) -> np.ndarray:
        return self.descriptor_alpha / (self.descriptor_alpha + self.descriptor_beta)

    def predict(self, rng: np.random.Generator, count: int, listener_center: np.ndarray,
                intent_target: np.ndarray, *, sample_response: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Equations 9 to 11; integrate assignments and R for equation 17.

        Dirichlet weights and Beta descriptor modes remain posterior random
        variables. The preference feature uses E[R | profile, listener, intent],
        integrating categorical assignment and response noise analytically.
        sample_response=True instead runs the generative kappa/R kernels.
        """
        mixture = rng.dirichlet(self.concentration, size=count)
        modes = rng.beta(self.descriptor_alpha, self.descriptor_beta,
                         size=(count, *self.descriptor_alpha.shape))
        center = (listener_center + intent_target) / 2
        compatibility = -8 * np.mean((modes - center[:, None, :]) ** 2, axis=-1)
        assignments = probabilities(np.log(mixture) + compatibility)
        response_mean = expit(logit(np.clip(modes, 1e-8, 1 - 1e-8))
                              + (listener_center[:, None, :] - 0.5)
                              + (intent_target[:, None, :] - 0.5))
        if sample_response:
            selected = np.array([rng.choice(self.concentration.size, p=p) for p in assignments])
            selected_mean = response_mean[np.arange(count), selected]
            response = rng.beta(np.maximum(selected_mean * 20, 1e-6),
                                np.maximum((1 - selected_mean) * 20, 1e-6))
        else:
            response = np.sum(assignments[..., None] * response_mean, axis=1)
        return response, assignments

    def update(self, descriptor_indices: dict[int, float], *, confidence: float,
               strength: float = 4.0, mode_probabilities=None) -> np.ndarray:
        """Soft-assignment fractional Beta/Dirichlet posterior update.

        This is a declared local approximation for bounded descriptor reports,
        not an exact posterior for the paper's general neural language kernel.
        Unknown coordinates receive no pseudo-observation.
        """
        if not 0 < confidence <= 1 or not np.isfinite(strength) or strength <= 0 or not descriptor_indices:
            raise ValueError("response update requires grounded descriptors and positive confidence/strength")
        indices, values = list(descriptor_indices), np.array(list(descriptor_indices.values()))
        if (any(not 0 <= i < self.descriptor_alpha.shape[1] for i in indices)
                or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
            raise ValueError("invalid response descriptor grounding")
        if mode_probabilities is None:
            likelihood = -strength * np.sum((self.mode_means[:, indices] - values) ** 2, axis=-1)
            responsibilities = probabilities(np.log(self.concentration) + likelihood)
        else:
            responsibilities = array(mode_probabilities, ndim=1)
            if responsibilities.shape != self.concentration.shape or np.any(responsibilities < 0) or responsibilities.sum() <= 0:
                raise ValueError("invalid grounded response-mode probabilities")
            responsibilities /= responsibilities.sum()
        counts = confidence * strength * responsibilities
        self.concentration += counts
        self.descriptor_alpha[:, indices] += counts[:, None] * values
        self.descriptor_beta[:, indices] += counts[:, None] * (1 - values)
        self.provenance = "posterior updated by grounded bounded descriptor observations"
        return responsibilities

    def to_dict(self) -> dict:
        return {"concentration": self.concentration.tolist(),
                "descriptor_alpha": self.descriptor_alpha.tolist(),
                "descriptor_beta": self.descriptor_beta.tolist(), "provenance": self.provenance}


@dataclass
class SharedResponsePrior:
    music_names: tuple[str, ...]
    response_names: tuple[str, ...]
    coefficients: np.ndarray  # mode, descriptor, intercept + music coordinates
    concentration: np.ndarray
    descriptor_strength: float = 2.0
    provenance: str = "uninformed shared response prior; not calibrated"

    def __post_init__(self):
        self.music_names, self.response_names = tuple(self.music_names), tuple(self.response_names)
        self.coefficients = array(self.coefficients, ndim=3)
        self.concentration = array(self.concentration, ndim=1)
        if (self.coefficients.shape != (self.concentration.size, len(self.response_names), len(self.music_names) + 1)
                or np.any(self.concentration <= 0) or not np.isfinite(self.descriptor_strength)
                or self.descriptor_strength <= 0):
            raise ValueError("invalid shared response prior")

    @classmethod
    def uninformed(cls, music_names, response_names, modes: int = 2):
        return cls(tuple(music_names), tuple(response_names),
                   np.zeros((modes, len(response_names), len(music_names) + 1)), np.ones(modes))

    def profile(self, music: np.ndarray) -> ResponseProfile:
        design = np.r_[1, array(music, ndim=1)]
        means = np.clip(expit(np.einsum("lqm,m->lq", self.coefficients, design)), 1e-6, 1 - 1e-6)
        return ResponseProfile(self.concentration.copy(), means * self.descriptor_strength,
                               (1 - means) * self.descriptor_strength, self.provenance)

    def fit(self, observations: list[dict], *, regularization: float = 1.0) -> dict:
        """Anchored descriptor supervision fits a regularized logit-mean encoder.

        Review mode IDs must already be aligned across training records. Mode
        order is arbitrary; downstream predictions integrate all modes.
        """
        if not observations or not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("shared learning requires grounded descriptor observations")
        counts = self.concentration.copy()
        for mode in range(self.concentration.size):
            rows = [r for r in observations if r["mode"] == mode]
            if not rows:
                continue
            design = np.array([np.r_[1, array(r["music"], ndim=1)] for r in rows])
            targets = array([r["descriptors"] for r in rows], ndim=2)
            weights = array([r.get("confidence", 1) for r in rows], ndim=1)
            if (targets.shape != (len(rows), len(self.response_names)) or np.any((targets < 0) | (targets > 1))
                    or np.any(weights <= 0) or np.any(weights > 1)):
                raise ValueError("invalid shared descriptor supervision")
            gram = design.T @ (weights[:, None] * design) + np.eye(design.shape[1]) * regularization
            response = design.T @ (weights[:, None] * logit(np.clip(targets, 0.01, 0.99)))
            self.coefficients[mode] = np.linalg.solve(gram, response).T
            counts[mode] += weights.sum()
        self.concentration = counts
        self.provenance = "regularized shared encoder fit to declared descriptor supervision; calibration unverified"
        return {"observations": len(observations), "provenance": self.provenance}

    def to_dict(self) -> dict:
        return {"music_names": list(self.music_names), "response_names": list(self.response_names),
                "coefficients": self.coefficients.tolist(), "concentration": self.concentration.tolist(),
                "descriptor_strength": self.descriptor_strength, "provenance": self.provenance}


@dataclass
class Region:
    region_id: str
    kind: str
    start_seconds: float
    end_seconds: float
    music: np.ndarray
    profile: ResponseProfile
    music_logit_variance: list[float] | None = None

    def __post_init__(self):
        self.music = array(self.music, ndim=1)
        if (not self.region_id or self.kind not in ("global", "section", "passage")
                or not np.isfinite([self.start_seconds, self.end_seconds]).all()
                or not 0 <= self.start_seconds < self.end_seconds or np.any((self.music < 0) | (self.music > 1))):
            raise ValueError("invalid bounded song region")
        if self.music_logit_variance is not None:
            variance = array(self.music_logit_variance, ndim=1)
            if variance.shape != self.music.shape or np.any(variance <= 0):
                raise ValueError("music uncertainty requires positive coordinate variances")
            self.music_logit_variance = variance.tolist()

    def to_dict(self) -> dict:
        return {"region_id": self.region_id, "kind": self.kind, "start_seconds": self.start_seconds,
                "end_seconds": self.end_seconds, "music": self.music.tolist(), "profile": self.profile.to_dict(),
                **({"music_logit_variance": self.music_logit_variance} if self.music_logit_variance is not None else {})}


@dataclass
class Song:
    song_id: str
    artist: str
    title: str
    duration_seconds: float
    regions: list[Region]
    observations: dict = field(default_factory=dict)

    def __post_init__(self):
        if (not self.song_id or not self.regions or not np.isfinite(self.duration_seconds)
                or self.duration_seconds <= 0 or sum(r.kind == "global" for r in self.regions) != 1
                or len({r.region_id for r in self.regions}) != len(self.regions)
                or any(r.end_seconds > self.duration_seconds + 1e-6 for r in self.regions)):
            raise ValueError("song must have unique aligned regions and exactly one global region")

    @property
    def global_region(self) -> Region:
        return next(r for r in self.regions if r.kind == "global")

    def predict(self, prior: SharedResponsePrior, listener: Listener, intent: Intent,
                centers: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        """Region aggregation (12) and augmented posterior features (17)."""
        if listener.feature_names != prior.music_names + prior.response_names:
            raise ValueError("song response prior and listener coordinates do not match")
        count, musical_dimension = centers.shape[0], len(prior.music_names)
        activation = listener.activation(intent, centers)
        listener_center = np.sum(activation[..., None] * centers, axis=1)
        target = np.array([intent.targets.get(name, 0.5) for name in prior.response_names])
        target = np.broadcast_to(target, (count, len(prior.response_names)))
        scores, responses, assignments = [], [], {}
        level_mass = {"global": 0.2, "section": 0.3, "passage": 0.5}
        level_duration = {kind: sum(r.end_seconds - r.start_seconds for r in self.regions if r.kind == kind)
                          for kind in level_mass}
        for region in self.regions:
            music = intent.map_music(region.music, prior.music_names)
            response, modes = region.profile.predict(rng, count, listener_center[:, musical_dimension:], target)
            responses.append(response)
            log_mass = np.log(level_mass[region.kind] * (region.end_seconds - region.start_seconds)
                              / level_duration[region.kind])
            scores.append(log_mass - 4 * np.mean((music - listener_center[:, :musical_dimension]) ** 2, axis=-1))
            assignments[region.region_id] = modes.mean(axis=0).tolist()
        region_weights = probabilities(np.stack(scores, axis=-1))
        response = np.sum(region_weights[..., None] * np.stack(responses, axis=1), axis=1)
        global_music = intent.map_music(self.global_region.music, prior.music_names)
        features = np.column_stack([np.broadcast_to(global_music, (count, musical_dimension)), response])
        return features, {"response_mode_weights": assignments,
                          "region_attention": dict(zip([r.region_id for r in self.regions],
                                                       region_weights.mean(axis=0).tolist()))}

    @classmethod
    def from_dict(cls, data: dict):
        regions = [Region(**{**r, "profile": ResponseProfile(**r["profile"])}) for r in data["regions"]]
        return cls(**{**data, "regions": regions})

    def to_dict(self) -> dict:
        return {"song_id": self.song_id, "artist": self.artist, "title": self.title,
                "duration_seconds": self.duration_seconds, "regions": [r.to_dict() for r in self.regions],
                **({"observations": self.observations} if self.observations else {})}


def condition_behavior_profiles(song: Song, prior: SharedResponsePrior, listener: Listener,
                                intent: Intent, event: dict, *, seed: int = 0, draws: int = 128) -> dict:
    """Local importance/moment projection of Omega under observed behavior.

    Each region is updated conditional on the other regions' predictive mean,
    current taste means, and recorded context. This explicitly approximates the
    structured posterior; it is not claimed to solve the full neural joint ELBO.
    """
    if event.get("exposed") is not True or event["kind"] != "behavior" or event["value"] not in (0, 1):
        raise ValueError("response learning requires exposed Bernoulli behavior")
    confidence = float(event.get("confidence", 1))
    if not 0 < confidence <= 1 or not 0 < float(event["propensity"]) <= 1:
        raise ValueError("invalid behavior confidence or exposure propensity")
    rng = np.random.default_rng(seed)
    center = np.stack([expit(c.state.mean) for c in listener.components])
    centers = np.broadcast_to(center, (draws, *center.shape))
    features, summary = song.predict(prior, listener, intent, centers, rng)
    baseline = features.mean(axis=0)
    m = len(prior.music_names)
    activation = listener.activation(intent, center)
    personal = np.sum(activation[:, None] * center, axis=0)[m:]
    target = np.array([intent.targets.get(name, 0.5) for name in prior.response_names])
    response_weights = array(event.get("response_weights", np.zeros(len(listener.feature_names))), ndim=1)
    if response_weights.shape != (len(listener.feature_names),):
        raise ValueError("behavior response coefficients do not match augmented features")
    effective_sizes = []
    for region in song.regions:
        profile = region.profile
        eta = rng.beta(profile.descriptor_alpha, profile.descriptor_beta,
                       size=(draws, *profile.descriptor_alpha.shape))
        mixture = rng.dirichlet(profile.concentration, size=draws)
        assignment = probabilities(np.log(mixture) - 8 * np.mean((eta - (personal + target) / 2) ** 2, axis=-1))
        perception = expit(logit(np.clip(eta, 1e-8, 1 - 1e-8)) + personal - 0.5 + target - 0.5)
        response = np.sum(assignment[..., None] * perception, axis=1)
        candidates = np.broadcast_to(baseline, (draws, baseline.size)).copy()
        attention = summary["region_attention"][region.region_id]
        candidates[:, m:] = np.clip(baseline[m:] + attention * (response - response.mean(axis=0)), 0, 1)
        logits = intent.beta * listener.utility(candidates, intent) + candidates @ response_weights + float(event.get("session_bias", 0))
        if np.ptp(logits) < 1e-12:
            continue
        importance = probabilities(confidence * (event["value"] * logits - np.logaddexp(0, logits)))
        means = np.sum(importance[:, None, None] * eta, axis=0)
        variances = np.sum(importance[:, None, None] * (eta - means) ** 2, axis=0)
        concentration = np.clip(means * (1 - means) / np.maximum(variances, 1e-8) - 1, 0.1, 10000)
        profile.descriptor_alpha = np.maximum(means * concentration, 1e-6)
        profile.descriptor_beta = np.maximum((1 - means) * concentration, 1e-6)
        profile.concentration += confidence * np.sum(importance[:, None] * assignment, axis=0)
        profile.provenance = "local importance/moment posterior from exposed behavior; calibration unverified"
        effective_sizes.append(float(1 / np.sum(importance ** 2)))
    return {"kind": "behavior response-profile posterior", "song_id": song.song_id,
            "regions_updated": len(effective_sizes),
            "mean_effective_sample_size": float(np.mean(effective_sizes)) if effective_sizes else None,
            "method": "conditional importance weighting and Beta/Dirichlet moment projection"}
