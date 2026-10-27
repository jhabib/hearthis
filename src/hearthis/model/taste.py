"""Intent, evolving taste states, energy, and posterior preference inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .probability import GaussianPosterior, array, probabilities


@dataclass
class Intent:
    request: str = ""
    targets: dict[str, float] = field(default_factory=dict)
    context_offsets: dict[str, float] = field(default_factory=dict)
    interactions: dict[str, float] = field(default_factory=dict)
    horizon_seconds: float = 600.0
    risk: float = 0.0
    beta: float = 4.0
    excluded_artists: list[str] = field(default_factory=list)
    minimum_features: dict[str, float] = field(default_factory=dict)
    maximum_features: dict[str, float] = field(default_factory=dict)

    def validate(self, names: tuple[str, ...]) -> None:
        for mapping in (self.targets, self.context_offsets, self.interactions,
                        self.minimum_features, self.maximum_features):
            if not set(mapping) <= set(names) or not np.isfinite(list(mapping.values())).all():
                raise ValueError("intent references unknown or nonfinite feature coordinates")
        for mapping in (self.targets, self.minimum_features, self.maximum_features):
            if any(not 0 <= value <= 1 for value in mapping.values()):
                raise ValueError("intent targets and bounds must be in the anchored unit cube")
        if (not np.isfinite([self.horizon_seconds, self.risk, self.beta]).all()
                or self.horizon_seconds <= 0 or self.beta <= 0):
            raise ValueError("intent requires finite risk and positive finite horizon and beta")
        for name in set(self.minimum_features) & set(self.maximum_features):
            if self.minimum_features[name] > self.maximum_features[name]:
                raise ValueError("intent has contradictory feasibility bounds")

    def map_music(self, features: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
        offsets = np.array([self.context_offsets.get(name, 0.0) for name in names])
        interaction = np.array([self.interactions.get(name, 0.0) * self.targets.get(name, 0.5)
                                for name in names])
        return np.clip(features + offsets + features * interaction, 0, 1)

    def feasible(self, features: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
        result = np.ones(features.shape[:-1], dtype=bool)
        for index, name in enumerate(names):
            result &= features[..., index] >= self.minimum_features.get(name, 0.0)
            result &= features[..., index] <= self.maximum_features.get(name, 1.0)
        return result


def parse_request(text: str, names: tuple[str, ...]) -> dict[str, float]:
    """Declared local grounding grammar: 'feature=0.8; other_feature=0.2'.

    Unsupported natural language is rejected rather than assigned invented
    semantic coordinates. Audio-provider reviews can supply richer groundings.
    """
    if not text.strip():
        return {}
    result = {}
    for clause in text.split(";"):
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9_.]*)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*", clause)
        if not match or match[1] not in names or not 0 <= float(match[2]) <= 1:
            raise ValueError("ungrounded request; use declared feature=value clauses or structured intent targets")
        if match[1] in result:
            raise ValueError("request contains conflicting duplicate feature targets")
        result[match[1]] = float(match[2])
    return result


@dataclass
class TasteComponent:
    name: str
    state: GaussianPosterior  # Z in logit coordinates; sigmoid(Z) is the preferred center.
    weights: GaussianPosterior
    precision: np.ndarray
    interactions: np.ndarray
    activation_bias: float = 0.0
    negative_design: np.ndarray | None = None
    negative_offset: np.ndarray | None = None
    negative_strength: np.ndarray | None = None
    transition_matrix: np.ndarray | None = None
    innovation: np.ndarray | None = None
    transition_covariance: np.ndarray | None = None
    previous_state: GaussianPosterior | None = None
    transition_noise: np.ndarray | None = None

    def __post_init__(self):
        dimension = self.state.mean.size
        self.precision = array(self.precision, ndim=1)
        self.interactions = array(self.interactions, ndim=2)
        if (self.weights.mean.size != dimension or self.precision.shape != (dimension,)
                or np.any(self.precision < 0) or self.interactions.shape != (dimension, dimension)
                or not np.allclose(self.interactions, self.interactions.T)
                or not np.allclose(np.diag(self.interactions), 0)
                or not np.isfinite(self.activation_bias)):
            raise ValueError("taste requires matching dimensions, PSD diagonal range precision and symmetric zero-diagonal interactions")
        self.negative_design = (np.empty((0, dimension)) if self.negative_design is None
                                else array(self.negative_design, ndim=2))
        count = self.negative_design.shape[0]
        self.negative_offset = (np.zeros(count) if self.negative_offset is None
                                else array(self.negative_offset, ndim=1))
        self.negative_strength = (np.zeros(count) if self.negative_strength is None
                                  else array(self.negative_strength, ndim=1))
        if (self.negative_design.shape[1] != dimension or self.negative_offset.shape != (count,)
                or self.negative_strength.shape != (count,) or np.any(self.negative_strength < 0)):
            raise ValueError("negative regions require matching design, offset and nonnegative strengths")
        self.transition_matrix = (np.eye(dimension) if self.transition_matrix is None
                                  else array(self.transition_matrix, ndim=2))
        self.innovation = (np.eye(dimension) * 0.002 if self.innovation is None
                           else array(self.innovation, ndim=2))
        # Verify transition and innovation without changing state.
        self.state.transition(self.transition_matrix, self.innovation)
        if self.transition_covariance is not None:
            from .probability import covariance
            self.transition_covariance = covariance(self.transition_covariance, dimension * dimension)
        if self.transition_noise is not None:
            from .probability import covariance
            self.transition_noise = covariance(self.transition_noise, dimension)
        if self.scale(self.weights.mean) <= 0:
            raise ValueError("taste utility scale must be positive")

    def scale(self, weights: np.ndarray) -> np.ndarray:
        """The paper's utility gauge, applied to every posterior draw."""
        return np.sqrt(np.sum(weights ** 2, axis=-1) + np.mean(self.precision) ** 2
                       + np.sum(self.interactions ** 2))

    def energy(self, features: np.ndarray, center=None, weights=None) -> np.ndarray:
        features = array(features)
        if features.shape[-1] != self.state.mean.size or np.any((features < 0) | (features > 1)):
            raise ValueError("augmented features must lie in the anchored unit cube")
        center = expit(self.state.mean) if center is None else center
        weights = self.weights.mean if weights is None else weights
        scale = self.scale(weights)
        if np.any(scale <= 0):
            raise ValueError("zero utility scale")
        linear = np.sum(features * weights, axis=-1)
        range_penalty = 0.5 * np.sum((features - center) ** 2 * self.precision, axis=-1)
        interactions = 0.5 * np.sum(features * (features @ self.interactions), axis=-1)
        negatives = np.logaddexp(0, features @ self.negative_design.T + self.negative_offset)
        negatives = np.sum(negatives * self.negative_strength, axis=-1)
        return (linear - range_penalty + interactions - negatives) / scale

    def to_dict(self) -> dict:
        return {"name": self.name, "state": self.state.to_dict(), "weights": self.weights.to_dict(),
                "precision": self.precision.tolist(), "interactions": self.interactions.tolist(),
                "activation_bias": self.activation_bias,
                "negative_design": self.negative_design.tolist(),
                "negative_offset": self.negative_offset.tolist(),
                "negative_strength": self.negative_strength.tolist(),
                "transition_matrix": self.transition_matrix.tolist(), "innovation": self.innovation.tolist(),
                "transition_covariance": None if self.transition_covariance is None else self.transition_covariance.tolist(),
                "previous_state": None if self.previous_state is None else self.previous_state.to_dict(),
                "transition_noise": None if self.transition_noise is None else self.transition_noise.tolist()}


@dataclass
class Listener:
    listener_id: str
    feature_names: tuple[str, ...]
    components: list[TasteComponent]
    step: int = 0
    change_probability: float = 0.02
    change_variance_multiplier: float = 10.0
    activation_precision: float = 8.0
    provenance: str = "declared prior"
    observed_event_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.feature_names = tuple(self.feature_names)
        if (not self.listener_id or not self.components or len(set(self.feature_names)) != len(self.feature_names)
                or any(c.state.mean.size != len(self.feature_names) for c in self.components)
                or len({c.name for c in self.components}) != len(self.components)
                or not 0 <= self.change_probability <= 1
                or not np.isfinite([self.change_variance_multiplier, self.activation_precision]).all()
                or self.change_variance_multiplier < 1 or self.activation_precision < 0):
            raise ValueError("invalid listener components, dimensions or dynamics")
        if (any(not isinstance(identifier, str) or not identifier for identifier in self.observed_event_ids)
                or len(set(self.observed_event_ids)) != len(self.observed_event_ids)):
            raise ValueError("listener event ledger requires distinct nonempty IDs")

    @classmethod
    def from_centers(cls, listener_id: str, names, centers, *, variance=0.05, provenance="declared prior"):
        centers = array(centers, ndim=2)
        if np.any((centers <= 0) | (centers >= 1)) or variance <= 0:
            raise ValueError("listener centers must be strictly inside the unit cube")
        dimension = centers.shape[1]
        components = [TasteComponent(f"taste-{index}",
                                     GaussianPosterior(logit(center), np.eye(dimension) * variance),
                                     GaussianPosterior(np.zeros(dimension), np.eye(dimension) * variance),
                                     np.ones(dimension), np.zeros((dimension, dimension)))
                      for index, center in enumerate(centers)]
        return cls(listener_id, tuple(names), components, provenance=provenance)

    @classmethod
    def from_dict(cls, data: dict) -> "Listener":
        components = []
        for record in data["components"]:
            record = dict(record)
            record["state"] = GaussianPosterior(**record["state"])
            record["weights"] = GaussianPosterior(**record["weights"])
            if record.get("previous_state") is not None:
                record["previous_state"] = GaussianPosterior(**record["previous_state"])
            # JSON has no way to preserve the shape of an empty design matrix.
            if record.get("negative_design") == []:
                record["negative_design"] = None
            components.append(TasteComponent(**record))
        return cls(**{**data, "components": components})

    def to_dict(self) -> dict:
        return {"listener_id": self.listener_id, "feature_names": list(self.feature_names),
                "components": [c.to_dict() for c in self.components], "step": self.step,
                "change_probability": self.change_probability,
                "change_variance_multiplier": self.change_variance_multiplier,
                "activation_precision": self.activation_precision, "provenance": self.provenance,
                **({"observed_event_ids": self.observed_event_ids} if self.observed_event_ids else {})}

    def activation(self, intent: Intent, centers: np.ndarray) -> np.ndarray:
        intent.validate(self.feature_names)
        logits = np.broadcast_to([c.activation_bias for c in self.components], centers.shape[:-1]).copy()
        indices = [self.feature_names.index(name) for name in intent.targets]
        if indices:
            target = np.array(list(intent.targets.values()))
            logits -= self.activation_precision * np.mean((centers[..., indices] - target) ** 2, axis=-1)
        return probabilities(logits)

    def draws(self, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
        if count <= 0:
            raise ValueError("posterior draw count must be positive")
        centers = np.stack([expit(c.state.sample(rng, count)) for c in self.components], axis=1)
        weights = np.stack([c.weights.sample(rng, count) for c in self.components], axis=1)
        return centers, weights

    def utility(self, features: np.ndarray, intent: Intent, *, centers=None, weights=None) -> np.ndarray:
        """Equations 20 to 22, sharing posterior state draws across candidates."""
        if centers is None:
            centers = np.stack([expit(c.state.mean) for c in self.components])
        if weights is None:
            weights = np.stack([c.weights.mean for c in self.components])
        activation = self.activation(intent, centers)
        component_utilities = np.stack([component.energy(features, centers[..., k, :], weights[..., k, :])
                                       for k, component in enumerate(self.components)], axis=-1)
        return np.sum(activation * component_utilities, axis=-1)

    def condition_prompt(self, targets: dict[str, float], *, confidence: float = 1.0,
                         noise_variance: float = 0.25) -> None:
        if not targets:
            return
        if not 0 < confidence <= 1 or not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("prompt confidence and observation noise must be positive")
        design = np.zeros((len(targets), len(self.feature_names)))
        observations = []
        for row, (name, target) in enumerate(targets.items()):
            if name not in self.feature_names or not np.isfinite(target) or not 0 <= target <= 1:
                raise ValueError("invalid grounded prompt coordinate")
            design[row, self.feature_names.index(name)] = 1
            observations.append(2 * target - 1)
        noise = np.eye(len(targets)) * noise_variance / confidence
        for component in self.components:
            component.weights = component.weights.update(design, observations, noise)

    def advance(self, rng: np.random.Generator, *, change_point: bool | None = None) -> bool:
        changed = bool(rng.random() < self.change_probability) if change_point is None else change_point
        multiplier = self.change_variance_multiplier if changed else 1.0
        for component in self.components:
            component.previous_state = component.state
            component.transition_noise = component.innovation * multiplier
            predicted = component.state.transition(component.transition_matrix, component.transition_noise)
            if component.transition_covariance is not None:
                d = component.state.mean.size
                second_moment = component.state.covariance + np.outer(component.state.mean, component.state.mean)
                additional = np.einsum("abcd,bd->ac", component.transition_covariance.reshape(d, d, d, d), second_moment)
                predicted = GaussianPosterior(predicted.mean, predicted.covariance + additional)
            component.state = predicted
        self.step += 1
        return changed


def fit_feedback(listener: Listener, events: list[dict], song_features: dict[str, np.ndarray],
                 intent: Intent, *, seed: int = 0, draws: int = 64,
                 iterations: int = 100, propensity_weighted: bool = False) -> dict:
    """Structured Gaussian variational inference for pairwise and behavior data.

    The encoder is conditioned on the measured/response feature posterior means.
    Full correlated Gaussian priors are whitened, then a diagonal q is fitted in
    whitened coordinates with a reparameterized Monte Carlo ELBO (equation 34).
    Exposure likelihood is explicit; unexposed items never create a behavior
    likelihood. Optional inverse-propensity weighting is a generalized ELBO.
    """
    intent.validate(listener.feature_names)
    rng = np.random.default_rng(seed)
    priors = [posterior for component in listener.components for posterior in (component.state, component.weights)]
    dimension = len(listener.feature_names)
    means = np.concatenate([p.mean for p in priors])
    cholesky = [np.linalg.cholesky(p.covariance) for p in priors]
    noise = rng.standard_normal((draws, means.size))
    observations, exposure_log_likelihood = [], 0.0
    for event in events:
        exposed = event.get("exposed")
        propensity = float(event.get("propensity", 0))
        if type(exposed) is not bool or not 0 < propensity <= 1:
            raise ValueError("feedback requires explicit exposure and propensity in (0,1]")
        exposure_log_likelihood += np.log(propensity if exposed else 1 - propensity) if (exposed or propensity < 1) else -np.inf
        if not exposed:
            if "value" in event:
                raise ValueError("an unexposed song cannot have observed behavior")
            continue
        kind = event["kind"]
        if kind not in ("pairwise", "behavior") or event["song_id"] not in song_features:
            raise ValueError("unknown feedback kind or song")
        value = float(event["value"])
        if value not in (0.0, 1.0):
            raise ValueError("Bernoulli feedback must be zero or one")
        if kind == "pairwise" and event["other_song_id"] not in song_features:
            raise ValueError("unknown pairwise comparison song")
        confidence = float(event.get("confidence", 1))
        if not 0 < confidence <= 1:
            raise ValueError("feedback confidence must be in (0,1]")
        observations.append((event, value, confidence / propensity if propensity_weighted else confidence))
    if not np.isfinite(exposure_log_likelihood):
        raise ValueError("exposure record has zero probability under its declared policy")
    if not observations:
        return {"updated": False, "observations": 0, "unexposed": len(events),
                "exposure_log_likelihood": float(exposure_log_likelihood)}

    def unpack(parameters):
        locations, scales = parameters[:means.size], np.exp(parameters[means.size:])
        white = locations + scales * noise
        values = np.concatenate([white[:, i * dimension:(i + 1) * dimension] @ factor.T + prior.mean
                                 for i, (factor, prior) in enumerate(zip(cholesky, priors))], axis=1)
        centers = np.stack([expit(values[:, 2 * k * dimension:(2 * k + 1) * dimension])
                            for k in range(len(listener.components))], axis=1)
        weights = np.stack([values[:, (2 * k + 1) * dimension:(2 * k + 2) * dimension]
                            for k in range(len(listener.components))], axis=1)
        return locations, scales, centers, weights

    def objective(parameters):
        locations, scales, centers, weights = unpack(parameters)
        likelihood = np.zeros(draws)
        for event, value, strength in observations:
            feature = np.broadcast_to(song_features[event["song_id"]], (draws, dimension))
            utility = listener.utility(feature, intent, centers=centers, weights=weights)
            if event["kind"] == "pairwise":
                other = np.broadcast_to(song_features[event["other_song_id"]], (draws, dimension))
                logits = utility - listener.utility(other, intent, centers=centers, weights=weights)
            else:
                response_weights = array(event.get("response_weights", np.zeros(dimension)), ndim=1)
                if response_weights.shape != (dimension,):
                    raise ValueError("behavior response weights must match augmented features")
                logits = intent.beta * utility + feature @ response_weights + float(event.get("session_bias", 0))
            likelihood += strength * (value * logits - np.logaddexp(0, logits))
        kl = 0.5 * np.sum(locations ** 2 + scales ** 2 - 1 - 2 * np.log(scales))
        return float(kl - likelihood.mean() - exposure_log_likelihood)

    initial = np.zeros(2 * means.size)
    initial_loss = objective(initial)
    result = minimize(objective, initial, method="L-BFGS-B",
                      bounds=[(-10, 10)] * means.size + [(-5, 2)] * means.size,
                      options={"maxiter": iterations, "ftol": 1e-9})
    if not np.isfinite(result.fun) or result.fun > initial_loss + 1e-7:
        raise RuntimeError("variational inference failed to improve a finite objective")
    locations, scales, _, _ = unpack(result.x)
    for index, (prior, factor) in enumerate(zip(priors, cholesky)):
        subset = slice(index * dimension, (index + 1) * dimension)
        prior.mean = prior.mean + factor @ locations[subset]
        prior.covariance = factor @ np.diag(scales[subset] ** 2) @ factor.T
    return {"updated": True, "observations": len(observations),
            "unexposed": len(events) - len(observations), "iterations": int(result.nit),
            "converged": bool(result.success), "initial_elbo": -initial_loss,
            "final_elbo": -float(result.fun), "exposure_log_likelihood": float(exposure_log_likelihood),
            "objective": "inverse-propensity generalized ELBO" if propensity_weighted else "Monte Carlo ELBO"}
