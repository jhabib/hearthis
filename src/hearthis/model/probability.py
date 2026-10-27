"""Probability primitives for the paper's conjugate and sampling programs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import truncnorm


def array(value, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or (ndim is not None and result.ndim != ndim):
        raise ValueError("model arrays must be finite and have the declared dimensions")
    return result


def probabilities(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("invalid log weights")
    normalizer = logsumexp(values, axis=-1, keepdims=True)
    if not np.isfinite(normalizer).all():
        raise ValueError("distribution has no positive feasible mass")
    return np.exp(values - normalizer)


def covariance(value, dimension: int) -> np.ndarray:
    result = array(value, ndim=2)
    if result.shape != (dimension, dimension) or not np.allclose(result, result.T, atol=1e-10):
        raise ValueError("covariance must be square and symmetric")
    if np.linalg.eigvalsh(result).min() <= 0:
        raise ValueError("covariance must be positive definite")
    return result


@dataclass
class GaussianPosterior:
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        self.mean = array(self.mean, ndim=1)
        self.covariance = covariance(self.covariance, self.mean.size)

    def update(self, design, observations, noise) -> "GaussianPosterior":
        """Exact linear Gaussian conditioning, paper equations 36 to 39."""
        design, observations = array(design, ndim=2), array(observations, ndim=1)
        if design.shape != (observations.size, self.mean.size):
            raise ValueError("prompt design dimensions do not match observations and weights")
        noise = covariance(noise, observations.size)
        prior_precision = np.linalg.solve(self.covariance, np.eye(self.mean.size))
        weighted_design = np.linalg.solve(noise, design)
        precision = prior_precision + design.T @ weighted_design
        posterior_covariance = np.linalg.solve(precision, np.eye(self.mean.size))
        natural_mean = prior_precision @ self.mean + design.T @ np.linalg.solve(noise, observations)
        return GaussianPosterior(np.linalg.solve(precision, natural_mean),
                                 (posterior_covariance + posterior_covariance.T) / 2)

    def transition(self, matrix, innovation) -> "GaussianPosterior":
        """Gaussian state dynamics, equations 18 and 19, for a declared A."""
        matrix = array(matrix, ndim=2)
        if matrix.shape != self.covariance.shape:
            raise ValueError("state transition dimensions do not match")
        innovation = covariance(innovation, self.mean.size)
        return GaussianPosterior(matrix @ self.mean,
                                 matrix @ self.covariance @ matrix.T + innovation)

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.covariance, size=count)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "covariance": self.covariance.tolist()}


@dataclass
class FeaturePrior:
    """Mixture of diagonal truncated Gaussians on the anchored unit cube.

    Conditional component weights include the likelihood of retained coordinates.
    This supplies a proper plausible-feature base and on-support imputations.
    """

    means: np.ndarray
    standard_deviation: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self):
        self.means = array(self.means, ndim=2)
        self.standard_deviation = array(self.standard_deviation, ndim=2)
        if (not self.means.size or self.standard_deviation.shape != self.means.shape
                or np.any(self.standard_deviation <= 0) or np.any((self.means < 0) | (self.means > 1))):
            raise ValueError("feature prior requires bounded means and positive matching standard deviations")
        self.weights = (np.full(self.means.shape[0], 1 / self.means.shape[0])
                        if self.weights is None else array(self.weights, ndim=1))
        if self.weights.shape != (self.means.shape[0],) or np.any(self.weights <= 0):
            raise ValueError("feature prior mixture weights must be positive")
        self.weights = self.weights / self.weights.sum()

    @property
    def dimension(self) -> int:
        return self.means.shape[1]

    def sample(self, rng: np.random.Generator, count: int,
               retained: dict[int, float] | None = None) -> np.ndarray:
        if count <= 0:
            raise ValueError("sample count must be positive")
        retained = retained or {}
        template = np.zeros((count, self.dimension))
        indices = []
        for index, value in retained.items():
            if not 0 <= index < self.dimension or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("retained feature is outside the anchored image")
            template[:, index] = value
            indices.append(index)
        return self.sample_conditioned(rng, template, indices)

    def sample_conditioned(self, rng: np.random.Generator, candidates: np.ndarray,
                           retained_indices: list[int]) -> np.ndarray:
        candidates = array(candidates, ndim=2)
        if (candidates.shape[1] != self.dimension or candidates.shape[0] == 0
                or any(not 0 <= i < self.dimension for i in retained_indices)
                or np.any((candidates < 0) | (candidates > 1))):
            raise ValueError("invalid conditioned feature candidates")
        count = candidates.shape[0]
        log_weights = np.broadcast_to(np.log(self.weights), (count, self.weights.size)).copy()
        for index in retained_indices:
            mean, sd = self.means[:, index], self.standard_deviation[:, index]
            log_weights += truncnorm.logpdf(candidates[:, index, None], -mean / sd, (1 - mean) / sd,
                                            loc=mean, scale=sd)
        distribution = probabilities(log_weights)
        uniforms = rng.random(count)
        components = np.sum(uniforms[:, None] > distribution.cumsum(axis=-1), axis=-1)
        components = np.minimum(components, self.means.shape[0] - 1)
        means, sd = self.means[components], self.standard_deviation[components]
        result = truncnorm.rvs(-means / sd, (1 - means) / sd, loc=means, scale=sd,
                               random_state=rng)
        result[:, retained_indices] = candidates[:, retained_indices]
        return result

    def log_density(self, features) -> np.ndarray:
        features = array(features, ndim=2)
        if features.shape[1] != self.dimension:
            raise ValueError("prior feature dimension mismatch")
        means, sd = self.means, self.standard_deviation
        densities = truncnorm.logpdf(features[:, None, :], -means / sd, (1 - means) / sd,
                                      loc=means, scale=sd).sum(axis=-1)
        return logsumexp(densities + np.log(self.weights), axis=-1)

    def to_dict(self) -> dict:
        return {"means": self.means.tolist(), "standard_deviation": self.standard_deviation.tolist(),
                "weights": self.weights.tolist()}
