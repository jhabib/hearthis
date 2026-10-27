"""Joint differentiable inference over music, response profiles and tastes.

A low-rank Gaussian variational family couples all continuous blocks. Finite
segmentation and critique-grounding alternatives are enumerated; response mode
assignments and response noise are integrated for the augmented feature map.
The ELBO includes normalized waveform, profile and observation likelihoods.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import digamma, logit, polygamma

from .probability import GaussianPosterior
from .waveform import WaveformModel, torch_modules, complete_window_data


def segmentation_options(song):
    """Same finite region topology with three complete boundary hypotheses."""
    base = np.array([[r.start_seconds, r.end_seconds] for r in song.regions])
    sections = [i for i, r in enumerate(song.regions) if r.kind == "section"]
    if len(sections) < 2:
        return base[None]
    delta = min(1.0, min(base[i, 1] - base[i, 0] for i in sections) * 0.2)
    options = [base]
    for shift in (delta, -delta):
        intervals = base.copy()
        boundaries = np.r_[0, [base[i, 1] for i in sections[:-1]], song.duration_seconds]
        boundaries[1:-1] += shift
        for j, index in enumerate(sections):
            intervals[index] = boundaries[j:j + 2]
        for i, region in enumerate(song.regions):
            if region.kind != "passage":
                continue
            section = next(j for j in sections if base[j, 0] <= base[i, 0] + 1e-6 and base[i, 1] <= base[j, 1] + 1e-6)
            fraction = (base[i] - base[section, 0]) / np.diff(base[section])[0]
            intervals[i] = intervals[section, 0] + fraction * np.diff(intervals[section])[0]
        options.append(intervals)
    return np.asarray(options)


def topology_options(song):
    """Complete timelines with uncertain section counts on candidate nodes."""
    base = segmentation_options(song)
    sections = [i for i, r in enumerate(song.regions) if r.kind == "section"]
    options = list(base)
    for count in range(1, len(sections)):
        intervals = base[0].copy()
        for group in np.array_split(sections, count):
            intervals[group[0]] = [base[0][group[0], 0], base[0][group[-1], 1]]
            for index in group[1:]:
                intervals[index] = [base[0][index, 0]] * 2
        options.append(intervals)
    if len(song.regions) > 1:
        global_only = base[0].copy()
        for i, region in enumerate(song.regions):
            if region.kind != "global":
                global_only[i] = [region.start_seconds] * 2
        options.append(global_only)
    return np.asarray(options)


class JointModel:
    def __init__(self, songs, shared, listener, waveform_path=None, waveform_observations="training-windows",
                 acoustic_posterior="explicit-latents", variable_topology=False):
        self.torch, _ = torch_modules()
        self.dtype = self.torch.float32
        self.songs, self.shared, self.listener = songs, shared, listener
        self.m, self.q, self.l = len(shared.music_names), len(shared.response_names), len(shared.concentration)
        self.blocks, self.dimension = [], 0
        self.waveform_path = waveform_path
        self.waveform_observations = waveform_observations
        self.acoustic_posterior = acoustic_posterior
        self.variable_topology = variable_topology
        self.acoustic_noise = None
        self.waveform, self.data = None, None
        self.hypotheses = {s.song_id: (topology_options if variable_topology else segmentation_options)(s) for s in songs}
        self.h_prior = {s.song_id: np.r_[0.8, np.full(len(self.hypotheses[s.song_id]) - 1, 0.2 / (len(self.hypotheses[s.song_id]) - 1))]
                        if len(self.hypotheses[s.song_id]) > 1 else np.array([1.0]) for s in songs}
        self.region_blocks, self.taste_blocks, self.dynamics_blocks = {}, [], {}
        for song in songs:
            for region in song.regions:
                music = self.add(region.region_id + "/X", logit(np.clip(region.music, 0.001, 0.999)),
                                 np.diag(region.music_logit_variance) if region.music_logit_variance is not None else np.eye(self.m) * 0.04)
                profile = region.profile
                eta_mean = digamma(profile.descriptor_alpha) - digamma(profile.descriptor_beta)
                eta_variance = polygamma(1, profile.descriptor_alpha) + polygamma(1, profile.descriptor_beta)
                eta = self.add(region.region_id + "/eta", eta_mean.ravel(), np.diag(eta_variance.ravel()))
                if self.l > 1:
                    alpha = profile.concentration
                    mixture = self.add(region.region_id + "/pi", digamma(alpha[:-1]) - digamma(alpha[-1]),
                                       np.diag(polygamma(1, alpha[:-1])) + polygamma(1, alpha[-1]))
                else:
                    mixture = None
                self.region_blocks[region.region_id] = (music, eta, mixture)
        for k, component in enumerate(listener.components):
            self.taste_blocks.append((self.add(component.name + "/Z", component.state.mean, component.state.covariance),
                                      self.add(component.name + "/w", component.weights.mean, component.weights.covariance)))
            if component.transition_covariance is not None and component.previous_state is not None:
                previous = self.add(component.name + "/Z_previous", component.previous_state.mean, component.previous_state.covariance)
                transition = self.add(component.name + "/A", component.transition_matrix.ravel(), component.transition_covariance)
                self.dynamics_blocks[k] = (previous, transition)
        self.shared_block = self.add("shared/descriptor-coefficients", shared.coefficients.ravel(),
                                     np.eye(shared.coefficients.size) * 0.04)
        self.style_block = None
        self.taste_kernels = None
        self.observation_kernel = None
        if any(s.observations for s in songs):
            from .modalities import ObservationKernel
            self.observation_kernel = ObservationKernel(songs, shared.music_names)
        if waveform_path is not None:
            self.waveform = WaveformModel.load(Path(waveform_path))
            manifest = json.loads(Path(waveform_path).read_text())
            with np.load(manifest["artifacts"]["training-data"]["path"], allow_pickle=False) as data:
                self.data = {name: data[name].copy() for name in data.files}
            if waveform_observations == "all-complete-windows":
                self.data = complete_window_data(self.waveform, manifest, songs)
            elif waveform_observations != "training-windows":
                raise ValueError("unknown waveform observation protocol")
            if acoustic_posterior == "explicit-latents":
                latent_size = len(self.data["audio"]) * self.waveform.config["latent_dimension"]
                self.style_block = self.add("waveform/acoustic-latents", np.zeros(latent_size), None)
            elif acoustic_posterior != "amortized":
                raise ValueError("unknown acoustic posterior")

    def tensor(self, value):
        return self.torch.as_tensor(value, dtype=self.dtype)

    def add(self, name, mean, covariance):
        mean = np.asarray(mean)
        factor = None if covariance is None else np.linalg.cholesky(covariance)
        block = {"name": name, "start": self.dimension, "stop": self.dimension + len(mean),
                 "mean": mean.tolist(), "factor": None if factor is None else factor.tolist()}
        self.blocks.append(block)
        self.dimension += len(mean)
        return len(self.blocks) - 1

    def get(self, white, index):
        block = self.blocks[index]
        values = white[:, block["start"]:block["stop"]]
        if block["factor"] is not None:
            values = values @ self.tensor(block["factor"]).T
        return values + self.tensor(block["mean"])

    def parameters(self, white):
        torch = self.torch
        centers = torch.stack([self.get(white, z).sigmoid() for z, w in self.taste_blocks], dim=1)
        weights = torch.stack([self.get(white, w) for z, w in self.taste_blocks], dim=1)
        coefficients = self.get(white, self.shared_block).reshape(-1, self.l, self.q, self.m + 1)
        regions = {}
        for region_id, (music, eta, mixture) in self.region_blocks.items():
            pi = (torch.cat([self.get(white, mixture), torch.zeros((len(white), 1))], dim=-1).softmax(-1)
                  if mixture is not None else torch.ones((len(white), 1)))
            regions[region_id] = (self.get(white, music).sigmoid(), self.get(white, eta).reshape(-1, self.l, self.q).sigmoid(), pi)
        return centers, weights, coefficients, regions

    def activation(self, centers, intent):
        bias = self.tensor([c.activation_bias for c in self.listener.components]) if self.taste_kernels is None else self.taste_kernels["activation_bias"]
        logits = bias.expand(centers.shape[:2])
        indices = [self.listener.feature_names.index(name) for name in intent.targets]
        if indices:
            target = self.tensor(list(intent.targets.values()))
            logits = logits - self.listener.activation_precision * ((centers[:, :, indices] - target) ** 2).mean(-1)
        return logits.softmax(-1)

    def features(self, centers, regions, song, intent, hypothesis=0):
        torch = self.torch
        personal = (self.activation(centers, intent)[:, :, None] * centers).sum(1)
        target = self.tensor([intent.targets.get(name, 0.5) for name in self.shared.response_names])
        intervals = self.hypotheses[song.song_id][hypothesis]
        masses = {kind: sum(intervals[i, 1] - intervals[i, 0] for i, r in enumerate(song.regions) if r.kind == kind)
                  for kind in ("global", "section", "passage")}
        offsets = self.tensor([intent.context_offsets.get(n, 0) for n in self.shared.music_names])
        interactions = self.tensor([intent.interactions.get(n, 0) * intent.targets.get(n, 0.5) for n in self.shared.music_names])
        scores, responses, assignments, regional = [], [], {}, {}
        for i, region in enumerate(song.regions):
            raw_music, eta, pi = regions[region.region_id]
            music = (raw_music + offsets + raw_music * interactions).clamp(0, 1)
            compatible = -8 * ((eta - (personal[:, None, self.m:] + target) / 2) ** 2).mean(-1)
            modes = (pi.clamp_min(1e-12).log() + compatible).softmax(-1)
            perception = (torch.logit(eta.clamp(1e-6, 1 - 1e-6)) + personal[:, None, self.m:] - 0.5 + target - 0.5).sigmoid()
            response = (modes[:, :, None] * perception).sum(1)
            responses.append(response)
            level = {"global": 0.2, "section": 0.3, "passage": 0.5}[region.kind]
            duration = np.diff(intervals[i])[0]
            log_mass = np.log(level * duration / masses[region.kind]) if duration > 0 else -np.inf
            scores.append(log_mass
                          - 4 * ((music - personal[:, :self.m]) ** 2).mean(-1))
            assignments[region.region_id] = modes
            regional[region.region_id] = torch.cat([music, response], dim=-1)
        attention = torch.stack(scores, dim=-1).softmax(-1)
        aggregated = (attention[:, :, None] * torch.stack(responses, dim=1)).sum(1)
        global_music = regional[song.global_region.region_id][:, :self.m]
        return torch.cat([global_music, aggregated], dim=-1), attention, assignments, regional

    def utility(self, features, centers, weights, intent):
        torch = self.torch
        activation = self.activation(centers, intent)
        values = []
        for k, component in enumerate(self.listener.components):
            precision, interaction = self.tensor(component.precision), self.tensor(component.interactions)
            if self.taste_kernels is not None:
                precision = self.taste_kernels["log_precision"][k].clamp(-8, 8).exp()
                raw = self.taste_kernels["interactions"][k]
                interaction = (raw + raw.T) / 2
                interaction = interaction - self.torch.diag(self.torch.diag(interaction))
            w = weights[:, k]
            scale = (w.square().sum(-1) + precision.mean().square() + interaction.square().sum()).sqrt()
            value = (features * w).sum(-1) - 0.5 * ((features - centers[:, k]) ** 2 * precision).sum(-1)
            value += 0.5 * (features * (features @ interaction)).sum(-1)
            if component.negative_design.size:
                negative = torch.nn.functional.softplus(features @ self.tensor(component.negative_design).T + self.tensor(component.negative_offset))
                value -= (negative * self.tensor(component.negative_strength)).sum(-1)
            values.append(value / scale.clamp_min(1e-12))
        return (activation * torch.stack(values, dim=-1)).sum(-1)

    def log_prior(self, white, parameters):
        torch = self.torch
        centers, weights, coefficients, regions = parameters
        gaussian = torch.ones(self.dimension, dtype=torch.bool)
        result = torch.zeros(len(white), dtype=self.dtype)
        ordinary, changed = torch.zeros(len(white), dtype=self.dtype), torch.zeros(len(white), dtype=self.dtype)
        for k, (previous_index, transition_index) in self.dynamics_blocks.items():
            component = self.listener.components[k]
            z_index = self.taste_blocks[k][0]
            dimension = len(self.listener.feature_names)
            previous = self.get(white, previous_index)
            transition = self.get(white, transition_index).reshape(-1, dimension, dimension)
            mean = torch.einsum("nab,nb->na", transition, previous)
            value = self.get(white, z_index)
            ordinary += torch.distributions.MultivariateNormal(mean, covariance_matrix=self.tensor(component.innovation)).log_prob(value)
            changed += torch.distributions.MultivariateNormal(mean, covariance_matrix=self.tensor(component.innovation * self.listener.change_variance_multiplier)).log_prob(value)
            block = self.blocks[z_index]
            result += float(np.linalg.slogdet(np.asarray(block["factor"]))[1])
            gaussian[block["start"]:block["stop"]] = False
        if self.dynamics_blocks:
            probability = self.listener.change_probability
            if probability == 0:
                result += ordinary
            elif probability == 1:
                result += changed
            else:
                # One listener change point is shared across taste components.
                result += torch.logaddexp(ordinary + np.log1p(-probability), changed + np.log(probability))
        for song in self.songs:
            for region in song.regions:
                music_index, eta_index, mixture_index = self.region_blocks[region.region_id]
                music, eta, pi = regions[region.region_id]
                design = torch.cat([torch.ones((len(white), 1)), music], dim=-1)
                shared_mean = torch.einsum("nlqm,nm->nlq", coefficients, design)
                old_design = np.r_[1, region.music]
                offset = logit(np.clip(region.profile.mode_means, 1e-6, 1 - 1e-6)) - np.einsum("lqm,m->lq", self.shared.coefficients, old_design)
                mean = (shared_mean + self.tensor(offset)).sigmoid().clamp(1e-5, 1 - 1e-5)
                strength = self.tensor(region.profile.descriptor_alpha + region.profile.descriptor_beta)
                alpha, beta = mean * strength, (1 - mean) * strength
                raw_eta = self.get(white, eta_index).reshape(-1, self.l, self.q)
                log_eta, log_complement = torch.nn.functional.logsigmoid(raw_eta), torch.nn.functional.logsigmoid(-raw_eta)
                log_beta = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
                # Logistic Jacobian and whitening determinant preserve density.
                result += (alpha * log_eta + beta * log_complement - log_beta).sum((1, 2))
                block = self.blocks[eta_index]
                result += float(np.linalg.slogdet(np.asarray(block["factor"]))[1])
                gaussian[block["start"]:block["stop"]] = False
                if mixture_index is not None:
                    concentration = self.tensor(region.profile.concentration)
                    result += (concentration * pi.clamp_min(1e-30).log()).sum(-1)
                    result += torch.lgamma(concentration.sum()) - torch.lgamma(concentration).sum()
                    block = self.blocks[mixture_index]
                    result += float(np.linalg.slogdet(np.asarray(block["factor"]))[1])
                    gaussian[block["start"]:block["stop"]] = False
        result += -0.5 * (white[:, gaussian].square() + np.log(2 * np.pi)).sum(-1)
        return result

    def log_observations(self, parameters, feedback, intent, h_weights):
        torch = self.torch
        centers, weights, coefficients, regions = parameters
        result = torch.zeros(len(centers), dtype=self.dtype)
        lookup = {s.song_id: s for s in self.songs}
        predictions = {}
        def bernoulli(logits, value, confidence):
            log_probability = -torch.nn.functional.softplus((1 - 2 * value) * logits)
            if confidence == 1:
                return log_probability
            # A reliability mixture is normalized over both outcomes.
            # Raising the Bernoulli likelihood to confidence would not be.
            noise = torch.full_like(log_probability, np.log((1 - confidence) / 2))
            return torch.logaddexp(log_probability + np.log(confidence), noise)
        for song in self.songs:
            predictions[song.song_id] = [self.features(centers, regions, song, intent, h) for h in range(len(self.hypotheses[song.song_id]))]
        for event in feedback:
            exposed, propensity = event.get("exposed"), float(event.get("propensity", 0))
            if type(exposed) is not bool or not 0 < propensity <= 1 or (not exposed and propensity == 1):
                raise ValueError("joint inference requires possible explicit exposure and propensity")
            result += np.log(propensity if exposed else 1 - propensity)
            if not exposed:
                if "value" in event:
                    raise ValueError("an unexposed song cannot have observed behavior")
                continue
            song = lookup[event["song_id"]]
            kind, confidence = event["kind"], float(event.get("confidence", 1))
            if not 0 < confidence <= 1:
                raise ValueError("invalid feedback confidence")
            for h, (features, attention, modes, regional) in enumerate(predictions[song.song_id]):
                mass = h_weights[song.song_id][h]
                if kind in ("pairwise", "behavior"):
                    value = event["value"]
                    if value not in (0, 1):
                        raise ValueError("Bernoulli value must be zero or one")
                    logits = self.utility(features, centers, weights, intent)
                    if kind == "pairwise":
                        other = event["other_song_id"]
                        for other_h, other_prediction in enumerate(predictions[other]):
                            difference = logits - self.utility(other_prediction[0], centers, weights, intent)
                            result += mass * h_weights[other][other_h] * bernoulli(difference, value, confidence)
                        continue
                    response_weights = self.tensor(event.get("response_weights", np.zeros(len(self.listener.feature_names))))
                    if response_weights.shape != (len(self.listener.feature_names),):
                        raise ValueError("invalid behavior response coefficients")
                    logits = intent.beta * logits + features @ response_weights + float(event.get("session_bias", 0))
                    result += mass * bernoulli(logits, value, confidence)
                elif kind in ("critique", "feature"):
                    grounding = event.get("region_probabilities", {event.get("region_id", song.global_region.region_id): 1.0})
                    if not set(grounding) <= regional.keys() or any(p < 0 for p in grounding.values()) or not np.isclose(sum(grounding.values()), 1):
                        raise ValueError("invalid uncertain region grounding")
                    alternatives = []
                    for region_id, probability in grounding.items():
                        if probability == 0:
                            continue
                        terms = torch.zeros(len(centers), dtype=self.dtype)
                        if kind == "feature":
                            j = self.listener.feature_names.index(event["feature"])
                            mean = float(event.get("coefficient", 1)) * regional[region_id][:, j] - float(event.get("offset", 0))
                            sd = float(event.get("noise_sd", 0.1)) / np.sqrt(confidence)
                            if not np.isfinite(sd) or sd <= 0 or not np.isfinite(event["value"]):
                                raise ValueError("invalid feature observation")
                            terms += -0.5 * ((mean - float(event["value"])) / sd).square() - np.log(sd * np.sqrt(2 * np.pi))
                        else:
                            j = self.listener.feature_names.index(event["feature"])
                            if event["polarity"] not in (-1, 1) or not 0 <= event["magnitude"] <= 1:
                                raise ValueError("invalid grounded critique")
                            target = event["polarity"] * event["magnitude"]
                            pseudo = (self.activation(centers, intent) * weights[:, :, j]).sum(1)
                            terms += -0.5 * confidence * (pseudo - target).square() / 0.25 - 0.5 * np.log(2 * np.pi * 0.25 / confidence)
                            for name, value in event.get("descriptor_reports", {}).items():
                                if not np.isfinite(value) or not 0 <= value <= 1:
                                    raise ValueError("invalid descriptor report")
                                j = self.m + self.shared.response_names.index(name)
                                terms += -0.5 * confidence * ((regional[region_id][:, j] - value) / 0.1).square() - np.log(0.1 * np.sqrt(2 * np.pi / confidence))
                        alternatives.append(terms + np.log(probability))
                    result += mass * torch.logsumexp(torch.stack(alternatives, dim=-1), dim=-1)
                else:
                    raise ValueError("unknown joint observation kind")
        return result

    def log_waveform(self, white, parameters, indices, h_weights):
        torch = self.torch
        if self.waveform is None:
            return torch.zeros(len(white))
        centers, weights, coefficients, regions = parameters
        data, config = self.data, self.waveform.config
        styles = (self.get(white, self.style_block).reshape(len(white), len(data["audio"]), config["latent_dimension"])
                  if self.style_block is not None else None)
        observed = self.tensor(data["audio"][indices])
        log_jacobian = float(np.log(data["scale"]).sum())
        result = torch.zeros(len(white))
        for song_index, song in enumerate(self.songs):
            selected = np.flatnonzero(data["song_indices"][indices] == song_index)
            if not len(selected):
                continue
            actual = indices[selected]
            for h, intervals in enumerate(self.hypotheses[song.song_id]):
                combinations = np.zeros((len(actual), len(song.regions)))
                for row, time in enumerate(data["times_seconds"][actual]):
                    for kind, level in (("global", 0.2), ("section", 0.3), ("passage", 0.5)):
                        matches = [i for i, r in enumerate(song.regions) if r.kind == kind and intervals[i, 0] <= time < intervals[i, 1]]
                        if matches:
                            combinations[row, matches] = level / len(matches)
                    combinations[row] /= combinations[row].sum()
                music = torch.einsum("br,nrm->nbm", self.tensor(combinations),
                                     torch.stack([regions[r.region_id][0] for r in song.regions], dim=1))
                if styles is None:
                    waveform = observed[selected].expand(len(white), -1, -1)
                    context = self.tensor(data["context"][actual]).expand(len(white), -1, -1) if "context" in data else None
                    mean, log_sd = self.waveform.model.encode(waveform, music, context)
                    noise = self.acoustic_noise[:, actual]
                    acoustic = mean + log_sd.exp() * noise
                    kl = 0.5 * (mean.square() + (2 * log_sd).exp() - 1 - 2 * log_sd).sum(-1)
                else:
                    acoustic, kl = styles[:, actual], 0
                    context = self.tensor(data["context"][actual]).expand(len(white), -1, -1) if "context" in data else None
                decoded = self.waveform.model.decode(acoustic, music, context)
                log_probability = -0.5 * (((decoded - observed[selected]) / config["observation_sd"]) ** 2
                                          + np.log(2 * np.pi * config["observation_sd"] ** 2)).sum(-1) - log_jacobian
                result += h_weights[song.song_id][h] * (log_probability - kl).sum(-1)
        return result * len(data["train"]) / len(indices)


class JointPosterior:
    def __init__(self, model, location, diagonal, factor, segmentation, report=None):
        self.model = model
        self.location, self.diagonal, self.factor = np.asarray(location), np.asarray(diagonal), np.asarray(factor)
        self.segmentation = {key: (np.asarray(value, dtype=float) / np.asarray(value, dtype=float).sum()).tolist()
                             for key, value in segmentation.items()}
        self.report = report or {}
        d = model.dimension
        if (self.location.shape != (d,) or self.diagonal.shape != (d,) or self.factor.ndim != 2
                or self.factor.shape[0] != d or np.any(self.diagonal <= 0)
                or not all(np.isfinite(a).all() for a in (self.location, self.diagonal, self.factor))):
            raise ValueError("invalid joint posterior dimensions or covariance")
        for block in model.blocks:
            size = block["stop"] - block["start"]
            mean = np.asarray(block["mean"])
            matrix = None if block["factor"] is None else np.asarray(block["factor"])
            if (mean.shape != (size,) or not np.isfinite(mean).all()
                    or (matrix is not None and (matrix.shape != (size, size) or not np.isfinite(matrix).all()
                                                or np.any(np.diag(matrix) <= 0) or not np.allclose(matrix, np.tril(matrix))))):
                raise ValueError("invalid joint coordinate transform")
        for song in model.songs:
            p = np.asarray(segmentation[song.song_id])
            if p.shape != model.h_prior[song.song_id].shape or np.any(p < 0) or not np.isfinite(p).all() or not np.isclose(p.sum(), 1):
                raise ValueError("invalid segmentation posterior")

    def sample(self, rng, count, targets=None):
        samples = self.location + rng.standard_normal((count, len(self.location))) * self.diagonal
        samples += rng.standard_normal((count, self.factor.shape[1])) @ self.factor.T
        if targets:
            # Exact conditional simulation for session pseudo-observations,
            # retaining cross-block covariance instead of refitting marginals.
            rows, values = [], []
            for z, w in self.model.taste_blocks:
                block = self.model.blocks[w]
                for name, target in targets.items():
                    j = self.model.listener.feature_names.index(name)
                    row = np.zeros(self.model.dimension)
                    row[block["start"]:block["stop"]] = np.asarray(block["factor"])[j]
                    rows.append(row)
                    values.append(2 * target - 1 - block["mean"][j])
            design = np.asarray(rows)
            covariance_design = self.diagonal[:, None] ** 2 * design.T + self.factor @ (design @ self.factor).T
            innovation = design @ covariance_design + np.eye(len(rows)) * 0.25
            gain = np.linalg.solve(innovation, covariance_design.T).T
            errors = np.asarray(values) - samples @ design.T - rng.standard_normal((count, len(rows))) * 0.5
            samples += errors @ gain.T
        return samples

    def forecast(self, listener, *, maximum_rank=64):
        """Propagate the joint Gaussian across a taste transition.

        Bilinear A*Z uses a first-order moment approximation. Sparse affine
        propagation keeps cross-block covariance; truncated factor variance is
        returned to the diagonal. This approximation is recorded explicitly.
        """
        d = len(listener.feature_names)
        location, diagonal, factor = self.location.copy(), self.diagonal.copy(), self.factor.copy()
        additional = []
        for k, (z_index, w_index) in enumerate(self.model.taste_blocks):
            z_block = self.model.blocks[z_index]
            z_slice = slice(z_block["start"], z_block["stop"])
            z_factor = np.asarray(z_block["factor"])
            z_mean = np.asarray(z_block["mean"]) + z_factor @ self.location[z_slice]
            component = listener.components[k]
            matrix = component.transition_matrix
            columns = np.arange(z_block["start"], z_block["stop"])
            jacobian_z = np.linalg.solve(z_factor, matrix @ z_factor)
            jacobian_a = None
            previous_slice = None
            if k in self.model.dynamics_blocks:
                previous_index, a_index = self.model.dynamics_blocks[k]
                a_block = self.model.blocks[a_index]
                a_slice = slice(a_block["start"], a_block["stop"])
                a_factor = np.asarray(a_block["factor"])
                matrix = (np.asarray(a_block["mean"]) + a_factor @ self.location[a_slice]).reshape(d, d)
                jacobian_z = np.linalg.solve(z_factor, matrix @ z_factor)
                jacobian_a = np.linalg.solve(z_factor, np.kron(np.eye(d), z_mean[None]) @ a_factor)
                columns = np.r_[columns, np.arange(a_block["start"], a_block["stop"])]
                previous_block = self.model.blocks[previous_index]
                previous_slice = slice(previous_block["start"], previous_block["stop"])
                previous_factor = np.asarray(previous_block["factor"])
                location[previous_slice] = np.linalg.solve(previous_factor, z_mean - previous_block["mean"])
                factor[previous_slice] = np.linalg.solve(previous_factor, z_factor @ self.factor[z_slice])
                diagonal[previous_slice] = 1e-6
            location[z_slice] = np.linalg.solve(z_factor, matrix @ z_mean - z_block["mean"])
            factor[z_slice] = jacobian_z @ self.factor[z_slice]
            if jacobian_a is not None:
                factor[z_slice] += jacobian_a @ self.factor[a_slice]
            extra = np.zeros((self.model.dimension, len(columns)))
            for j, column in enumerate(columns):
                extra[column, j] = self.diagonal[column]
            extra[z_slice] = np.column_stack([jacobian_z, jacobian_a]) if jacobian_a is not None else jacobian_z
            extra[z_slice] *= self.diagonal[columns]
            if previous_slice is not None:
                extra[previous_slice] = 0
                extra[previous_slice, :d] = np.linalg.solve(previous_factor, z_factor) * self.diagonal[z_slice]
            diagonal[columns] = 1e-6
            transformed_noise = np.linalg.solve(z_factor, np.linalg.cholesky(component.transition_noise))
            noise = np.zeros((self.model.dimension, d))
            noise[z_slice] = transformed_noise
            additional.extend([extra, noise])
        factor = np.column_stack([factor, *additional])
        if factor.shape[1] > maximum_rank:
            u, singular, _ = np.linalg.svd(factor, full_matrices=False)
            retained = u[:, :maximum_rank] * singular[:maximum_rank]
            diagonal = np.sqrt(diagonal ** 2 + np.maximum((factor ** 2).sum(-1) - (retained ** 2).sum(-1), 0))
            factor = retained
        self.location, self.diagonal, self.factor = location, np.maximum(diagonal, 0.003), factor
        self.model.listener = listener
        self.report = {**self.report, "forecast": "first-order joint Gaussian dynamics with covariance rank compression"}

    def predict(self, songs, prior, listener, intent, seed, draws):
        torch = self.model.torch
        rng = np.random.default_rng(seed)
        white = self.sample(rng, draws, intent.targets)
        with torch.no_grad():
            centers, weights, coefficients, regions = self.model.parameters(self.model.tensor(white))
            predictions = {}
            summaries = {}
            for song in songs:
                hypotheses = rng.choice(len(self.segmentation[song.song_id]), size=draws, p=self.segmentation[song.song_id])
                features = torch.zeros((draws, len(listener.feature_names)), dtype=centers.dtype)
                attention = torch.zeros((draws, len(song.regions)), dtype=centers.dtype)
                mode_assignments = {r.region_id: torch.zeros((draws, self.model.l), dtype=centers.dtype) for r in song.regions}
                for h in np.unique(hypotheses):
                    f, a, assignments, regional = self.model.features(centers, regions, song, intent, h)
                    selected = hypotheses == h
                    features[selected], attention[selected] = f[selected], a[selected]
                    for region_id in assignments:
                        mode_assignments[region_id][selected] = assignments[region_id][selected]
                predictions[song.song_id] = features.numpy()
                summaries[song.song_id] = {"region_attention": dict(zip([r.region_id for r in song.regions], attention.mean(0).tolist())),
                                          "response_mode_weights": {key: value.mean(0).tolist() for key, value in mode_assignments.items()},
                                          "segmentation_posterior": self.segmentation[song.song_id]}
        return {"centers": centers.numpy(), "weights": weights.numpy(), "features": predictions,
                "summaries": summaries, "shared_coefficients": coefficients.numpy()}

    def project(self, songs, shared, listener, *, targets=None, seed=0, draws=2048):
        """Compatibility summaries; persisted joint samples remain authoritative."""
        torch = self.model.torch
        white = self.sample(np.random.default_rng(seed), draws, targets)
        with torch.no_grad():
            if self.model.taste_kernels is not None:
                for k, component in enumerate(listener.components):
                    component.precision = self.model.taste_kernels["log_precision"][k].clamp(-8, 8).exp().numpy()
                    raw = self.model.taste_kernels["interactions"][k].numpy()
                    component.interactions = (raw + raw.T) / 2
                    np.fill_diagonal(component.interactions, 0)
                    component.activation_bias = float(self.model.taste_kernels["activation_bias"][k])
            parameters = self.model.parameters(self.model.tensor(white))
            regions = parameters[-1]
            for k, (z, w) in enumerate(self.model.taste_blocks):
                for index, attr in ((z, "state"), (w, "weights")):
                    values = self.model.get(self.model.tensor(white), index).numpy()
                    setattr(listener.components[k], attr, GaussianPosterior(values.mean(0), np.cov(values.T) + np.eye(values.shape[1]) * 1e-8))
                if k in self.model.dynamics_blocks:
                    previous_index, transition_index = self.model.dynamics_blocks[k]
                    values = self.model.get(self.model.tensor(white), transition_index).numpy()
                    listener.components[k].transition_matrix = values.mean(0).reshape(len(listener.feature_names), -1)
                    listener.components[k].transition_covariance = np.cov(values.T) + np.eye(values.shape[1]) * 1e-8
            shared.coefficients = parameters[2].mean(0).numpy()
            for song in songs:
                for region in song.regions:
                    music, eta, pi = (value.numpy() for value in regions[region.region_id])
                    region.music = music.mean(0)
                    mean, variance = eta.mean(0), eta.var(0)
                    strength = np.clip(mean * (1 - mean) / np.maximum(variance, 1e-8) - 1, 0.1, 10000)
                    region.profile.descriptor_alpha = np.maximum(mean * strength, 1e-6)
                    region.profile.descriptor_beta = np.maximum((1 - mean) * strength, 1e-6)
                    strength_pi = np.median(pi.mean(0) * (1 - pi.mean(0)) / np.maximum(pi.var(0), 1e-8) - 1)
                    region.profile.concentration = np.maximum(pi.mean(0) * max(0.1, strength_pi), 1e-6)
                    region.profile.provenance = "joint waveform/feature/profile/listener posterior; Gaussian compatibility moment summary"

    def to_dict(self):
        return {"schema": "hearthis.joint-posterior/v1", "blocks": self.model.blocks,
                "location": self.location.tolist(), "diagonal": self.diagonal.tolist(), "factor": self.factor.tolist(),
                "segmentation": self.segmentation, "report": self.report,
                "waveform_observations": self.model.waveform_observations,
                "acoustic_posterior": self.model.acoustic_posterior,
                "variable_topology": self.model.variable_topology,
                "precision_bits": 64 if self.model.dtype == self.model.torch.float64 else 32,
                "observation_kernel": None if self.model.observation_kernel is None else self.model.observation_kernel.to_dict(),
                "taste_kernels": None if self.model.taste_kernels is None else {key: value.detach().tolist() for key, value in self.model.taste_kernels.items()},
                "waveform_path": None if self.model.waveform_path is None else str(Path(self.model.waveform_path).resolve())}

    @classmethod
    def from_dict(cls, data, songs, shared, listener):
        if data["schema"] != "hearthis.joint-posterior/v1":
            raise ValueError("invalid joint posterior schema")
        model = JointModel(songs, shared, listener, data.get("waveform_path"), data.get("waveform_observations", "training-windows"),
                           data.get("acoustic_posterior", "explicit-latents"), data.get("variable_topology", False))
        if [(b["name"], b["start"], b["stop"]) for b in model.blocks] != [(b["name"], b["start"], b["stop"]) for b in data["blocks"]]:
            raise ValueError("joint posterior belongs to a different catalog or listener")
        model.blocks = data["blocks"]
        if data.get("observation_kernel") is not None:
            from .modalities import ObservationKernel
            model.observation_kernel = ObservationKernel(songs, shared.music_names, data["observation_kernel"])
        model.dtype = model.torch.float64 if data.get("precision_bits") == 64 else model.torch.float32
        if data.get("taste_kernels") is not None:
            kernels = data["taste_kernels"]
            k, d = len(listener.components), len(listener.feature_names)
            expected = {"log_precision": (k, d), "interactions": (k, d, d), "activation_bias": (k,)}
            if set(kernels) != set(expected) or any(np.shape(kernels[key]) != shape or not np.isfinite(kernels[key]).all() for key, shape in expected.items()):
                raise ValueError("invalid learned taste kernel parameters")
            model.taste_kernels = model.torch.nn.ParameterDict({key: model.torch.nn.Parameter(model.tensor(value), requires_grad=False)
                                                              for key, value in data["taste_kernels"].items()})
        return cls(model, data["location"], data["diagonal"], data["factor"], data["segmentation"], data.get("report"))


def fit_joint(songs, shared, listener, feedback, intent, *, waveform_path=None, previous=None,
              seed=0, iterations=120, draws=16, rank=6, batch_size=32, learning_rate=0.01,
              train_neural=False, neural_output_dir=None, neural_learning_rate=0.0001, learn_taste_kernels=True):
    """Reparameterized joint ELBO, with exact finite-state expectations."""
    torch, _ = torch_modules()
    if min(iterations, draws, rank, batch_size) <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("invalid joint inference configuration")
    intent.validate(listener.feature_names)
    if train_neural and (previous is not None or waveform_path is None or neural_output_dir is None):
        raise ValueError("joint neural training requires a waveform model, new checkpoint directory and initial joint fit")
    if not np.isfinite(neural_learning_rate) or neural_learning_rate <= 0:
        raise ValueError("invalid joint neural learning rate")
    model = JointModel(songs, shared, listener, waveform_path, "all-complete-windows", "amortized", True)
    if previous is not None:
        model = previous.model
        model.listener = listener
        model.dtype = torch.float64
        if model.taste_kernels is not None:
            model.taste_kernels.double()
            model.taste_kernels.requires_grad_(False)
        if not any(event.get("exposed") is True for event in feedback):
            with torch.no_grad():
                white = model.tensor(previous.location[None])
                model.log_observations(model.parameters(white), feedback, intent,
                                       {key: model.tensor(value) for key, value in previous.segmentation.items()})
            return JointPosterior(model, previous.location.copy(), previous.diagonal.copy(), previous.factor.copy(),
                                  previous.segmentation, {"updated": False, "observations": 0,
                                                          "method": "prior retained; no exposed feedback"})
    d = model.dimension
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    initial_location = np.zeros(d, dtype=np.float32)
    initial_diagonal = np.ones(d, dtype=np.float32)
    initial_factor = rng.normal(0, 0.005, (d, rank)).astype(np.float32)
    if model.waveform is not None and previous is None:
        if model.style_block is not None:
            with torch.no_grad():
                location_style, log_sd = model.waveform.model.encode(model.tensor(model.data["audio"]), model.tensor(model.data["music"]))
            block = model.blocks[model.style_block]
            initial_location[block["start"]:block["stop"]] = location_style.numpy().ravel()
            initial_diagonal[block["start"]:block["stop"]] = log_sd.exp().numpy().ravel()
        else:
            model.acoustic_noise = model.tensor(rng.standard_normal((draws, len(model.data["audio"]), model.waveform.config["latent_dimension"])))
        for parameter in model.waveform.model.parameters():
            parameter.requires_grad_(train_neural)
    location = torch.nn.Parameter(model.tensor(initial_location).clone())
    log_diagonal = torch.nn.Parameter(model.tensor(initial_diagonal).log().clone())
    factor = torch.nn.Parameter(model.tensor(initial_factor).clone())
    factor_scale = None
    if previous is not None:
        # Parameterize the prior's independent innovations and shared factors.
        # This whitens a narrow forecast without building a dense covariance.
        rank = previous.factor.shape[1]
        location = torch.nn.Parameter(model.tensor(np.zeros(d)))
        log_diagonal = torch.nn.Parameter(model.tensor(np.zeros(d)))
        factor = torch.nn.Parameter(model.tensor(np.zeros(rank)))
        factor_scale = torch.nn.Parameter(model.tensor(np.zeros(rank)))
        base_location, base_diagonal, base_factor = (model.tensor(value) for value in
                                                     (previous.location, previous.diagonal, previous.factor))

    def variational():
        if previous is None:
            return location, log_diagonal.clamp(-6, 3).exp(), factor
        return (base_location + base_diagonal * location + base_factor @ factor,
                base_diagonal * log_diagonal.clamp(-6, 3).exp(),
                base_factor * factor_scale.clamp(-6, 3).exp()[None])
    h_logits = {s.song_id: torch.nn.Parameter(model.tensor(previous.segmentation[s.song_id] if previous else model.h_prior[s.song_id]).log()) for s in songs}
    optimized = [location, log_diagonal, factor, *h_logits.values()]
    neural_parameters = list(model.waveform.model.parameters()) if train_neural else []
    initial_neural = [p.detach().clone() for p in neural_parameters]
    observation_parameters = list(model.observation_kernel.model.parameters()) if model.observation_kernel is not None and previous is None else []
    if learn_taste_kernels and previous is None:
        model.taste_kernels = torch.nn.ParameterDict({
            "log_precision": torch.nn.Parameter(model.tensor(np.log(np.maximum(np.array([c.precision for c in listener.components]), 1e-6)))),
            "interactions": torch.nn.Parameter(model.tensor(np.array([c.interactions for c in listener.components]))),
            "activation_bias": torch.nn.Parameter(model.tensor([c.activation_bias for c in listener.components]))})
    kernel_parameters = list(model.taste_kernels.parameters()) if model.taste_kernels is not None and previous is None else []
    initial_kernels = [p.detach().clone() for p in kernel_parameters]
    if factor_scale is not None:
        optimized.append(factor_scale)
    optimizer = (torch.optim.Adam([{"params": optimized}, {"params": observation_parameters},
                                  {"params": neural_parameters, "lr": neural_learning_rate}, {"params": kernel_parameters}], lr=learning_rate) if previous is None
                 else torch.optim.LBFGS(optimized, lr=1, max_iter=1, history_size=10, line_search_fn="strong_wolfe"))
    fixed_noise = model.tensor(rng.standard_normal((draws, d)))
    fixed_rank_noise = model.tensor(rng.standard_normal((draws, rank)))
    train = model.data["train"] if model.data is not None else None
    evaluation_batch = rng.choice(train, min(len(train), max(batch_size, 64)), replace=False) if train is not None else None
    def loss(noise, rank_noise, indices):
        mean, diagonal, covariance_factor = variational()
        white = mean + diagonal * noise + rank_noise @ covariance_factor.T
        parameters = model.parameters(white)
        h_weights = {key: value.softmax(-1) for key, value in h_logits.items()}
        observation_probability = model.log_observations(parameters, feedback, intent, h_weights)
        prior = model.log_prior(white, parameters) if previous is None else None
        log_probability = prior + observation_probability if prior is not None else observation_probability
        if previous is None:
            log_probability += model.log_waveform(white, parameters, indices, h_weights)
            if model.observation_kernel is not None:
                log_probability += model.observation_kernel.log_probability(songs, parameters[-1], shared.music_names)
        h_kl = sum((mass * (mass.clamp_min(1e-30).log() - model.tensor(previous.segmentation[key] if previous else model.h_prior[key]).log())).sum() for key, mass in h_weights.items())
        if previous is not None:
            auxiliary_mean = torch.cat([location, factor])
            auxiliary_log_sd = torch.cat([log_diagonal, factor_scale]).clamp(-6, 3)
            kl = 0.5 * (auxiliary_mean.square() + (2 * auxiliary_log_sd).exp() - 1 - 2 * auxiliary_log_sd).sum()
            return kl - observation_probability.mean() + h_kl
        distribution = torch.distributions.LowRankMultivariateNormal(mean, covariance_factor, diagonal.square())
        regularization = sum(0.5 * ((parameter - initial) / 0.1).square().sum()
                             for parameter, initial in zip(neural_parameters, initial_neural))
        regularization += sum(0.5 * ((parameter - initial) / 0.5).square().sum()
                              for parameter, initial in zip(kernel_parameters, initial_kernels))
        return -log_probability.mean() - distribution.entropy() + h_kl + regularization

    with torch.no_grad():
        initial = float(loss(fixed_noise, fixed_rank_noise, evaluation_batch))
    best, best_state, history, best_neural, best_observation, best_kernels = initial, None, [], None, None, None
    for iteration in range(iterations):
        batch = rng.choice(train, min(batch_size, len(train)), replace=False) if train is not None else None
        if previous is not None:
            def closure():
                optimizer.zero_grad()
                objective = loss(fixed_noise, fixed_rank_noise, evaluation_batch)
                if not torch.isfinite(objective):
                    raise RuntimeError("joint replay encountered a nonfinite ELBO")
                objective.backward()
                return objective
            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            objective = loss(fixed_noise, fixed_rank_noise, batch)
            if not torch.isfinite(objective):
                raise RuntimeError("joint inference encountered a nonfinite ELBO")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(optimized + neural_parameters + observation_parameters + kernel_parameters, 100)
            optimizer.step()
        with torch.no_grad():
            value = float(loss(fixed_noise, fixed_rank_noise, evaluation_batch))
        history.append(value)
        if value < best:
            best = value
            mean, diagonal, covariance_factor = variational()
            best_state = (mean.detach().clone(), diagonal.detach().log().clone(), covariance_factor.detach().clone(),
                          {key: value.detach().softmax(-1).tolist() for key, value in h_logits.items()})
            if train_neural:
                best_neural = {key: value.detach().clone() for key, value in model.waveform.model.state_dict().items()}
            if model.observation_kernel is not None:
                best_observation = model.observation_kernel.to_dict()
            if model.taste_kernels is not None:
                best_kernels = {key: value.detach().clone() for key, value in model.taste_kernels.items()}
    if best_state is None:
        raise RuntimeError(f"joint inference did not improve its finite paired ELBO (initial {initial}, best attempt {min(history)})")
    if best_observation is not None:
        from .modalities import ObservationKernel
        model.observation_kernel = ObservationKernel(songs, shared.music_names, best_observation)
    if best_kernels is not None:
        model.taste_kernels.load_state_dict(best_kernels)
    if train_neural:
        from .waveform import save_joint_waveform
        model.waveform.model.load_state_dict(best_neural)
        model.waveform_path = save_joint_waveform(model.waveform, Path(waveform_path), Path(neural_output_dir),
                                                  {"initial_elbo": -initial, "final_elbo": -best,
                                                   "iterations": iterations, "learning_rate": neural_learning_rate})
    report = {"method": "joint low-rank Gaussian variational ELBO with exact finite segmentation/grounding expectations",
              "initial_elbo": -initial, "final_elbo": -best, "iterations": iterations, "draws": draws,
              "dimension": d, "covariance_rank": rank, "observations": len(feedback),
              "replay_family": "whitened independent innovations and shared Gaussian factors" if previous is not None else None,
              "waveform_likelihood": model.waveform is not None and previous is None,
              "neural_parameters_learned": train_neural,
              "taste_kernels_learned": bool(kernel_parameters),
              "lyric_metadata_likelihood": model.observation_kernel is not None and previous is None,
              "waveform_windows": 0 if model.data is None else len(model.data["train"]),
              "waveform_protocol": "resampled complete nonoverlapping windows; final partial window excluded",
              "acoustic_posterior": model.acoustic_posterior,
              "estimator": "common random numbers for continuous reparameterization; waveform minibatches",
              "nodes": ["X", "H", "Omega", "Z", "w", "shared_response_parameters", "acoustic_latents"] + (["A", "Z_previous"] if model.dynamics_blocks else []),
              "integrated_nodes": ["kappa", "R", "Gamma"] + (["chi"] if model.dynamics_blocks else []), "history_negative_elbo": history,
              "convergence": "Unverified; finite stochastic variational optimization"}
    return JointPosterior(model, best_state[0].numpy(), best_state[1].exp().numpy(), best_state[2].numpy(), best_state[3], report)
