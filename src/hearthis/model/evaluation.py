"""Held-out preference, annotation, waveform and sampling evaluation."""
from __future__ import annotations

from collections import Counter
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logsumexp

from ..audio.ingest import sha256_file
from ..audio.separation import _audio_geometry
from .response import SharedResponsePrior, Song
from .taste import Intent, Listener
from .validation import validate_model_run


def load_run(path):
    validate_model_run(Path(path))
    manifest = json.loads(Path(path).read_text())
    data = {name: json.loads(Path(ref["path"]).read_text()) for name, ref in manifest["artifacts"].items()}
    catalog = data["catalog.json"]
    shared = SharedResponsePrior(**catalog["shared_prior"])
    songs = [Song.from_dict(value) for value in catalog["songs"]]
    listener = Listener.from_dict(data["listener-posterior.json"])
    joint = None
    if "joint-posterior.json" in data:
        from .joint import JointPosterior
        joint = JointPosterior.from_dict(data["joint-posterior.json"], songs, shared, listener)
    return data, songs, shared, listener, joint


def preference_metrics(model_path, events, *, seed=0, draws=1024, inverse_propensity=False):
    from .recommend import rank_catalog
    data, songs, shared, listener, joint = load_run(model_path)
    if draws < 2 or not events:
        raise ValueError("held-out evaluation requires events and at least two draws")
    training = data["feedback.json"]
    training_ids = set(listener.observed_event_ids) | {e["event_id"] for e in training if e.get("event_id")}
    fingerprints = lambda e: json.dumps({key: value for key, value in e.items() if key not in ("event_id", "provenance", "step")}, sort_keys=True)
    seen = set()
    probabilities, outcomes, weights, predictions, unexposed = [], [], [], [], 0
    contexts = {}
    for event in events:
        if not event.get("event_id") or event["event_id"] in seen or event["event_id"] in training_ids:
            raise ValueError("evaluation event IDs must be unique and held out")
        seen.add(event["event_id"])
        if fingerprints(event) in {fingerprints(e) for e in training}:
            raise ValueError("evaluation event duplicates a training observation")
        propensity = event.get("propensity")
        if (type(event.get("exposed")) is not bool or propensity is None
                or not np.isfinite(propensity) or not 0 < propensity <= 1 or (not event["exposed"] and propensity == 1)):
            raise ValueError("evaluation requires possible recorded exposure probabilities")
        if not event["exposed"]:
            if "value" in event:
                raise ValueError("unexposed events cannot supply preference outcomes")
            unexposed += 1
            continue
        if event.get("kind") not in ("pairwise", "behavior") or event.get("value") not in (0, 1):
            raise ValueError("held-out preference outcomes must be Bernoulli")
        intent = Intent(**event.get("intent", {}))
        key = json.dumps(event.get("intent", {}), sort_keys=True)
        if key not in contexts:
            _, posterior = rank_catalog(songs, shared, listener, intent, seed=seed, draws=draws, joint_posterior=joint)
            contexts[key] = posterior
        posterior = contexts[key]
        features = posterior["features"][event["song_id"]]
        utility = listener.utility(features, intent, centers=posterior["centers"], weights=posterior["weights"])
        if event["kind"] == "pairwise":
            other = posterior["features"][event["other_song_id"]]
            logits = utility - listener.utility(other, intent, centers=posterior["centers"], weights=posterior["weights"])
        else:
            response_weights = np.asarray(event.get("response_weights", np.zeros(features.shape[1])))
            if response_weights.shape != (features.shape[1],) or not np.isfinite(response_weights).all():
                raise ValueError("invalid evaluation response coefficients")
            logits = intent.beta * utility + features @ response_weights + event.get("session_bias", 0)
        prediction = float(expit(logits).mean())
        probabilities.append(prediction)
        outcomes.append(event["value"])
        weights.append(1 / propensity if inverse_propensity else 1)
        predictions.append({"event_id": event["event_id"], "probability": prediction, "value": event["value"]})
    if not outcomes:
        raise ValueError("evaluation has no exposed outcomes")
    p, y, w = np.clip(probabilities, 1e-12, 1 - 1e-12), np.asarray(outcomes), np.asarray(weights)
    bins = []
    for i in range(10):
        mask = (p >= i / 10) & (p < (i + 1) / 10)
        if mask.any():
            bins.append({"lower": i / 10, "count": int(mask.sum()), "mean_probability": float(np.average(p[mask], weights=w[mask])),
                         "observed_rate": float(np.average(y[mask], weights=w[mask]))})
    return {"observations": len(y), "unexposed_without_outcome": unexposed,
            "log_loss": float(np.average(-y * np.log(p) - (1 - y) * np.log1p(-p), weights=w)),
            "brier_score": float(np.average((p - y) ** 2, weights=w)),
            "accuracy": float(np.average((p >= 0.5) == y, weights=w)),
            "effective_sample_size": float(w.sum() ** 2 / (w ** 2).sum()),
            "inverse_propensity": inverse_propensity, "calibration_bins": bins, "predictions": predictions,
            "data_provenance": sorted({e.get("provenance", "unspecified") for e in events}),
            "generalization": "Unverified beyond the supplied held-out events"}


def annotation_metrics(predicted, reference, *, tolerance_seconds=0.5):
    if not np.isfinite(tolerance_seconds) or tolerance_seconds <= 0:
        raise ValueError("annotation tolerance must be positive")
    def validate(sections):
        end = 0
        for section in sections:
            if (not np.isfinite([section["start_seconds"], section["end_seconds"]]).all()
                    or not np.isclose(section["start_seconds"], end) or section["end_seconds"] <= end):
                raise ValueError("annotation evaluation requires complete contiguous timelines")
            end = section["end_seconds"]
        return end
    duration = validate(predicted)
    if not reference or not predicted or not np.isclose(duration, validate(reference)):
        raise ValueError("annotation timelines must cover the same recording")
    a, b = np.array([s["end_seconds"] for s in predicted[:-1]]), np.array([s["end_seconds"] for s in reference[:-1]])
    matches, errors = 0, []
    if len(a) and len(b):
        distance = np.abs(a[:, None] - b)
        # Maximize valid matches before minimizing distance.
        cost = np.where(distance <= tolerance_seconds, distance / tolerance_seconds, (len(a) + len(b) + 1))
        rows, cols = linear_sum_assignment(cost)
        errors = [float(distance[i, j]) for i, j in zip(rows, cols) if distance[i, j] <= tolerance_seconds]
        matches = len(errors)
    precision, recall = matches / len(a) if len(a) else float(not len(b)), matches / len(b) if len(b) else float(not len(a))
    agreement = sum(max(0, min(p["end_seconds"], r["end_seconds"]) - max(p["start_seconds"], r["start_seconds"]))
                    for p in predicted for r in reference if p["label"] == r["label"]) / duration
    return {"boundary_precision": precision, "boundary_recall": recall,
            "boundary_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "matched_boundary_mae_seconds": float(np.mean(errors)) if errors else None,
            "duration_weighted_label_accuracy": agreement, "tolerance_seconds": tolerance_seconds}


def waveform_metrics(path, *, window_samples=1024):
    rate, audio = _audio_geometry(Path(path))
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    diff = np.diff(mono.astype(float))
    seam = np.arange(window_samples - 1, len(diff), window_samples)
    return {"sample_rate_hz": rate, "duration_seconds": len(audio) / rate,
            "rms": float(np.sqrt(np.mean(audio ** 2))), "peak": float(np.max(np.abs(audio))),
            "clipped_fraction": float(np.mean(np.abs(audio) >= 1)),
            "mean_adjacent_sample_difference": float(np.mean(np.abs(diff))) if len(diff) else 0,
            "mean_window_seam_difference": float(np.mean(np.abs(diff[seam]))) if len(seam) else None,
            "perceived_musical_quality": "Unverified; requires independent listening evaluation"}


def split_rhat(chains):
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2 or min(chains.shape) < 4:
        raise ValueError("R-hat needs at least four chains and four draws")
    n = chains.shape[1] // 2
    split = np.vstack([chains[:, :n], chains[:, -n:]])
    within = np.var(split, axis=1, ddof=1).mean()
    between = n * np.var(split.mean(axis=1), ddof=1)
    if within == 0:
        return 1.0 if between == 0 else None
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def held_out_waveform_metrics(model_path, catalog_path, *, seed=0, draws=8, batch_size=32):
    """Evaluate likelihood on distinct recordings excluded from model fitting."""
    from .waveform import WaveformModel, complete_window_data
    from .modalities import validate_observations
    model = WaveformModel.load(Path(model_path))
    torch = __import__("torch")
    if min(draws, batch_size) <= 0:
        raise ValueError("held-out waveform evaluation requires positive sampling and batch size")
    manifest = json.loads(Path(model_path).read_text())
    catalog = json.loads(Path(catalog_path).read_text())
    shared = SharedResponsePrior(**catalog["shared_prior"])
    songs = [Song.from_dict(song) for song in catalog["songs"]]
    if len(shared.music_names) != model.config["music_dimension"]:
        raise ValueError("held-out waveform coordinates do not match decoder")
    sources = []
    training_hashes = {source["sha256"] for source in manifest["sources"]}
    for source in catalog["provenance"]:
        if not source.get("recording_id"):
            continue
        ingestion = source["references"]["ingestion"]
        if sha256_file(Path(ingestion["path"])) != ingestion["sha256"]:
            raise ValueError("held-out ingestion changed")
        audio = json.loads(Path(ingestion["path"]).read_text())["canonical_audio"]
        if sha256_file(Path(audio["path"])) != audio["sha256"] or audio["sha256"] in training_hashes:
            raise ValueError("held-out recordings overlap training or validation sources")
        sources.append({"song_id": source["recording_id"], "path": audio["path"], "sha256": audio["sha256"]})
    lookup = {source["song_id"]: source for source in sources}
    if not songs or set(lookup) != {song.song_id for song in songs}:
        raise ValueError("held-out catalog lacks complete source lineage")
    for song in songs:
        if song.observations:
            validate_observations(song, song.observations, shared.music_names)
    data = complete_window_data(model, {"sources": [lookup[song.song_id] for song in songs]}, songs)
    if not len(data["audio"]):
        raise ValueError("held-out recordings must contain complete decoder windows")
    rng = np.random.default_rng(seed)
    values = []
    with torch.no_grad():
        for start in range(0, len(data["audio"]), batch_size):
            observed = torch.from_numpy(data["audio"][start:start + batch_size])
            music = torch.from_numpy(data["music"][start:start + batch_size])
            context = torch.from_numpy(data["context"][start:start + batch_size])
            mean, log_sd = model.model.encode(observed, music, context)
            noise = torch.tensor(rng.normal(size=(draws, *mean.shape)), dtype=mean.dtype)
            latent = mean + log_sd.exp() * noise
            decoded = model.model.decode(latent, music.expand(draws, -1, -1), context.expand(draws, -1, -1))
            logp = -0.5 * (((decoded - observed) / model.config["observation_sd"]) ** 2 + np.log(2 * np.pi * model.config["observation_sd"] ** 2)).sum(-1)
            logp -= float(np.log(model.scale).sum())
            kl = 0.5 * (mean.square() + (2 * log_sd).exp() - 1 - 2 * log_sd).sum(-1)
            values.extend((logp.mean(0) - kl).tolist())
    return {"recordings": len(songs), "windows": len(values), "mean_window_elbo": float(np.mean(values)),
            "mean_elbo_nats_per_sample": float(np.mean(values) / model.config["samples"]),
            "protocol": "complete resampled windows with teacher-forced previous audio; independent recording hashes",
            "perceived_quality": "Unverified; waveform likelihood does not establish musical quality"}


def sampler_metrics(model_path, *, seed=0, chains=4, sweeps=400, draws=128, explain_count=1, explanation_permutations=32):
    from .recommend import rank_catalog, sample_playlist, sample_ideal_songs, explain_song
    from .probability import FeaturePrior
    if chains < 4 or sweeps < 8 or draws < 2 or explain_count < 0 or explanation_permutations <= 0:
        raise ValueError("sampling diagnostics need four chains, eight sweeps and two posterior draws")
    data, songs, shared, listener, joint = load_run(model_path)
    intent = Intent(**data["intent.json"])
    means, targets, traces, explanations = [], [], [], {}
    augmented = np.array([row["features_mean"] for row in data["ranking.json"]])
    conditional_prior = FeaturePrior(augmented, np.full_like(augmented, 0.15))
    for chain in range(chains):
        ranking, posterior = rank_catalog(songs, shared, listener, intent, seed=seed + chain, draws=draws, joint_posterior=joint)
        means.append({r["song_id"]: r["expected_utility"] for r in ranking})
        base = FeaturePrior(np.array([s.global_region.music for s in songs]), np.full((len(songs), len(shared.music_names)), 0.12))
        ideal = sample_ideal_songs(shared, base, listener, intent, seed=seed + 100 + chain, particles=max(128, draws), count=1, joint_posterior=joint)
        targets.append(ideal)
        for row in [r for r in data["ranking.json"] if r["score"] is not None][:explain_count]:
            song_id = row["song_id"]
            explanation = explain_song(posterior["features"][song_id], listener, intent, conditional_prior,
                                       posterior["centers"], posterior["weights"], seed=seed + 10000 + chain,
                                       permutations=explanation_permutations)
            explanations.setdefault(song_id, []).append(explanation)
        playlist = sample_playlist(data["ranking.json"], intent, seed=seed + chain, sweeps=sweeps, record_every=1)
        retained = playlist["trace"][sweeps // 2:]
        traces.append([t["log_density"] for t in retained])
    standard_errors = {s.song_id: float(np.std([m[s.song_id] for m in means], ddof=1) / np.sqrt(chains)) for s in songs}
    exact = None
    eligible = [r for r in data["ranking.json"] if r["score"] is not None and r["catalog_target_probability"] > 0]
    size = data["playlist.json"]["item_count"]
    if len(eligible) ** size <= 100000:
        f = np.array([r["features_mean"] for r in eligible])
        distance = np.mean((f[:, None] - f[None]) ** 2, axis=-1)
        logp = np.log([r["catalog_target_probability"] for r in eligible])
        states = list(itertools.product(range(len(eligible)), repeat=size))
        energy = np.array([sum(logp[i] for i in state) - sum(np.exp(-distance[state[i], state[j]] / 0.05) for i in range(size) for j in range(i + 1, size))
                           - sum(distance[state[i], state[i + 1]] for i in range(size - 1)) for state in states])
        probability = np.exp(energy - logsumexp(energy))
        # Use a fresh chain with the same fixed ranking as the exact target.
        trace = sample_playlist(data["ranking.json"], intent, seed=seed + 1000, sweeps=sweeps * 4, record_every=1)["trace"][sweeps * 2:]
        observed = Counter(tuple(t["songs"]) for t in trace)
        measured = np.array([observed[tuple(eligible[i]["song_id"] for i in state)] / len(trace) for state in states])
        exact = {"states": len(states), "total_variation_distance": float(np.abs(probability - measured).sum() / 2),
                 "samples": len(trace), "log_normalizer": float(logsumexp(energy))}
    return {"chains": chains, "retained_draws_per_chain": len(traces[0]), "split_rhat_log_density": split_rhat(traces),
            "ranking_mean_standard_errors": standard_errors, "ideal_sampling_reports": targets,
            "explanation_shapley_standard_errors": {song_id: {name: float(np.std([e["features"][name]["conditional_shapley"] for e in values], ddof=1) / np.sqrt(chains))
                                                             for name in listener.feature_names} for song_id, values in explanations.items()},
            "ideal_log_normalizer_standard_error": float(np.std([target["diagnostics"][0]["log_normalizer_estimate"] for target in targets], ddof=1) / np.sqrt(chains)),
            "finite_playlist_exact_crosscheck": exact,
            "interpretation": "diagnostics for this configuration; finite diagnostics do not prove convergence"}


def evaluate_configuration(config_path, output_dir):
    from ..pipeline import write_json
    config_path, output_dir = Path(config_path), Path(output_dir)
    if output_dir.exists():
        raise ValueError("evaluation requires a new output directory")
    config = json.loads(config_path.read_text())
    reports, references = {}, {"configuration": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)}}
    if config.get("model_path"):
        references["model"] = {"path": str(Path(config["model_path"]).resolve()), "sha256": sha256_file(Path(config["model_path"]))}
    if config.get("preferences_path"):
        events_path = Path(config["preferences_path"])
        references["preferences"] = {"path": str(events_path.resolve()), "sha256": sha256_file(events_path)}
        reports["preferences"] = preference_metrics(config["model_path"], json.loads(events_path.read_text()), **config.get("preference_options", {}))
    if config.get("sampling") is not None:
        reports["sampling"] = sampler_metrics(config["model_path"], **config["sampling"])
    if config.get("waveform_test_catalog_path"):
        for key in ("waveform_model_path", "waveform_test_catalog_path"):
            path = Path(config[key])
            references[key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        reports["held_out_waveform"] = held_out_waveform_metrics(config["waveform_model_path"], config["waveform_test_catalog_path"], **config.get("waveform_test_options", {}))
    reports["annotations"] = []
    for record in config.get("annotations", []):
        for key in ("predicted", "reference"):
            path = Path(record[key])
            references[f"annotation-{len(references)}"] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        reports["annotations"].append(annotation_metrics(json.loads(Path(record["predicted"]).read_text())["sections"],
                                                          json.loads(Path(record["reference"]).read_text())["sections"],
                                                          tolerance_seconds=record.get("tolerance_seconds", 0.5)))
    reports["waveforms"] = []
    for path in config.get("waveforms", []):
        references[f"waveform-{len(references)}"] = {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
        reports["waveforms"].append({"path": path, **waveform_metrics(path)})
    if not reports.get("preferences") and not reports.get("sampling") and not reports.get("held_out_waveform") and not reports["annotations"] and not reports["waveforms"]:
        raise ValueError("evaluation configuration has no observations")
    path = output_dir / "manifest.json"
    pending = output_dir / ".manifest.pending.json"
    write_json(pending, {"schema": "hearthis.evaluation/v1", "references": references, "reports": reports})
    try:
        validate_evaluation(pending)
        pending.replace(path)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    return path


def validate_evaluation(path):
    data = json.loads(Path(path).read_text())
    if data["schema"] != "hearthis.evaluation/v1":
        raise ValueError("invalid evaluation schema")
    for ref in data["references"].values():
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("evaluation input provenance mismatch")
    if "model" in data["references"]:
        validate_model_run(Path(data["references"]["model"]["path"]))
    json.dumps(data["reports"], allow_nan=False)
    return {"valid": True, "reports": list(data["reports"])}
