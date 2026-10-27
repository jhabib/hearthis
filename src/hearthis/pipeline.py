"""Audio-to-taste orchestration and versioned, inspectable model runs."""

from __future__ import annotations

from dataclasses import asdict
import itertools
import json
from pathlib import Path

import numpy as np

from .audio.corrections import apply_structure_correction
from .audio.ingest import ingest_audio, sha256_file
from .audio.listening import review_audio
from .audio.passages import build_passages
from .audio.separation import separate_sources
from .audio.spectrogram import generate_spectrograms
from .audio.validation import validate_recording
from .model.features import FEATURE_SCHEMA, MUSIC_NAMES, BROAD_MUSIC_NAMES, RESPONSE_NAMES, extract_song
from .model.probability import FeaturePrior
from .model.recommend import explain_song, rank_catalog, sample_ideal_songs, sample_playlist
from .model.response import SharedResponsePrior, Song, condition_behavior_profiles
from .model.taste import Intent, Listener, fit_feedback, parse_request
from .model.validation import validate_model_run, validate_end_to_end


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def process_audio(source: Path, artifact_root: Path, *, recording_id: str | None = None,
                  device: str = "cpu", separation_backend=None, structure_backend=None) -> Path:
    """Complete audio pipeline; new recording directories protect prior artifacts."""
    identifier = recording_id or sha256_file(source)[:16]
    if (artifact_root / identifier).exists():
        raise ValueError("audio processing requires a new recording directory; use validated existing recordings to resume")
    if separation_backend is None:
        from .audio.demucs_backend import DemucsBackend
        separation_backend = DemucsBackend(device=device)
    if structure_backend is None:
        from .audio.structure import LibrosaStructureBackend
        structure_backend = LibrosaStructureBackend()
    from .audio.structure import analyze_structure
    ingestion_path = ingest_audio(source, artifact_root, recording_id=identifier)
    root = ingestion_path.parent
    canonical = Path(json.loads(ingestion_path.read_text())["canonical_audio"]["path"])
    specs = generate_spectrograms(canonical, root / "spectrograms")
    separation = separate_sources(canonical, root / "stems", separation_backend)
    for name, record in json.loads(separation.read_text())["stems"].items():
        generate_spectrograms(Path(record["path"]), root / "stems" / "spectrograms" / name)
    structure = analyze_structure(canonical, root / "structure", structure_backend, specs)
    build_passages(structure, root / "passages" / "manifest.json",
                   spectrogram_manifests=specs, separation_manifest=separation)
    validate_recording(root)
    return root


def apply_listening_review(root: Path, correction: Path, *, output_dir: Path | None = None) -> tuple[Path, Path]:
    if output_dir is not None and output_dir.exists():
        raise ValueError("corrected timeline output must be a new directory")
    destination = correction.parent if output_dir is None else output_dir
    specs = sorted((root / "spectrograms").glob("*.json"))
    structure = apply_structure_correction(root / "structure" / "manifest.json", correction,
                                          destination / "structure" / "manifest.json",
                                          spectrogram_manifests=specs)
    passages = build_passages(structure, destination / "passages" / "manifest.json",
                              spectrogram_manifests=specs, separation_manifest=root / "stems" / "manifest.json")
    validate_recording(root, structure_manifest=structure, passages_manifest=passages)
    return structure, passages


def make_catalog(recordings: list[dict], *, shared_prior: SharedResponsePrior | None = None, broad_features=False) -> tuple[list[Song], SharedResponsePrior, list[dict]]:
    prior = shared_prior or SharedResponsePrior.uninformed(BROAD_MUSIC_NAMES if broad_features else MUSIC_NAMES, RESPONSE_NAMES)
    songs, provenance = [], []
    for record in recordings:
        song, evidence = extract_song(Path(record["recording_dir"]), prior,
                                      structure_manifest=Path(record["structure_manifest"]) if record.get("structure_manifest") else None,
                                      passages_manifest=Path(record["passages_manifest"]) if record.get("passages_manifest") else None,
                                      metadata=record.get("metadata"),
                                      listening_evidence=Path(record["listening_evidence"]) if record.get("listening_evidence") else None)
        if record.get("observations_path"):
            from .model.modalities import attach_observations
            evidence["references"]["observations"] = attach_observations(song, record["observations_path"], prior.music_names)
            evidence["lyrics_status"] = "supplied" if "lyrics" in song.observations else "absent"
        songs.append(song)
        provenance.append(evidence)
    if not songs or len({s.song_id for s in songs}) != len(songs):
        raise ValueError("catalog requires distinct validated recordings")
    return songs, prior, provenance


def _critiques(events, songs, listener, prior) -> list[dict]:
    lookup = {song.song_id: song for song in songs}
    reports = []
    for event in events:
        if event["kind"] != "critique":
            continue
        if (event.get("exposed") is not True or not 0 < float(event.get("propensity", 0)) <= 1
                or event["song_id"] not in lookup):
            raise ValueError("critique requires exposure, propensity and a known recording")
        song = lookup[event["song_id"]]
        region_by_id = {r.region_id: r for r in song.regions}
        grounding = event.get("region_probabilities", {event.get("region_id"): 1.0})
        if (not grounding or not set(grounding) <= region_by_id.keys()
                or any(not np.isfinite(p) or p < 0 for p in grounding.values())
                or not np.isclose(sum(grounding.values()), 1)):
            raise ValueError("critique must ground a normalized posterior over known regions")
        confidence = float(event["confidence"])
        magnitude = float(event["magnitude"])
        if not 0 < confidence <= 1 or not 0 <= magnitude <= 1 or event["polarity"] not in (-1, 1):
            raise ValueError("critique confidence, magnitude or polarity is invalid")
        listener.condition_prompt({event["feature"]: (1 + event["polarity"] * magnitude) / 2}, confidence=confidence)
        descriptors = event.get("descriptor_reports", {})
        indices = {prior.response_names.index(name): value for name, value in descriptors.items()}
        if indices:
            for region_id, probability in grounding.items():
                if probability > 0:
                    region_by_id[region_id].profile.update(indices, confidence=confidence,
                                                         strength=4 * probability,
                                                         mode_probabilities=event.get("mode_probabilities"))
        reports.append({"kind": "grounded critique", "song_id": song.song_id, "feature": event["feature"],
                        "region_posterior": grounding, "descriptor_profiles_updated": bool(indices)})
    return reports


def run_model(songs: list[Song], prior: SharedResponsePrior, listener: Listener, intent: Intent,
              output_dir: Path, *, feedback: list[dict] | None = None, provenance=None,
              seed: int = 0, draws: int = 128, particles: int = 2048,
              ideal_count: int = 8, explain_count: int = 3, fit_iterations: int = 100,
              joint: dict | None = None, joint_posterior_path: Path | None = None,
              waveform_path: Path | None = None, generation: dict | None = None) -> Path:
    """Tu=(O,Ou,Gu,Pu): conditioning, posterior prediction, target and retrieval."""
    if output_dir.exists():
        raise ValueError("model run output must be a new directory")
    if listener.feature_names != prior.music_names + prior.response_names:
        raise ValueError("shared and personal model feature coordinates do not match")
    intent = Intent(**asdict(intent))
    parsed = parse_request(intent.request, listener.feature_names)
    if any(name in intent.targets and intent.targets[name] != value for name, value in parsed.items()):
        raise ValueError("text request conflicts with structured intent targets")
    intent.targets = {**intent.targets, **parsed}
    intent.validate(listener.feature_names)
    # Isolate all mutable persistent posterior state from caller-owned priors.
    working = Listener.from_dict(listener.to_dict())
    songs = [Song.from_dict(song.to_dict()) for song in songs]
    shared = SharedResponsePrior(**prior.to_dict())
    prior = shared
    joint_posterior = None
    if joint_posterior_path is not None:
        from .model.joint import JointPosterior
        joint_posterior = JointPosterior.from_dict(json.loads(Path(joint_posterior_path).read_text()), songs, prior, working)
        if waveform_path is None:
            waveform_path = joint_posterior.model.waveform_path
    feedback = [{**event, "step": event.get("step", working.step)} for event in (feedback or [])]
    event_ids = [event["event_id"] for event in feedback if "event_id" in event]
    if (any(not isinstance(identifier, str) or not identifier for identifier in event_ids)
            or len(event_ids) != len(set(event_ids)) or set(event_ids) & set(working.observed_event_ids)):
        raise ValueError("feedback event IDs must be unique and previously unobserved")
    steps = [event.get("step", working.step) for event in feedback]
    if any(type(step) is not int or step < working.step or step - working.step > 10000 for step in steps) or steps != sorted(steps):
        raise ValueError("feedback steps must be ordered, current and within the reference dynamics limit")
    rng = np.random.default_rng(seed)
    inference = []
    group_key = lambda event: (event["step"], json.dumps(event.get("intent", {}), sort_keys=True))
    for (step, historical_intent), group in itertools.groupby(feedback, key=group_key):
        while working.step < step:
            changed = working.advance(rng)
            if joint_posterior is not None:
                joint_posterior.forecast(working)
            inference.append({"kind": "state transition", "step": working.step, "change_point": changed})
        events = list(group)
        historical = Intent(**json.loads(historical_intent))
        historical.targets = {**historical.targets, **parse_request(historical.request, working.feature_names)}
        historical.validate(working.feature_names)
        if joint is not None or joint_posterior is not None:
            from .model.joint import fit_joint
            options = dict(joint or {})
            if waveform_path is not None and joint_posterior is None:
                options.setdefault("train_neural", True)
                options.setdefault("neural_output_dir", output_dir / "joint-waveform")
            else:
                options["train_neural"] = False
            joint_posterior = fit_joint(songs, prior, working, events, historical,
                                        waveform_path=waveform_path, previous=joint_posterior,
                                        seed=seed + step, **options)
            if joint_posterior.model.waveform_path is not None:
                waveform_path = joint_posterior.model.waveform_path
            joint_posterior.project(songs, prior, working, seed=seed + step)
            inference.append(joint_posterior.report)
            continue
        inference.extend(_critiques(events, songs, working, prior))
        behavior = [event for event in events if event["kind"] != "critique"]
        if behavior:
            states, _ = working.draws(rng, draws)
            features = {song.song_id: song.predict(prior, working, historical, states, rng)[0].mean(axis=0)
                        for song in songs}
            inference.append(fit_feedback(working, behavior, features, historical, seed=seed + step,
                                          draws=draws, iterations=fit_iterations))
            for event in behavior:
                if event.get("exposed") is True and event["kind"] == "behavior":
                    song = next(s for s in songs if s.song_id == event["song_id"])
                    inference.append(condition_behavior_profiles(song, prior, working, historical, event,
                                                                  seed=seed + step, draws=draws))
    if joint is not None and joint_posterior is None:
        from .model.joint import fit_joint
        options = dict(joint)
        if waveform_path is not None:
            options.setdefault("train_neural", True)
            options.setdefault("neural_output_dir", output_dir / "joint-waveform")
        joint_posterior = fit_joint(songs, prior, working, [], Intent(), waveform_path=waveform_path, seed=seed, **options)
        if joint_posterior.model.waveform_path is not None:
            waveform_path = joint_posterior.model.waveform_path
        joint_posterior.project(songs, prior, working, seed=seed)
        inference.append(joint_posterior.report)
    working.observed_event_ids.extend(event_ids)
    persistent = working.to_dict()
    # Prompt pseudo-observations are session-local, preserving learned long-term priors.
    if joint_posterior is not None:
        # Project the session listener only. Persistent song/shared summaries
        # remain separate, while prediction uses the complete joint posterior.
        session_shared = SharedResponsePrior(**prior.to_dict())
        session_songs = [Song.from_dict(song.to_dict()) for song in songs]
        joint_posterior.project(session_songs, session_shared, working, targets=intent.targets, seed=seed + 17)
    else:
        working.condition_prompt(intent.targets)
    ranking, posterior = rank_catalog(songs, prior, working, intent, seed=seed, draws=draws, joint_posterior=joint_posterior)
    music_means = np.array([song.global_region.music for song in songs])
    music_base = FeaturePrior(music_means, np.full_like(music_means, 0.12))
    ideal = sample_ideal_songs(prior, music_base, working, intent, seed=seed + 1,
                              count=ideal_count, particles=particles, joint_posterior=joint_posterior)
    playlist = sample_playlist(ranking, intent, seed=seed + 2)
    augmented_means = np.array([r["features_mean"] for r in ranking])
    conditional_prior = FeaturePrior(augmented_means, np.full_like(augmented_means, 0.15))
    explanations = {}
    for row in [r for r in ranking if r["score"] is not None][:explain_count]:
        explanations[row["song_id"]] = explain_song(posterior["features"][row["song_id"]], working, intent,
                                                    conditional_prior, posterior["centers"], posterior["weights"], seed=seed + 3)
    files = {
        "catalog.json": {"schema": FEATURE_SCHEMA, "shared_prior": prior.to_dict(),
                         "songs": [song.to_dict() for song in songs], "provenance": provenance or []},
        "listener-prior.json": listener.to_dict(), "listener-posterior.json": persistent,
        "listener-session.json": working.to_dict(), "intent.json": asdict(intent),
        "feedback.json": feedback, "inference.json": inference, "ranking.json": ranking,
        "ideal-songs.json": ideal, "playlist.json": playlist, "explanations.json": explanations,
    }
    if joint_posterior is not None:
        files["joint-posterior.json"] = joint_posterior.to_dict()
    if waveform_path is not None:
        from .model.waveform import generate_ideal_audio
        files["generated-audio.json"] = generate_ideal_audio(ideal, Path(waveform_path), output_dir / "generated", seed=seed + 4, **(generation or {}))
        if joint_posterior is not None and joint_posterior.model.observation_kernel is not None:
            files["generated-audio.json"]["generated_modalities"] = [joint_posterior.model.observation_kernel.sample(sample["music"], seed=seed + 100 + index)
                                                                      for index, sample in enumerate(ideal["samples"])]
    for name, value in files.items():
        write_json(output_dir / name, value)
    manifest = {
        "schema": "hearthis.taste-program/v1", "seed": seed,
        "feature_names": list(working.feature_names), "listener_id": working.listener_id,
        "song_count": len(songs), "posterior_draws": draws,
        "artifacts": {name: {"path": str((output_dir / name).resolve()), "sha256": sha256_file(output_dir / name)}
                      for name in files},
        "factor_graph": {"conditioned_song_state": ["W", "F"] if joint_posterior else ["W", "H", "F", "X"],
                         "latent_song_state": ["X", "H"] if joint_posterior else [],
                         "response": ["Omega -> kappa -> R -> aggregated_R"],
                         "taste": ["Z_previous -> Z -> activation -> U"],
                         "observations": ["exposure -> behavior", "grounded_critique -> response_and_weight_posterior"],
                         "outputs": ["posterior -> ideal_feature_density", "posterior -> ranking -> playlist", "conditional_reference -> explanations"]},
        "boundaries": {"song_encoder": "joint uncertain music state and learned waveform likelihood" if joint_posterior and waveform_path else "anchored measurements",
                       "response_inference": "joint logistic-normal profile inference" if joint_posterior else "Dirichlet/Beta local approximation",
                       "preference_inference": "joint low-rank Gaussian Monte Carlo ELBO" if joint_posterior else "correlated-prior, whitened Gaussian Monte Carlo variational inference",
                       "target": "bounded feature-state density and conditional neural waveform sampling" if waveform_path else "bounded feature-state density",
                       "language_grounding": "explicit feature targets and supplied probabilistic region/mode groundings",
                       "listener_provenance": working.provenance},
    }
    path = output_dir / "manifest.json"
    pending = output_dir / ".manifest.pending.json"
    write_json(pending, manifest)
    try:
        validate_model_run(pending)
        pending.replace(path)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    return path


def learn_shared_descriptions(songs: list[Song], records: list[dict], prior: SharedResponsePrior) -> dict:
    """Weak shared supervision from direct listening descriptions, clustered by anchored descriptor coordinates."""
    from scipy.cluster.vq import kmeans2
    observations = []
    for song, record in zip(songs, records, strict=True):
        if not record.get("listening_evidence"):
            continue
        evidence = json.loads(Path(record["listening_evidence"]).read_text())
        for mode in evidence["review"].get("descriptor_modes", []):
            if mode["weight"] > 0 and mode["confidence"] > 0:
                observations.append({"music": song.global_region.music.tolist(),
                                     "descriptors": [mode["descriptors"][name] for name in prior.response_names],
                                     "confidence": mode["confidence"] * mode["weight"]})
        for section in evidence["review"]["sections"]:
            regions = [r for r in song.regions if r.kind == "section"]
            overlap = np.array([max(0, min(r.end_seconds, section["end_seconds"])
                                    - max(r.start_seconds, section["start_seconds"])) for r in regions])
            if overlap.sum() == 0:
                continue
            music = np.average(np.array([r.music for r in regions]), axis=0, weights=overlap)
            for mode in section.get("descriptor_modes", []):
                if mode["weight"] > 0 and mode["confidence"] > 0:
                    observations.append({"music": music.tolist(),
                                         "descriptors": [mode["descriptors"][name] for name in prior.response_names],
                                         "confidence": mode["confidence"] * mode["weight"]})
    mode_count = prior.concentration.size
    if len(observations) < mode_count:
        return {"updated": False, "reason": "insufficient direct listening descriptor observations"}
    descriptors = np.array([r["descriptors"] for r in observations])
    # Fix only a reporting order. Permuting modes leaves the model's predictions unchanged.
    order = np.argsort(descriptors[:, 0])
    initial = descriptors[order[np.linspace(0, len(order) - 1, mode_count).astype(int)]]
    _, assignments = kmeans2(descriptors, initial, minit="matrix", seed=0)
    for record, mode in zip(observations, assignments, strict=True):
        record["mode"] = int(mode)
    report = prior.fit(observations)
    report.update(updated=True, supervision="weak direct-audio LLM descriptors; no human calibration")
    return report


def run_configuration(config_path: Path, output_dir: Path, *, device: str = "cpu") -> Path:
    """One command from recordings or source audio through all model outputs."""
    if output_dir.exists():
        raise ValueError("end-to-end output must be a new directory")
    config = json.loads(config_path.read_text())
    if config.get("catalog_path") and (config.get("recordings") or config.get("sources") or config.get("learn_shared_descriptions")):
        raise ValueError("resuming a catalog cannot also rebuild recordings or refit its shared prior")
    records = [dict(record) for record in config.get("recordings", [])]
    for source in config.get("sources", []):
        root = process_audio(Path(source["path"]), output_dir / "audio",
                             recording_id=source.get("recording_id"), device=device)
        records.append({"recording_dir": str(root), "metadata": source.get("metadata", {})})
    for record in records:
        root = Path(record["recording_dir"])
        correction = record.get("listening_correction")
        if correction is None and config.get("listening"):
            identifier = json.loads((root / "manifest.json").read_text())["recording_id"]
            correction = review_audio(root, output_dir / "listening" / identifier, **config["listening"])
        if correction is not None:
            correction = Path(correction)
            identifier = json.loads((root / "manifest.json").read_text())["recording_id"]
            structure, passages = apply_listening_review(root, correction, output_dir=output_dir / "corrected" / identifier)
            record.update(structure_manifest=str(structure), passages_manifest=str(passages))
            annotation = json.loads(correction.read_text())
            if annotation.get("annotation_source") == "llm" and annotation["provenance"].get("audio_listened") is True:
                evidence_path = Path(annotation["provenance"]["evidence_path"])
                record["listening_evidence"] = str(evidence_path if evidence_path.is_absolute() else correction.parent / evidence_path)
    if config.get("catalog_path"):
        if records or config.get("learn_shared_descriptions"):
            raise ValueError("resuming a catalog cannot also rebuild recordings or refit its shared prior")
        catalog_path = Path(config["catalog_path"])
        stored = json.loads(catalog_path.read_text())
        prior = SharedResponsePrior(**stored["shared_prior"])
        songs = [Song.from_dict(song) for song in stored["songs"]]
        provenance = stored["provenance"] + [{"references": {"resumed_catalog": {
            "path": str(catalog_path.resolve()), "sha256": sha256_file(catalog_path)}}}]
    else:
        songs, prior, provenance = make_catalog(records, broad_features=config.get("broad_features", False))
    shared_learning = {"updated": False, "reason": "not requested"}
    if config.get("learn_shared_descriptions", False):
        shared_learning = learn_shared_descriptions(songs, records, prior)
        songs, prior, provenance = make_catalog(records, shared_prior=prior)
    names = prior.music_names + prior.response_names
    listener_config = config.get("listener", {})
    if config.get("listener_path"):
        listener = Listener.from_dict(json.loads(Path(config["listener_path"]).read_text()))
    else:
        listener = Listener.from_centers(listener_config.get("listener_id", "cold-start"), names,
                                         listener_config.get("centers", [[0.5] * len(names)]),
                                         variance=listener_config.get("variance", 0.05),
                                         provenance=listener_config.get("provenance", "uninformed cold-start prior; no listener evidence"))
    feedback = (json.loads(Path(config["feedback_path"]).read_text()) if config.get("feedback_path")
                else config.get("feedback", []))
    if config.get("dynamics"):
        variance = float(config["dynamics"]["transition_variance"])
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("hierarchical transition variance must be positive")
        for component in listener.components:
            if component.transition_covariance is None:
                component.transition_covariance = np.eye(len(names) ** 2) * variance
    intent_config = config.get("intent", {})
    if config.get("language"):
        from .model.language import ground_configuration
        language_options = dict(config["language"])
        response_cache = language_options.pop("response_cache_path", output_dir / "language-response.json")
        intent_config, feedback, language_evidence = ground_configuration(intent_config, feedback, songs, names,
                                                                         response_cache_path=response_cache, **language_options)
        write_json(output_dir / "language-grounding.json", language_evidence)
        provenance.append({"references": {"language_grounding": {"path": str((output_dir / "language-grounding.json").resolve()),
                                                                   "sha256": sha256_file(output_dir / "language-grounding.json")}}})
    elif config.get("language_grounding_path"):
        grounding_path = Path(config["language_grounding_path"])
        json.loads(grounding_path.read_text())
        provenance.append({"references": {"language_grounding": {"path": str(grounding_path.resolve()),
                                                                   "sha256": sha256_file(grounding_path)}}})
    waveform_path = config.get("waveform_path")
    if config.get("waveform") is not None:
        if waveform_path:
            raise ValueError("choose waveform training or an existing waveform model")
        from .model.waveform import train_waveform
        waveform_path = train_waveform(songs, provenance, output_dir / "waveform", seed=config.get("sampling", {}).get("seed", 0), **config["waveform"])
    if waveform_path is not None:
        waveform_path = Path(waveform_path)
        provenance.append({"references": {"waveform_model": {"path": str(waveform_path.resolve()), "sha256": sha256_file(waveform_path)}}})
    model_path = run_model(songs, prior, listener, Intent(**intent_config), output_dir / "taste",
                           feedback=feedback, provenance=provenance, joint=config.get("joint"),
                           joint_posterior_path=config.get("joint_posterior_path"), waveform_path=waveform_path,
                           generation=config.get("generation"), **config.get("sampling", {}))
    extensions = {}
    if config.get("edits"):
        from .model.editing import edit_audio
        stored = json.loads((model_path.parent / "catalog.json").read_text())
        music_names = stored["shared_prior"]["music_names"]
        catalog_by_id = {song["song_id"]: Song.from_dict(song) for song in stored["songs"]}
        evidence_by_id = {record.get("recording_id"): record for record in stored["provenance"]}
        decoder_path = Path(json.loads((model_path.parent / "generated-audio.json").read_text())["model"]["path"])
        edit_results = []
        for index, request in enumerate(config["edits"]):
            song = catalog_by_id[request["song_id"]]
            music = song.global_region.music
            intervals = []
            for interval in request["intervals"]:
                if not set(interval["targets"]) <= set(music_names):
                    raise ValueError("edit targets contain unsupported coordinates")
                target = music.copy()
                for name, value in interval["targets"].items():
                    target[music_names.index(name)] = value
                intervals.append({"start_seconds": interval["start_seconds"], "end_seconds": interval["end_seconds"],
                                  "strength": interval.get("strength", 1.0), "music": target.tolist()})
            ingestion_path = Path(evidence_by_id[song.song_id]["references"]["ingestion"]["path"])
            source = Path(json.loads(ingestion_path.read_text())["canonical_audio"]["path"])
            path = edit_audio(source, decoder_path, output_dir / "edits" / str(index), music=music,
                              edits=intervals, fade_seconds=request.get("fade_seconds", 0.02))
            edit_results.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        extensions["edits"] = edit_results
    if config.get("evaluation") is not None:
        from .model.evaluation import evaluate_configuration
        evaluation_config = {**config["evaluation"], "model_path": str(model_path.resolve())}
        evaluation_config.setdefault("waveforms", [record["path"] for record in json.loads((model_path.parent / "generated-audio.json").read_text())["waveforms"]]
                                     if (model_path.parent / "generated-audio.json").exists() else [])
        evaluation_path = output_dir / "evaluation-configuration.json"
        write_json(evaluation_path, evaluation_config)
        path = evaluate_configuration(evaluation_path, output_dir / "evaluation")
        extensions["evaluation"] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    write_json(output_dir / "shared-learning.json", shared_learning)
    path = output_dir / "manifest.json"
    pending = output_dir / ".manifest.pending.json"
    write_json(pending, {"schema": "hearthis.end-to-end/v1", "configuration": {
        "path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "model": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)},
        "shared_learning": {"path": str((output_dir / "shared-learning.json").resolve()), "sha256": sha256_file(output_dir / "shared-learning.json")},
        "recordings": records, "validation": validate_model_run(model_path),
        "listening_review_count": sum(bool(r.get("listening_evidence")) for r in records), **extensions})
    try:
        validate_end_to_end(pending)
        pending.replace(path)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    return path
