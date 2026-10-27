from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from hearthis.model.joint import JointModel, JointPosterior, fit_joint, segmentation_options
from hearthis.model.language import ground_configuration
from hearthis.model.taste import Intent
from hearthis.model.waveform import WaveformModel, train_waveform
from hearthis.model.validation import validate_model_run
from hearthis.pipeline import make_catalog, process_audio, run_model
from test_taste_model import small_catalog
from test_end_to_end import ExactFourStemBackend, FixedFourSecondStructureBackend, _source_recording


def test_joint_energy_matches_existing_numpy_contract():
    songs, prior, listener = small_catalog()
    model = JointModel(songs, prior, listener)
    white = model.tensor(np.random.default_rng(0).normal(size=(8, model.dimension)))
    centers, weights, coefficients, regions = model.parameters(white)
    intent = Intent(targets={"arousal": 0.7})
    features = model.features(centers, regions, songs[0], intent)[0]
    values = model.utility(features, centers, weights, intent).detach().numpy()
    expected = listener.utility(features.detach().numpy(), intent, centers=centers.detach().numpy(), weights=weights.detach().numpy())
    np.testing.assert_allclose(values, expected, atol=1e-6)


def test_confidence_adjusted_joint_bernoulli_is_normalized():
    songs, prior, listener = small_catalog()
    model = JointModel(songs, prior, listener)
    parameters = model.parameters(model.tensor(np.zeros((3, model.dimension))))
    masses = {s.song_id: model.tensor([1]) for s in songs}
    probabilities = []
    for outcome in (0, 1):
        event = {"kind": "behavior", "song_id": "0", "value": outcome, "exposed": True,
                 "propensity": 0.5, "confidence": 0.25}
        likelihood = model.log_observations(parameters, [event], Intent(), masses)
        probabilities.append(np.exp(likelihood.detach().numpy()) / 0.5)
    np.testing.assert_allclose(np.sum(probabilities, axis=0), 1, atol=1e-6)


def test_joint_feedback_updates_profiles_music_and_listener_with_finite_elbo():
    songs, prior, listener = small_catalog()
    events = [{"kind": "behavior", "song_id": "1", "value": 1, "exposed": True, "propensity": 0.5,
               "response_weights": [0, 8]} for _ in range(8)]
    posterior = fit_joint(songs, prior, listener, events, Intent(), iterations=35, draws=32, rank=3, seed=4)
    assert posterior.report["final_elbo"] > posterior.report["initial_elbo"]
    assert np.any(np.abs(posterior.factor) > 0.01)
    saved = posterior.to_dict()
    restored = JointPosterior.from_dict(saved, songs, prior, listener)
    a = posterior.sample(np.random.default_rng(0), 10)
    b = restored.sample(np.random.default_rng(0), 10)
    np.testing.assert_allclose(a, b, atol=1e-7)
    alpha = songs[1].global_region.profile.descriptor_alpha.copy()
    weights = listener.components[0].weights.mean.copy()
    posterior.project(songs, prior, listener)
    assert not np.allclose(songs[1].global_region.profile.descriptor_alpha, alpha)
    assert not np.allclose(listener.components[0].weights.mean, weights)


def test_session_conditioning_retains_cross_block_covariance():
    songs, prior, listener = small_catalog()
    model = JointModel(songs, prior, listener)
    factor = np.zeros((model.dimension, 1))
    z_block = model.blocks[model.taste_blocks[0][0]]
    w_block = model.blocks[model.taste_blocks[0][1]]
    factor[z_block["start"], 0] = factor[w_block["start"], 0] = 2
    posterior = JointPosterior(model, np.zeros(model.dimension), np.ones(model.dimension), factor,
                               {s.song_id: [1] for s in songs})
    before = posterior.sample(np.random.default_rng(0), 20000)
    after = posterior.sample(np.random.default_rng(0), 20000, {"brightness": 0.9})
    assert after[:, z_block["start"]].mean() > before[:, z_block["start"]].mean() + 0.2
    assert after[:, w_block["start"]].var() < before[:, w_block["start"]].var()


def test_impossible_exposure_is_rejected_by_joint_likelihood():
    songs, prior, listener = small_catalog()
    with pytest.raises(ValueError, match="exposure"):
        fit_joint(songs, prior, listener, [{"kind": "behavior", "exposed": False, "propensity": 1}], Intent(), iterations=1, draws=2)


def test_hierarchical_dynamics_forecast_and_joint_replay():
    songs, prior, listener = small_catalog()
    for component in listener.components:
        component.transition_covariance = np.eye(4) * 0.002
    old_covariance = listener.components[0].state.covariance.copy()
    old_mean = listener.components[0].state.mean.copy()
    expected = old_covariance + listener.components[0].innovation
    expected += np.eye(2) * 0.002 * np.trace(old_covariance + np.outer(old_mean, old_mean))
    listener.advance(np.random.default_rng(0), change_point=False)
    np.testing.assert_allclose(listener.components[0].state.covariance, expected)
    events = [{"kind": "pairwise", "song_id": "1", "other_song_id": "0", "value": 1,
               "exposed": True, "propensity": 0.5}]
    posterior = fit_joint(songs, prior, listener, events, Intent(), iterations=12, draws=16, rank=3)
    assert "A" in posterior.report["nodes"] and "Z_previous" in posterior.report["nodes"]
    posterior.project(songs, prior, listener)
    listener.advance(np.random.default_rng(1), change_point=False)
    posterior.forecast(listener, maximum_rank=12)
    assert posterior.factor.shape[1] <= 12
    updated = fit_joint(songs, prior, listener, events, Intent(), previous=posterior, iterations=12, draws=16)
    assert updated.report["final_elbo"] > updated.report["initial_elbo"]
    assert updated.report["waveform_likelihood"] is False
    assert listener.step == 2
    unexposed = {"kind": "behavior", "song_id": "unknown", "exposed": False, "propensity": 0.5}
    retained = fit_joint(songs, prior, listener, [unexposed], Intent(), previous=updated)
    np.testing.assert_array_equal(retained.location, updated.location)
    np.testing.assert_array_equal(retained.factor, updated.factor)
    assert retained.report["updated"] is False


def test_language_grounding_preserves_exposure_and_rejects_unknown_coordinates():
    songs, prior, listener = small_catalog()
    event = {"kind": "critique", "song_id": "0", "text": "too intense here", "exposed": True, "propensity": 0.2}
    grounded = {"targets": {"arousal": 0.3}, "excluded_artists": [], "intent_confidence": 0.9, "intent_notes": "lower intensity",
                "critiques": [{"index": 0, "feature": "arousal", "polarity": -1, "magnitude": 0.7,
                               "confidence": 0.8, "region_probabilities": {"0/global": 1}, "descriptor_reports": {"arousal": 0.8}, "notes": "global"}]}
    def transport(payload):
        assert "0/global" in payload["messages"][0]["content"]
        return {"choices": [{"message": {"content": json.dumps(grounded)}}]}
    intent, feedback, evidence = ground_configuration({"request": "something calmer"}, [event], songs,
                                                     listener.feature_names, model="fixture", transport=transport)
    assert intent["targets"] == {"arousal": 0.3}
    assert feedback[0]["propensity"] == 0.2 and feedback[0]["exposed"] is True
    assert event.get("feature") is None
    assert evidence["execution"] == "test_transport"
    grounded["targets"] = {"invented": 0.1}
    with pytest.raises(ValueError, match="unknown"):
        ground_configuration({"request": "something calmer"}, [event], songs, listener.feature_names, model="fixture", transport=transport)


def test_audio_training_joint_inference_generation_and_validation(tmp_path: Path):
    source = tmp_path / "source.wav"
    _source_recording(source)
    root = process_audio(source, tmp_path / "audio", recording_id="synthetic-neural",
                         separation_backend=ExactFourStemBackend(), structure_backend=FixedFourSecondStructureBackend())
    songs, prior, evidence = make_catalog([{"recording_dir": str(root)}])
    from hearthis.model.taste import Listener
    listener = Listener.from_centers("synthetic", prior.music_names + prior.response_names, [[0.5] * 9])
    decoder = train_waveform(songs, evidence, tmp_path / "decoder", epochs=6, samples=128, latent_dimension=3, hidden=16, max_clips=24)
    data = json.loads(decoder.read_text())
    assert data["training"]["final_train_negative_elbo"] < data["training"]["initial_train_negative_elbo"]
    model = WaveformModel.load(decoder)
    a = model.generate(songs[0].global_region.music, duration_seconds=0.1, seed=3)
    b = model.generate(songs[0].global_region.music, duration_seconds=0.1, seed=3)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (800,) and np.isfinite(a).all() and a.std() > 0
    alternatives = segmentation_options(songs[0])
    assert np.all(alternatives[:, :, 1] > alternatives[:, :, 0])
    manifest = run_model(songs, prior, listener, Intent(request="arousal=0.6"), tmp_path / "run", provenance=evidence,
                         draws=16, particles=32, ideal_count=2, explain_count=1,
                         joint={"iterations": 30, "draws": 16, "rank": 2, "batch_size": 32},
                         waveform_path=decoder, generation={"duration_seconds": 0.1})
    assert validate_model_run(manifest)["valid"]
    generated = json.loads((manifest.parent / "generated-audio.json").read_text())
    assert len(generated["waveforms"]) == 2
    assert (manifest.parent / "joint-posterior.json").exists()
    saved_joint = json.loads((manifest.parent / "joint-posterior.json").read_text())
    assert saved_joint["waveform_observations"] == "all-complete-windows"
    assert saved_joint["report"]["waveform_windows"] > data["training"]["training_windows"]
    from scipy.io import wavfile
    path = Path(generated["waveforms"][0]["path"])
    wavfile.write(path, 8000, np.zeros(800, dtype=np.float32))
    with pytest.raises(ValueError, match="waveform hash mismatch"):
        validate_model_run(manifest)


def test_validation_failure_does_not_publish_success_manifest(tmp_path, monkeypatch):
    import hearthis.pipeline as pipeline
    songs, prior, listener = small_catalog()
    def reject(_):
        raise ValueError("injected validation failure")
    monkeypatch.setattr(pipeline, "validate_model_run", reject)
    with pytest.raises(ValueError, match="injected"):
        run_model(songs, prior, listener, Intent(), tmp_path / "failed", draws=8, particles=16,
                  ideal_count=1, explain_count=0)
    assert not (tmp_path / "failed" / "manifest.json").exists()
    assert not (tmp_path / "failed" / ".manifest.pending.json").exists()
