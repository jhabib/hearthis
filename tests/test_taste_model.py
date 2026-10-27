from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logit

from hearthis.audio.ingest import sha256_file
from hearthis.model.probability import FeaturePrior, GaussianPosterior, probabilities
from hearthis.model.recommend import explain_song, rank_catalog, sample_ideal_songs, sample_playlist
from hearthis.model.response import Region, ResponseProfile, SharedResponsePrior, Song
from hearthis.model.taste import Intent, Listener, TasteComponent, fit_feedback
from hearthis.pipeline import make_catalog, process_audio, run_model
from hearthis.model.response import condition_behavior_profiles
from hearthis.model.validation import validate_model_run
from test_end_to_end import ExactFourStemBackend, FixedFourSecondStructureBackend, _source_recording


def small_catalog():
    prior = SharedResponsePrior.uninformed(("brightness",), ("arousal",))
    songs = [Song(str(i), f"artist-{i}", f"song-{i}", 60,
                  [Region(f"{i}/global", "global", 0, 60, np.array([x]), prior.profile(np.array([x])))])
             for i, x in enumerate([0.2, 0.8])]
    listener = Listener.from_centers("synthetic-test", ("brightness", "arousal"), [[0.25, 0.5], [0.7, 0.7]])
    return songs, prior, listener


def test_paper_worked_example_reproduces_all_energies_and_odds():
    names = ("B", "G", "NC", "NL", "V", "P", "E")
    diamonds = np.array([0.45, 0.80, 0.80, 0.55, 0.75, 0.95, 0.20])
    after_love = np.array([0.65, 0.85, 0.75, 0.75, 0.85, 0.25, 0.95])
    listeners = [Listener.from_centers("A", names, [[0.70, 0.90, 0.90, 0.80, 0.90, 0.90, 0.20]]),
                 Listener.from_centers("B", names, [[0.70, 0.90, 0.90, 0.80, 0.90, 0.20, 0.95]])]
    energies = np.array([[-listener.utility(song, Intent()) for song in [diamonds, after_love]]
                         for listener in listeners])
    np.testing.assert_allclose(energies, [[0.085, 0.50875], [0.64625, 0.0175]], atol=1e-12)
    odds_a = np.exp(4 * (energies[0, 1] - energies[0, 0]))
    odds_b = np.exp(4 * (energies[1, 0] - energies[1, 1]))
    assert odds_a == pytest.approx(5.45, abs=0.01)
    assert odds_b == pytest.approx(12.37, abs=0.01)


def test_exact_conjugate_update_preserves_unobserved_personalization():
    covariance = np.eye(2)
    a = GaussianPosterior(np.array([0.0, 1.0]), covariance)
    b = GaussianPosterior(np.array([0.0, -1.0]), covariance)
    design, noise, observation = np.array([[1.0, 0.0]]), np.array([[0.25]]), np.array([0.8])
    posterior_a, posterior_b = a.update(design, observation, noise), b.update(design, observation, noise)
    np.testing.assert_allclose(posterior_a.mean, [0.64, 1.0])
    np.testing.assert_allclose(posterior_a.covariance, [[0.2, 0], [0, 1]])
    gain = covariance @ design.T @ np.linalg.inv(design @ covariance @ design.T + noise)
    np.testing.assert_allclose(posterior_a.mean - posterior_b.mean,
                               (np.eye(2) - gain @ design) @ (a.mean - b.mean))
    assert posterior_a.mean[1] != posterior_b.mean[1]


def test_interactions_have_exact_half_factor_and_scale_gauge():
    component = TasteComponent("interactions", GaussianPosterior(logit([0.5, 0.5]), np.eye(2)),
                               GaussianPosterior(np.zeros(2), np.eye(2)), np.ones(2),
                               np.array([[0, 2], [2, 0]]))
    features = np.array([0.2, 0.8])
    expected = (-0.5 * np.sum((features - 0.5) ** 2) + 2 * features[0] * features[1]) / 3
    assert component.energy(features) == pytest.approx(expected)
    normalized_precision = component.precision / component.scale(component.weights.mean)
    normalized_interactions = component.interactions / component.scale(component.weights.mean)
    assert normalized_precision.mean() ** 2 + np.sum(normalized_interactions ** 2) == pytest.approx(1)
    with pytest.raises(ValueError, match="zero-diagonal"):
        TasteComponent("invalid", component.state, component.weights, np.ones(2), np.eye(2))


def test_soft_negative_region_lowers_utility():
    _, _, listener = small_catalog()
    component = listener.components[0]
    before = component.energy(np.array([0.8, 0.8]))
    component.negative_design = np.array([[1.0, 0]])
    component.negative_offset = np.array([-0.4])
    component.negative_strength = np.array([2.0])
    assert component.energy(np.array([0.8, 0.8])) < before


def test_response_mode_permutation_preserves_predictive_distribution():
    alpha, beta = np.array([[2.0], [8.0]]), np.array([[8.0], [2.0]])
    original = ResponseProfile(np.array([2.0, 8.0]), alpha, beta)
    permuted = ResponseProfile(np.array([8.0, 2.0]), alpha[::-1].copy(), beta[::-1].copy())
    count = 20000
    center = np.full((count, 1), 0.5)
    a, _ = original.predict(np.random.default_rng(1), count, center, center)
    b, _ = permuted.predict(np.random.default_rng(2), count, center, center)
    assert a.mean() == pytest.approx(b.mean(), abs=0.01)
    assert a.var() == pytest.approx(b.var(), abs=0.01)
    assert np.all((a >= 0) & (a <= 1))


def test_grounded_response_updates_only_observed_coordinates():
    profile = ResponseProfile(np.ones(2), np.ones((2, 2)), np.ones((2, 2)))
    profile.update({0: 0.9}, confidence=0.8)
    assert profile.mode_means[:, 0].mean() > 0.5
    np.testing.assert_array_equal(profile.mode_means[:, 1], [0.5, 0.5])
    assert profile.concentration.sum() > 2


def test_shared_descriptor_encoder_learns_declared_supervision():
    prior = SharedResponsePrior.uninformed(("brightness",), ("arousal",))
    records = [{"mode": 0, "music": [x], "descriptors": [x], "confidence": 1.0}
               for x in [0.1, 0.2, 0.8, 0.9]]
    report = prior.fit(records, regularization=0.05)
    assert report["observations"] == 4
    assert prior.profile(np.array([0.9])).mode_means[0, 0] > prior.profile(np.array([0.1])).mode_means[0, 0]


def test_artist_exclusions_change_ranking_without_changing_target():
    songs, prior, listener = small_catalog()
    original, _ = rank_catalog(songs, prior, listener, Intent(), seed=2, draws=64)
    filtered, _ = rank_catalog(songs, prior, listener, Intent(excluded_artists=["artist-0"]), seed=2, draws=64)
    original_by_id, filtered_by_id = ({r["song_id"]: r for r in rows} for rows in [original, filtered])
    for song_id in original_by_id:
        assert original_by_id[song_id]["catalog_target_probability"] == filtered_by_id[song_id]["catalog_target_probability"]
    assert filtered_by_id["0"]["score"] is None
    assert filtered_by_id["0"]["ranking_probability"] == 0
    assert filtered_by_id["0"]["catalog_target_probability"] > 0


def test_risk_uses_posterior_standard_deviation_with_correct_sign():
    songs, prior, listener = small_catalog()
    conservative, _ = rank_catalog(songs, prior, listener, Intent(risk=2), seed=1, draws=128)
    exploratory, _ = rank_catalog(songs, prior, listener, Intent(risk=-2), seed=1, draws=128)
    for rows, risk in [(conservative, 2), (exploratory, -2)]:
        for row in rows:
            assert row["score"] == pytest.approx(4 * row["expected_utility"] - risk * 4 * row["utility_sd"])


def test_ideal_song_sampler_is_bounded_feasible_and_ignores_artist_filters():
    _, prior, listener = small_catalog()
    base = FeaturePrior(np.array([[0.3], [0.7]]), np.full((2, 1), 0.15))
    intent = Intent(minimum_features={"brightness": 0.2}, maximum_features={"brightness": 0.8})
    a = sample_ideal_songs(prior, base, listener, intent, seed=0, particles=256, count=3)
    b = sample_ideal_songs(prior, base, listener,
                           Intent(**{**intent.__dict__, "excluded_artists": ["artist-0"]}),
                           seed=0, particles=256, count=3)
    assert a == b
    for row in a["samples"]:
        assert 0.2 <= row["augmented_features"][0] <= 0.8
        assert np.all((np.array(row["augmented_features"]) >= 0) & (np.array(row["augmented_features"]) <= 1))
    for diagnostic in a["diagnostics"]:
        assert np.isfinite(diagnostic["log_normalizer_estimate"])
        assert 1 <= diagnostic["effective_sample_size"] <= 256


def test_zero_feasible_mass_is_rejected():
    with pytest.raises(ValueError, match="no positive feasible mass"):
        probabilities(np.array([-np.inf, -np.inf]))
    _, prior, listener = small_catalog()
    base = FeaturePrior(np.array([[0.3]]), np.full((1, 1), 0.1))
    with pytest.raises(ValueError, match="no positive feasible mass"):
        sample_ideal_songs(prior, base, listener, Intent(minimum_features={"brightness": 1}), particles=32, count=1)


def test_conditional_prior_uses_retained_coordinate_to_select_modes():
    prior = FeaturePrior(np.array([[0.1, 0.1], [0.9, 0.9]]), np.full((2, 2), 0.04))
    samples = prior.sample(np.random.default_rng(0), 1000, retained={0: 0.9})
    np.testing.assert_array_equal(samples[:, 0], np.full(1000, 0.9))
    assert samples[:, 1].mean() > 0.85


def test_conditional_shapley_values_sum_to_posterior_utility_difference():
    songs, prior, listener = small_catalog()
    ranking, posterior = rank_catalog(songs, prior, listener, Intent(), seed=0, draws=64)
    base = FeaturePrior(np.array([r["features_mean"] for r in ranking]), np.full((2, 2), 0.15))
    features = posterior["features"]["0"]
    explanation = explain_song(features, listener, Intent(), base, posterior["centers"], posterior["weights"])
    assert explanation["shapley_total"] == pytest.approx(explanation["utility_difference"], abs=1e-10)
    expected = next(r["expected_utility"] for r in ranking if r["song_id"] == "0")
    assert explanation["candidate_expected_utility"] == pytest.approx(expected)


def test_temporal_change_point_increases_state_uncertainty():
    _, _, listener = small_catalog()
    original = listener.components[0].state.covariance.copy()
    innovation = listener.components[0].innovation.copy()
    assert listener.advance(np.random.default_rng(0), change_point=True)
    np.testing.assert_allclose(listener.components[0].state.covariance, original + 10 * innovation)
    assert listener.step == 1


def test_variational_feedback_improves_elbo_and_pairwise_preference():
    names = ("brightness", "arousal")
    listener = Listener.from_centers("synthetic", names, [[0.5, 0.5]], variance=0.4)
    features = {"low": np.array([0.1, 0.5]), "high": np.array([0.9, 0.5])}
    before = listener.utility(features["high"], Intent()) - listener.utility(features["low"], Intent())
    events = [{"kind": "pairwise", "song_id": "high", "other_song_id": "low", "value": 1,
               "exposed": True, "propensity": 0.5} for _ in range(12)]
    report = fit_feedback(listener, events, features, Intent(), seed=0, draws=32, iterations=80)
    after = listener.utility(features["high"], Intent()) - listener.utility(features["low"], Intent())
    assert report["final_elbo"] > report["initial_elbo"]
    assert after > before


def test_unexposed_song_does_not_become_negative_evidence():
    _, _, listener = small_catalog()
    before = listener.to_dict()
    event = {"kind": "behavior", "song_id": "not-in-catalog", "exposed": False, "propensity": 0.5}
    report = fit_feedback(listener, [event], {}, Intent())
    assert report["updated"] is False
    assert listener.to_dict() == before
    with pytest.raises(ValueError, match="unexposed song"):
        fit_feedback(listener, [{**event, "value": 0}], {}, Intent())


def test_playlist_uses_horizon_and_respects_exclusions():
    songs, prior, listener = small_catalog()
    intent = Intent(horizon_seconds=121, excluded_artists=["artist-0"])
    ranking, _ = rank_catalog(songs, prior, listener, intent, seed=0, draws=16)
    playlist = sample_playlist(ranking, intent, seed=0, sweeps=10)
    assert playlist["item_count"] == 3
    assert playlist["songs"] == ["1", "1", "1"]


def test_behavior_conditions_response_posterior_and_requires_exposure():
    songs, prior, listener = small_catalog()
    before = songs[0].global_region.profile.descriptor_alpha.copy()
    event = {"kind": "behavior", "value": 1, "exposed": True, "propensity": 0.5,
             "response_weights": [0, 15]}
    result = condition_behavior_profiles(songs[0], prior, listener, Intent(), event, draws=4096)
    assert result["regions_updated"] == 1
    assert 1 <= result["mean_effective_sample_size"] <= 4096
    assert not np.allclose(before, songs[0].global_region.profile.descriptor_alpha)
    profile = songs[0].global_region.profile
    assert np.mean(profile.descriptor_alpha / (profile.descriptor_alpha + profile.descriptor_beta)) > 0.5
    with pytest.raises(ValueError, match="exposed"):
        condition_behavior_profiles(songs[0], prior, listener, Intent(), {**event, "exposed": False})


def test_saved_model_can_resume_and_validator_detects_tampering(tmp_path):
    from hearthis.pipeline import run_configuration
    songs, prior, listener = small_catalog()
    first = run_model(songs, prior, listener, Intent(), tmp_path / "first",
                      draws=16, particles=32, ideal_count=2, explain_count=0)
    config = tmp_path / "resume.json"
    config.write_text(json.dumps({"catalog_path": str(first.parent / "catalog.json"),
                                 "listener_path": str(first.parent / "listener-posterior.json"),
                                 "intent": {"request": "brightness=0.7"},
                                 "sampling": {"draws": 16, "particles": 32, "ideal_count": 2, "explain_count": 0}}))
    resumed = run_configuration(config, tmp_path / "resumed")
    manifest = Path(json.loads(resumed.read_text())["model"]["path"])
    assert validate_model_run(manifest)["valid"]
    assert json.loads((manifest.parent / "listener-posterior.json").read_text()) == listener.to_dict()
    ranking_path = manifest.parent / "ranking.json"
    ranking_path.write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_model_run(manifest)


def test_audio_to_mathematical_outputs_with_synthetic_backends(tmp_path: Path):
    source = tmp_path / "source.wav"
    _source_recording(source)
    root = process_audio(source, tmp_path / "audio", recording_id="synthetic-e2e",
                         separation_backend=ExactFourStemBackend(), structure_backend=FixedFourSecondStructureBackend())
    songs, prior, evidence = make_catalog([{"recording_dir": str(root)}])
    names = prior.music_names + prior.response_names
    listener = Listener.from_centers("synthetic-e2e", names, [[0.5] * len(names)], provenance="synthetic fixture")
    original = listener.to_dict()
    manifest = run_model(songs, prior, listener, Intent(request="vocal_share=0.7", horizon_seconds=5),
                         tmp_path / "model", provenance=evidence, draws=16, particles=64,
                         ideal_count=2, explain_count=1)
    assert listener.to_dict() == original
    data = json.loads(manifest.read_text())
    for artifact in data["artifacts"].values():
        assert sha256_file(Path(artifact["path"])) == artifact["sha256"]
    ranking = json.loads((manifest.parent / "ranking.json").read_text())
    assert ranking[0]["ranking_probability"] == 1
    assert ranking[0]["region_attention"]
    persistent = json.loads((manifest.parent / "listener-posterior.json").read_text())
    session = json.loads((manifest.parent / "listener-session.json").read_text())
    assert persistent == original
    assert session["components"][0]["weights"]["mean"] != original["components"][0]["weights"]["mean"]
    with pytest.raises(ValueError, match="new directory"):
        run_model(songs, prior, listener, Intent(), manifest.parent)
