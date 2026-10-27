from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from hearthis.audio.ingest import sha256_file
from hearthis.bundle import export_bundle
from hearthis.model.evaluation import annotation_metrics, preference_metrics, split_rhat
from hearthis.model.features import BROAD_MUSIC_NAMES
from hearthis.model.modalities import ObservationKernel, attach_observations
from hearthis.model.response import SharedResponsePrior
from hearthis.model.taste import Intent, Listener
from hearthis.pipeline import make_catalog, process_audio, run_model, write_json
from hearthis.service import api_server, check_configuration, execute_job, validate_result
from test_end_to_end import ExactFourStemBackend, FixedFourSecondStructureBackend, _source_recording
from test_taste_model import small_catalog


@pytest.fixture(scope="module")
def audio_model(tmp_path_factory):
    pytest.importorskip("torch")
    from hearthis.model.waveform import train_waveform
    root = tmp_path_factory.mktemp("completion-audio")
    source = root / "source.wav"
    _source_recording(source)
    recording = process_audio(source, root / "audio", recording_id="fixture",
                              separation_backend=ExactFourStemBackend(), structure_backend=FixedFourSecondStructureBackend())
    songs, prior, provenance = make_catalog([{"recording_dir": str(recording)}])
    model = train_waveform(songs, provenance, root / "decoder", epochs=6, samples=128, hidden=16, latent_dimension=3, max_clips=24)
    return root, source, recording, songs, prior, provenance, model


def test_broad_features_keep_missing_semantics_uncertain(audio_model, tmp_path):
    _, _, recording, _, _, _, _ = audio_model
    observation = tmp_path / "observations.json"
    write_json(observation, {"schema": "hearthis.song-observations/v1", "song_id": "fixture", "provenance": "synthetic fixture",
                             "lyrics": {"text": "soft light", "segments": [{"start_seconds": 0, "end_seconds": 4, "text": "soft light"}]},
                             "metadata_tags": ["synthetic"], "semantic_reports": [
                                 {"region_id": "fixture/global/0", "feature": "lyric_valence", "value": 0.8,
                                  "noise_sd": 0.02, "source": "synthetic"}]})
    songs, prior, provenance = make_catalog([{"recording_dir": str(recording), "observations_path": str(observation)}], broad_features=True)
    assert prior.music_names == BROAD_MUSIC_NAMES
    assert songs[0].global_region.music[-1] == 0.5
    assert songs[0].global_region.music_logit_variance[-1] == 2
    assert provenance[0]["references"]["observations"]["sha256"] == sha256_file(observation)
    kernel = ObservationKernel(songs, prior.music_names)
    logits = kernel.model["tokens"](kernel.torch.zeros((1, len(prior.music_names))))
    assert float(logits.softmax(-1).sum().detach()) == pytest.approx(1)
    synthetic = kernel.sample(songs[0].global_region.music)
    assert set(synthetic["metadata_tags"]) <= {"synthetic"}
    assert "not a source transcript" in synthetic["provenance"]
    from hearthis.model.joint import fit_joint, JointPosterior
    listener = Listener.from_centers("fixture", prior.music_names + prior.response_names, [[0.5] * 21])
    fitted = fit_joint(songs, prior, listener, [], Intent(), iterations=25, draws=16, rank=2, seed=2)
    assert fitted.report["lyric_metadata_likelihood"]
    restored = JointPosterior.from_dict(fitted.to_dict(), songs, prior, listener)
    fitted.project(songs, prior, listener)
    assert songs[0].global_region.music[-2] > 0.55
    assert restored.model.observation_kernel.vocabulary == fitted.model.observation_kernel.vocabulary


def test_joint_neural_weights_are_persisted_and_used(audio_model, tmp_path):
    _, _, _, songs, prior, provenance, model_path = audio_model
    listener = Listener.from_centers("fixture", prior.music_names + prior.response_names, [[0.5] * 9])
    manifest = run_model(songs, prior, listener, Intent(), tmp_path / "joint", provenance=provenance,
                         draws=8, particles=32, ideal_count=1, explain_count=0,
                         joint={"iterations": 12, "draws": 4, "rank": 2, "batch_size": 8},
                         waveform_path=model_path, generation={"duration_seconds": 0.05})
    assert validate_result(manifest)["valid"]
    joint = json.loads((manifest.parent / "joint-posterior.json").read_text())
    assert joint["report"]["neural_parameters_learned"]
    assert joint["report"]["taste_kernels_learned"]
    learned = json.loads(Path(joint["waveform_path"]).read_text())
    original = json.loads(model_path.read_text())
    assert learned["artifacts"]["checkpoint"]["sha256"] != original["artifacts"]["checkpoint"]["sha256"]
    generated = json.loads((manifest.parent / "generated-audio.json").read_text())
    assert generated["model"]["path"] == joint["waveform_path"]
    from hearthis.model.joint import JointPosterior
    from hearthis.model.validation import validate_model_run
    catalog = json.loads((manifest.parent / "catalog.json").read_text())
    from hearthis.model.response import Song
    saved_songs = [Song.from_dict(s) for s in catalog["songs"]]
    saved_prior = SharedResponsePrior(**catalog["shared_prior"])
    saved_listener = Listener.from_dict(json.loads((manifest.parent / "listener-posterior.json").read_text()))
    posterior = JointPosterior.from_dict(joint, saved_songs, saved_prior, saved_listener)
    white = posterior.model.tensor(posterior.sample(np.random.default_rng(4), 8))
    centers, weights, _, regions = posterior.model.parameters(white)
    features = posterior.model.features(centers, regions, saved_songs[0], Intent())[0]
    expected = saved_listener.utility(features.detach().numpy(), Intent(), centers=centers.detach().numpy(), weights=weights.detach().numpy())
    np.testing.assert_allclose(posterior.model.utility(features, centers, weights, Intent()).detach().numpy(), expected, atol=1e-5)
    assert validate_model_run(manifest)["valid"]


def test_autoregressive_generation_and_local_edit_invariants(audio_model, tmp_path):
    from hearthis.model.waveform import WaveformModel
    from hearthis.model.editing import edit_audio, validate_edit
    from hearthis.audio.separation import _audio_geometry
    _, source, _, songs, _, _, model_path = audio_model
    model = WaveformModel.load(model_path)
    assert model.config["autoregressive"]
    music = songs[0].global_region.music
    target = np.clip(music + 0.1, 0, 1)
    baseline = model.generate(music, duration_seconds=0.1, seed=2)
    changed = model.generate(music, duration_seconds=0.1, seed=2, sections=[
        {"start_seconds": 0, "end_seconds": 0.05, "music": music.tolist()},
        {"start_seconds": 0.05, "end_seconds": 0.1, "music": target.tolist()}])
    np.testing.assert_array_equal(baseline[:128], changed[:128])
    assert not np.array_equal(baseline[-128:], changed[-128:])
    manifest = edit_audio(source, model_path, tmp_path / "edit", music=music, edits=[
        {"start_seconds": 1, "end_seconds": 2, "music": target.tolist()}])
    assert validate_edit(manifest)["outside_intervals_unchanged"]
    rate, original = _audio_geometry(source)
    edited = _audio_geometry(manifest.parent / "edited.wav")[1]
    assert np.any(edited[rate:2 * rate] != original[rate:2 * rate])
    noop = edit_audio(source, model_path, tmp_path / "noop", music=music, edits=[
        {"start_seconds": 1, "end_seconds": 2, "music": music.tolist()}])
    np.testing.assert_array_equal(_audio_geometry(noop.parent / "edited.wav")[1], original)
    with pytest.raises(ValueError, match="ordered"):
        edit_audio(source, model_path, tmp_path / "bad", music=music, edits=[
            {"start_seconds": -1, "end_seconds": 2, "music": music.tolist()}])


def test_evaluation_metrics_and_held_out_ledger(tmp_path):
    sections = [{"start_seconds": 0, "end_seconds": 1, "label": "intro"},
                {"start_seconds": 1, "end_seconds": 2, "label": "verse"}]
    metrics = annotation_metrics(sections, sections)
    assert metrics["boundary_f1"] == metrics["duration_weighted_label_accuracy"] == 1
    assert split_rhat(np.random.default_rng(2).normal(size=(4, 1000))) < 1.02
    assert split_rhat(np.repeat(np.arange(4)[:, None], 10, axis=1)) is None
    songs, prior, listener = small_catalog()
    listener.observed_event_ids = ["training"]
    manifest = run_model(songs, prior, listener, Intent(), tmp_path / "run", draws=16, particles=32, ideal_count=1, explain_count=0)
    events = [{"event_id": "held-out", "kind": "pairwise", "song_id": "0", "other_song_id": "1", "exposed": True,
               "propensity": 0.5, "value": 1, "provenance": "synthetic evaluation fixture"}]
    report = preference_metrics(manifest, events, draws=64)
    p = report["predictions"][0]["probability"]
    assert report["brier_score"] == pytest.approx((1 - p) ** 2)
    assert report["log_loss"] == pytest.approx(-np.log(p))
    events[0]["event_id"] = "training"
    with pytest.raises(ValueError, match="held out"):
        preference_metrics(manifest, events)
    with pytest.raises(ValueError, match="previously unobserved"):
        run_model(songs, prior, listener, Intent(), tmp_path / "duplicate", feedback=events)


def test_large_feature_explanation_preserves_efficiency():
    from hearthis.model.probability import FeaturePrior
    from hearthis.model.recommend import explain_song
    names = tuple(f"feature-{i}" for i in range(14))
    listener = Listener.from_centers("synthetic", names, [[0.5] * 14])
    rng = np.random.default_rng(1)
    centers, weights = listener.draws(rng, 16)
    features = rng.uniform(0.1, 0.9, (16, 14))
    prior = FeaturePrior(np.full((2, 14), 0.5), np.full((2, 14), 0.15))
    result = explain_song(features, listener, Intent(), prior, centers, weights, permutations=8)
    assert result["permutations"] == 8
    assert sum(item["conditional_shapley"] for item in result["features"].values()) == pytest.approx(result["utility_difference"])


def test_bundle_rebases_nested_and_detached_hashes(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    audio = root / "audio.bin"
    audio.write_bytes(b"audio")
    base = root / "base.json"
    write_json(base, {"audio": {"path": str(audio), "sha256": sha256_file(audio)}})
    annotation = root / "annotation.json"
    write_json(annotation, {"base_manifest_sha256": sha256_file(base)})
    entry = root / "entry.json"
    write_json(entry, {"base_manifest_path": str(base), "base_manifest_sha256": sha256_file(base),
                       "annotation": {"path": str(annotation), "sha256": sha256_file(annotation)}})
    output = tmp_path / "bundle"
    export_bundle(entry, root, output, Path("/catalog"))
    stored = json.loads((output / "entry.json").read_text())
    assert stored["base_manifest_path"] == "/catalog/base.json"
    assert stored["base_manifest_sha256"] == sha256_file(output / "base.json")
    assert json.loads((output / "annotation.json").read_text())["base_manifest_sha256"] == sha256_file(output / "base.json")
    assert stored["annotation"]["sha256"] == sha256_file(output / "annotation.json")


def test_http_auth_path_and_idempotency_contract(tmp_path):
    from hearthis.jobs import JobConflict
    class Queue:
        def ready(self): return True
        def get(self, identifier): return None
        def submit(self, kind, configuration, idempotency_key=None):
            if idempotency_key == "conflict": raise JobConflict("conflict")
            return {"id": "fixture", "kind": kind, "state": "queued"}
    artifact = tmp_path / "model.json"
    artifact.write_text("{}")
    token = "test-token-abcdefghijklmnopqrstuvwxyz"
    server = api_server(Queue(), [tmp_path], token, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert json.load(urllib.request.urlopen(url + "/readyz"))["ready"]
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(url + "/jobs/id")
        assert denied.value.code == 401
        payload = {"kind": "generate", "configuration": {"model_path": str(artifact), "music": [0.5], "duration_seconds": 1}}
        request = urllib.request.Request(url + "/jobs", data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + token})
        assert urllib.request.urlopen(request).status == 202
        request.add_header("Idempotency-Key", "conflict")
        with pytest.raises(urllib.error.HTTPError) as conflict:
            urllib.request.urlopen(request)
        assert conflict.value.code == 409
        payload["configuration"]["model_path"] = "/etc/passwd"
        with pytest.raises(ValueError, match="outside"):
            check_configuration("generate", payload["configuration"], [tmp_path])
        with pytest.raises(ValueError, match="workers own"):
            check_configuration("run", {"joint": {"neural_output_dir": str(tmp_path)}}, [tmp_path])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_generation_job_publishes_only_valid_artifacts(audio_model, tmp_path):
    _, _, _, songs, _, _, model = audio_model
    config = tmp_path / "generation.json"
    write_json(config, {"model_path": str(model), "music": songs[0].global_region.music.tolist(), "duration_seconds": 0.05})
    manifest = execute_job("generate", config, tmp_path / "result")
    assert validate_result(manifest)["valid"]
    data = json.loads(manifest.read_text())
    Path(data["waveforms"][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        validate_result(manifest)


def test_external_corrections_are_read_only_inputs(audio_model, tmp_path):
    from hearthis.pipeline import run_configuration
    _, _, recording, _, _, _, _ = audio_model
    root = tmp_path / "input"
    root.mkdir()
    correction = root / "correction.json"
    write_json(correction, {"schema": "hearthis.structure-correction/v1", "annotation_source": "human",
                            "annotator": "fixture", "created_at": "2026-10-27T00:00:00Z",
                            "base_manifest_sha256": sha256_file(recording / "structure/manifest.json"),
                            "sections": [{"start_seconds": 0, "end_seconds": 4, "label": "instrumental"}]})
    original_hash = sha256_file(correction)
    config = tmp_path / "configuration.json"
    write_json(config, {"recordings": [{"recording_dir": str(recording), "listening_correction": str(correction)}],
                        "intent": {"horizon_seconds": 4},
                        "sampling": {"draws": 8, "particles": 32, "ideal_count": 1, "explain_count": 0}})
    manifest = run_configuration(config, tmp_path / "run")
    assert validate_result(manifest)["valid"]
    assert sha256_file(correction) == original_hash
    assert sorted(path.name for path in root.iterdir()) == ["correction.json"]
    assert (manifest.parent / "corrected/fixture/structure/manifest.json").exists()


def test_waveform_evaluation_rejects_training_and_accepts_distinct_sources(audio_model, tmp_path):
    from hearthis.model.evaluation import held_out_waveform_metrics
    from scipy.io import wavfile
    _, source, _, songs, prior, provenance, model = audio_model
    catalog = tmp_path / "training-catalog.json"
    write_json(catalog, {"songs": [song.to_dict() for song in songs], "shared_prior": prior.to_dict(), "provenance": provenance})
    with pytest.raises(ValueError, match="overlap"):
        held_out_waveform_metrics(model, catalog)
    rate, audio = wavfile.read(source)
    independent = tmp_path / "independent.wav"
    wavfile.write(independent, rate, np.roll(audio, 100, axis=0))
    recording = process_audio(independent, tmp_path / "audio", recording_id="held-out-fixture",
                              separation_backend=ExactFourStemBackend(), structure_backend=FixedFourSecondStructureBackend())
    held_out_songs, held_out_prior, evidence = make_catalog([{"recording_dir": str(recording)}])
    catalog = tmp_path / "held-out-catalog.json"
    write_json(catalog, {"songs": [song.to_dict() for song in held_out_songs], "shared_prior": held_out_prior.to_dict(), "provenance": evidence})
    report = held_out_waveform_metrics(model, catalog, draws=2)
    assert report["recordings"] == 1 and report["windows"] > 0
    assert np.isfinite(report["mean_window_elbo"])


def test_generated_modalities_use_the_matching_joint_run(audio_model, tmp_path):
    from hearthis.model.response import Song
    _, _, _, originals, prior, evidence, decoder = audio_model
    songs = [Song.from_dict(song.to_dict()) for song in originals]
    songs[0].observations = {"schema": "hearthis.song-observations/v1", "song_id": "fixture", "provenance": "synthetic fixture",
                             "lyrics": {"text": "soft light"}, "metadata_tags": ["synthetic"]}
    listener = Listener.from_centers("fixture", prior.music_names + prior.response_names, [[0.5] * 9])
    manifest = run_model(songs, prior, listener, Intent(), tmp_path / "model", provenance=evidence,
                         draws=8, particles=32, ideal_count=1, explain_count=0, waveform_path=decoder,
                         joint={"iterations": 8, "draws": 4, "rank": 2, "batch_size": 8}, generation={"duration_seconds": 0.05})
    generated = json.loads((manifest.parent / "generated-audio.json").read_text())
    assert "synthetic" in generated["generated_modalities"][0]["provenance"]
    config = tmp_path / "generation.json"
    write_json(config, {"model_path": generated["model"]["path"], "model_run_path": str(manifest),
                        "music": songs[0].global_region.music.tolist(), "duration_seconds": 0.05})
    result = execute_job("generate", config, tmp_path / "result")
    assert validate_result(result)["valid"]
    assert "generated_modalities" in json.loads(result.read_text())
