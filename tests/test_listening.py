from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import wave

import numpy as np
import pytest
from scipy.io import wavfile

from hearthis.audio.ingest import sha256_file
from hearthis.audio.listening import ListeningReviewError, review_audio


@pytest.fixture
def recording(tmp_path):
    root = tmp_path / "recording"
    (root / "structure").mkdir(parents=True)
    source = root / "mixture.wav"
    wavfile.write(source, 8000, np.round(16000 * np.sin(np.arange(32000) * 0.1)).astype(np.int16))
    (root / "manifest.json").write_text(json.dumps({"canonical_audio": {
        "path": str(source), "sha256": sha256_file(source)}}))
    (root / "structure" / "manifest.json").write_text('{"label": "secret-classifier-label"}')
    return root


def reply(end=4.0):
    mode = {"weight": 1, "descriptors": {"arousal": 0.6, "intimacy": 0.3, "tension": 0.4}, "confidence": 0.7}
    return {"sections": [{"start_seconds": 0, "end_seconds": end, "label": "instrumental_section",
                          "confidence": 0.7, "notes": "Audible repeating tone", "descriptor_modes": [mode]}],
            "descriptor_modes": [mode], "audio_observations": "A repeating tone."}


def response(review):
    return {"model": "fixture", "choices": [{"message": {"content": "```json\n" + json.dumps(review) + "\n```"}}]}


def test_listening_submits_waveform_blind_and_marks_test_provenance(recording, tmp_path):
    def transport(payload):
        content = payload["messages"][0]["content"]
        assert "secret-classifier-label" not in json.dumps(payload)
        audio = base64.b64decode(content[1]["input_audio"]["data"])
        with wave.open(io.BytesIO(audio)) as decoded:
            assert decoded.getframerate() == 16000
            assert decoded.getnframes() == 64000
            assert decoded.getnchannels() == 1
            assert np.frombuffer(decoded.readframes(64000), dtype="<i2").std() > 1000
        return response(reply(3.9997))
    output = tmp_path / "review"
    correction = review_audio(recording, output, model="fixture", transport=transport)
    annotation = json.loads(correction.read_text())
    evidence = json.loads((output / "evidence.json").read_text())
    assert annotation["annotation_source"] == "llm"
    assert annotation["provenance"]["audio_listened"] is False
    assert evidence["execution"] == "test_transport"
    assert evidence["endpoint_adjustment"]["reported_end_seconds"] == 3.9997
    assert annotation["sections"][-1]["end_seconds"] == 4
    assert sha256_file(output / "input.wav") == evidence["input_audio"]["sha256"]


@pytest.mark.parametrize("bad", ["gap", "duration", "confidence", "descriptor"])
def test_invalid_review_never_publishes_annotation(recording, tmp_path, bad):
    review = reply()
    if bad == "gap":
        review["sections"][0]["start_seconds"] = 1
    elif bad == "duration":
        review["sections"][0]["end_seconds"] = 3.9
    elif bad == "confidence":
        review["sections"][0]["confidence"] = float("nan")
    else:
        review["descriptor_modes"][0]["weight"] = 0.5
    output = tmp_path / "bad"
    with pytest.raises(ListeningReviewError):
        review_audio(recording, output, model="fixture", transport=lambda _: response(review))
    assert not output.exists()


def test_audio_hash_mismatch_prevents_provider_call(recording, tmp_path):
    (recording / "mixture.wav").write_bytes(b"corrupt")
    with pytest.raises(ListeningReviewError, match="hash mismatch"):
        review_audio(recording, tmp_path / "bad", model="fixture", transport=lambda _: pytest.fail("provider called"))


@pytest.mark.parametrize("supported", [True, False])
def test_semantic_listening_uses_defined_vocabulary_and_abstention(recording, tmp_path, supported):
    review = reply()
    review["semantic_reports"] = [{"feature": "harmonic_complexity" if supported else "invented",
                                   "value": 0.2, "confidence": 0.7, "notes": "Synthetic repeating tone fixture"}]
    def transport(payload):
        prompt = payload["messages"][0]["content"][0]["text"]
        assert "0 simple repetitive harmony" in prompt
        assert "secret-classifier-label" not in json.dumps(payload)
        return response(review)
    output = tmp_path / "semantic-review"
    if supported:
        correction = review_audio(recording, output, model="fixture", transport=transport,
                                  semantic_features={"harmonic_complexity": "0 simple repetitive harmony, 1 complex harmony"})
        evidence = json.loads((correction.parent / "evidence.json").read_text())
        assert evidence["review"]["semantic_reports"][0]["value"] == 0.2
        assert evidence["execution"] == "test_transport"
    else:
        with pytest.raises(ListeningReviewError, match="semantic"):
            review_audio(recording, output, model="fixture", transport=transport,
                         semantic_features={"harmonic_complexity": "0 simple repetitive harmony, 1 complex harmony"})
        assert not output.exists()
