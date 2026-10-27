"""Direct audio-input LLM review, with auditable bytes and blind annotations."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import wave

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .ingest import sha256_file
from .separation import _to_float32
from .validation import _contiguous
from ..providers import complete, response_json


class ListeningReviewError(RuntimeError):
    pass


def review_audio(recording_dir: Path, output_dir: Path, *, model: str,
                 endpoint: str | None = None, key_environment: str = "HEARTHIS_AUDIO_API_KEY",
                 provider: str = "https", transport=None, timeout: float = 180.0, semantic_features=None) -> Path:
    """Submit waveform audio, never draft labels or model probabilities.

    The transport is Chat-Completions-compatible input_audio. Real provider
    capability and authentication must be verified by a successful call; a text
    reply without an audio-capable provider is insufficient listening evidence.
    The callable transport seam permits byte-level integration tests.
    """
    if not model.strip():
        raise ValueError("an explicit audio-capable model is required")
    recording_dir, output_dir = recording_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise ValueError("listening review output must be a new directory")
    base = recording_dir / "structure" / "manifest.json"
    base_hash = sha256_file(base)
    ingestion = json.loads((recording_dir / "manifest.json").read_text())
    canonical = ingestion["canonical_audio"]
    source = Path(canonical["path"])
    if sha256_file(source) != canonical["sha256"]:
        raise ListeningReviewError("canonical audio hash mismatch")
    sample_rate, samples = wavfile.read(source)
    samples = _to_float32(samples)
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    if not mono.size or not np.isfinite(mono).all():
        raise ListeningReviewError("audio must contain finite nonempty samples")
    duration = len(mono) / sample_rate
    divisor = int(np.gcd(sample_rate, 16000))
    mono = resample_poly(mono, 16000 // divisor, sample_rate // divisor)
    pcm = np.round(np.clip(mono, -1, 1 - 1 / 32768) * 32768).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as encoded:
        encoded.setnchannels(1)
        encoded.setsampwidth(2)
        encoded.setframerate(16000)
        encoded.writeframes(pcm.tobytes())
    audio = buffer.getvalue()
    prompt = (
        f"Listen to the supplied complete music recording ({duration:.9f} seconds). "
        "You are an AI music reviewer. Analyze the audio directly, without claiming human authorship. "
        "Return ONLY a JSON object with sections, an array of contiguous positive intervals covering "
        "exactly 0 through the recording duration. Each section must contain start_seconds, end_seconds, "
        "label, confidence in [0,1], and notes describing audible evidence and uncertainty. "
        "Use broad instrumental or vocal_section labels when a functional verse/chorus/solo label "
        "cannot be supported by listening. Do not invent lyrics. Optionally add descriptor_modes to "
        "a section, an array of {weight, descriptors: {arousal, intimacy, tension}, confidence}. "
        "Each number must be in [0,1], mode weights must sum to 1, arousal means perceived activation, "
        "intimacy means perceived closeness, and tension means perceived unresolved musical pressure. "
        "These are subjective AI estimates, not measured listener reports. "
        "Also return audio_observations, a nonempty string describing audible instruments, voice, "
        "temporal changes and limitations. If you cannot access or analyze the audio, return an error "
        "object and no sections. No draft annotations or classifier outputs have been supplied."
    )
    if semantic_features is not None:
        if (not isinstance(semantic_features, dict) or not semantic_features
                or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in semantic_features.items())):
            raise ValueError("semantic features require explicit coordinate definitions")
        prompt += (" Optionally provide semantic_reports globally and within each section, as an array of "
                   "{feature, value, confidence, notes}. Use only the defined coordinates below and values/confidence "
                   "in [0,1]. Omit a coordinate when the audio does not support a judgment. These are uncertain AI "
                   "reports, not measurements or human labels. Definitions: " + json.dumps(semantic_features))
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "input_audio", "input_audio": {
            "data": base64.b64encode(audio).decode("ascii"), "format": "wav"}},
    ]}], "max_tokens": 8192}
    request_bytes = json.dumps(payload).encode("utf-8")
    request_hash = hashlib.sha256(request_bytes).hexdigest()
    response_cache = output_dir.with_suffix(".provider-response.json")
    if transport is None and response_cache.exists():
        cached = json.loads(response_cache.read_text())
        if cached["request_sha256"] != request_hash:
            raise ListeningReviewError("cached response belongs to a different audio review request")
        provider_response, endpoint = cached["response"], cached["endpoint"]
    else:
        try:
            provider_response, endpoint = complete(payload, provider=provider, endpoint=endpoint,
                                                    key_environment=key_environment, timeout=timeout,
                                                    transport=transport, purpose="hearthis audio listening review")
        except (RuntimeError, ValueError) as error:
            raise ListeningReviewError(str(error)) from error
    if transport is None and not response_cache.exists():
        response_cache.parent.mkdir(parents=True, exist_ok=True)
        response_cache.write_text(json.dumps({"request_sha256": request_hash, "endpoint": endpoint,
                                              "response": provider_response}, indent=2) + "\n")
    try:
        review = response_json(provider_response)
        sections = review["sections"]
        # Providers often print millisecond endpoints. Snap only this rounding
        # error, retain the raw reply, and document the adjustment explicitly.
        endpoint_adjustment = None
        if sections and abs(float(sections[-1]["end_seconds"]) - duration) <= 0.0005:
            reported_end = sections[-1]["end_seconds"]
            sections[-1]["end_seconds"] = duration
            endpoint_adjustment = {"reported_end_seconds": reported_end,
                                   "canonical_end_seconds": duration,
                                   "reason": "provider endpoint rounded to milliseconds"}
        if not review.get("audio_observations") or not _contiguous(sections, duration):
            raise ValueError("review requires audible observations and a complete positive timeline")
        correction_sections = []
        for group in [review, *sections]:
            for report in group.get("semantic_reports", []):
                if (semantic_features is None or report["feature"] not in semantic_features
                        or not np.isfinite([report["value"], report["confidence"]]).all()
                        or not 0 <= report["value"] <= 1 or not 0 < report["confidence"] <= 1
                        or not isinstance(report.get("notes"), str) or not report["notes"].strip()):
                    raise ValueError("invalid listening semantic report")
        for section in sections:
            confidence = float(section["confidence"])
            if not np.isfinite(confidence) or not 0 <= confidence <= 1 or not section["label"].strip() or not section["notes"].strip():
                raise ValueError("invalid listening section label, notes or confidence")
            correction_sections.append({key: section[key] for key in (
                "start_seconds", "end_seconds", "label", "confidence", "notes")})
            mode_groups = [section.get("descriptor_modes", [])]
            if section is sections[0]:
                mode_groups.append(review.get("descriptor_modes", []))
            for modes in mode_groups:
                if not modes:
                    continue
                mode_weights = []
                for mode in modes:
                    mode_weights.append(float(mode["weight"]))
                    values = [float(mode["descriptors"][name]) for name in ("arousal", "intimacy", "tension")]
                    values += [float(mode["confidence"]), float(mode["weight"])]
                    if not np.isfinite(values).all() or any(not 0 <= value <= 1 for value in values):
                        raise ValueError("invalid listening descriptor mode")
                if not np.isclose(sum(mode_weights), 1):
                    raise ValueError("listening mode weights must sum to one")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ListeningReviewError(f"provider did not return a valid audio listening review: {error}") from error

    # Write nothing resembling a completed annotation until the call validates.
    output_dir.mkdir(parents=True)
    (output_dir / "input.wav").write_bytes(audio)
    evidence = {"method": "direct waveform-input LLM listening review; blind to learned structure",
                "canonical_wave_sha256": canonical["sha256"],
                "input_audio": {"path": "input.wav", "sha256": hashlib.sha256(audio).hexdigest(),
                                "sample_rate_hz": 16000, "channels": 1, "duration_seconds": duration},
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(), "prompt": prompt,
                "model": provider_response.get("model", model), "requested_model": model,
                "provider_response_id": provider_response.get("id"), "endpoint": endpoint,
                "usage": provider_response.get("usage"), "review": review,
                "endpoint_adjustment": endpoint_adjustment,
                "execution": "test_transport" if transport is not None else "live_provider",
                "provenance": "LLM listening judgment; not a human or calibrated population observation"}
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    correction = {"schema": "hearthis.structure-correction/v1", "annotation_source": "llm",
                  "annotator": "audio-llm-reviewer", "created_at": datetime.now(timezone.utc).isoformat(),
                  "base_manifest_sha256": base_hash, "sections": correction_sections,
                  "provenance": {"model": str(evidence["model"]), "method": evidence["method"],
                                 "audio_listened": transport is None, "independent_reference": transport is None,
                                 "evidence_path": evidence_path.name, "evidence_sha256": sha256_file(evidence_path)}}
    path = output_dir / "correction.json"
    path.write_text(json.dumps(correction, indent=2, sort_keys=True) + "\n")
    return path
