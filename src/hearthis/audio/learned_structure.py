"""Learned functional structure inference using All-In-One checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ingest import sha256_file
from .legacy_natten import install_legacy_natten_adapter

LEARNED_STRUCTURE_SCHEMA = "hearthis.learned-song-structure/v1"
HARMONIX_LABELS = (
    "start",
    "end",
    "intro",
    "outro",
    "break",
    "bridge",
    "instrumental",
    "solo",
    "verse",
    "chorus",
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _spectrogram_axes(recording_dir: Path) -> dict[str, np.ndarray]:
    axes = {}
    for path in sorted((recording_dir / "spectrograms").glob("*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        with np.load(metadata["tensor"]["path"]) as tensor:
            axes[metadata["name"]] = tensor["time_seconds"].copy()
    return axes


def _normalize_segments(segments, duration: float) -> list[dict[str, float | str]]:
    normalized = [
        {"start": float(segment.start), "end": float(segment.end), "label": str(segment.label)}
        for segment in segments
    ]
    if len(normalized) > 1 and normalized[0]["label"] == "start":
        normalized[1]["start"] = 0.0
        normalized.pop(0)
    if len(normalized) > 1 and normalized[-1]["label"] == "end":
        normalized[-2]["end"] = duration
        normalized.pop()
    for segment in normalized:
        if segment["label"] == "inst":
            segment["label"] = "instrumental"
    return normalized


def analyze_learned_structure(
    recording_dir: Path,
    learned_input_path: Path,
    output_dir: Path,
    *,
    model_name: str = "harmonix-fold0",
    device: str = "cpu",
) -> Path:
    """Run one published checkpoint and preserve frame-level uncertainty."""
    install_legacy_natten_adapter()
    from allin1.models import load_pretrained_model
    from allin1.postprocessing import (
        estimate_tempo_from_beats,
        postprocess_functional_structure,
        postprocess_metrical_structure,
    )

    recording_dir = recording_dir.expanduser().resolve()
    learned_input_path = learned_input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    ingestion = json.loads((recording_dir / "manifest.json").read_text(encoding="utf-8"))
    canonical = ingestion["canonical_audio"]
    sample_rate = int(canonical["sample_rate_hz"])
    frame_count = int(canonical["frame_count"])
    duration = frame_count / sample_rate
    axes = _spectrogram_axes(recording_dir)

    learned_input = np.load(learned_input_path)
    model = load_pretrained_model(model_name, device=device)
    inputs = torch.from_numpy(learned_input).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(inputs)
        metrical = postprocess_metrical_structure(logits, model.cfg)
        predicted_segments = postprocess_functional_structure(logits, model.cfg)
        beat_probability = torch.sigmoid(logits.logits_beat[0]).cpu().numpy()
        downbeat_probability = torch.sigmoid(logits.logits_downbeat[0]).cpu().numpy()
        boundary_probability = torch.sigmoid(logits.logits_section[0]).cpu().numpy()
        label_probability = torch.softmax(logits.logits_function[0], dim=0).cpu().numpy()

    activations_path = output_dir / "activations.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        activations_path,
        beat_probability=beat_probability.astype(np.float32),
        downbeat_probability=downbeat_probability.astype(np.float32),
        boundary_probability=boundary_probability.astype(np.float32),
        label_probability=label_probability.astype(np.float32),
    )

    sections = []
    frames_per_second = float(model.cfg.fps)
    normalized_segments = _normalize_segments(predicted_segments, duration)
    for index, segment in enumerate(normalized_segments):
        start = max(0.0, min(float(segment["start"]), duration))
        end = max(start, min(float(segment["end"]), duration))
        if index == 0:
            start = 0.0
        if index == len(normalized_segments) - 1:
            end = duration
        start_frame = min(label_probability.shape[1] - 1, max(0, round(start * frames_per_second)))
        end_frame = min(label_probability.shape[1], max(start_frame + 1, round(end * frames_per_second)))
        distribution = label_probability[:, start_frame:end_frame].mean(axis=1)
        best = np.argsort(distribution)[::-1][:3]
        boundary_frame = min(boundary_probability.size - 1, start_frame)
        label = str(segment["label"])
        sections.append(
            {
                "index": index,
                "start_seconds": start,
                "end_seconds": end,
                "start_sample": round(start * sample_rate),
                "end_sample": min(frame_count, round(end * sample_rate)),
                "label": label,
                "label_kind": "learned_functional_label",
                "confidence": float(distribution[HARMONIX_LABELS.index(label)])
                if label in HARMONIX_LABELS
                else float(distribution[best[0]]),
                "boundary_confidence": float(boundary_probability[boundary_frame]),
                "label_hypotheses": [
                    {"label": HARMONIX_LABELS[int(label)], "probability": float(distribution[label])}
                    for label in best
                ],
                "spectrogram_frames": {
                    name: {
                        "start": int(np.searchsorted(times, start, side="left")),
                        "end": int(np.searchsorted(times, end, side="left")),
                    }
                    for name, times in axes.items()
                },
            }
        )

    manifest = {
        "schema": LEARNED_STRUCTURE_SCHEMA,
        "canonical_wave": {
            "path": canonical["path"],
            "sha256": canonical["sha256"],
            "sample_rate_hz": sample_rate,
            "frame_count": frame_count,
            "duration_seconds": duration,
        },
        "backend": {
            "name": "all-in-one-compatible",
            "model": model_name,
            "checkpoint_repository": "taejunkim/allinone",
            "label_vocabulary": list(HARMONIX_LABELS),
            "legacy_natten_adapter": True,
            "instrument_axis_padding": {
                "original_stems": 4,
                "attention_kernel": 5,
                "padded_tokens": 1,
            },
            "device": device,
        },
        "learned_input": {
            "path": str(learned_input_path),
            "sha256": sha256_file(learned_input_path),
            "shape": list(learned_input.shape),
        },
        "activations": {
            "path": str(activations_path),
            "sha256": sha256_file(activations_path),
            "frames_per_second": frames_per_second,
        },
        "tempo_bpm": float(estimate_tempo_from_beats(metrical["beats"])),
        "beats_seconds": metrical["beats"],
        "downbeats_seconds": metrical["downbeats"],
        "beat_positions": metrical["beat_positions"],
        "sections": sections,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path
