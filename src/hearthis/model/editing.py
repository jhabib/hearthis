"""Localized posterior audio edits preserving the original timeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from scipy.io import wavfile
from scipy.signal import resample_poly

from ..audio.ingest import sha256_file
from ..audio.separation import _audio_geometry
from .waveform import WaveformModel, torch_modules


def edit_audio(source: Path, model_path: Path, output_dir: Path, *, music, edits, fade_seconds=0.02):
    """Decode posterior acoustic latents under changed controls.

    Apply only the decoded difference inside requested intervals. This retains
    source channels, sample rate, duration, and exact float PCM outside edits.
    Feature adherence requires evaluation; it is never assumed from a control.
    """
    from ..pipeline import write_json
    if output_dir.exists():
        raise ValueError("editing requires a new output directory")
    model = WaveformModel.load(Path(model_path))
    torch, _ = torch_modules()
    rate, original = _audio_geometry(Path(source))
    music = np.asarray(music, dtype=np.float32)
    if (music.shape != (model.config["music_dimension"],) or not np.isfinite(music).all()
            or np.any((music < 0) | (music > 1)) or not edits
            or not np.isfinite(fade_seconds) or fade_seconds < 0):
        raise ValueError("edits require bounded music coordinates, intervals and nonnegative fade")
    final = original.copy()
    last = 0
    validated = []
    for edit in edits:
        start, end, strength = edit["start_seconds"], edit["end_seconds"], edit.get("strength", 1.0)
        target = np.asarray(edit["music"], dtype=np.float32)
        if (not np.isfinite([start, end, strength]).all() or not last <= start < end <= len(original) / rate
                or not 0 <= strength <= 1 or target.shape != music.shape
                or not np.isfinite(target).all() or np.any((target < 0) | (target > 1))):
            raise ValueError("edits must be bounded, ordered, nonoverlapping and inside the recording")
        last = end
        first, stop = round(start * rate), round(end * rate)
        if stop <= first:
            raise ValueError("edit interval must contain source samples")
        mono = original.mean(axis=1) if original.ndim == 2 else original
        divisor = int(np.gcd(rate, model.config["sample_rate_hz"]))
        resampled = resample_poly(mono, model.config["sample_rate_hz"] // divisor, rate // divisor)
        size = model.config["samples"]
        windows = np.pad(resampled, (0, (-len(resampled)) % size)).reshape(-1, size)
        whitened = ((dct(windows, type=2, norm="ortho", axis=-1) - model.location) / model.scale).astype(np.float32)
        context = np.vstack([np.zeros((1, size), dtype=np.float32), whitened[:-1]])
        selected = np.arange(max(0, int(start * model.config["sample_rate_hz"]) // size),
                             min(len(windows), int(np.ceil(end * model.config["sample_rate_hz"] / size))))
        with torch.no_grad():
            audio = torch.from_numpy(whitened[selected])
            old = torch.from_numpy(np.tile(music, (len(selected), 1)))
            new = torch.from_numpy(np.tile(target, (len(selected), 1)))
            ctx = torch.from_numpy(context[selected])
            latent, _ = model.model.encode(audio, old, ctx)
            difference = (model.model.decode(latent, new, ctx) - model.model.decode(latent, old, ctx)).numpy()
        changed = np.zeros_like(windows)
        changed[selected] = idct(difference * model.scale, type=2, norm="ortho", axis=-1)
        delta = resample_poly(changed.ravel()[:len(resampled)], rate // divisor, model.config["sample_rate_hz"] // divisor)
        delta = np.pad(delta, (0, max(0, len(original) - len(delta))))[:len(original)]
        gain = np.ones(stop - first) * strength
        fade = min(len(gain) // 2, round(fade_seconds * rate))
        if fade:
            ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
            gain[:fade] *= ramp
            gain[-fade:] *= ramp[::-1]
        adjustment = delta[first:stop] * gain
        final[first:stop] += adjustment[:, None] if final.ndim == 2 else adjustment
        validated.append({**edit, "source_sample_interval": [first, stop]})
    if not np.isfinite(final).all():
        raise RuntimeError("edit produced nonfinite waveform")
    output_dir.mkdir(parents=True)
    path = output_dir / "edited.wav"
    wavfile.write(path, rate, final.astype(np.float32))
    manifest = output_dir / "manifest.json"
    pending = output_dir / ".manifest.pending.json"
    write_json(pending, {"schema": "hearthis.audio-edit/v1", "source": {"path": str(Path(source).resolve()), "sha256": sha256_file(Path(source))},
                         "model": {"path": str(Path(model_path).resolve()), "sha256": sha256_file(Path(model_path))},
                         "audio": {"path": str(path.resolve()), "sha256": sha256_file(path)},
                         "sample_rate_hz": rate, "frame_count": len(final), "channels": 1 if final.ndim == 1 else final.shape[1],
                         "music": music.tolist(), "edits": validated, "fade_seconds": fade_seconds,
                         "method": "posterior acoustic latent conditional decoder difference with localized fade",
                         "attribute_adherence": "Unverified; evaluate changed musical attributes independently"})
    try:
        validate_edit(pending)
        pending.replace(manifest)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    return manifest


def validate_edit(path):
    manifest = json.loads(Path(path).read_text())
    if manifest["schema"] != "hearthis.audio-edit/v1":
        raise ValueError("invalid edit schema")
    for key in ("source", "model", "audio"):
        ref = manifest[key]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("edit artifact hash mismatch")
    WaveformModel.load(Path(manifest["model"]["path"]))
    rate, source = _audio_geometry(Path(manifest["source"]["path"]))
    edited_rate, edited = _audio_geometry(Path(manifest["audio"]["path"]))
    if rate != edited_rate or source.shape != edited.shape or not np.isfinite(edited).all():
        raise ValueError("edit changed source geometry")
    mask = np.ones(len(source), dtype=bool)
    for edit in manifest["edits"]:
        first, stop = edit["source_sample_interval"]
        if [first, stop] != [round(edit["start_seconds"] * rate), round(edit["end_seconds"] * rate)]:
            raise ValueError("edit sample alignment mismatch")
        mask[first:stop] = False
    if not np.array_equal(source[mask], edited[mask]):
        raise ValueError("edit modified audio outside requested intervals")
    if (manifest["frame_count"] != len(edited) or manifest["sample_rate_hz"] != rate
            or manifest["channels"] != (1 if edited.ndim == 1 else edited.shape[1])):
        raise ValueError("edit manifest geometry mismatch")
    return {"valid": True, "frame_count": len(edited), "outside_intervals_unchanged": True}
