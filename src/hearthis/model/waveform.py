"""Conditional neural waveform VAE with a normalized time-domain likelihood.

An orthonormal DCT keeps the waveform likelihood invertible. Whitening has a
recorded Jacobian; neither spectrogram inversion nor copying catalog clips is
used for generation. Audio quality requires separate listening evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from scipy.io import wavfile
from scipy.signal import resample_poly

from ..audio.ingest import sha256_file
from ..audio.separation import _audio_geometry


def torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("install hearthis[analysis] for neural training and generation") from error
    return torch, nn


def network(samples: int, music_dimension: int, latent_dimension: int, hidden: int, autoregressive=False):
    torch, nn = torch_modules()
    class WaveformVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.context = nn.Sequential(nn.Linear(samples, hidden), nn.Tanh()) if autoregressive else None
            extra = hidden if autoregressive else 0
            self.encoder = nn.Sequential(nn.Linear(samples + music_dimension + extra, hidden), nn.Tanh(),
                                         nn.Linear(hidden, 2 * latent_dimension))
            self.decoder = nn.Sequential(nn.Linear(latent_dimension + music_dimension + extra, hidden), nn.Tanh(),
                                         nn.Linear(hidden, samples))

        def inputs(self, values, music, context):
            parts = [values, music]
            if self.context is not None:
                context = values.new_zeros((*values.shape[:-1], samples)) if context is None else context
                parts.append(self.context(context))
            return torch.cat(parts, dim=-1)

        def encode(self, audio, music, context=None):
            location, log_scale = self.encoder(self.inputs(audio, music, context)).chunk(2, dim=-1)
            return location, log_scale.clamp(-5, 2)

        def decode(self, latent, music, context=None):
            return self.decoder(self.inputs(latent, music, context))
    return WaveformVAE()


class WaveformModel:
    def __init__(self, model, config, location, scale):
        self.model, self.config = model, config
        self.location, self.scale = np.asarray(location), np.asarray(scale)

    @classmethod
    def load(cls, manifest_path: Path):
        torch, _ = torch_modules()
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema"] != "hearthis.waveform-model/v1":
            raise ValueError("invalid waveform model schema")
        if manifest.get("parent_model") and sha256_file(Path(manifest["parent_model"]["path"])) != manifest["parent_model"]["sha256"]:
            raise ValueError("joint waveform parent model hash mismatch")
        for record in manifest["artifacts"].values():
            if sha256_file(Path(record["path"])) != record["sha256"]:
                raise ValueError("waveform model artifact hash mismatch")
        for record in manifest["sources"]:
            if sha256_file(Path(record["path"])) != record["sha256"]:
                raise ValueError("waveform model source hash mismatch")
        config = manifest["configuration"]
        required = ("samples", "music_dimension", "latent_dimension", "hidden", "sample_rate_hz")
        if (any(type(config.get(key)) is not int or config[key] <= 0 for key in required)
                or not np.isfinite(config.get("observation_sd", np.nan)) or config["observation_sd"] <= 0):
            raise ValueError("invalid waveform model configuration")
        model = network(config["samples"], config["music_dimension"], config["latent_dimension"], config["hidden"], config.get("autoregressive", False))
        state = torch.load(manifest["artifacts"]["checkpoint"]["path"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with np.load(manifest["artifacts"]["training-data"]["path"], allow_pickle=False) as data:
            location, scale = data["location"].copy(), data["scale"].copy()
        if (location.shape != (config["samples"],) or scale.shape != location.shape
                or not np.isfinite(location).all() or not np.isfinite(scale).all() or np.any(scale <= 0)
                or any(not torch.isfinite(value).all() for value in model.parameters())):
            raise ValueError("invalid waveform normalization or network parameters")
        return cls(model, config, location, scale)


    def generate(self, music, *, duration_seconds: float, seed: int = 0, sections=None):
        torch, _ = torch_modules()
        music = np.asarray(music, dtype=np.float32)
        if (music.shape != (self.config["music_dimension"],) or not np.isfinite(music).all()
                or np.any((music < 0) | (music > 1)) or not np.isfinite(duration_seconds)
                or not 0 < duration_seconds <= 3600):
            raise ValueError("generation requires bounded music coordinates and a duration in (0,3600]")
        sections = sections or [{"start_seconds": 0, "end_seconds": duration_seconds, "music": music.tolist()}]
        end = 0
        for section in sections:
            values = np.asarray(section["music"])
            if (values.shape != music.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
                    or not np.isfinite([section["start_seconds"], section["end_seconds"]]).all()
                    or not np.isclose(section["start_seconds"], end, rtol=0, atol=1e-9) or section["end_seconds"] <= end):
                raise ValueError("generation sections require a complete bounded coordinate trajectory")
            end = section["end_seconds"]
        if not np.isclose(end, duration_seconds, rtol=0, atol=1e-9):
            raise ValueError("generation sections must cover the requested duration")
        rng = np.random.default_rng(seed)
        frames = round(duration_seconds * self.config["sample_rate_hz"])
        if frames < 1:
            raise ValueError("generation duration must include at least one sample")
        count = int(np.ceil(frames / self.config["samples"]))
        latent = rng.standard_normal((count, self.config["latent_dimension"])).astype(np.float32)
        controls = np.array([next(s["music"] for s in sections if s["start_seconds"] <= min((i + 0.5) * self.config["samples"] / self.config["sample_rate_hz"], duration_seconds - 1e-9) < s["end_seconds"]) for i in range(count)], dtype=np.float32)
        with torch.no_grad():
            if self.config.get("autoregressive", False):
                previous = torch.zeros((1, self.config["samples"]))
                windows = []
                for i in range(count):
                    mean = self.model.decode(torch.from_numpy(latent[i:i + 1]), torch.from_numpy(controls[i:i + 1]), previous).numpy()
                    whitened = mean + rng.standard_normal(mean.shape) * self.config["observation_sd"]
                    windows.append(whitened[0])
                    previous = torch.from_numpy(whitened.astype(np.float32))
                coefficients = np.asarray(windows) * self.scale + self.location
            else:
                mean = self.model.decode(torch.from_numpy(latent), torch.from_numpy(controls)).numpy()
                coefficients = (mean + rng.standard_normal(mean.shape) * self.config["observation_sd"]) * self.scale + self.location
        samples = idct(coefficients, type=2, norm="ortho", axis=-1).ravel()[:frames]
        if not np.isfinite(samples).all():
            raise RuntimeError("decoder generated nonfinite waveform")
        return samples.astype(np.float32)


def save_joint_waveform(model, parent_path, output_dir, report):
    """Publish the learned theta/phi checkpoint without modifying its parent."""
    from ..pipeline import write_json
    torch, _ = torch_modules()
    if output_dir.exists():
        raise ValueError("joint neural checkpoint requires a new directory")
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "checkpoint.pt"
    torch.save(model.model.state_dict(), checkpoint)
    parent = json.loads(parent_path.read_text())
    parent["artifacts"]["checkpoint"] = {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}
    parent["parent_model"] = {"path": str(parent_path.resolve()), "sha256": sha256_file(parent_path)}
    parent["joint_training"] = {**report, "method": "joint acoustic encoder/decoder and variational ELBO; Gaussian MAP regularization around pretrained weights"}
    path = output_dir / "manifest.json"
    pending = output_dir / ".manifest.pending.json"
    write_json(pending, parent)
    try:
        WaveformModel.load(pending)
        pending.replace(path)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    return path


def train_waveform(songs, provenance, output_dir: Path, *, seed=0, epochs=80,
                   sample_rate_hz=8000, samples=2048, latent_dimension=12, hidden=96,
                   max_clips=256, batch_size=32, observation_sd=0.5, learning_rate=0.002, autoregressive=True):
    """Fit a conditional VAE on actual waveform crops with a held-out split."""
    from ..pipeline import write_json
    torch, _ = torch_modules()
    if output_dir.exists() or min(epochs, samples, latent_dimension, hidden, max_clips, batch_size, sample_rate_hz) <= 0:
        raise ValueError("waveform training requires a new output and positive configuration")
    if not np.isfinite([observation_sd, learning_rate]).all() or min(observation_sd, learning_rate) <= 0:
        raise ValueError("invalid waveform likelihood or optimizer scale")
    lookup = {p.get("recording_id"): p for p in provenance}
    crops, contexts, coordinates, song_indices, times, sources = [], [], [], [], [], []
    for song_index, song in enumerate(songs):
        ingestion_ref = lookup[song.song_id]["references"]["ingestion"]
        if sha256_file(Path(ingestion_ref["path"])) != ingestion_ref["sha256"]:
            raise ValueError("training ingestion provenance changed")
        ingestion = json.loads(Path(ingestion_ref["path"]).read_text())
        audio_ref = ingestion["canonical_audio"]
        if sha256_file(Path(audio_ref["path"])) != audio_ref["sha256"]:
            raise ValueError("training waveform hash mismatch")
        rate, audio = _audio_geometry(Path(audio_ref["path"]))
        audio = audio.mean(axis=1) if audio.ndim == 2 else audio
        divisor = int(np.gcd(rate, sample_rate_hz))
        audio = resample_poly(audio, sample_rate_hz // divisor, rate // divisor)
        starts = np.arange(0, len(audio) - samples + 1, samples)
        if not len(starts):
            raise ValueError("training recording is shorter than one decoder window")
        starts = starts[np.linspace(0, len(starts) - 1, min(max_clips, len(starts))).astype(int)]
        for start in starts:
            midpoint = (start + samples / 2) / sample_rate_hz
            region = next((r for r in song.regions if r.kind == "passage" and r.start_seconds <= midpoint < r.end_seconds), song.global_region)
            crops.append(dct(audio[start:start + samples], type=2, norm="ortho"))
            contexts.append(dct(audio[start - samples:start], type=2, norm="ortho") if start >= samples else None)
            coordinates.append(region.music)
            song_indices.append(song_index)
            times.append(midpoint)
        sources.append({"song_id": song.song_id, "path": audio_ref["path"], "sha256": audio_ref["sha256"]})
    audio = np.asarray(crops, dtype=np.float32)
    music = np.asarray(coordinates, dtype=np.float32)
    if len(audio) < 5:
        raise ValueError("waveform training needs at least five windows for a held-out split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(audio))
    test, train = order[:max(1, len(order) // 5)], order[max(1, len(order) // 5):]
    location = audio[train].mean(axis=0)
    scale = np.maximum(audio[train].std(axis=0), 0.001)
    whitened = (audio - location) / scale
    context = np.asarray([(value - location) / scale if value is not None else np.zeros(samples) for value in contexts], dtype=np.float32)
    torch.manual_seed(seed)
    model = network(samples, music.shape[1], latent_dimension, hidden, autoregressive)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tensor, features = torch.from_numpy(whitened), torch.from_numpy(music)
    context_tensor = torch.from_numpy(context)
    # Fixed validation noise makes the initial/final likelihood comparison paired.
    generator = torch.Generator().manual_seed(seed + 1)
    fixed_noise = torch.randn((len(audio), latent_dimension), generator=generator)

    def negative_elbo(indices, noise):
        mean, log_sd = model.encode(tensor[indices], features[indices], context_tensor[indices])
        latent = mean + log_sd.exp() * noise
        predicted = model.decode(latent, features[indices], context_tensor[indices])
        log_probability = -0.5 * (((tensor[indices] - predicted) / observation_sd) ** 2
                                  + np.log(2 * np.pi * observation_sd ** 2)).sum(dim=-1)
        log_probability -= float(np.log(scale).sum())
        kl = 0.5 * (mean.square() + (2 * log_sd).exp() - 1 - 2 * log_sd).sum(dim=-1)
        return (kl - log_probability).mean()

    with torch.no_grad():
        initial_train = float(negative_elbo(train, fixed_noise[train]))
        initial_test = float(negative_elbo(test, fixed_noise[test]))
    history, validation_history = [], []
    best, best_validation, best_epoch, best_state = initial_train, np.inf, None, None
    for epoch in range(epochs):
        epoch_order = rng.permutation(train)
        for offset in range(0, len(train), batch_size):
            batch = epoch_order[offset:offset + batch_size]
            optimizer.zero_grad()
            loss = negative_elbo(batch, torch.randn((len(batch), latent_dimension)))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite waveform training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()
        with torch.no_grad():
            value = float(negative_elbo(train, fixed_noise[train]))
            validation = float(negative_elbo(test, fixed_noise[test]))
        history.append(value)
        validation_history.append(validation)
        if value < initial_train and validation < best_validation:
            best, best_validation, best_epoch = value, validation, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("waveform training did not improve the paired training ELBO")
    model.load_state_dict(best_state)
    with torch.no_grad():
        final_test = float(negative_elbo(test, fixed_noise[test]))
    if not best < initial_train:
        raise RuntimeError("waveform training did not improve the paired training ELBO")
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    dataset = output_dir / "training-data.npz"
    np.savez_compressed(dataset, audio=whitened, context=context, music=music, song_indices=song_indices,
                        times_seconds=times, location=location, scale=scale, train=train, test=test)
    config = {"samples": samples, "sample_rate_hz": sample_rate_hz, "music_dimension": music.shape[1],
              "latent_dimension": latent_dimension, "hidden": hidden, "observation_sd": observation_sd,
              "autoregressive": bool(autoregressive)}
    manifest = output_dir / "manifest.json"
    write_json(manifest, {"schema": "hearthis.waveform-model/v1", "configuration": config, "sources": sources,
                         "artifacts": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                                       for name, path in (("checkpoint", checkpoint), ("training-data", dataset))},
                         "training": {"seed": seed, "epochs": epochs, "training_windows": len(train),
                                      "held_out_windows": len(test), "initial_train_negative_elbo": initial_train,
                                      "final_train_negative_elbo": best, "initial_held_out_negative_elbo": initial_test,
                                      "final_held_out_negative_elbo": final_test, "history": history,
                                      "validation_history": validation_history, "selected_epoch": best_epoch,
                                      "selection": "best paired held-out ELBO among checkpoints improving training ELBO",
                                      "held_out_improved": final_test < initial_test},
                         "likelihood": "Gaussian in orthonormal DCT coordinates with whitening Jacobian; normalized waveform density",
                         "quality": "Unverified; evaluate generated audio independently"})
    return manifest


def complete_window_data(model, manifest, songs):
    """All complete source windows for the joint likelihood, independent of
    the subset selected for decoder pretraining and validation."""
    if [s.song_id for s in songs] != [r["song_id"] for r in manifest["sources"]]:
        raise ValueError("waveform model belongs to a different ordered catalog")
    windows, contexts, music, owners, times = [], [], [], [], []
    rate, size = model.config["sample_rate_hz"], model.config["samples"]
    for index, (song, reference) in enumerate(zip(songs, manifest["sources"], strict=True)):
        source_rate, samples = _audio_geometry(Path(reference["path"]))
        samples = samples.mean(axis=1) if samples.ndim == 2 else samples
        divisor = int(np.gcd(source_rate, rate))
        samples = resample_poly(samples, rate // divisor, source_rate // divisor)
        count = len(samples) // size
        coefficients = dct(samples[:count * size].reshape(count, size), type=2, norm="ortho", axis=-1)
        whitened = (coefficients - model.location) / model.scale
        windows.extend(whitened)
        contexts.extend(np.vstack([np.zeros(size), whitened[:-1]]) if count else np.empty((0, size)))
        for i in range(count):
            time = (i + 0.5) * size / rate
            region = next((r for r in song.regions if r.kind == "passage" and r.start_seconds <= time < r.end_seconds), song.global_region)
            music.append(region.music)
            owners.append(index)
            times.append(time)
    return {"audio": np.asarray(windows, dtype=np.float32), "context": np.asarray(contexts, dtype=np.float32), "music": np.asarray(music, dtype=np.float32),
            "song_indices": np.asarray(owners), "times_seconds": np.asarray(times),
            "location": model.location, "scale": model.scale, "train": np.arange(len(windows))}


def generate_ideal_audio(ideal, model_path: Path, output_dir: Path, *, duration_seconds=4.0, seed=0, sections=None):
    model = WaveformModel.load(model_path)
    records = []
    output_dir.mkdir(parents=True, exist_ok=False)
    for index, sample in enumerate(ideal["samples"]):
        audio = model.generate(sample["music"], duration_seconds=duration_seconds, seed=seed + index, sections=sections)
        path = output_dir / f"ideal-{index:03d}.wav"
        wavfile.write(path, model.config["sample_rate_hz"], audio)
        records.append({"path": str(path.resolve()), "sha256": sha256_file(path),
                        "sample_rate_hz": model.config["sample_rate_hz"], "frame_count": len(audio),
                        "duration_seconds": len(audio) / model.config["sample_rate_hz"],
                        "peak": float(np.max(np.abs(audio))), "rms": float(np.sqrt(np.mean(audio ** 2))),
                        "music": sample["music"], "sections": sections, "ideal_sample_index": index})
    return {"model": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)},
            "waveforms": records, "method": "autoregressive conditional neural waveform decoder" if model.config.get("autoregressive") else "conditional learned waveform decoder with sampled acoustic latent and observation noise",
            "quality": "Unverified; small-corpus generation is not validated music synthesis"}
