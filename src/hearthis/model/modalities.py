"""Observed lyrics, metadata and semantic reports with normalized kernels."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from ..audio.ingest import sha256_file


def tokens(text):
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", text.casefold(), flags=re.UNICODE)


def attach_observations(song, path, names):
    """Attach supplied observations. Absent lyrics and labels stay absent."""
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("schema") != "hearthis.song-observations/v1" or data.get("song_id") != song.song_id:
        raise ValueError("song observations require matching schema and song ID")
    validate_observations(song, data, names)
    song.observations = {**data, "semantic_reports": song.observations.get("semantic_reports", []) + data.get("semantic_reports", [])}
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def validate_observations(song, data, names):
    if not isinstance(data.get("provenance"), str) or not data["provenance"].strip():
        raise ValueError("song observations require source provenance")
    ids = {r.region_id for r in song.regions}
    for report in data.get("semantic_reports", []):
        if (report.get("feature") not in names or report.get("region_id") not in ids
                or report.get("source") not in ("audio_llm", "human", "metadata", "lyrics", "synthetic")
                or not np.isfinite([report.get("value", np.nan), report.get("noise_sd", np.nan)]).all()
                or not 0 <= report["value"] <= 1 or report["noise_sd"] <= 0):
            raise ValueError("invalid semantic observation or source")
    lyrics = data.get("lyrics")
    if lyrics is not None:
        if not isinstance(lyrics, dict) or not isinstance(lyrics.get("text"), str):
            raise ValueError("lyrics require supplied text")
        last = 0
        for segment in lyrics.get("segments", []):
            if (not isinstance(segment.get("text"), str)
                    or not np.isfinite([segment["start_seconds"], segment["end_seconds"]]).all()
                    or not last <= segment["start_seconds"] < segment["end_seconds"] <= song.duration_seconds):
                raise ValueError("lyrics must be ordered and aligned to the recording")
            last = segment["end_seconds"]
        if lyrics.get("segments") and tokens(lyrics["text"]) != tokens(" ".join(s["text"] for s in lyrics["segments"])):
            raise ValueError("aligned lyric segments must cover the supplied text")
    if "metadata_tags" in data and (not isinstance(data["metadata_tags"], list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in data["metadata_tags"])
            or len(set(data["metadata_tags"])) != len(data["metadata_tags"])):
        raise ValueError("metadata tags must be distinct strings")


class ObservationKernel:
    """p(L|X): Poisson length and categorical ordered tokens.

    p(M|X): independent Bernoulli tags in the declared vocabulary. Semantic
    reports use Gaussian observation noise. These are explicit learned kernels;
    they do not claim linguistic understanding from a small training corpus.
    """
    def __init__(self, songs, names, saved=None):
        from .waveform import torch_modules
        self.torch, nn = torch_modules()
        for song in songs:
            if song.observations:
                validate_observations(song, song.observations, names)
        counts = Counter(t for song in songs for t in tokens(song.observations.get("lyrics", {}).get("text", "")))
        self.has_lyrics = any("lyrics" in song.observations for song in songs)
        self.vocabulary = ["<unk>"] + [word for word, _ in counts.most_common(2047)]
        self.tags = sorted({tag for song in songs for tag in song.observations.get("metadata_tags", [])})
        self.model = nn.ModuleDict({"tokens": nn.Linear(len(names), len(self.vocabulary)),
                                    "length": nn.Linear(len(names), 1),
                                    "tags": nn.Linear(len(names), max(1, len(self.tags)))})
        if saved is not None:
            if saved["vocabulary"] != self.vocabulary or saved["tags"] != self.tags:
                raise ValueError("observation kernel vocabulary changed")
            self.model.load_state_dict({key: self.torch.tensor(value) for key, value in saved["parameters"].items()})

    def to_dict(self):
        return {"vocabulary": self.vocabulary, "tags": self.tags, "has_lyrics": self.has_lyrics,
                "parameters": {key: value.detach().tolist() for key, value in self.model.state_dict().items()}}

    def sample(self, music, *, seed=0, max_tokens=512):
        """Generate model lyrics and metadata, preserving synthetic provenance."""
        music = np.asarray(music, dtype=np.float32)
        dimension = self.model["length"].in_features
        if music.shape != (dimension,) or not np.isfinite(music).all() or np.any((music < 0) | (music > 1)) or max_tokens <= 0:
            raise ValueError("modality generation requires bounded music and a positive token budget")
        rng = np.random.default_rng(seed)
        with self.torch.no_grad():
            value = self.torch.from_numpy(music[None])
            rate = float(self.torch.nn.functional.softplus(self.model["length"](value)).squeeze()) + 1e-6
            count = int(rng.poisson(rate)) if self.has_lyrics else 0
            if count > max_tokens:
                raise ValueError("sampled lyrics exceed generation token budget")
            probability = self.model["tokens"](value).softmax(-1).numpy()[0]
            words = [self.vocabulary[i] for i in rng.choice(len(self.vocabulary), size=count, p=probability)]
            tag_probability = self.model["tags"](value).sigmoid().numpy()[0][:len(self.tags)]
        return {"lyrics": " ".join(words) if self.has_lyrics else None, "metadata_tags": [tag for tag, probability in zip(self.tags, tag_probability) if rng.random() < probability],
                "provenance": "synthetic conditional model output; not a source transcript or observed annotation",
                "lyric_model": "Poisson length and categorical token sequence", "quality": "Unverified; evaluate generated language independently"}

    def log_probability(self, songs, regions, names):
        torch = self.torch
        example = next(iter(regions.values()))[0]
        result = torch.zeros(len(example), dtype=example.dtype)
        index = {word: i for i, word in enumerate(self.vocabulary)}
        for song in songs:
            data = song.observations
            for report in data.get("semantic_reports", []):
                mean = regions[report["region_id"]][0][:, names.index(report["feature"])]
                sd = report["noise_sd"]
                result += -0.5 * ((mean - report["value"]) / sd).square() - np.log(sd * np.sqrt(2 * np.pi))
            music = regions[song.global_region.region_id][0]
            if "metadata_tags" in data:
                if self.tags:
                    target = music.new_tensor([tag in data["metadata_tags"] for tag in self.tags])
                    logits = self.model["tags"](music)
                    result -= torch.nn.functional.binary_cross_entropy_with_logits(logits, target.expand_as(logits), reduction="none").sum(-1)
            if "lyrics" not in data:
                continue
            lyrics = data["lyrics"]
            segments = lyrics.get("segments") or [{"text": lyrics["text"]}]
            for segment in segments:
                text_tokens = tokens(segment["text"])
                conditioned = music
                if "start_seconds" in segment:
                    midpoint = (segment["start_seconds"] + segment["end_seconds"]) / 2
                    region = next((r for r in song.regions if r.kind == "passage" and r.start_seconds <= midpoint < r.end_seconds), song.global_region)
                    conditioned = regions[region.region_id][0]
                rate = torch.nn.functional.softplus(self.model["length"](conditioned).squeeze(-1)) + 1e-6
                count = len(text_tokens)
                result += count * rate.log() - rate - torch.lgamma(rate.new_tensor(count + 1))
                if text_tokens:
                    positions = torch.tensor([index.get(word, 0) for word in text_tokens])
                    result += self.model["tokens"](conditioned).log_softmax(-1)[:, positions].sum(-1)
        return result
