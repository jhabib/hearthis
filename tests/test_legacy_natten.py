from __future__ import annotations

import torch

from hearthis.audio.legacy_natten import (
    natten1dav,
    natten1dqkrpb,
    natten2dav,
    natten2dqkrpb,
)
from hearthis.audio.learned_structure import _normalize_segments


def test_one_dimensional_attention_matches_manual_center_window() -> None:
    query = torch.ones((1, 1, 5, 1))
    key = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)
    bias = torch.zeros((1, 5))
    scores = natten1dqkrpb(query, key, bias, 3, 1)
    assert scores[0, 0].tolist() == [[0, 1, 2], [0, 1, 2], [1, 2, 3], [2, 3, 4], [2, 3, 4]]
    probabilities = torch.softmax(scores, dim=-1)
    output = natten1dav(probabilities, key, 3, 1)
    assert output.shape == query.shape


def test_two_dimensional_adapter_pads_four_instruments_for_kernel_five() -> None:
    query = torch.ones((1, 1, 4, 5, 1))
    key = torch.ones_like(query)
    value = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5, 1)
    bias = torch.zeros((1, 9, 9))
    scores = natten2dqkrpb(query, key, bias, 5, 1)
    assert scores.shape == (1, 1, 4, 5, 25)
    output = natten2dav(torch.softmax(scores, dim=-1), value, 5, 1)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_marker_segments_are_folded_into_musical_sections() -> None:
    class Segment:
        def __init__(self, start, end, label):
            self.start, self.end, self.label = start, end, label

    result = _normalize_segments(
        [
            Segment(0, 0.05, "start"),
            Segment(0.05, 9.5, "inst"),
            Segment(9.5, 10, "end"),
        ],
        10,
    )
    assert result == [{"start": 0.0, "end": 10, "label": "instrumental"}]
