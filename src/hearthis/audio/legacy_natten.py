"""Portable inference implementation of NATTEN 0.14 neighborhood operators."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def _window_start(index: int, length: int, kernel: int, dilation: int) -> int:
    neighborhood = kernel // 2
    if dilation <= 1:
        return max(index - neighborhood, 0) + int(index + neighborhood >= length) * (
            length - index - neighborhood - 1
        )
    start = index - neighborhood * dilation
    if start < 0:
        return index % dilation
    if index + neighborhood * dilation >= length:
        remainder = index % dilation
        aligned = (length // dilation) * dilation
        tail = length - aligned
        if remainder < tail:
            return length - tail + remainder - 2 * neighborhood * dilation
        return aligned + remainder - kernel * dilation
    return start


def _bias_start(index: int, length: int, kernel: int, dilation: int) -> int:
    neighborhood = kernel // 2
    if dilation <= 1:
        return neighborhood + int(index < neighborhood) * (neighborhood - index) + int(
            index + neighborhood >= length
        ) * (length - index - 1 - neighborhood)
    if index - neighborhood * dilation < 0:
        return kernel - 1 - index // dilation
    if index + neighborhood * dilation >= length:
        return (length - index - 1) // dilation
    return neighborhood


def _axis_layout(length: int, kernel: int, dilation: int, device: torch.device):
    if length < kernel * dilation:
        raise ValueError(f"axis length {length} is smaller than {kernel * dilation}")
    starts = [_window_start(i, length, kernel, dilation) for i in range(length)]
    bias_starts = [_bias_start(i, length, kernel, dilation) for i in range(length)]
    offsets = torch.arange(kernel, device=device, dtype=torch.long)
    neighbors = torch.tensor(starts, device=device)[:, None] + offsets * dilation
    biases = torch.tensor(bias_starts, device=device)[:, None] + offsets
    return neighbors, biases


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    neighbors, biases = _axis_layout(query.shape[2], kernel_size, dilation, query.device)
    local_keys = key[:, :, neighbors, :]
    scores = (query.unsqueeze(-2) * local_keys).sum(dim=-1)
    return scores + rpb[:, biases].unsqueeze(0)


def natten1dav(attention, value, kernel_size, dilation):
    neighbors, _ = _axis_layout(value.shape[2], kernel_size, dilation, value.device)
    local_values = value[:, :, neighbors, :]
    return (attention.unsqueeze(-1) * local_values).sum(dim=-2)


def _pad_2d_axis(tensor: torch.Tensor, kernel_size: int, dilation: int):
    required = kernel_size * dilation
    height, width = tensor.shape[2:4]
    pad_height = max(0, required - height)
    pad_width = max(0, required - width)
    if pad_height or pad_width:
        tensor = functional.pad(tensor, (0, 0, 0, pad_width, 0, pad_height))
    return tensor, height, width


def _local_2d(tensor, rows, columns):
    height, width = rows.shape[0], columns.shape[0]
    kernel = rows.shape[1]
    row_grid = rows[:, None, :, None].expand(height, width, kernel, kernel)
    column_grid = columns[None, :, None, :].expand(height, width, kernel, kernel)
    return tensor[:, :, row_grid, column_grid, :]


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    query, original_height, original_width = _pad_2d_axis(query, kernel_size, dilation)
    key, _, _ = _pad_2d_axis(key, kernel_size, dilation)
    rows, row_biases = _axis_layout(query.shape[2], kernel_size, dilation, query.device)
    columns, column_biases = _axis_layout(query.shape[3], kernel_size, dilation, query.device)
    local_keys = _local_2d(key, rows, columns)
    scores = (query.unsqueeze(-2).unsqueeze(-2) * local_keys).sum(dim=-1)
    row_grid = row_biases[:, None, :, None].expand(
        query.shape[2], query.shape[3], kernel_size, kernel_size
    )
    column_grid = column_biases[None, :, None, :].expand(
        query.shape[2], query.shape[3], kernel_size, kernel_size
    )
    scores = scores + rpb[:, row_grid, column_grid].unsqueeze(0)
    scores = scores.flatten(start_dim=-2)
    return scores[:, :, :original_height, :original_width]


def natten2dav(attention, value, kernel_size, dilation):
    value, original_height, original_width = _pad_2d_axis(value, kernel_size, dilation)
    rows, _ = _axis_layout(value.shape[2], kernel_size, dilation, value.device)
    columns, _ = _axis_layout(value.shape[3], kernel_size, dilation, value.device)
    local_values = _local_2d(value, rows, columns)
    attention = attention.reshape(
        attention.shape[0], attention.shape[1], original_height, original_width, kernel_size, kernel_size
    )
    output = (attention.unsqueeze(-1) * local_values[:, :, :original_height, :original_width]).sum(
        dim=(-3, -2)
    )
    return output


def install_legacy_natten_adapter() -> None:
    """Expose the legacy split operators expected by All-In-One 1.1."""
    import natten.functional as natten_functional

    natten_functional.natten1dqkrpb = natten1dqkrpb
    natten_functional.natten1dav = natten1dav
    natten_functional.natten2dqkrpb = natten2dqkrpb
    natten_functional.natten2dav = natten2dav
