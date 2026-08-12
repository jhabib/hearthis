#!/usr/bin/env python3
"""Export All-In-One spectrograms using its original Madmom chain."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from madmom.audio.signal import FramedSignalProcessor, Signal
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor

STEMS = ("bass", "drums", "other", "vocals")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stem_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    processor = SequentialProcessor([
        FramedSignalProcessor(frame_size=2048, fps=100),
        ShortTimeFourierTransformProcessor(),
        FilteredSpectrogramProcessor(num_bands=12, fmin=30, fmax=17000, norm_filters=True),
        LogarithmicSpectrogramProcessor(mul=1, add=1),
    ])
    tensors = []
    for stem in STEMS:
        signal = Signal(args.stem_directory / f"{stem}.wav", num_channels=1)
        tensors.append(np.asarray(processor(signal), dtype=np.float32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.stack(tensors))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
