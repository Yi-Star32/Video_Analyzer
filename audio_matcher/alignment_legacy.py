"""Legacy chunk-based stitching (kept for backward compatibility)."""

from typing import Sequence, Union

import numpy as np
from pydub import AudioSegment

from .chunking import AudioChunk
from .io import numpy_to_audiosegment


def _chunk_waveform(chunk: Union[dict, AudioChunk]) -> np.ndarray:
    if isinstance(chunk, dict):
        return chunk["audio"]
    return chunk.audio


def stitch_audio(
    chunks: Sequence[Union[dict, AudioChunk]],
    sr: int = 16000,
    crossfade_ms: int = 20,
    max_duration_ms: int | None = None,
) -> AudioSegment:
    if not chunks:
        return AudioSegment.silent(duration=0)

    sorted_chunks = sorted(
        chunks,
        key=lambda c: c["start_time"] if isinstance(c, dict) else c.start_time,
    )

    total_ms = 0
    for chunk in sorted_chunks:
        end_time = chunk["end_time"] if isinstance(chunk, dict) else chunk.end_time
        ms = int(end_time * 1000)
        if ms > total_ms:
            total_ms = ms

    if max_duration_ms is not None:
        total_ms = max_duration_ms

    total_samples = int(total_ms * sr / 1000)
    output = np.zeros(total_samples, dtype=np.float64)
    weight = np.zeros(total_samples, dtype=np.float64)

    fade_len = int(crossfade_ms * sr / 1000)

    for chunk in sorted_chunks:
        audio = _chunk_waveform(chunk).astype(np.float64)
        start_ms = int(
            (chunk["start_time"] if isinstance(chunk, dict) else chunk.start_time) * 1000
        )
        start_s = int(start_ms * sr / 1000)
        if start_s >= total_samples:
            continue
        end_s = start_s + len(audio)
        if end_s > total_samples:
            audio = audio[: total_samples - start_s]
            end_s = total_samples

        if 0 < fade_len * 2 < len(audio):
            audio = audio.copy()
            audio[:fade_len] *= np.linspace(0.0, 1.0, fade_len)
            audio[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

        output[start_s:end_s] += audio
        weight[start_s:end_s] += 1.0

    mask = weight > 0
    output[mask] /= weight[mask]

    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.95

    return numpy_to_audiosegment(output.astype(np.float32), sr)
