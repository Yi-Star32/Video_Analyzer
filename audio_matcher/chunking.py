from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class AudioChunk:
    audio: np.ndarray
    start_time: float
    end_time: float
    rms: float
    index: int


def compute_rms(audio_chunk: np.ndarray) -> float:
    if len(audio_chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_chunk**2)))


def chunk_audio_fixed_overlap(
    audio: np.ndarray,
    sr: int,
    chunk_ms: int = 300,
    hop_ms: int = 100,
    rms_threshold: float = 0.01,
    normalize: bool = True,
) -> List[AudioChunk]:
    chunk_size = int(sr * chunk_ms / 1000)
    hop_size = int(sr * hop_ms / 1000)

    chunks: List[AudioChunk] = []
    index = 0

    for start in range(0, len(audio) - chunk_size + 1, hop_size):
        end = start + chunk_size
        chunk = audio[start:end]
        rms = compute_rms(chunk)

        if rms < rms_threshold:
            continue

        if normalize:
            peak = np.max(np.abs(chunk))
            if peak > 0:
                chunk = chunk / peak

        chunks.append(
            AudioChunk(
                audio=chunk,
                start_time=start / sr,
                end_time=end / sr,
                rms=rms,
                index=index,
            )
        )
        index += 1

    return chunks
