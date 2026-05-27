"""Audio alignment: match song segments to a reference library via phoneme alignment + FAISS.

Pipeline:
  Audio → WhisperX forced alignment → phoneme timestamps → FAISS per phoneme → stitch
"""

from .alignment import (
    build_phoneme_index_from_audio,
    build_phoneme_index_from_episodes,
    build_reference_index,
    run_phoneme_pipeline,
    run_pipeline,
    stitch_phonemes,
)
from .chunking import AudioChunk, chunk_audio_fixed_overlap
from .embedding import AudioEmbeddingPipeline
from .index import IndexEntry, PhonemeIndex, build_faiss_index, search, search_phoneme_index
from .io import export_audio, load_audio
from .phonemes import AlignedWord, PhonemeAligner, PhonemeData, AlignmentResult

__all__ = [
    "AlignedWord",
    "AlignmentResult",
    "AudioChunk",
    "AudioEmbeddingPipeline",
    "PhonemeAligner",
    "PhonemeData",
    "PhonemeIndex",
    "IndexEntry",
    "build_faiss_index",
    "build_phoneme_index_from_audio",
    "build_phoneme_index_from_episodes",
    "build_reference_index",
    "chunk_audio_fixed_overlap",
    "export_audio",
    "load_audio",
    "run_phoneme_pipeline",
    "run_pipeline",
    "search",
    "search_phoneme_index",
    "stitch_phonemes",
]
