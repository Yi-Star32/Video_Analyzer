"""FAISS index — supports both raw embeddings and phoneme-tagged entries."""

from dataclasses import dataclass, field
from typing import List, Optional

import faiss
import numpy as np


@dataclass
class IndexEntry:
    phoneme: str
    start_time: float
    end_time: float
    word: str = ""
    audio_idx: int = 0
    source_file: str = ""


@dataclass
class PhonemeIndex:
    index: faiss.IndexFlatIP
    entries: List[IndexEntry] = field(default_factory=list)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(l2_normalize(embeddings).astype("float32"))
    return index


def search(index: faiss.IndexFlatIP, query_embeddings: np.ndarray, k: int = 1):
    return index.search(l2_normalize(query_embeddings).astype("float32"), k)


def build_phoneme_index(
    embeddings: np.ndarray,
    entries: List[IndexEntry],
) -> PhonemeIndex:
    fi = build_faiss_index(embeddings)
    return PhonemeIndex(index=fi, entries=entries)


def search_phoneme_index(
    pindex: PhonemeIndex,
    query_embedding: np.ndarray,
    k: int = 5,
    phoneme_boost: float = 0.05,
) -> List[List[IndexEntry]]:
    search_k = k * 3
    distances, indices = pindex.index.search(
        l2_normalize(query_embedding).astype("float32"), search_k
    )
    results = []
    for dist_row, idx_row in zip(distances, indices):
        scored = []
        for d, idx in zip(dist_row, idx_row):
            if 0 <= idx < len(pindex.entries):
                entry = pindex.entries[idx]
                is_chunk = entry.phoneme.startswith("CHUNK_")
                adjusted = d - phoneme_boost if is_chunk else d
                scored.append((adjusted, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        matches = [entry for _, entry in scored[:k]]
        results.append(matches)
    return results
