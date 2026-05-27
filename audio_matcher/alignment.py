"""Phoneme-based audio alignment pipeline.

Audio → WhisperX forced alignment → phoneme timestamps → FAISS per phoneme → stitch.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from pydub import AudioSegment

from .embedding import AudioEmbeddingPipeline
from .index import IndexEntry, PhonemeIndex, build_phoneme_index, search_phoneme_index
from .io import load_audio, numpy_to_audiosegment
from .phonemes import PhonemeAligner, PhonemeData


# ── helpers ──────────────────────────────────────────────────────────


def _extract_audio_segment(
    audio: np.ndarray,
    sr: int,
    start: float,
    end: float,
) -> np.ndarray:
    lo = int(start * sr)
    hi = int(end * sr)
    if hi > len(audio):
        hi = len(audio)
    if lo >= hi:
        return np.zeros(int((end - start) * sr), dtype=audio.dtype)
    return audio[lo:hi].copy()


def _peak_normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak * target
    return audio


# ── reference index build ───────────────────────────────────────────


def build_phoneme_index_from_audio(
    reference_path: Union[str, Path],
    aligner: PhonemeAligner,
    embedder: AudioEmbeddingPipeline,
) -> PhonemeIndex:
    """Transcribe, force-align, embed each phoneme segment, build FAISS."""
    audio, sr = load_audio(str(reference_path))
    phonemes = aligner.get_phonemes(str(reference_path))
    if not phonemes:
        return PhonemeIndex(
            index=build_phoneme_index(np.empty((0, 768), dtype="float32"), [])
        )

    embeddings = []
    entries = []
    for ph in phonemes:
        seg = _extract_audio_segment(audio, sr, ph.start, ph.end)
        if len(seg) < sr * 0.05:
            continue
        try:
            emb = embedder.embed_audio(seg, sr=sr, normalize=True)
        except Exception as e:
            continue
        embeddings.append(emb)
        entries.append(
            IndexEntry(
                phoneme=ph.phoneme,
                start_time=ph.start,
                end_time=ph.end,
                word=ph.word,
                audio_idx=len(entries),
            )
        )

    all_embs = np.vstack(embeddings) if embeddings else np.empty((0, 768), dtype="float32")
    return build_phoneme_index(all_embs, entries)


# ── stitching ───────────────────────────────────────────────────────


def stitch_phonemes(
    phoneme_matches: List[PhonemeData],
    reference_audio: np.ndarray,
    sr: int = 16000,
    crossfade_ms: int = 10,
    audio_cache: Optional[dict] = None,
) -> AudioSegment:
    """Stitch reference audio segments at song phoneme positions."""
    if not phoneme_matches:
        return AudioSegment.silent(duration=0)

    total_ms = int(phoneme_matches[-1].end * 1000) + 10
    total_samples = int(total_ms * sr / 1000)
    output = np.zeros(total_samples, dtype=np.float64)
    weight = np.zeros(total_samples, dtype=np.float64)

    fade_len = int(crossfade_ms * sr / 1000)

    for pm in phoneme_matches:
        ref_start = pm.start
        ref_end = pm.end
        src_audio = reference_audio
        if pm.source_file and audio_cache is not None:
            if pm.source_file not in audio_cache:
                audio_cache[pm.source_file], _ = load_audio(pm.source_file)
            src_audio = audio_cache[pm.source_file]
        seg = _extract_audio_segment(src_audio, sr, ref_start, ref_end)
        if len(seg) == 0:
            continue
        seg = seg.astype(np.float64)

        out_start = int(pm.word_start * sr) if pm.word_start > 0 else int(pm.start * sr)
        out_end = out_start + len(seg)
        if out_end > total_samples:
            seg = seg[: total_samples - out_start]
            out_end = total_samples
        if out_start >= total_samples:
            break

        if 0 < fade_len * 2 < len(seg):
            seg = seg.copy()
            seg[:fade_len] *= np.linspace(0.0, 1.0, fade_len)
            seg[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

        output[out_start:out_end] += seg
        weight[out_start:out_end] += 1.0

    mask = weight > 0
    output[mask] /= weight[mask]
    output = _peak_normalize(output)

    return numpy_to_audiosegment(output.astype(np.float32), sr)


# ── multi-file phoneme index ────────────────────────────────────────


def build_phoneme_index_from_episodes(
    file_paths: List[Path],
    aligner: PhonemeAligner,
    embedder: AudioEmbeddingPipeline,
) -> PhonemeIndex:
    """Build a combined phoneme FAISS index from multiple episode audio files.

    Each phoneme entry tracks its source file so stitching can lazy-load
    the correct audio later.
    """
    import gc
    from tqdm import tqdm

    all_embeddings: list[np.ndarray] = []
    all_entries: list[IndexEntry] = []
    entry_counter = 0

    for path in tqdm(file_paths, desc="Building phoneme index from episodes"):
        try:
            phonemes = aligner.get_phonemes(str(path))
        except Exception as e:
            print(f"  Warning: {path} — phoneme extraction failed: {e}")
            phonemes = None

        n_phonemes = len(phonemes) if phonemes else 0
        print(f"  [{path.name}] {n_phonemes} phonemes")

        if not phonemes:
            print(f"  No phonemes found — {path}, falling back to chunk-based")
            try:
                audio, sr = load_audio(str(path))
                from .chunking import chunk_audio_fixed_overlap
                chunks = chunk_audio_fixed_overlap(audio, sr)
                if not chunks:
                    del audio
                    continue
                embs = embedder.embed_batch(chunks)
                for i, (c, emb) in enumerate(zip(chunks, embs)):
                    all_embeddings.append(emb.reshape(1, -1))
                    all_entries.append(
                        IndexEntry(
                            phoneme=f"CHUNK_{i}",
                            start_time=c.start_time,
                            end_time=c.end_time,
                            word="",
                            audio_idx=entry_counter,
                            source_file=str(path),
                        )
                    )
                    entry_counter += 1
                del audio, chunks
            except Exception as e2:
                print(f"  Warning: {path.name} — fallback also failed: {e2}")
                continue
        else:
            try:
                audio, sr = load_audio(str(path))
                for ph in phonemes:
                    seg = _extract_audio_segment(audio, sr, ph.start, ph.end)
                    if len(seg) < sr * 0.05:
                        continue
                    try:
                        emb = embedder.embed_audio(seg, sr=sr, normalize=True)
                    except Exception:
                        continue
                    all_embeddings.append(emb)
                    all_entries.append(
                        IndexEntry(
                            phoneme=ph.phoneme,
                            start_time=ph.start,
                            end_time=ph.end,
                            word=ph.word,
                            audio_idx=entry_counter,
                            source_file=str(path),
                        )
                    )
                    entry_counter += 1
                del audio
            except Exception as e:
                print(f"  Warning: {path.name}: {e}")
                continue
        gc.collect()
        if hasattr(embedder, "device") and "cuda" in str(embedder.device):
            import torch
            torch.cuda.empty_cache()

    all_embs = (
        np.vstack(all_embeddings)
        if all_embeddings
        else np.empty((0, 768), dtype="float32")
    )
    print(
        f"Built phoneme index from {len(file_paths)} files "
        f"-> {len(all_entries)} phonemes"
    )
    return build_phoneme_index(all_embs, all_entries)


# ── main phoneme pipeline ───────────────────────────────────────────


def run_phoneme_pipeline(
    song_path: Union[str, Path],
    reference_path: Union[str, Path],
    aligner: Optional[PhonemeAligner] = None,
    embedder: Optional[AudioEmbeddingPipeline] = None,
    *,
    pindex: Optional[PhonemeIndex] = None,
    reference_audio: Optional[np.ndarray] = None,
    reference_sr: int = 16000,
    match_top_k: int = 3,
) -> AudioSegment:
    """Transcribe song, match each phoneme to reference index, stitch.

    Usage:
        aligner = PhonemeAligner()
        embedder = AudioEmbeddingPipeline()
        pindex = build_phoneme_index_from_audio("ref.wav", aligner, embedder)
        result = run_phoneme_pipeline("song.wav", "ref.wav", pindex=pindex)
    """
    if pindex is None and reference_path is not None:
        aligner = aligner or PhonemeAligner()
        embedder = embedder or AudioEmbeddingPipeline()
        pindex = build_phoneme_index_from_audio(reference_path, aligner, embedder)

    if pindex is None:
        raise ValueError("Provide a PhonemeIndex or a reference_path")

    if reference_audio is None and reference_path is not None:
        reference_audio, reference_sr = load_audio(str(reference_path))
    elif reference_audio is None:
        reference_audio = np.empty(0)

    aligner = aligner or PhonemeAligner()
    embedder = embedder or AudioEmbeddingPipeline()

    song_phonemes = aligner.get_phonemes(str(song_path))
    if not song_phonemes:
        return AudioSegment.silent(duration=0)

    song_audio, song_sr = load_audio(str(song_path))

    matched: List[PhonemeData] = []
    for ph in song_phonemes:
        seg = _extract_audio_segment(song_audio, song_sr, ph.start, ph.end)
        if len(seg) < song_sr * 0.05:
            matched.append(
                PhonemeData(phoneme=ph.phoneme, start=ph.start, end=ph.end, confidence=0.0, word=ph.word)
            )
            continue
        try:
            q_emb = embedder.embed_audio(seg, sr=song_sr, normalize=True)
        except Exception:
            matched.append(
                PhonemeData(phoneme=ph.phoneme, start=ph.start, end=ph.end, confidence=0.0, word=ph.word)
            )
            continue
        candidates = search_phoneme_index(pindex, q_emb, k=match_top_k)
        best = candidates[0][0] if candidates and candidates[0] else None

        if best is not None:
            matched.append(
                PhonemeData(
                    phoneme=best.phoneme,
                    start=best.start_time,
                    end=best.end_time,
                    confidence=1.0,
                    word=best.word,
                    word_start=ph.start,
                    word_end=ph.end,
                    source_file=best.source_file,
                )
            )
        else:
            matched.append(
                PhonemeData(
                    phoneme=ph.phoneme,
                    start=ph.start,
                    end=ph.end,
                    confidence=0.0,
                    word=ph.word,
                    word_start=ph.start,
                    word_end=ph.end,
                )
            )

    audio_cache = {}
    if reference_audio is not None:
        audio_cache["_default"] = reference_audio
    result = stitch_phonemes(matched, reference_audio, sr=reference_sr, audio_cache=audio_cache)

    n_match = sum(1 for m in matched if m.confidence > 0)
    print(f"song phonemes: {len(song_phonemes)}, matched: {n_match}")
    print(f"output_ms: {len(result)}")

    return result


# ── backward-compatible chunk-based wrappers ────────────────────────


def build_reference_index(
    reference_path: Union[str, Path],
    pipeline: AudioEmbeddingPipeline,
    *,
    chunk_ms: int = 300,
    hop_ms: int = 100,
    rms_threshold: float = 0.01,
):
    """Original chunk-based reference index builder (deprecated)."""
    from .chunking import chunk_audio_fixed_overlap
    from .index import build_faiss_index as _build_faiss

    audio, sr = load_audio(str(reference_path))
    chunks = chunk_audio_fixed_overlap(
        audio, sr, chunk_ms=chunk_ms, hop_ms=hop_ms, rms_threshold=rms_threshold
    )
    embeddings = pipeline.embed_batch(chunks, normalize=False)
    index = _build_faiss(embeddings)
    return index, chunks


def _resolve_chunk_audio(chunk, cache, sr=16000):
    """Extract audio from a reference chunk, loading from file if needed."""
    from .chunking import AudioChunk
    if isinstance(chunk, AudioChunk):
        return chunk.audio
    if "audio" in chunk:
        return chunk["audio"]
    source = chunk.get("source_file")
    if not source:
        raise ValueError("Chunk has no audio, source_file, and is not AudioChunk")
    if source not in cache:
        cache[source], _ = load_audio(source)
    file_audio = cache[source]
    lo = int(chunk["start_time"] * sr)
    hi = int(chunk["end_time"] * sr)
    if hi > len(file_audio):
        hi = len(file_audio)
    if lo >= hi:
        return np.zeros(int((chunk["end_time"] - chunk["start_time"]) * sr), dtype=file_audio.dtype)
    return file_audio[lo:hi].copy()


def run_pipeline(
    song_path: Union[str, Path],
    pipeline: AudioEmbeddingPipeline,
    faiss_index,
    reference_chunks: Sequence,
    *,
    chunk_ms: int = 300,
    hop_ms: int = 100,
    rms_threshold: float = 0.01,
):
    """Original chunk-based pipeline (deprecated)."""
    from .chunking import AudioChunk, chunk_audio_fixed_overlap
    from .index import search as _search

    audio, sr = load_audio(str(song_path))
    song_chunks = chunk_audio_fixed_overlap(
        audio, sr, chunk_ms=chunk_ms, hop_ms=hop_ms, rms_threshold=rms_threshold
    )
    song_embeddings = pipeline.embed_batch(song_chunks, normalize=False)
    res = _search(faiss_index, song_embeddings, k=1)
    if isinstance(res, tuple) and len(res) == 2:
        distances, indices = res
    else:
        raise RuntimeError("Unexpected return from search()")

    max_duration_ms = int(len(audio) / sr * 1000)
    mapped: list[AudioChunk] = []
    audio_cache = {}
    for song_chunk, idx_raw in zip(song_chunks, indices[:, 0]):
        try:
            idx = int(idx_raw)
        except Exception:
            continue
        if idx < 0 or idx >= len(reference_chunks):
            continue
        rc = reference_chunks[idx]
        ref_audio = _resolve_chunk_audio(rc, audio_cache, sr)
        mapped.append(
            AudioChunk(
                audio=ref_audio,
                start_time=song_chunk.start_time,
                end_time=song_chunk.end_time,
                rms=song_chunk.rms,
                index=song_chunk.index,
            )
        )

    if not mapped:
        return AudioSegment.silent(duration=0), distances

    from .alignment_legacy import stitch_audio

    result = stitch_audio(mapped, sr, max_duration_ms=max_duration_ms)
    print(f"input_ms: {max_duration_ms}, output_ms: {len(result)}")
    return result, distances
