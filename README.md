# Audio Matcher

Align song audio against a reference library using **phoneme-level forced alignment + FAISS**.

## Pipeline

```
Audio
  ↓
WhisperX forced alignment (word + character timestamps)
  ↓
Phoneme extraction (dictionary-based G2P → phoneme timestamps)
  ↓
Per-phoneme audio embedding (Wav2Vec2) + FAISS index
  ↓
Match song phonemes → reference phonemes → stitch at song timing
```

## Package structure

```
audio_matcher/
├── audio_matcher/          # Python package
│   ├── __init__.py
│   ├── alignment.py        # build_phoneme_index_from_audio, run_phoneme_pipeline
│   ├── alignment_legacy.py # old chunk-based stitch_audio
│   ├── chunking.py         # old sliding-window chunking (kept for compat)
│   ├── embedding.py        # Wav2Vec2 embedding pipeline
│   ├── index.py            # FAISS index + phoneme metadata
│   ├── io.py               # audio load / export
│   └── phonemes.py         # PhonemeAligner (WhisperX + G2P)
├── notebooks/
│   └── alignment_pipeline.ipynb
└── data/
    └── episodes_audio/<id>/audio.webm
```

## Usage

### 1. Install dependencies

```bash
pip install whisperx torch transformers faiss-cpu pydub librosa
```

### 2. Phoneme pipeline (new)

```python
from pathlib import Path
from audio_matcher import (
    PhonemeAligner,
    AudioEmbeddingPipeline,
    build_phoneme_index_from_audio,
    run_phoneme_pipeline,
)

ref_path = "data/episodes_audio/h6LFdBPjbf4/audio.webm"
song_path = "data/episodes_audio/z3hMX65Khtg/audio.webm"

aligner = PhonemeAligner(device="cpu", whisper_model="base")
embedder = AudioEmbeddingPipeline()

pindex = build_phoneme_index_from_audio(ref_path, aligner, embedder)
result = run_phoneme_pipeline(song_path, ref_path, aligner, embedder, pindex=pindex)
export_audio(result, "aligned_output.wav")
```

### 3. Old chunk-based pipeline (deprecated)

```python
from audio_matcher import AudioEmbeddingPipeline, build_reference_index, run_pipeline

pipeline = AudioEmbeddingPipeline()
index, ref_chunks = build_reference_index("ref.wav", pipeline)
final, scores = run_pipeline("song.wav", pipeline, index, ref_chunks)
export_audio(final, "output.wav")
```

## How it works

1. **Reference processing**: WhisperX transcribes + force-aligns reference audio → word + character timestamps. A G2P dictionary converts words to phoneme sequences. Each phoneme's audio is extracted and embedded with Wav2Vec2 into FAISS.

2. **Song matching**: The same alignment + embedding is applied to the song. Each song phoneme embedding is searched against the reference FAISS index to find the best-matching reference phoneme (acoustically similar).

3. **Stitching**: Reference audio segments are placed at the song's phoneme timing positions with crossfade overlap-add, preserving the original song duration.
