"""WhisperX forced alignment → phoneme timestamps."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import whisperx


@dataclass
class PhonemeData:
    phoneme: str
    start: float
    end: float
    confidence: float = 1.0
    word: str = ""
    word_start: float = 0.0
    word_end: float = 0.0
    source_file: str = ""


@dataclass
class AlignedWord:
    text: str
    start: float
    end: float
    confidence: float
    phonemes: List[PhonemeData] = field(default_factory=list)


@dataclass
class AlignmentResult:
    words: List[AlignedWord]
    language: str
    audio_duration: float


_CMUDICT: dict[str, list[str]] = {}


def _load_cmudict(path: Optional[Path] = None) -> dict[str, list[str]]:
    global _CMUDICT
    if _CMUDICT:
        return _CMUDICT
    if path is None:
        path = Path(__file__).parent / "cmudict.json"
    if path.exists():
        with open(path) as f:
            raw = json.load(f)
        _CMUDICT = {k.lower(): v for k, v in raw.items()}
    return _CMUDICT


def _word_to_phonemes(word: str, lexicon: dict[str, list[str]]) -> list[str]:
    cleaned = "".join(c for c in word if c.isalpha()).lower()
    if cleaned in lexicon:
        return lexicon[cleaned]
    phonemes = []
    for ch in cleaned:
        phonemes.append(f"{ch.upper()}")
    return phonemes


def _distribute_by_chars(
    word_text: str,
    phoneme_seq: list[str],
    char_alignments: list[dict],
) -> list[PhonemeData]:
    if not phoneme_seq or not char_alignments:
        return []

    chars = [c["text"] for c in char_alignments]
    char_starts = [c["start"] for c in char_alignments]
    char_ends = [c["end"] for c in char_alignments]

    phoneme_map = {p: [] for p in phoneme_seq}
    p_idx = 0
    n_phonemes = len(phoneme_seq)
    n_chars = len(chars)

    if n_phonemes == 0:
        return []

    equal_shares = max(1, n_chars // n_phonemes)

    for i, ch in enumerate(chars):
        target = min(i // equal_shares, n_phonemes - 1) if equal_shares > 0 else 0
        key = phoneme_seq[target]
        phoneme_map[key].append(i)

    result = []
    for p_text in phoneme_seq:
        indices = phoneme_map.get(p_text, [])
        if not indices:
            continue
        start_t = char_starts[indices[0]]
        end_t = char_ends[indices[-1]]
        conf = np.mean([c.get("score", 1.0) for c in [char_alignments[i] for i in indices]])
        result.append(
            PhonemeData(
                phoneme=p_text,
                start=start_t,
                end=end_t,
                confidence=float(conf),
                word=word_text,
            )
        )
    return result


def _distribute_equally(
    word_text: str,
    phoneme_seq: list[str],
    word_start: float,
    word_end: float,
) -> list[PhonemeData]:
    duration = word_end - word_start
    n = len(phoneme_seq)
    if n == 0:
        return []
    seg_dur = duration / n
    result = []
    for i, p in enumerate(phoneme_seq):
        st = word_start + i * seg_dur
        en = st + seg_dur
        result.append(
            PhonemeData(phoneme=p, start=st, end=en, confidence=1.0, word=word_text)
        )
    return result


class PhonemeAligner:
    def __init__(
        self,
        device: Optional[str] = None,
        language: str = "en",
        whisper_model: str = "base",
        batch_size: int = 16,
        compute_type: str = "int8",
        cmudict_path: Optional[Path] = None,
    ):
        self.device = device or "cpu"
        self.language = language
        self.whisper_model_name = whisper_model
        self.batch_size = batch_size
        self.compute_type = compute_type
        self._whisper = None
        self._align_model = None
        self._align_metadata = None
        self._lexicon = _load_cmudict(cmudict_path)

    def _lazy_load_whisper(self):
        if self._whisper is None:
            self._whisper = whisperx.load_model(
                self.whisper_model_name,
                self.device,
                compute_type=self.compute_type,
            )

    def _lazy_load_aligner(self):
        if self._align_model is None:
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=self.language, device=self.device
            )

    def align(self, audio_path: str) -> AlignmentResult:
        self._lazy_load_whisper()
        self._lazy_load_aligner()

        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / 16000.0

        result = self._whisper.transcribe(audio, batch_size=self.batch_size)
        result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            return_char_alignments=True,
        )

        aligned_words: list[AlignedWord] = []
        if isinstance(result, dict):
            segments = result.get("segments", result.get("word_segments", []))
        else:
            segments = getattr(result, "segments", [])

        for seg in segments:
            words = seg.get("words", []) if isinstance(seg, dict) else []
            for w in words:
                if isinstance(w, str):
                    text = w
                    w_start = 0.0
                    w_end = 0.0
                    w_score = 1.0
                    w_chars = []
                else:
                    text = w.get("text", w.get("word", "")).strip()
                    w_start = w.get("start", 0.0)
                    w_end = w.get("end", 0.0)
                    w_score = w.get("score", w.get("confidence", 1.0))
                    w_chars = w.get("chars", [])
                if not text:
                    continue
                wa = AlignedWord(
                    text=text,
                    start=w_start,
                    end=w_end,
                    confidence=w_score,
                )
                phoneme_seq = _word_to_phonemes(text, self._lexicon)
                if w_chars:
                    wa.phonemes = _distribute_by_chars(text, phoneme_seq, w_chars)
                else:
                    wa.phonemes = _distribute_equally(
                        text, phoneme_seq, wa.start, wa.end
                    )
                aligned_words.append(wa)

        if not aligned_words:
            n_seg = len(segments)
            result_type = type(result).__name__
            keys = list(result.keys()) if isinstance(result, dict) else []
            print(f"  [diagnostic] align result: type={result_type}, "
                  f"segments={n_seg}, keys={keys}")
            if n_seg > 0 and isinstance(segments[0], dict):
                seg0 = segments[0]
                print(f"  [diagnostic] first seg keys={list(seg0.keys())}")
                first_words = seg0.get("words", [])
                print(f"  [diagnostic] first seg 'words' type={type(first_words).__name__}, len={len(first_words)}")
                if first_words:
                    w0 = first_words[0]
                    print(f"  [diagnostic] first word: type={type(w0).__name__}, val={w0}")
                    if isinstance(w0, dict):
                        print(f"  [diagnostic] word keys={list(w0.keys())}")
                        for wk in ['text','word','start','end','score','confidence']:
                            print(f"    '{wk}': {w0.get(wk, '<MISSING>')}")

        return AlignmentResult(
            words=aligned_words,
            language=self.language,
            audio_duration=duration,
        )

    def get_phonemes(self, audio_path: str) -> List[PhonemeData]:
        result = self.align(audio_path)
        phonemes = []
        for w in result.words:
            if w.phonemes:
                phonemes.extend(w.phonemes)
            else:
                phonemes.append(
                    PhonemeData(
                        phoneme=w.text.upper(),
                        start=w.start,
                        end=w.end,
                        confidence=w.confidence,
                        word=w.text,
                    )
                )
        return phonemes

    def get_word_phoneme_pairs(self, audio_path: str) -> List[AlignedWord]:
        return self.align(audio_path).words
