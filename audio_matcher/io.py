import librosa
import numpy as np
from pydub import AudioSegment


def load_audio(file_path: str, sr: int = 16000):
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    return audio, sr


def numpy_to_audiosegment(audio: np.ndarray, sr: int = 16000) -> AudioSegment:
    pcm = (audio * 32767).astype(np.int16)
    return AudioSegment(
        pcm.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )


def export_audio(audio_segment: AudioSegment, path: str = "output.wav") -> None:
    audio_segment.export(path, format="wav")
