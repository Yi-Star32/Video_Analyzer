"""Audio embedding pipeline (Wav2Vec2) — used to embed phoneme audio segments."""

import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class AudioEmbeddingPipeline:
    def __init__(self, model_name="facebook/wav2vec2-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_audio(self, audio: np.ndarray, sr: int = 16000, normalize: bool = True) -> np.ndarray:
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            hidden = self.model(input_values).last_hidden_state
            emb = hidden.mean(dim=1).cpu().numpy()
        if normalize:
            emb = self._l2_normalize(emb)
        return emb

    def embed_batch(self, audio_chunks, batch_size=16, sr=16000, normalize=True):
        all_embeddings = []
        for i in tqdm(range(0, len(audio_chunks), batch_size)):
            batch = audio_chunks[i : i + batch_size]
            waveforms = [c.audio if hasattr(c, "audio") else c["audio"] for c in batch]
            inputs = self.processor(waveforms, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)
            with torch.no_grad():
                hidden = self.model(input_values).last_hidden_state
                embeddings = hidden.mean(dim=1).cpu().numpy()
            if normalize:
                embeddings = self._l2_normalize(embeddings)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 768))

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return x / norms
