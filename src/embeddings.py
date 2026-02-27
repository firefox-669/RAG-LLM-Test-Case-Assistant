# -*- coding: utf-8 -*-
from typing import List
import numpy as np
from config import Config
import sys
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    pass
class SimpleEmbedding:
    def __init__(self):
        self.dimension = 384
    def encode(self, texts, convert_to_tensor=False, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            vector = np.zeros(self.dimension)
            words = text.lower().split()
            for i, word in enumerate(words[:100]):
                hash_val = abs(hash(word)) % self.dimension
                weight = 1.0 / (1.0 + i * 0.01)
                vector[hash_val] += weight
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            embeddings.append(vector)
        return np.array(embeddings)
    def get_sentence_embedding_dimension(self):
        return self.dimension
class EmbeddingModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.use_simple = False
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = SimpleEmbedding()
                self.use_simple = True
        else:
            self.model = SimpleEmbedding()
            self.use_simple = True
    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_tensor=False)
        if isinstance(embedding, np.ndarray):
            if embedding.ndim == 2:
                return embedding[0].tolist()
            return embedding.tolist()
        return embedding
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        show_progress = len(texts) > 10 and not self.use_simple
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=show_progress)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return embeddings
    def similarity(self, text1: str, text2: str) -> float:
        emb1 = self.model.encode(text1, convert_to_tensor=False)
        emb2 = self.model.encode(text2, convert_to_tensor=False)
        if isinstance(emb1, np.ndarray) and emb1.ndim == 2:
            emb1 = emb1[0]
        if isinstance(emb2, np.ndarray) and emb2.ndim == 2:
            emb2 = emb2[0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        return float(similarity)
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
_embedding_model = None
def get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
