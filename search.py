"""Semantic search over campus notices using sentence embeddings."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.classify import get_encoder

DATA = Path("data/notices.csv")


class NoticeIndex:
    def __init__(self, texts=None):
        if texts is None:
            df = pd.read_csv(DATA).dropna(subset=["text"])
            texts = df["text"].tolist()
        self.texts = list(texts)
        self.emb = (get_encoder().encode(self.texts, normalize_embeddings=True)
                    if self.texts else None)

    def add(self, text: str):
        """Add a new notice at runtime (used when a student pastes one in the app)."""
        vec = get_encoder().encode([text], normalize_embeddings=True)
        self.texts.append(text)
        self.emb = vec if self.emb is None else np.vstack([self.emb, vec])

    def search(self, query: str, k: int = 3):
        if not self.texts:
            return []
        q = get_encoder().encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q               # cosine similarity (vectors are normalized)
        order = np.argsort(-scores)[:k]
        return [{"idx": int(i), "text": self.texts[i], "score": float(scores[i])}
                for i in order]
