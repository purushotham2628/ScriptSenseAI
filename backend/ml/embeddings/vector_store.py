from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: Dict


class VectorIndexService:
    """Stores symbol/script/context embeddings for retrieval on unseen scripts."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.ids: List[str] = []
        self.metadata: List[Dict] = []
        self.index = faiss.IndexFlatIP(dim) if faiss else None
        self._fallback_vectors: List[np.ndarray] = []

    def add(self, item_id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> None:
        vector = self._normalize(vector.reshape(1, -1).astype("float32"))
        self.ids.append(item_id)
        self.metadata.append(metadata or {})
        if self.index is not None:
            self.index.add(vector)
        else:
            self._fallback_vectors.append(vector[0])

    def search(self, vector: np.ndarray, top_k: int = 8) -> List[SearchResult]:
        if not self.ids:
            return []
        query = self._normalize(vector.reshape(1, -1).astype("float32"))
        if self.index is not None:
            scores, indices = self.index.search(query, min(top_k, len(self.ids)))
            return [
                SearchResult(id=self.ids[idx], score=float(score), metadata=self.metadata[idx])
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            ]
        matrix = np.stack(self._fallback_vectors)
        scores = matrix @ query[0]
        order = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(id=self.ids[i], score=float(scores[i]), metadata=self.metadata[i]) for i in order]

    def anomaly_score(self, vector: np.ndarray) -> float:
        results = self.search(vector, top_k=1)
        if not results:
            return 1.0
        return float(1.0 - max(0.0, min(1.0, results[0].score)))

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector, axis=1, keepdims=True) + 1e-12
        return vector / norm
