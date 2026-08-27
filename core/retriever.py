import json
import os
import time
from typing import List, Dict, Any
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class LegalClauseRetriever:
    """Zero-recompute retrieval engine using serialized FAISS index and BGE embeddings."""

    def __init__(
        self,
        index_path: str = "artifacts/contract_index.faiss",
        meta_path: str = "artifacts/clauses_metadata.json",
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found at: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata JSON file not found at: {meta_path}")

        # 1. Load serialized FAISS index
        self.index = faiss.read_index(index_path)

        # 2. Load clause metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata: List[Dict[str, Any]] = json.load(f)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index count ({self.index.ntotal}) does not match metadata count ({len(self.metadata)})"
            )

        # 3. Load embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(model_name, device=self.device)
        self.query_prefix = "Represent this sentence for searching relevant passages: "

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant clauses with cosine similarity scores."""
        formatted_query = f"{self.query_prefix}{query}"
        
        # Compute normalized embedding
        query_vec = self.embedder.encode(
            [formatted_query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores, indices = self.index.search(query_vec, top_k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            clause = self.metadata[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "clause_id": clause["clause_id"],
                "header": clause["header"],
                "text": clause["text"],
                "start_char": clause.get("start_char", 0),
                "end_char": clause.get("end_char", 0),
            })

        return results
