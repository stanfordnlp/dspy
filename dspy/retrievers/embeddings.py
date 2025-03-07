import numpy as np
from typing import Any, List, Optional
from dspy.utils.unbatchify import Unbatchify

# TODO: Add .save and .load methods!


class Embeddings:
    def __init__(
        self,
        corpus: List[str],
        embedder,
        k: int = 5,
        callbacks: Optional[List[Any]] = None,
        cache: bool = False,
        brute_force_threshold: int = 20_000,
        normalize: bool = True
    ):
        assert cache is False, "Caching is not supported for embeddings-based retrievers"

        self.embedder = embedder
        self.k = k
        self.corpus = corpus
        self.normalize = normalize

        self.corpus_embeddings = self.embedder(self.corpus)
        self.corpus_embeddings = self._normalize(self.corpus_embeddings) if self.normalize else self.corpus_embeddings

        self.index = self._build_faiss() if len(corpus) >= brute_force_threshold else None
        self.search_fn = Unbatchify(self._batch_forward)

    def __call__(self, query: str):
        return self.forward(query)

    def forward(self, query: str):
        import dspy

        passages, indices = self.search_fn(query)
        return dspy.Prediction(passages=passages, indices=indices)

    def _batch_forward(self, queries: List[str]):
        q_embeds = self.embedder(queries)
        q_embeds = self._normalize(q_embeds) if self.normalize else q_embeds

        pids = self._faiss_search(q_embeds, self.k * 10) if self.index else None
        pids = np.tile(np.arange(len(self.corpus)), (len(queries), 1)) if pids is None else pids
        
        return self._rerank_and_predict(q_embeds, pids)

    def _build_faiss(self):
        nbytes = 32
        partitions = int(2 * np.sqrt(len(self.corpus)))
        dim = self.corpus_embeddings.shape[1]

        try:
            import faiss
        except ImportError:
            raise ImportError("Please `pip install faiss-cpu` or increase `brute_force_threshold` to avoid FAISS.")

        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, partitions, nbytes, 8)

        print(f"Training a {nbytes}-byte FAISS index with {partitions} partitions, based on "
              f"{len(self.corpus)} x {dim}-dim embeddings")
        index.train(self.corpus_embeddings)
        index.add(self.corpus_embeddings)
        index.nprobe = min(16, partitions)

        return index

    def _faiss_search(self, query_embeddings: np.ndarray, num_candidates: int):
        return self.index.search(query_embeddings, num_candidates)[1]

    def _rerank_and_predict(self, q_embeds: np.ndarray, candidate_indices: np.ndarray):
        candidate_embeddings = self.corpus_embeddings[candidate_indices]
        scores = np.einsum('qd,qkd->qk', q_embeds, candidate_embeddings)

        top_k_indices = np.argsort(-scores, axis=1)[:, :self.k]
        top_indices = candidate_indices[np.arange(len(q_embeds))[:, None], top_k_indices]

        return [([self.corpus[idx] for idx in indices], [idx for idx in indices]) for indices in top_indices]

    def _normalize(self, embeddings: np.ndarray):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-10)
