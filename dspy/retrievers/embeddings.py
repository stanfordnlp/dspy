import json
import os
from typing import Any

import numpy as np

from dspy.utils.unbatchify import Unbatchify


class Embeddings:
    def __init__(
        self,
        corpus: list[str],
        embedder,
        k: int = 5,
        callbacks: list[Any] | None = None,
        cache: bool = False,
        brute_force_threshold: int = 20_000,
        normalize: bool = True,
    ):
        """Initialize the Embedding retriever.

        Args:
            corpus: List of text passages to be indexed and searched.
            embedder: Embedding model that converts text to vector representations.
            k: Number of top relevant passages to retrieve for each query. Defaults to 5.
            callbacks: Reserved for future use. Currently ignored if provided.
            cache: MUST be False. Caching is explicitly unsupported and will raise AssertionError if enabled.
            brute_force_threshold: Threshold size for using brute-force search instead of FAISS. If corpus size is below this threshold, brute-force search is used. Defaults to 20,000.
            normalize: Whether to normalize embeddings to unit vectors before similarity calculation. Normalization ensures that cosine similarity equals dot product. Defaults to True.
        ```
        """
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
        """Retrieve relevant passages for a single query (alias for forward()).
    
        Args:
            query: Text query to search against the corpus.
        
        Returns:
            dspy.Prediction: Object containing 'passages' (list of retrieved texts) 
            and 'indices' (their positions in the original corpus).
        """
        return self.forward(query)

    def forward(self, query: str):
        """Retrieve relevant passages for a single query.

        Args:
            query: Text query to find relevant passages for.

        Returns:
            dspy.Prediction: Object with 'passages' (list of retrieved passages) and
            'indices' (list of their indices in the corpus).
        """
        import dspy

        passages, indices = self.search_fn(query)
        return dspy.Prediction(passages=passages, indices=indices)

    def _batch_forward(self, queries: list[str]):
        """
        Retrieve relevant passages for a batch of queries.

        Args:
            queries: List of text queries to find relevant passages for.

        Returns:
            List[Tuple[List[str], List[int]]]:
                For each query, a tuple containing:
                - Top-k retrieved passages (list of strings)
                - Corresponding corpus indices (list of integers)
        """
        q_embeds = self.embedder(queries)
        q_embeds = self._normalize(q_embeds) if self.normalize else q_embeds

        pids = self._faiss_search(q_embeds, self.k * 10) if self.index else None
        pids = np.tile(np.arange(len(self.corpus)), (len(queries), 1)) if pids is None else pids

        return self._rerank_and_predict(q_embeds, pids)

    def _build_faiss(self):
        """
        Build a FAISS index for efficient similarity search.

        Returns:
            faiss.Index: Trained FAISS index object (concrete type depends on configuration).

        Note:
            Index type is IndexIVFPQ in current implementation, but this may change.
        
        Raises:
            ImportError: If FAISS library is not installed.
        """
        nbytes = 32
        partitions = int(2 * np.sqrt(len(self.corpus)))
        dim = self.corpus_embeddings.shape[1]

        try:
            import faiss
        except ImportError:
            raise ImportError("Please `pip install faiss-cpu` or increase `brute_force_threshold` to avoid FAISS.")

        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, partitions, nbytes, 8)

        print(
            f"Training a {nbytes}-byte FAISS index with {partitions} partitions, based on "
            f"{len(self.corpus)} x {dim}-dim embeddings"
        )
        index.train(self.corpus_embeddings)
        index.add(self.corpus_embeddings)
        index.nprobe = min(16, partitions)

        return index

    def _faiss_search(self, query_embeddings: np.ndarray, num_candidates: int):
        """Perform similarity search using the FAISS index.

        Args:
            query_embeddings: Numpy array of query embeddings with shape (n_queries, embedding_dim).
            num_candidates: Number of candidates to retrieve for each query.
            
        Returns:
            np.ndarray: Candidate indices array of shape (n_queries, num_candidates).
        """
        return self.index.search(query_embeddings, num_candidates)[1]

    def _rerank_and_predict(self, q_embeds: np.ndarray, candidate_indices: np.ndarray):
        """
        Rerank candidates using dot-product similarity (equivalent to cosine similarity when embeddings are normalized).

        Args:
            q_embeds: Numpy array of query embeddings with shape (n_queries, embedding_dim).
            candidate_indices: Numpy array of candidate indices with shape (n_queries, num_candidates).

        Returns:
            List[Tuple[List[str], List[int]]]:
                For each query, a tuple containing:
                - Top-k retrieved passages (list of strings)
                - Corresponding corpus indices (list of integers)
        """
        candidate_embeddings = self.corpus_embeddings[candidate_indices]
        scores = np.einsum("qd,qkd->qk", q_embeds, candidate_embeddings)

        top_k_indices = np.argsort(-scores, axis=1)[:, : self.k]
        top_indices = candidate_indices[np.arange(len(q_embeds))[:, None], top_k_indices]

        return [([self.corpus[idx] for idx in indices], [idx for idx in indices]) for indices in top_indices]  # noqa: C416

    def _normalize(self, embeddings: np.ndarray):
        """
        Normalize embeddings to unit vectors.
        
        Args:
            embeddings: Numpy array of embeddings to normalize.
            
        Returns:
            np.ndarray: Normalized embeddings with unit length.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-10)

    def save(self, path: str):
        """
        Save the embeddings index to disk.

        This saves the corpus, embeddings, FAISS index (if present), and configuration
        to allow for fast loading without recomputing embeddings.

        Args:
            path: Directory path where the embeddings will be saved
        """
        os.makedirs(path, exist_ok=True)

        # Save configuration and corpus
        config = {
            "k": self.k,
            "normalize": self.normalize,
            "corpus": self.corpus,
            "has_faiss_index": self.index is not None,
        }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save embeddings
        np.save(os.path.join(path, "corpus_embeddings.npy"), self.corpus_embeddings)

        # Save FAISS index if it exists
        if self.index is not None:
            try:
                import faiss
                faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))
            except ImportError:
                # If FAISS is not available, we can't save the index
                # but we can still save the embeddings for brute force search
                pass

    def load(self, path: str, embedder):
        """
        Load the embeddings index from disk into the current instance.

        Args:
            path: Directory path where the embeddings were saved
            embedder: The embedder function to use for new queries

        Returns:
            self: Returns self for method chaining

        Raises:
            FileNotFoundError: If the save directory or required files don't exist
            ValueError: If the saved config is invalid or incompatible
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Save directory not found: {path}")

        config_path = os.path.join(path, "config.json")
        embeddings_path = os.path.join(path, "corpus_embeddings.npy")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        # Load configuration and corpus
        with open(config_path) as f:
            config = json.load(f)

        # Validate required config fields
        required_fields = ["k", "normalize", "corpus", "has_faiss_index"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Invalid config: missing required field '{field}'")

        # Restore configuration
        self.k = config["k"]
        self.normalize = config["normalize"]
        self.corpus = config["corpus"]
        self.embedder = embedder

        # Load embeddings
        self.corpus_embeddings = np.load(embeddings_path)

        # Load FAISS index if it was saved and FAISS is available
        faiss_index_path = os.path.join(path, "faiss_index.bin")
        if config["has_faiss_index"] and os.path.exists(faiss_index_path):
            try:
                import faiss
                self.index = faiss.read_index(faiss_index_path)
            except ImportError:
                # If FAISS is not available, fall back to brute force
                self.index = None
        else:
            self.index = None

        return self

    @classmethod
    def from_saved(cls, path: str, embedder):
        """
        Create an Embeddings instance from a saved index.

        This is the recommended way to load saved embeddings as it creates a new
        instance without unnecessarily computing embeddings.

        Args:
            path: Directory path where the embeddings were saved
            embedder: The embedder function to use for new queries

        Returns:
            Embeddings instance loaded from disk

        Example:
            ```python
            # Save embeddings
            embeddings = Embeddings(corpus, embedder)
            embeddings.save("./saved_embeddings")

            # Load embeddings later
            loaded_embeddings = Embeddings.from_saved("./saved_embeddings", embedder)
            ```
        """
        # Create a minimal instance without triggering embedding computation
        instance = cls.__new__(cls)
        # Initialize the search function (required since we bypassed __init__)
        instance.search_fn = Unbatchify(instance._batch_forward)
        instance.load(path, embedder)
        return instance
