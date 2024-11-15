"""Retriever model for faiss: https://github.com/facebookresearch/faiss.
Author: Jagane Sundar: https://github.com/jagane.
(modified to support `dspy.Retriever` interface)
"""

import logging
from typing import List, Any, Optional

import numpy as np

import dspy
from dspy import Embedder

try:
    import faiss
except ImportError:
    faiss = None

if faiss is None:
    raise ImportError(
        """
        The faiss package is required. Install it using `pip install dspy-ai[faiss-cpu]`
        """,
    )

logger = logging.getLogger(__name__)

class FaissRetriever(dspy.Retriever):
    """A retrieval module that uses an in-memory Faiss index to return the top passages for a given query.

    Args:
        document_chunks: The input text chunks.
        embedder: An instance of `dspy.Embedder` to compute embeddings.
        k (int, optional): The number of top passages to retrieve. Defaults to 3.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriever:

        ```python
        import dspy
        from dspy.retriever.faiss_retriever import FaissRetriever

        # Custom embedding function using SentenceTransformers and dspy.Embedder
        def sentence_transformers_embedder(texts):
            #(pip install sentence-transformers)
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts, batch_size=256, normalize_embeddings=True)
            return embeddings.tolist()
        embedder = dspy.Embedder(embedding_function=sentence_transformers_embedder)

        document_chunks = [
            "The superbowl this year was played between the San Francisco 49ers and the Kansas City Chiefs",
            "Pop corn is often served in a bowl",
            "The Rice Bowl is a Chinese Restaurant located in the city of Tucson, Arizona",
            "Mars is the fourth planet in the Solar System",
            "An aquarium is a place where children can learn about marine life",
            "The capital of the United States is Washington, D.C",
            "Rock and Roll musicians are honored by being inducted in the Rock and Roll Hall of Fame",
            "Music albums were published on Long Play Records in the 70s and 80s",
            "Sichuan cuisine is a spicy cuisine from central China",
            "The interest rates for mortgages are considered to be very high in 2024",
        ]

        retriever = FaissRetriever(document_chunks, embedder=embedder)
        results = retriever("I am in the mood for Chinese food").passages
        print(results)
        ```
    """

    def __init__(
        self,
        document_chunks: List[str],
        embedder: Optional[Embedder] = None,
        k: int = 3,
        callbacks: Optional[List[Any]] = None,
    ):
        """Inits the faiss retriever.

        Args:
            document_chunks: A list of input strings.
            embedder: An instance of `dspy.Embedder` to compute embeddings.
            k: Number of matches to return.
        """
        if embedder is not None and not isinstance(embedder, dspy.Embedder):
            raise ValueError("If provided, the embedder must be of type `dspy.Embedder`.")
        self.embedder = embedder
        embeddings = self.embedder(document_chunks)
        xb = np.array(embeddings)
        d = xb.shape[1]
        logger.info(f"FaissRetriever: embedding size={d}")
        if len(xb) < 100:
            self._faiss_index = faiss.IndexFlatL2(d)
            self._faiss_index.add(xb)
        else:
            # If we have at least 100 vectors, we use Voronoi cells
            nlist = 100
            quantizer = faiss.IndexFlatL2(d)
            self._faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self._faiss_index.train(xb)
            self._faiss_index.add(xb)

        logger.info(f"{self._faiss_index.ntotal} vectors in faiss index")
        self._document_chunks = document_chunks  # Save the input document chunks

        super().__init__(embedder=self.embedder, k=k, callbacks=callbacks)

    def _dump_raw_results(self, queries, index_list, distance_list) -> None:
        for i in range(len(queries)):
            indices = index_list[i]
            distances = distance_list[i]
            logger.debug(f"Query: {queries[i]}")
            for j in range(len(indices)):
                logger.debug(
                    f"    Hit {j} = {indices[j]}/{distances[j]}: {self._document_chunks[indices[j]]}"
                )
        return

    def forward(self, query: str, k: Optional[int] = None, **kwargs) -> dspy.Prediction:
        """Search the faiss index for k or self.k top passages for query.

        Args:
            query (str): The query to search for.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        k = k or self.k
        embeddings = self.embedder([query])
        emb_npa = np.array(embeddings)
        distance_list, index_list = self._faiss_index.search(emb_npa, k)
        # self._dump_raw_results([query], index_list, distance_list)
        passages = [self._document_chunks[ind] for ind in index_list[0]]
        doc_ids = [ind for ind in index_list[0]]
        return dspy.Prediction(passages=passages, doc_ids=doc_ids)
