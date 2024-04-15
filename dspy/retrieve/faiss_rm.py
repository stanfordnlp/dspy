"""Retriever model for faiss: https://github.com/facebookresearch/faiss.
Author: Jagane Sundar: https://github.com/jagane.
"""

from typing import Optional, Union

import numpy as np

import dspy
from dsp.modules.sentence_vectorizer import SentenceTransformersVectorizer
from dsp.utils import dotdict

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


class FaissRM(dspy.Retrieve):
    """A retrieval module that uses an in-memory Faiss to return the top passages for a given query.

    Args:
        document_chunks: the input text chunks
        vectorizer: an object that is a subclass of BaseSentenceVectorizer
        k (int, optional): The number of top passages to retrieve. Defaults to 3.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriver:
        ```python
        import dspy
        from dspy.retrieve import faiss_rm

        document_chunks = [
            "The superbowl this year was played between the San Francisco 49ers and the Kanasas City Chiefs",
            "Pop corn is often served in a bowl",
            "The Rice Bowl is a Chinese Restaurant located in the city of Tucson, Arizona",
            "Mars is the fourth planet in the Solar System",
            "An aquarium is a place where children can learn about marine life",
            "The capital of the United States is Washington, D.C",
            "Rock and Roll musicians are honored by being inducted in the Rock and Roll Hall of Fame",
            "Music albums were published on Long Play Records in the 70s and 80s",
            "Sichuan cuisine is a spicy cuisine from central China",
            "The interest rates for mortgages is considered to be very high in 2024",
        ]

        frm = faiss_rm.FaissRM(document_chunks)
        turbo = dspy.OpenAI(model="gpt-3.5-turbo")
        dspy.settings.configure(lm=turbo, rm=frm)
        print(frm(["I am in the mood for Chinese food"]))
        ```

        Below is a code snippet that shows how to use this in the forward() function of a module
        ```python
        self.retrieve = FaissRM(k=num_passages)
        ```
    """

    def __init__(self, document_chunks, vectorizer=None, k: int = 3):
        """Inits the faiss retriever.

        Args:
            document_chunks: a list of input strings.
            vectorizer: an object that is a subclass of BaseTransformersVectorizer.
            k: number of matches to return.
        """
        if vectorizer:
            self._vectorizer = vectorizer
        else:
            self._vectorizer = SentenceTransformersVectorizer()
        embeddings = self._vectorizer(document_chunks)
        xb = np.array(embeddings)
        d = len(xb[0])
        dspy.logger.info(f"FaissRM: embedding size={d}")
        if len(xb) < 100:
            self._faiss_index = faiss.IndexFlatL2(d)
            self._faiss_index.add(xb)
        else:
            # if we have at least 100 vectors, we use Voronoi cells
            nlist = 100
            quantizer = faiss.IndexFlatL2(d)
            self._faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self._faiss_index.train(xb)
            self._faiss_index.add(xb)

        dspy.logger.info(f"{self._faiss_index.ntotal} vectors in faiss index")
        self._document_chunks = document_chunks  # save the input document chunks

        super().__init__(k=k)

    def _dump_raw_results(self, queries, index_list, distance_list) -> None:
        for i in range(len(queries)):
            indices = index_list[i]
            distances = distance_list[i]
            dspy.logger.debug(f"Query: {queries[i]}")
            for j in range(len(indices)):
                dspy.logger.debug(f"    Hit {j} = {indices[j]}/{distances[j]}: {self._document_chunks[indices[j]]}")
        return

    def forward(self, query_or_queries: Union[str, list[str]], k: Optional[int] = None, **kwargs) -> dspy.Prediction:
        """Search the faiss index for k or self.k top passages for query.

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]  # Filter empty queries
        embeddings = self._vectorizer(queries)
        emb_npa = np.array(embeddings)
        # For single query, just look up the top k passages
        if len(queries) == 1:
            distance_list, index_list = self._faiss_index.search(emb_npa, k or self.k)
            # self._dump_raw_results(queries, index_list, distance_list)
            passages = [(self._document_chunks[ind], ind) for ind in index_list[0]]
            return [dotdict({"long_text": passage[0], "index": passage[1]}) for passage in passages]

        distance_list, index_list = self._faiss_index.search(emb_npa, (k or self.k) * 3, **kwargs)
        # self._dump_raw_results(queries, index_list, distance_list)
        passage_scores = {}
        for emb in range(len(embeddings)):
            indices = index_list[emb]  # indices of neighbors for embeddings[emb] - this is an array of k*3 integers
            distances = distance_list[
                emb
            ]  # distances of neighbors for embeddings[emb] - this is an array of k*3 floating point numbers
            for res in range((k or self.k) * 3):
                neighbor = indices[res]
                distance = distances[res]
                if neighbor in passage_scores:
                    passage_scores[neighbor].append(distance)
                else:
                    passage_scores[neighbor] = [distance]
        # Note re. sorting:
        # first degree sort: number of queries that got a hit with any particular document chunk. More
        # is a better match. This is len(queries)-len(x[1])
        # second degree sort: sum of the distances of each hit returned by faiss. Smaller distance is a better match
        sorted_passages = sorted(passage_scores.items(), key=lambda x: (len(queries) - len(x[1]), sum(x[1])))[
            : k or self.k
        ]
        return [
            dotdict({"long_text": self._document_chunks[passage_index], "index": passage_index})
            for passage_index, _ in sorted_passages
        ]
