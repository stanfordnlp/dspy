import abc
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np

import dspy
from dspy.dsp.utils import dotdict
from dspy.primitives.example import Example

try:
    from qdrant_client import QdrantClient, models
except ImportError as e:
    raise ImportError(
        "The 'qdrant' extra is required to use QdrantRM. Install it with `pip install dspy-ai[qdrant]`",
    ) from e


class BaseSentenceVectorizer(abc.ABC):
    """
    Base Class for Vectorizers.
    """

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        pass

    def _extract_text_from_examples(self, inp_examples: List["Example"]) -> List[str]:
        if isinstance(inp_examples[0], str):
            return inp_examples
        return [" ".join([example[key] for key in example._input_keys]) for example in inp_examples]


class FastEmbedVectorizer(BaseSentenceVectorizer):
    """Sentence vectorizer implementaion using FastEmbed - https://qdrant.github.io/fastembed."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 256,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        parallel: Optional[int] = None,
        **kwargs,
    ):
        """Initialize fastembed.TextEmbedding.

        Args:
            model_name (str): The name of the model to use. Defaults to `"BAAI/bge-small-en-v1.5"`.
            batch_size (int): Batch size for encoding. Higher values will use more memory, but be faster.\
                                        Defaults to 256.
            cache_dir (str, optional): The path to the model cache directory.\
                                       Can also be set using the `FASTEMBED_CACHE_PATH` env variable.
            threads (int, optional): The number of threads single onnxruntime session can use.
            parallel (int, optional): If `>1`, data-parallel encoding will be used, recommended for large datasets.\
                                      If `0`, use all available cores.\
                                      If `None`, don't use data-parallel processing, use default onnxruntime threading.\
                                      Defaults to None.
            **kwargs: Additional options to pass to fastembed.TextEmbedding
        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-small-en-v1.5.
        """
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ValueError(
                "The 'fastembed' package is not installed. Please install it with `pip install fastembed`",
            ) from e
        self._batch_size = batch_size
        self._parallel = parallel
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads, **kwargs)

    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        texts_to_vectorize = self._extract_text_from_examples(inp_examples)
        embeddings = self._model.embed(texts_to_vectorize, batch_size=self._batch_size, parallel=self._parallel)

        return np.array([embedding.tolist() for embedding in embeddings], dtype=np.float32)


class QdrantRM(dspy.Retrieve):
    """A retrieval module that uses Qdrant to return the top passages for a given query.

    Args:
        qdrant_collection_name (str): The name of the Qdrant collection.
        qdrant_client (QdrantClient): An instance of `qdrant_client.QdrantClient`.
        k (int, optional): The default number of top passages to retrieve. Default: 3.
        document_field (str, optional): The key in the Qdrant payload with the content. Default: `"document"`.
        vectorizer (BaseSentenceVectorizer, optional): An implementation `BaseSentenceVectorizer`.
                                                           Default: `FastEmbedVectorizer`.
        vector_name (str, optional): Name of the vector in the collection. Default: The first available vector name.

    Examples:
        Below is a code snippet that shows how to use Qdrant as the default retriver:
        ```python
        from qdrant_client import QdrantClient

        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        qdrant_client = QdrantClient()
        retriever_model = QdrantRM("my_collection_name", qdrant_client=qdrant_client)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use Qdrant in the forward() function of a module
        ```python
        self.retrieve = QdrantRM(question, k=num_passages, filter=filter)
        ```
    """

    def __init__(
        self,
        qdrant_collection_name: str,
        qdrant_client: QdrantClient,
        k: int = 3,
        document_field: str = "document",
        vectorizer: Optional[BaseSentenceVectorizer] = None,
        vector_name: Optional[str] = None,
    ):
        self._collection_name = qdrant_collection_name
        self._client = qdrant_client

        self._vectorizer = vectorizer or FastEmbedVectorizer(self._client.embedding_model_name)

        self._document_field = document_field

        self._vector_name = vector_name or self._get_first_vector_name()

        super().__init__(k=k)

    def forward(
        self, query_or_queries: Union[str, list[str]], k: Optional[int] = None, filter: Optional[models.Filter] = None
    ) -> dspy.Prediction:
        """Search with Qdrant for self.k top passages for query.

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
            filter (Optional["Filter"]): "Look only for points which satisfies this conditions". Default: None.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]  # Filter empty queries

        vectors = self._vectorizer(queries)

        search_requests = [
            models.QueryRequest(
                query=vector,
                using=self._vector_name,
                limit=k or self.k,
                with_payload=[self._document_field],
                filter=filter,
            )
            for vector in vectors
        ]
        batch_results = self._client.query_batch_points(self._collection_name, requests=search_requests)

        passages_scores = defaultdict(float)
        for batch in batch_results:
            for result in batch.points:
                # If a passage is returned multiple times, the score is accumulated.
                document = result.payload.get(self._document_field)
                passages_scores[document] += result.score

        # Sort passages by their accumulated scores in descending order
        sorted_passages = sorted(passages_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Wrap each sorted passage in a dotdict with 'long_text'
        return [dotdict({"long_text": passage}) for passage, _ in sorted_passages]

    def _get_first_vector_name(self) -> Optional[str]:
        vectors = self._client.get_collection(self._collection_name).config.params.vectors

        if not isinstance(vectors, dict):
            # The collection only has the default, unnamed vector
            return None

        first_vector_name = list(vectors.keys())[0]

        # The collection has multiple vectors. Could also include the falsy unnamed vector - Empty string("")
        return first_vector_name or None
