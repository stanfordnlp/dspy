import dspy
from dspy.dsp.utils import dotdict

try:
    import chromadb
except ImportError as err:
    raise ImportError(
        "The 'chromadb' extra is required to use ChromadbRM. Install it with `pip install dspy[chromadb]`",
    ) from err


class ChromadbRM(dspy.Retrieve):
    """A retrieval module that uses ChromaDB to return the top passages for a given query.

    Assumes that a ChromaDB collection has been created and populated with documents, where
    each document's text is stored as the collection's document content.

    Args:
        collection_name (str): The name of the ChromaDB collection to search.
        chromadb_client (chromadb.Client): An instance of the ChromaDB client.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.

    Examples:
        Below is a code snippet that shows how to use ChromaDB as the default retriever:
        ```python
        import chromadb

        chromadb_client = chromadb.Client()
        retriever_model = ChromadbRM("my_collection_name", chromadb_client=chromadb_client)
        dspy.configure(lm=llm, rm=retriever_model)

        retrieve = dspy.Retrieve(k=1)
        topK_passages = retrieve("what are the stages in planning, sanctioning and execution of public works").passages
        ```

        Below is a code snippet that shows how to use ChromaDB in the forward() function of a module
        ```python
        self.retrieve = ChromadbRM("my_collection_name", chromadb_client=chromadb_client, k=num_passages)
        ```
    """

    def __init__(
        self,
        collection_name: str,
        chromadb_client: chromadb.Client,
        k: int = 3,
    ):
        self._collection_name = collection_name
        self._chromadb_client = chromadb_client
        self._chromadb_collection = self._chromadb_client.get_collection(name=collection_name)
        super().__init__(k=k)

    def forward(self, query_or_queries: str | list[str], k: int | None = None, **kwargs) -> list[dotdict]:
        """Search with ChromaDB for the top k passages for the given query or queries.

        Args:
            query_or_queries (Union[str, list[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.

        Returns:
            list[dotdict]: One dotdict per retrieved passage, each with a `long_text` field.
        """
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        passages = []
        for query in queries:
            results = self._chromadb_collection.query(query_texts=[query], n_results=k)
            parsed_results = results["documents"][0]
            passages.extend(dotdict({"long_text": d}) for d in parsed_results)
        return passages
