# import functools
# import os
# from typing import List, Optional

# import openai

# import dspy
# from dspy.dsp.modules.cache_utils import NotebookCacheMemory, cache_turn_on
# from dspy.dsp.utils import dotdict

# # Check for necessary libraries and suggest installation if not found.
# try:
#     import clickhouse_connect
# except ImportError:
#     raise ImportError(
#         "The 'myscale' extra is required to use MyScaleRM. Install it with `pip install dspy-ai[myscale]`",
#     )

# # Verify the compatibility of the OpenAI library version installed.
# try:
#     major, minor, _ = map(int, openai.__version__.split("."))
#     OPENAI_VERSION_COMPATIBLE = major >= 1 and minor >= 16
# except Exception:
#     OPENAI_VERSION_COMPATIBLE = False

# if not OPENAI_VERSION_COMPATIBLE:
#     raise ImportError(
#         "An incompatible OpenAI library version is installed. Ensure you have version 1.16.1 or later.",
#     )

# # Attempt to handle specific OpenAI errors; fallback to general ones if necessary.
# try:
#     import openai.error

#     ERRORS = (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError)
# except Exception:
#     ERRORS = (openai.RateLimitError, openai.APIError)


# class MyScaleRM(dspy.Retrieve):
#     """
#     A retrieval module that uses MyScaleDB to return the top passages for a given query.

#     MyScaleDB is a fork of ClickHouse that focuses on vector similarity search and full
#     text search.  MyScaleRM is designed to facilitate easy retrieval of information from
#     MyScaleDB using embeddings.  It supports embedding generation through either a local
#     model or the OpenAI API. This class abstracts away the complexities of connecting to
#     MyScaleDB, managing API keys, and processing queries to return semantically
#     relevant results.

#     Assumes that a table named `database.table` exists in MyScaleDB, and that the
#     table has column named `vector_column` that stores vector data and a vector index has
#     been created on this column. Other metadata are stored in `metadata_columns`.

#     Args:
#         client (clickhouse_connect.driver.client.Client): A client connection to the MyScaleDB.
#         table (str): Name of the table within the database to perform queries against.
#         database (str, optional): Name of the database to query within MyScaleDB.
#         metadata_columns(List[str], optional): A list of columns to include in the results.
#         vector_column (str, optional): The name of the column in the table that stores vector data.
#         k (int, optional): The number of closest matches to retrieve for a given query.
#         openai_api_key (str, optional): The API key for accessing OpenAI's services.
#         model (str, optional): Specifies the particular OpenAI model to use for embedding generation.
#         use_local_model (bool): Flag indicating whether a local model is used for embeddings.

#     """

#     def __init__(
#         self,
#         client: clickhouse_connect.driver.client.Client,
#         table: str,
#         database: str = "default",
#         metadata_columns: List[str] = ["text"],
#         vector_column: str = "vector",
#         k: int = 3,
#         openai_api_key: Optional[str] = None,
#         openai_model: Optional[str] = None,
#         local_embed_model: Optional[str] = None,
#     ):
#         self.client = client
#         self.database = database
#         self.table = table
#         if not metadata_columns:
#             raise ValueError("metadata_columns is required")
#         self.metadata_columns = metadata_columns
#         self.vector_column = vector_column
#         self.k = k
#         self.openai_api_key = openai_api_key
#         self.model = openai_model
#         self.use_local_model = False

#         if local_embed_model:
#             self.setup_local_model(local_embed_model)
#         elif openai_api_key:
#             os.environ["OPENAI_API_KEY"] = self.openai_api_key

#     def setup_local_model(self, model_name: str):
#         """
#         Configures a local model for embedding generation, including model and tokenizer loading.

#         Args:
#             model_name: The name or path to the pre-trained model to load.

#         Raises:
#             ModuleNotFoundError: If necessary libraries (torch or transformers) are not installed.
#         """
#         try:
#             import torch
#             from transformers import AutoModel, AutoTokenizer
#         except ImportError as exc:
#             raise ModuleNotFoundError(
#                 """You need to install PyTorch and Hugging Face's transformers library to use a local embedding model.
#                 Install the pytorch using `pip install torch` and transformers using `pip install transformers` """,
#             ) from exc

#         try:
#             self._local_embed_model = AutoModel.from_pretrained(model_name)
#             self._local_tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.use_local_model = True
#         except Exception as e:
#             raise ValueError(f"Failed to load model or tokenizer. Error: {str(e)}")

#         if torch.cuda.is_available():
#             self.device = torch.device("cuda:0")
#         elif torch.backends.mps.is_available():
#             self.device = torch.device("mps")
#         else:
#             self.device = torch.device("cpu")

#         self._local_embed_model.to(self.device)

#     @functools.lru_cache(maxsize=None if cache_turn_on else 0)
#     @NotebookCacheMemory.cache
#     def get_embeddings(self, query: str) -> List[float]:
#         """
#         Determines the appropriate source (OpenAI or local model) for embedding generation based on class configuration,
#         and retrieves embeddings for the provided queries.

#         Args:
#             query: A query to generate embeddings for.

#         Returns:
#             A list of embeddings corresponding to the query in the input list.

#         Raises:
#             ValueError: If neither an OpenAI API key nor a local model has been configured.
#         """
#         if self.openai_api_key and self.model:
#             return self._get_embeddings_from_openai(query)
#         elif self.use_local_model:
#             return self._get_embedding_from_local_model(query)
#         else:
#             raise ValueError("No valid method for obtaining embeddings is configured.")

#     # TO DO Add this method as Util method outside MyScaleRM
#     def _get_embeddings_from_openai(self, query: str) -> List[float]:
#         """
#         Uses the OpenAI API to generate embeddings for the given query.

#         Args:
#             query: A string for which to generate embeddings.

#         Returns:
#             A list containing the embedding of a query.
#         """
#         response = openai.embeddings.create(model=self.model, input=query)
#         return response.data[0].embedding

#     # TO DO Add this method as Util method outside MyScaleRM
#     def _get_embedding_from_local_model(self, query: str) -> List[float]:
#         """
#         Generates embeddings for a single query using the configured local model.

#         Args:
#             query: The text query to generate an embedding for.

#         Returns:
#             A list representing the query's embedding.
#         """
#         import torch

#         self._local_embed_model.eval()  # Ensure the model is in evaluation mode

#         inputs = self._local_tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.device)
#         with torch.no_grad():
#             output = self._local_embed_model(**inputs)

#         return output.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

#     def forward(self, user_query: str, k: Optional[int] = None) -> List[dotdict]:
#         """
#         Executes a retrieval operation based on a user's query and returns the top k relevant results.

#         Args:
#             user_query: The query text to search for.
#             k: Optional; The number of top matches to return. Defaults to the class's configured k value.

#         Returns:
#             A list of dotdict objects containing the formatted retrieval results.

#         Raises:
#             ValueError: If the user_query is None.
#         """
#         if user_query is None:
#             raise ValueError("Query is required")
#         k = k or self.k
#         embeddings = self.get_embeddings(user_query)
#         columns_string = ", ".join(self.metadata_columns)
#         result = self.client.query(f"""
#         SELECT {columns_string},
#         distance({self.vector_column}, {embeddings}) as dist FROM {self.database}.{self.table} ORDER BY dist LIMIT {k}
#         """)
#         result = self.client.query(f"""
#             SELECT {columns_string}, distance({self.vector_column}, {embeddings}) AS dist
#             FROM {self.database}.{self.table}
#             ORDER BY dist
#             LIMIT {k}
#         """)

#         # Convert the metadata into strings to pass to dspy.Prediction
#         results = []
#         for row in result.named_results():
#             if len(self.metadata_columns) == 1:
#                 results.append(row[self.metadata_columns[0]])
#             else:
#                 row_strings = [f"{column}: {row[column]}" for column in self.metadata_columns]  # Format row data
#                 row_string = "\n".join(row_strings)  # Combine formatted data
#                 results.append(row_string)  # Append to results

#         # Return list of dotdict
#         return [dotdict({"long_text": passage}) for passage in results]
