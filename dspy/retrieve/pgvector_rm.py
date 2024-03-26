import warnings
from typing import Callable, Optional

import dspy

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    from psycopg2 import sql
except ImportError:
    raise ImportError(
        "The 'pgvector' extra is required to use PgVectorRM. Install it with `pip install dspy-ai[pgvector]`",
    )
try:
    import openai
except ImportError:
    warnings.warn("`openai` is not installed. Install it with `pip install openai` to use OpenAI embedding models.",
                  category=ImportWarning)


class PgVectorRM(dspy.Retrieve):
    """
    Implements a retriever that (as the name suggests) uses pgvector to retrieve passages,
    using a raw SQL query and a postgresql connection managed by psycopg2.

    It needs to register the pgvector extension with the psycopg2 connection

    Returns a list of dspy.Example objects

    Args:
        db_url (str): A PostgreSQL database URL in psycopg2's DSN format
        pg_table_name (Optional[str]): name of the table containing passages
        openai_client (openai.OpenAI): OpenAI client to use for computing query embeddings. Either openai_client or embedding_func must be provided.
        embedding_func (Callable): A function to use for computing query embeddings. Either openai_client or embedding_func must be provided.
        k (Optional[int]): Default number of top passages to retrieve. Defaults to 20
        embedding_field (str = "embedding"): Field containing passage embeddings. Defaults to "embedding"
        fields (List[str] = ['text']): Fields to retrieve from the table. Defaults to "text"
        embedding_model (str = "text-embedding-ada-002"): Field containing the OpenAI embedding model to use. Defaults to "text-embedding-ada-002"

    Examples:
        Below is a code snippet that shows how to use PgVector as the default retriever

        ```python
        import dspy
        import openai
        import psycopg2

        openai.api_key = os.environ.get("OPENAI_API_KEY", None)
        openai_client = openai.OpenAI()

        llm = dspy.OpenAI(model="gpt-3.5-turbo")

        DATABASE_URL should be in the format postgresql://user:password@host/database
        db_url=os.getenv("DATABASE_URL")

        retriever_model = PgVectorRM(conn, openai_client=openai_client, "paragraphs", fields=["text", "document_id"], k=20)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use PgVector in the forward() function of a module
        ```python
        self.retrieve = PgVectorRM(db_url, openai_client=openai_client, "paragraphs", fields=["text", "document_id"], k=20)
        ```
    """
    def __init__(
            self,
            db_url: str,
            pg_table_name: str,
            openai_client: Optional[openai.OpenAI] = None,
            embedding_func: Optional[Callable] = None,
            k: int = 20,
            embedding_field: str = "embedding",
            fields: Optional[list[str]] = None,
            embedding_model: str = "text-embedding-ada-002",
            include_similarity: bool = False,
    ):
        """
        k = 20 is the number of paragraphs to retrieve
        """
        assert openai_client or embedding_func, "Either openai_client or embedding_func must be provided."
        self.openai_client = openai_client
        self.embedding_func = embedding_func

        self.conn = psycopg2.connect(db_url)
        register_vector(self.conn)
        self.pg_table_name = pg_table_name
        self.fields = fields or ['text']
        self.embedding_field = embedding_field
        self.embedding_model = embedding_model
        self.include_similarity = include_similarity

        super().__init__(k=k)

    def forward(self, query: str):
        """Search with PgVector for self.k top passages for query using cosine similarity

        Args:
            query  (str): The query to search for
            include_similarity (bool): Whether or not to include the similarity for each record
        Returns:
            dspy.Prediction: an object containing the retrieved passages.
        """
        # Embed query
        query_embedding = self._get_embeddings(query)

        retrieved_docs = []

        fields = sql.SQL(',').join([
            sql.Identifier(f)
            for f in self.fields
        ])
        if self.include_similarity:
            similarity_field = (
                sql.SQL(',') +
                sql.SQL(
                    '1 - ({embedding_field} <=> %s) AS similarity',
                ).format(embedding_field=sql.Identifier(self.embedding_field))
            )
            fields += similarity_field
            args = (query_embedding, query_embedding, self.k)
        else:
            args = (query_embedding, self.k)

        sql_query = sql.SQL(
            "select {fields} from {table} order by {embedding_field} <=> %s::vector limit %s",
        ).format(
            fields=fields,
            table=sql.Identifier(self.pg_table_name),
            embedding_field=sql.Identifier(self.embedding_field),
        )

        with self.conn as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql_query,
                    args)
                rows = cur.fetchall()
                columns = [descrip[0] for descrip in cur.description]
                for row in rows:
                    data = dict(zip(columns, row))
                    retrieved_docs.append(dspy.Example(**data))
        # Return Prediction
        return retrieved_docs

    def _get_embeddings(self, query: str) -> list[float]:
        if self.openai_client is not None:
            return self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query,
                encoding_format="float",
            ).data[0].embedding
        else:
            return self.embedding_func(query)
