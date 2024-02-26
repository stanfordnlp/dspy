import dspy
import openai
from typing import List, Union, Optional

try:
    from pgvector.psycopg2 import register_vector
    import psycopg2
    from psycopg2 import sql
except ImportError:
    raise ImportError(
        "The 'pgvector' extra is required to use PgVectorRM. Install it with `pip install dspy-ai[pgvector]`"
    )


class PgVectorRM(dspy.Retrieve):
    """
    Implements a retriever that (as the name suggests) uses pgvector to retrieve passages,
    using a raw SQL query and a postgresql connection managed by psycopg2.

    It needs to register the pgvector extension with the psycopg2 connection

    Returns a list of dspy.Example objects

    Args:
        db_url (str): A PostgreSQL database URL in psycopg2's DSN format
        pg_table_name (Optional[str]): name of the table containing passages
        openai_client (openai.OpenAI): OpenAI client to use for computing query embeddings
        k (Optional[int]): Default number of top passages to retrieve. Defaults to 20
        embedding_field (str = "embedding"): Field containing passage embeddings. Defaults to "embedding"
        fields (List[str] = ['text']): Fields to retrieve from the table. Defaults to "text"

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
            openai_client: openai.OpenAI,
            k: Optional[int]=20,
            embedding_field: str = "embedding",
            fields: List[str] = ['text']
    ):
        """
        k = 20 is the number of paragraphs to retrieve
        """
        self.openai_client = openai_client
        
        self.conn = psycopg2.connect(db_url)
        register_vector(self.conn)
        self.pg_table_name = pg_table_name
        self.fields = fields
        self.embedding_field = embedding_field

        super().__init__(k=k)

    def forward(self, query: str, k: Optional[int]=20):
        """Search with PgVector for self.k top passages for query
        
        Args:
            query  (str): The query to search for
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k
        Returns: 
            dspy.Prediction: an object containing the retrieved passages.
        """
        # Embed query
        query_embedding = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query,
            encoding_format="float"
        ).data[0].embedding

        related_paragraphs = []

        sql_query = sql.SQL(
            "select {fields} from {table} order by {embedding_field} <-> %s::vector limit %s").format(
            fields=sql.SQL(',').join([
                sql.Identifier(f)
                for f in self.fields
            ]),
            table=sql.Identifier(self.pg_table_name),
            embedding_field=sql.Identifier(self.embedding_field)
        )

        with self.conn as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql_query,
                    (query_embedding, self.k))
                rows = cur.fetchall()
                for row in rows:
                    related_paragraphs.append(dspy.Example(long_text=row[0], document_id=row[1]))
        # Return Prediction
        return related_paragraphs