from typing import Callable, Optional, Literal

import dspy
import array

try:
    import mariadb
except ImportError:
    raise ImportError(
        "The 'mariadb' extra is required to use MariadbRM. Install it with `pip install mariadb~=1.1.12`.",
    )


class MariadbRM(dspy.Retrieve):
    """
    Implements a retriever that can retrieve passages from MariaDB vector database.

    Returns a list of dspy.Example objects.

    Args:
        embedding_func: A function to use for computing query embeddings. Should return iterable
        table: A name of the table containing passages
        content_field: Name of column containing the passages.
        embedding_field: Name of column containing embeddings
        host: Host for MariaDB server
        port: The TCP/IP port number to use for the MariaDB connection
        user: The MariaDB user
        password: The password of the MariaDB account
        database: The MariaDB database
        mariadb_connector: An active connection to a MariaDB
        k: Number of top passages to retrieve
        fields: Other fields which will be returned with passages. Format is list of strings
        where_clause: A valid SQL where clause starting with 'WHERE' for possible filtering of passages before retrieval
        distance_measure: A metric used to measure distance between vectors. Either Euclidean, cosine or VEC_DISTANCE which
                          will behave as one of aforementioned meioned depending on the underlying index type and
                          results in error if the underlying index cannot be determined

    Examples:
        Below is a code snippet that shows how to use MariaDB as the default retriever

        ```python
        import openai
        from dspy.retrieve.mariadb_rm import MariadbRM

        openai.api_key = "OPENAI_API_KEY"

        def embedding_function(text):
            response = openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
            )

            return response.data[0].embedding



        retriever_model = MariadbRM(embedding_function, host='localhost', port = 3306, user = 'root', 
                                    password='12345', database = 'test', k=3, table = 'passages',
                                    content_field = 'text', embedding_field = 'embedding', fields=['id'],
                                    distance_measure='VEC_DISTANCE_EUCLIDEAN')

        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use this in the forward() function of a module
        ```python
        from dspy.retrieve.mariadb_rm import MariadbRM
        self.retrieve = MariadbRM(embedding_function, host='localhost', port = 3306, user = 'root', 
                                    password='12345', database = 'test', k=3, table = 'passages',
                                    content_field = 'text', embedding_field = 'embedding', fields=['id'],
                                    distance_measure='VEC_DISTANCE_EUCLIDEAN')
        ```
    """

    def __init__(
        self,
        embedding_func: Callable,
        table: str,
        content_field: str,
        embedding_field: str,
        host: Optional[str] = None,
        port: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        mariadb_connector: Optional[mariadb.Connection] = None,
        k: int = 20,
        fields: Optional[list[str]] = None,
        where_clause: str = "",
        distance_measure: Literal['VEC_DISTANCE_EUCLIDEAN', 'VEC_DISTANCE_COSINE', 'VEC_DISTANCE'] = 'VEC_DISTANCE_COSINE'
    ):
        if mariadb_connector is not None:
            self.cursor = mariadb_connector.cursor()
        else:
            try:
                conn = mariadb.connect(
                    user=user,
                    password=password,
                    host=host,
                    port=port,
                    database=database
                )
                self.cursor = conn.cursor()
            except mariadb.Error as e:
                print(f"Error connecting to MariaDB Platform: {e}")

        self.embedding_func = embedding_func
        self.table = table
        self.content_field = content_field
        self.embedding_field = embedding_field
        self.fields = fields
        self.where_clause = where_clause
        self.distance_measure = distance_measure

        super().__init__(k=k)

    def forward(
        self,
        query: str,
        k: Optional[int] = None
    ):

        """Search with for k top passages for query using distance between vector representation of text

        Args:
            query: The query to search for
            k: The number of top passages to retrieve. Defaults to the value set in the constructor.
        Returns:
            dspy.Prediction: an object containing the retrieved passages.
        """

        if k:
            self.k = k
        if self.fields:
            final_fields = self.content_field + "," + ",".join(self.fields)
        else:
            final_fields = self.content_field
        embedding = array.array("f", self.embedding_func(query)).tobytes()
        sql_query = f'''
                    SELECT {final_fields} FROM {self.table}
                    {self.where_clause}
                    ORDER BY {self.distance_measure}({self.embedding_field},%s)
                    LIMIT %s
                    '''

        self.cursor.execute(sql_query, (embedding, self.k ))
        rows = self.cursor.fetchall()
        column_names = [desc[0] for desc in self.cursor.description]

        retrieved_docs = []
        for row in rows:
            data = dict(zip(column_names, row))
            data["long_text"] = data[self.content_field]
            retrieved_docs.append(dspy.Example(**data))

        return retrieved_docs
