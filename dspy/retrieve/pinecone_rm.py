"""
Retriever model for Pinecone
Author: Dhar Rawal (@drawal1)
"""

import pinecone  #you will need to install pinecone # type: ignore
import openai   # type: ignore
import dspy     # type: ignore

PINECONE_API_KEY = 'YOUR_PINECONE_API_KEY'
PINECONE_ENVIRONMENT = 'YOUR PINCONE ENVIRONMENT' # for example 'us-east4-gcp'
INDEX_NAME = "YOUR PINECONE INDEX NAME" # You should have an index build already. See Pinecone docs
EMBED_MODEL = "YOUR EMBEDDING MODEL" # For example 'text-embedding-ada-002' for OpenAI gpt-3.5-turbo

def init_pinecone(pinecone_api_key, pinecone_env, index_name):
    """Initialize pinecone and load the index"""
    pinecone.init(
        api_key=pinecone_api_key,  # find at app.pinecone.io
        environment=pinecone_env,  # next to api key in console
    )

    return pinecone.Index(index_name)

PINECONE_INDEX = init_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT, INDEX_NAME)

class PineconeRM(dspy.Retrieve):
    """
        A class that uses Pinecone to retrieve the top passages for a given query.

        Args:
            openai_api_key (str): The API key for OpenAI.
            k (int, optional): The number of top passages to retrieve. Defaults to 3.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.

        Examples:
            Below is a code snippet that shows how to use this as the default retriver:
            ```python
            llm = dspy.OpenAI(model='gpt-3.5-turbo')
            retriever_model = PineconeRM(openai.api_key)
            dspy.settings.configure(lm=llm, rm=retriever_model)
            ```

            Below is a code snippet that shows how to use this in the forward() function of a module
            ```python
            self.retrieve = PineconeRM(k=num_passages)
            ```
    """
    def __init__(self, openai_api_key, k=3):
        self._openai_api_key = openai_api_key
        super().__init__(k=k)

    def forward(self, query_or_queries):
        """ search with pinecone for self.k top passages for query"""
        # convert query_or_queries to a python list if it is not
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries

        embedding = openai.Embedding.create(input=queries, engine=EMBED_MODEL, openai_api_key=self._openai_api_key)
        query_vec = embedding['data'][0]['embedding']

        # retrieve relevant contexts from Pinecone (including the questions)
        results_dict = PINECONE_INDEX.query(query_vec, top_k=self.k, include_metadata=True)

        passages = [result['metadata']['text'] for result in results_dict['matches']]
        return dspy.Prediction(passages=passages)
