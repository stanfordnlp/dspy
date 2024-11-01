from typing import Callable, List, Optional 


class Embedder:
    def __init__(self, embedding_model: str = 'text-embedding-ada-002', embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None):
        self.embedding_model = embedding_model
        self.embedding_function = embedding_function or self.default_embedding_function

    def default_embedding_function(self, texts: List[str]) -> List[List[float]]:
        from litellm import embedding
        embeddings_response = embedding(model=self.embedding_model, input=texts)
        embeddings = [data['embedding'] for data in embeddings_response.data]
        return embeddings

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_function(texts)