import litellm
import numpy as np
from typing import Callable, List, Union

class Embedding:
    """DSPy embedding class.

    The class for computing embeddings for text inputs. This class provides a unified interface for both:

    1. Hosted embedding models (e.g. OpenAI's text-embedding-3-small) via litellm integration
    2. Custom embedding functions that you provide

    For hosted models, simply pass the model name as a string (e.g. "openai/text-embedding-3-small"). The class will use
    litellm to handle the API calls and caching.

    For custom embedding models, pass a callable as `embedding_model` that:
    - Takes a list of strings as input.
    - Returns embeddings as either:
        - A 2D numpy array of float32 values.
        - A 2D list of float32 values.
    - Each row represents one embedding vector.

    Args:
        embedding_model: The embedding model to use, either a string (for hosted models supported by litellm) or
            a callable that returns custom embeddings.

    Examples:
        Example 1: Using a hosted model.

        ```python
        import dspy

        embedder = dspy.Embedding(embedding_model="openai/text-embedding-3-small")
        embeddings = embedder(["hello", "world"])

        assert embeddings.shape == (2, 1536)
        ```

        Example 2: Using a custom function.

        ```python
        import dspy
        import numpy as np

        def my_embedder(texts):
            return np.random.rand(len(texts), 10)

        embedder = dspy.Embedding(embedding_model=my_embedder)
        embeddings = embedder(["hello", "world"])

        assert embeddings.shape == (2, 10)
        ```
    """

    def __init__(self, embedding_model: Union[str, Callable] = 'openai/text-embedding-3-small'):
        self.embedding_model = embedding_model

    def default_embedding_model(self, texts: List[str], caching: bool = True, **kwargs) -> List[List[float]]:
        embeddings_response = litellm.embedding(model=self.embedding_model, input=texts, caching=caching, **kwargs)
        return [data['embedding'] for data in embeddings_response.data]

    def __call__(self, inputs: Union[str, List[str]], caching: bool = True, **kwargs) -> np.ndarray:
        """Compute embeddings for the given inputs.

        Args:
            inputs: Query inputs to compute embeddings for, can be a single string or a list of strings.
            caching: Cache flag for embedding response when using an embedding model.
            kwargs: Additional keyword arguments to pass to the embedding model.

        Returns:
            A 2-D numpy array of embeddings, one embedding per row.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        if callable(self.embedding_model):
            embeddings = self.embedding_model(inputs, **kwargs)
        elif isinstance(self.embedding_model, str):
            embeddings = self.default_embedding_model(inputs, caching=caching, **kwargs)
        else:
            raise ValueError(
                f"`embedding_model` must be a string or a callable, but got type: {type(self.embedding_model)}."
            )
        return np.array(embeddings, dtype=np.float32)
