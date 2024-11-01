import litellm


class Embedding:
    """DSPy embedding class.

    The class for computing embeddings for text inputs. This class supports both hosted embedding
    models like OpenAI's text-embedding-3-small as well as custom embedding models that are provided as a function.
    When hosted embedding models are used, this class relies on litellm to call the embedding model. When a custom
    embedding model is used, this class directly passes the inputs to the model function.

    Args:
        model: The embedding model to use. This can be either a string (representing the name of the hosted embedding
            model, must be an embedding model supported by litellm) or a callable that represents a custom embedding
            model.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, caching=True, **kwargs):
        """Compute embeddings for the given inputs.

        Args:
            inputs: The inputs to compute embeddings for, can be a single string or a list of strings.
            caching: Whether to cache the embedding response, only valid when using a hosted embedding model.
            kwargs: Additional keyword arguments to pass to the embedding model.

        Returns:
            A list of embeddings, one for each input, in the same order as the inputs. Or the output of the custom
            embedding model.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(self.model, str):
            embedding_response = litellm.embedding(model=self.model, input=inputs, caching=caching, **kwargs)
            return [data["embedding"] for data in embedding_response.data]
        elif callable(self.model):
            return self.model(inputs, **kwargs)
        else:
            raise ValueError(f"`model` in `dspy.Embedding` must be a string or a callable, but got {type(self.model)}.")
