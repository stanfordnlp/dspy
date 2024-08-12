from typing import Any

# Pydantic data model for the LLM class
import litellm
from litellm.types.utils import EmbeddingResponse, Usage

from dspy.utils.logging import logger
from dsp.modules.schemas import EncoderParams, DSPyEncoderModelResponse

# Use this for development testing
# litellm.set_verbose = True


class Encoder:
    """Wrapper around the Embedding API.

    Usage:
    ```python
    import dspy
    from dspy import EncoderModelParams

    encoder_params = EncoderModelParams(
        model="databricks/databricks-bge-large-en", DATABRICKS_API_KEY=XXX, DATABRICKS_API_BASE=XXX, provider="databricks"
    )
    encoder = dspy.Encoder(encoder_params)
    encoder(texts=["Please embed this text."])
    ```
    """

    encoder_params: EncoderParams
    history: list[dict[str, Any]] = []

    def __init__(
        self,
        encoder_params: EncoderParams,
    ):
        super().__init__()
        self.encoder_params = encoder_params

    def basic_request(self, inputs: list[str], **kwargs) -> EmbeddingResponse:
        self.update_messages_with_prompt(inputs)

        response = litellm.embedding(**self.encoder_params.to_json(), **kwargs)

        self.history.append(
            {
                "inputs": inputs,
                "response": response.to_dict(),
                "raw_kwargs": kwargs,
                "kwargs": self.encoder_params.to_json(ignore_sensitive=True),
            }
        )

        return response

    # TODO: enable caching
    def request(self, prompt: str, **kwargs) -> EmbeddingResponse:
        return self.basic_request(prompt, **kwargs)

    def log_usage(self, response: EmbeddingResponse):
        """Log the total tokens from the API response."""
        usage_data: Usage = response.get("usage")
        if usage_data:
            total_tokens = usage_data.get("total_tokens")
            logger.debug("Encoder Token Usage: %s", total_tokens)

    def transform_choices_to_dspy_encoder_model_response(
        self, embedding_response: EmbeddingResponse
    ) -> list[DSPyEncoderModelResponse]:
        """Transforms the choices to DSPyModelResponse."""
        return DSPyEncoderModelResponse(embedding=embedding_response.data)

    def update_messages_with_prompt(self, inputs: str):
        """Updates the messages with the prompt."""
        self.encoder_params.input = inputs

    def __call__(
        self,
        texts: list[str],
        **kwargs,
    ) -> list[DSPyEncoderModelResponse]:
        """Retrieves embeddings from the API."""

        response = self.request(texts, **kwargs)

        self.log_usage(response)

        return self.transform_choices_to_dspy_encoder_model_response(response)

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.encoder_params.to_json(), **kwargs}

        return self.__class__(encoder_params=EncoderParams(**kwargs), **kwargs)
