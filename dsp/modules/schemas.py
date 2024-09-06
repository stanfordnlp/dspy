import os
from typing import Any, Optional, Union, List
from typing_extensions import NotRequired, TypedDict
from pydantic import BaseModel

import litellm


class StreamOptions(BaseModel):
    """Pydantic data model for the Stream Options."""

    include_usage: bool = False


class ToolFunction(BaseModel):
    """Pydantic data model for the Tool Function."""

    description: Optional[str] = None
    name: str
    parameters: Optional[dict[str, Any]] = None
    strict: Optional[bool] = False


class ToolChoiceFunction(BaseModel):
    """Pydantic data model for the Tool Choice Function."""

    name: str


class ToolChoice(BaseModel):
    """Pydantic data model for the Tool Choice."""

    function: ToolChoiceFunction
    type: str


class Tool(BaseModel):
    """Pydantic data model for the Tool."""

    type: str
    function: ToolFunction


class ChatMessage(BaseModel):
    """Pydantic data model for the Chat Message."""

    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Tool] = None


class DSPyModelResponse(TypedDict):
    """LLM response data model that is trimmed down to the essentials."""

    text: str
    logprobs: NotRequired[dict[str, Any]]


class DSPyEncoderModelResponse(TypedDict):
    """Encoder response data model that is trimmed down to the essentials."""

    embedding: list[float]


class LLMModelParams(BaseModel):
    """Pydantic data model for the LLM Model Params."""

    model: str = "openai/gpt-4o-mini"
    messages: Optional[list[ChatMessage]] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None  # TODO: our previous default are varied across providers. Clarify!
    top_p: float = 1
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: int = 1
    stop: Optional[Union[str, list[str]]] = None
    safety_settings: Optional[dict[str, Any]] = None
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[str] = None  # support strict schema
    seed: Optional[int] = None
    tools: Optional[ToolFunction] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    stream: bool = False  # TODO: can we support this?
    stream_options: Optional[StreamOptions] = None

    # TODO: Custom parameter: figure out best way to delete these
    system_prompt: Optional[str] = None
    prompt: Optional[str] = None
    only_completed: bool = False  # TODO: previously set to True, which is counterintuitive. Clarify!
    return_sorted: bool = False

    class Config:
        """Allows for extra parameters to be added."""

        extra = "allow"

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return None


    def _init_messages(self):
        if self.messages:
            if self.messages[0].role != "system" and self.system_prompt:
                self.messages = [
                    ChatMessage(role="system", content=self.system_prompt),
                    *self.messages
                ]
            return

        if self.system_prompt:
            self.messages = [
                ChatMessage(role="system", content=self.system_prompt),
            ]
        else:
            self.messages = []
    
    def get_only_valid_model_params(self):
        litellm_model_params = litellm.get_supported_openai_params(self.model)
        litellm_model_params += ["model", "messages"]
        return self.to_json(include_only=litellm_model_params, exclude_none=True)

    def to_json(self, include_only: list[str] = [], exclude_none: bool = False) -> dict[str, Any]:
        """Converts the LLMModelParams to a JSON object."""
        return LLMModelParams(**self.model_dump()).model_dump(include=include_only, exclude_none=exclude_none)

    def get_copy(self) -> dict[str, Any]:
        return LLMModelParams(**self.model_dump())

    def __init__(self, **data):
        super().__init__(**data)

        # DSPy mainly focuses on single prompt text, so we do a minor transformation to format it as messages
        self._init_messages()


class LiteLLMParams(BaseModel):
    """Pydantic data model for the Lite LLM Params."""

    # Litellm specific
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None  # Azure
    num_retries: Optional[int] = (
        None  # The number of times to retry the API call if an APIError, TimeoutError or ServiceUnavailableError occurs
    )
    context_window_fallback_dict: Optional[dict[str, Any]] = (
        None  # A mapping of model to use if call fails due to context window error
    )
    fallbacks: Optional[list[dict[str, Any]]] = (
        None  # A list of model names + params to be used, in case the initial call fails
    )
    metadata: Optional[dict[str, Any]] = (
        None  # Any additional data you want to be logged when the call is made (sent to logging integrations, eg. promptlayer and accessible via custom callback function)
    )
    api_key: Optional[str] = None
    # provider: Optional[str] = ""
    extra_headers: Optional[dict[str, str]] = None

    # Provider specific params

    # Vertex specific params
    vertex_credentials: Optional[str] = None
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None

    # WatsonX specific params
    decoding_method: Optional[str] = None  # WatsonX - greedy
    project_id: Optional[str] = None  # WatsonX

    # AWS specific params
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION_NAME: Optional[str] = None

    # Cloudflare specific params
    CLOUDFLARE_API_KEY: Optional[str] = None
    CLOUDFLARE_ACCOUNT_ID: Optional[str] = None

    # Anyscale specific params
    ANYSCALE_API_KEY: Optional[str] = None

    # TogetherAI specific params
    TOGETHERAI_API_KEY: Optional[str] = None


    class Config:
        """Allows for extra parameters to be added."""

        extra = "allow"

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return None

    def to_json(self, exclude_none: bool = True) -> dict[str, Any]:
        """Converts the LLMParams to a JSON object."""
        return self.model_dump(
            exclude_none=exclude_none,
        )     

class EncoderModelParams(BaseModel):
    """Pydantic data model for the Encoder Model Params."""

    model: str
    input: Optional[List[str]] = None
    # instructions: str
    dimension: Optional[int] = None

    class Config:
        """Allows for extra parameters to be added."""

        extra = "allow"

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return None


class EncoderParams(EncoderModelParams):
    """Pydantic data model for the Encoder Params."""

    provider: Optional[str] = ""
    custom_provider: Optional[bool] = None
    DATABRICKS_API_KEY: Optional[str] = None
    DATABRICKS_API_BASE: Optional[str] = None

    # Litellm specific
    timeout: int = 600
    user: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None  # Azure
    num_retries: Optional[int] = (
        None  # The number of times to retry the API call if an APIError, TimeoutError or ServiceUnavailableError occurs
    )
    metadata: Optional[dict[str, Any]] = (
        None  # Any additional data you want to be logged when the call is made (sent to logging integrations, eg. promptlayer and accessible via custom callback function)
    )
    class Config:
        """Allows for extra parameters to be added."""

        extra = "allow"

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return None

    def to_json(self, ignore_sensitive: bool = False) -> dict[str, Any]:
        """Converts the EncoderParams to a JSON object."""
        return self.model_dump(exclude=(["api_base"] if ignore_sensitive else []))
