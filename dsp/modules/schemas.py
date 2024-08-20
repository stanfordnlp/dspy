import os
from typing import Any, Optional, Union
from typing_extensions import NotRequired, TypedDict
from pydantic import BaseModel

import litellm
from dsp.modules.provider_kwargs import get_google_ai_safety_settings


class StreamOptions(BaseModel):
    """Pydantic data model for the Stream Options."""

    include_usage: bool = False


class Functions(BaseModel):
    """Pydantic data model for the Functions."""

    description: Optional[str] = ""
    name: str
    parameters: Optional[dict[str, Any]] = None


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

    model: str = "gpt-4o-mini"
    messages: Optional[list[ChatMessage]] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None  # TODO: our previous default are varied across providers. Clarify!
    top_p: float = 1
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: int = 1
    stop: Optional[Union[str, list[str]]] = None
    functions: Optional[list[Functions]] = None  # TODO: openai depricated this
    function_call: Optional[Union[Optional[str], Functions]] = None  # TODO: openai depricated this
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

    def to_json(self, ignore_sensitive: bool = False, exclude_none: bool = False) -> dict[str, Any]:
        return LLMModelParams(**self.model_dump()).model_dump(exclude_none=exclude_none)

    def get_copy(self) -> dict[str, Any]:
        return LLMModelParams(**self.model_dump())


class LiteLLMParams:
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
    drop_params: bool = True


class LLMParams(LLMModelParams, LiteLLMParams):
    """Pydantic data model for the LLM Params."""

    provider: Optional[str] = ""
    extra_headers: Optional[dict[str, str]] = None
    vertex_credentials: Optional[str] = None
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    custom_provider: Optional[bool] = None
    decoding_method: Optional[str] = None  # WatsonX - greedy
    project_id: Optional[str] = None  # WatsonX

    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION_NAME: Optional[str] = None

    CLOUDFLARE_API_KEY: Optional[str] = None
    CLOUDFLARE_ACCOUNT_ID: Optional[str] = None

    ANYSCALE_API_KEY: Optional[str] = None

    TOGETHERAI_API_KEY: Optional[str] = None
    model_all_params: Optional[list[str]] = None
    litellm_model: Optional[str] = None

    class Config:
        """Allows for extra parameters to be added."""

        extra = "allow"

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return None

    def to_json(self, ignore_sensitive: bool = False, exclude_none: bool = True) -> dict[str, Any]:
        """Converts the LLMParams to a JSON object."""
        param_json = self.model_dump(
            include=self.model_all_params,
            exclude=["api_base"] if ignore_sensitive else [],
            exclude_none=exclude_none,
        )
        return param_json

    def get_model_params(
        self, return_json: bool = False, exclude_none: bool = False
    ) -> Union[dict[str, Any], LLMModelParams]:
        """Returns the model parameters."""
        if return_json:
            return super().to_json(exclude_none=exclude_none)
        return super().get_copy()

    def __init__(self, **data):
        super().__init__(**data)
        if self.provider is None:
            _, provider, _, _ = litellm.get_llm_provider(self.model)
            self.provider = provider
        elif self.provider not in litellm.provider_list:
            self.custom_provider = True
            if self.provider == "premai":
                self.api_base = "https://api.premai.com/v1"
            if self.provider == "you.com":

                if self.model == "research":
                    self.model = "custom/research"
                else:
                    self.model = "custom/smart"

                if self.api_base is None:
                    self.api_base = "https://chat-api.you.com/"
                    self.api_base = self.api_base + "/" + self.model.split("/")[1]

            if self.api_base is None:
                raise ValueError(
                    f"""Please provide an API base for the custom provider {self.provider}.
                    Please visit https://docs.litellm.ai/docs/providers to see all supported LLM providers.
                    You can still use your custom provider by setting the api_base parameter."""
                )
            if "/" in self.model.split("/")[0]:
                self.model = "custom/" + self.model

        self.litellm_model = f"{self.provider}/{self.model}"

        if self.system_prompt:
            self.messages = [ChatMessage(role="system", content=self.system_prompt)]
        if self.prompt:
            if not self.messages:
                self.messages = []
            self.messages.append(ChatMessage(role="user", content=self.prompt))

        if "vertex_ai" in self.provider and self.safety_settings is None:
            self.safety_settings = get_google_ai_safety_settings()

            self.vertex_credentials = (
                self.api_key or os.environ.get("VERTEX_CREDENTIALS") or self.extra_args.get("vertex_credentials")
            )
            self.vertex_project = self.extra_args.get("vertex_project") or os.environ.get("VERTEX_PROJECT")
            self.vertex_location = self.extra_args.get("vertex_location") or os.environ.get("VERTEX_LOCATION")

        self.model_all_params = (
            litellm.get_supported_openai_params(self.litellm_model)
            + list(LiteLLMParams.__annotations__.keys())
            + ["model", "messages"]
        )


class EncoderModelParams(BaseModel):
    """Pydantic data model for the Encoder Model Params."""

    model: str
    input: list[str]
    instructions: str
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
    drop_params: bool = True

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
