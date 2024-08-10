from typing import Any, Optional, Union
from typing_extensions import NotRequired, TypedDict
from pydantic import BaseModel, ConfigDict

import litellm


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


class LLMModelParams(BaseModel):
    """
    This class represents the parameters for any LLM model.

    Attributes:
    ----------
    model : str
        The model to be used, default is "gpt-4o-mini".
    messages : Optional[list[dict[str, Any]]]
        The messages to be processed, default is None.
    provider : str
        The provider of the model, default is "openai".
    temperature : float
        The temperature for the model, default is 0.0.
    max_tokens : Optional[int]
        The maximum number of tokens, default is None.
    top_p : float
        The top_p value for the model, default is 1.
    stream : bool
        Whether to stream the output, default is False.
    stream_options : Optional[StreamOptions]
        The stream options, default is None.
    stop : Optional[Union[str, list[str]]]
        The stop condition, default is None.
    n : int
        The number of outputs, default is 1.
    presence_penalty : Optional[float]
        The presence penalty, default is None.
    frequency_penalty : Optional[float]
        The frequency penalty, default is None.
    functions : Optional[list[Functions]]
        The functions to be used, default is None.
    function_call : Optional[Union[Optional[str], Functions]]
        The function call, default is None.
    logit_bias : Optional[dict[str, float]]
        The logit bias, default is None.
    user : Optional[str]
        The user, default is None.
    response_format : Optional[str]
        The response format, default is None.
    seed : Optional[int]
        The seed for the random number generator, default is None.
    tools : Optional[ToolFunction]
        The tools to be used, default is None.
    tool_choice : Optional[Union[str, ToolChoice]]
        The tool choice, default is None.
    logprobs : Optional[int]
        The logprobs, default is False.
    top_logprobs : Optional[int]
        The top logprobs, default is None.
    extra_headers : Optional[dict[str, str]]
        The extra headers, default is None.
    extra_args : dict[str, Any]
        The extra arguments, these are automatically added when you pass items not defined here.
    api_base : Optional[str]
        The API base, default is None.
    api_version : Optional[str]
        The API version, default is None.
    num_retries : Optional[int]
        The number of retries, default is None.
    context_window_fallback_dict : Optional[dict[str, Any]]
        The context window fallback dictionary, default is None.
    fallbacks : Optional[list[dict[str, Any]]]
        The fallbacks, default is None.
    metadata : Optional[dict[str, Any]]
        The metadata, default is None.
    system_prompt : Optional[str]
        The system prompt, default is None.
    prompt : Optional[str]
        The prompt, default is None.
    only_completed : bool
        Whether to only complete, default is False.
    return_sorted : bool
        Whether to return sorted, default is False.
    drop_params : bool
    """

    model: str = "gpt-4o-mini"
    messages: Optional[list[ChatMessage]] = None
    provider: str = "openai"
    temperature: float = 0.0
    max_tokens: Optional[int] = (
        None  # TODO: our previous default was set to 150. Clarify!
    )
    top_p: float = 1
    stream: bool = False  # TODO: can we support this?
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, list[str]]] = None
    n: int = 1
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    functions: Optional[list[Functions]] = (
        None  # TODO: can we support this? | openai depricated this
    )
    function_call: Optional[Union[Optional[str], Functions]] = (
        None  # TODO: can we support this? | openai depricated this
    )
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[str] = None  # support strict schema
    seed: Optional[int] = None
    tools: Optional[ToolFunction] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    logprobs: Optional[int] = False
    top_logprobs: Optional[int] = None
    extra_headers: Optional[dict[str, str]] = None
    extra_args: dict[str, Any] = {}

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

    # TODO: Custom parameter: figure out best way to delete these
    system_prompt: Optional[str] = None
    prompt: Optional[str] = None
    only_completed: bool = (
        False  # TODO: previously set to True, which is counterintuitive. Clarify!
    )
    return_sorted: bool = False

    def to_json(self, ignore_sensitive: bool = False) -> dict[str, Any]:
        return self.model_dump(
            exclude=[
                "system_prompt",
                "prompt",
                "only_completed",
                "return_sorted",
            ]
            + (["api_base"] if ignore_sensitive else [])
            + [
                "extra_args",
                "provider",
            ],  # this should automatically work when drop_params is set to True but it doesn't.
            exclude_none=True,
        )

    def __init__(self, **data):
        # filter extra_kwargs and pass it separately
        # would have been easier if ConfigDict(allow="extra") worked but doesn't seem to work with custom __init__
        extra_args = data.get("extra_args", {})
        to_delete = []
        for k, v in data.items():
            if k not in self.__annotations__.keys():
                extra_args[k] = v
                to_delete.append(k)
        for k in to_delete:
            del data[k]

        super().__init__(**data)
        _, provider, _, _ = litellm.get_llm_provider(self.model)
        self.provider = provider

        if self.system_prompt:
            self.messages = [
                ChatMessage(role="system", content=self.system_prompt)
            ]
        if self.prompt:
            if not self.messages:
                self.messages = []
            self.messages.append(ChatMessage(role="user", content=self.prompt))
