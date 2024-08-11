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


class LLMModelParams(BaseModel):
    model: str = "gpt-4o-mini"
    messages: Optional[list[ChatMessage]] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = (
        None  # TODO: our previous default are varied across providers. Clarify!
    )
    top_p: float = 1
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: int = 1
    stop: Optional[Union[str, list[str]]] = None
    functions: Optional[list[Functions]] = None  # TODO: openai depricated this
    function_call: Optional[Union[Optional[str], Functions]] = (
        None  # TODO: openai depricated this
    )
    safety_settings: Optional[dict[str, Any]] = None
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[str] = None  # support strict schema
    seed: Optional[int] = None
    tools: Optional[ToolFunction] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    logprobs: Optional[int] = False
    top_logprobs: Optional[int] = None
    stream: bool = False  # TODO: can we support this?
    stream_options: Optional[StreamOptions] = None

    # TODO: Custom parameter: figure out best way to delete these
    system_prompt: Optional[str] = None
    prompt: Optional[str] = None
    only_completed: bool = (
        False  # TODO: previously set to True, which is counterintuitive. Clarify!
    )
    return_sorted: bool = False

    def to_json(self, exclude_none: bool = False) -> dict[str, Any]:
        return LLMModelParams(**self.model_dump()).model_dump(
            exclude_none=exclude_none
        )

    def get_copy(self) -> dict[str, Any]:
        return LLMModelParams(**self.model_dump())


class LLMParams(LLMModelParams):
    provider: Optional[str] = ""
    extra_headers: Optional[dict[str, str]] = None
    extra_args: dict[str, Any] = {}
    vertex_credentials: Optional[str] = None
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    custom_provider: Optional[bool] = None
    decoding_method: Optional[str] = None  # WatsonX - greedy
    project_id: Optional[str] = None  # WatsonX

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

    def to_json(self, ignore_sensitive: bool = False) -> dict[str, Any]:
        return {
            **self.model_dump(
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
            ),
            **self.extra_args,
        }

    def get_model_params(
        self, return_json: bool = False, exclude_none: bool = False
    ) -> Union[dict[str, Any], LLMModelParams]:
        if return_json:
            return super().to_json(exclude_none=exclude_none)
        return super().get_copy()

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
                    self.api_base = (
                        self.api_base + "/" + self.model.split("/")[1]
                    )

            if self.api_base is None:
                raise ValueError(
                    f"""Please provide an API base for the custom provider {self.provider}.
                    Please visit https://docs.litellm.ai/docs/providers to see all supported LLM providers.
                    You can still use your custom provider by setting the api_base parameter."""
                )
            if "/" in self.model.split("/")[0]:
                self.model = "custom/" + self.model

        if self.system_prompt:
            self.messages = [
                ChatMessage(role="system", content=self.system_prompt)
            ]
        if self.prompt:
            if not self.messages:
                self.messages = []
            self.messages.append(ChatMessage(role="user", content=self.prompt))

        if "vertex_ai" in self.provider and self.safety_settings is None:
            self.safety_settings = get_google_ai_safety_settings()

            self.vertex_credentials = (
                self.api_key
                or os.environ.get("VERTEX_CREDENTIALS")
                or self.extra_args.get("vertex_credentials")
            )
            self.vertex_project = self.extra_args.get(
                "vertex_project"
            ) or os.environ.get("VERTEX_PROJECT")
            self.vertex_location = self.extra_args.get(
                "vertex_location"
            ) or os.environ.get("VERTEX_LOCATION")
