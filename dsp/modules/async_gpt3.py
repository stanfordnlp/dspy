import json
from typing import Any, cast

import backoff
import openai
import openai.error
from openai.openai_object import OpenAIObject


from dsp.modules.gpt3 import GPT3, backoff_hdlr


class AsyncGPT3(GPT3):
    """Wrapper around OpenAI's GPT API. Supports both the OpenAI and Azure APIs.

    Args:
        model (str, optional): OpenAI or Azure supported LLM model to use. Defaults to "text-davinci-002".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        api_provider (Literal["openai", "azure"], optional): The API provider to use. Defaults to "openai".
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        **kwargs: Additional arguments to pass to the API provider.
    """

    async def basic_request(self, prompt: str, **kwargs) -> OpenAIObject:
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == "chat":
            # caching mechanism requires hashable kwargs
            kwargs["messages"] = [{"role": "user", "content": prompt}]
            kwargs = {"stringify_request": json.dumps(kwargs)}
            response = await _a_gpt3_chat_request(**kwargs)

        else:
            kwargs["prompt"] = prompt
            response = await _a_gpt3_completion_request(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        (openai.error.RateLimitError, openai.error.ServiceUnavailableError),
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    async def request(self, prompt: str, **kwargs) -> OpenAIObject:
        """Handles retreival of GPT-3 completions whilst handling rate limiting and caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return await self.basic_request(prompt, **kwargs)

    async def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from GPT-3.

        Args:
            prompt (str): prompt to send to GPT-3
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        response = await self.request(prompt, **kwargs)
        completions = self._get_completions_from_response(
            response=response,
            only_completed=only_completed,
            return_sorted=return_sorted,
            **kwargs,
        )
        return completions


async def _a_gpt3_completion_request(**kwargs):
    return openai.Completion.create(**kwargs)


async def _a_gpt3_chat_request(**kwargs) -> OpenAIObject:
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    res = await openai.ChatCompletion.acreate(**kwargs)
    return cast(OpenAIObject, res)
