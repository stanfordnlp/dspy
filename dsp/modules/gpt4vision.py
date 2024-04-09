import functools
import json
import logging
import os
from typing import Any, Literal, Optional, Union

import backoff
import numpy as np
import openai

import dsp
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.vlm import VLM
from dspy.primitives.vision import Image, SupportsImage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("openai_usage.log")],
    force=True, # Don't log to azure_openai_usage.log
)

try:
    OPENAI_LEGACY = int(openai.version.__version__[0]) == 0
except Exception:
    OPENAI_LEGACY = True

try:
    import openai.error
    from openai.openai_object import OpenAIObject

    ERRORS = (
        openai.error.RateLimitError,
    )
except Exception:
    ERRORS = (openai.RateLimitError,)
    OpenAIObject = dict


def backoff_hdlr(details) -> None:
  """Handler from https://pypi.org/project/backoff/ ."""
  logging.info(
      "Backing off {wait:0.1f} seconds after {tries} tries "
      "calling function {target} with kwargs "
      "{kwargs}".format(**details),)


class GPT4Vision(VLM):
  """Wrapper around OpenAI's GPT API.

  Args:
      model (str, optional): OpenAI supported LLM model to use. Defaults to "text-davinci-002".
      api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
      api_provider (Literal["openai"], optional): The API provider to use. Defaults to "openai".
      model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
      **kwargs: Additional arguments to pass to the API provider.
  """

  def __init__(
      self,
      model: str = "gpt-4-vision-preview",
      api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
      api_provider: Literal["openai"] = "openai",
      api_base: Optional[str] = None,
      system_prompt: Optional[str] = None,
      **kwargs,
  ):
    super().__init__(model)
    self.provider = "openai"
    openai.api_type = api_provider

    self.system_prompt = system_prompt

    self.model_type = "chat"

    if api_key:
      openai.api_key = api_key

    if api_base:
      openai.base_url = api_base

    self.kwargs = {
        "temperature": 0.0,
        "max_tokens": 150,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        **kwargs,
    }

    self.kwargs["model"] = model
    self.history: list[dict[str, Any]] = []

  def _openai_client(self):
    return openai

  def log_usage(self, response) -> None:
    """Log the total tokens from the OpenAI API response."""
    usage_data = response.get("usage")
    if usage_data:
      total_tokens = usage_data.get("total_tokens")
      logging.info(f"{total_tokens}")

  def basic_request(self, prompt: str, image: SupportsImage = None, **kwargs) -> Any:
    """Handles retrieval of GPT-4 completions and chats. Use the Image class to specify encoding of image data."""
    """Image data can also be passed as a numpy array, base64 string, or file path."""
    raw_kwargs = kwargs

    kwargs = {**self.kwargs, **kwargs}
   
    if image is not None:
      image = Image(image) if not isinstance(image, Image) else image
      content = [
          {
              "type": "text",
              "text": prompt,
          },
          {
              "type": "image_url",
              "image_url": {
                  "url": image.url,
              },
          },
      ]
    else:
      content = prompt
    messages = [{"role": "user", "content": content}]
    if self.system_prompt:
        messages.insert(0, {"role": "system", "content": self.system_prompt})
   
    kwargs["messages"] = messages
    kwargs = {"stringify_request": json.dumps(kwargs)}
    response = chat_request(**kwargs)


    history = {
        "prompt": prompt,
        "image": image,
        "response": response,
        "kwargs": kwargs,
        "raw_kwargs": raw_kwargs,
    }
    self.history.append(history)

    return response


  @backoff.on_exception(
      backoff.expo,
      ERRORS,
      max_time=10,
      on_backoff=backoff_hdlr,
  )
  def request(self, prompt: str, image: SupportsImage = None, **kwargs) -> Any:
    """Handles retreival of GPT-4 completions whilst handling rate limiting and caching."""
    if "model_type" in kwargs:
      del kwargs["model_type"]

    return self.basic_request(prompt, image, **kwargs)

  def _get_choice_content(self, choice: dict[str, Any]) -> str:
    if self.model_type == "chat":
      return choice["message"]["content"]
    return choice["text"]

  def __call__(
      self,
      prompt: str,
      image: Union[np.ndarray, str, Image, None] = None,
      only_completed: bool = True,
      return_sorted: bool = False,
      **kwargs,
  ) -> list[dict[str, Any]]:
    """Retrieves completions from GPT-4.

    Args:
      prompt (str): prompt to send to GPT-4
      only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
      return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.
      image (Union[np.ndarray, str, Image, None], optional): image to send to GPT-4 as a path, base64 string, or numpy array. Defaults to None.
      **kwargs: additional arguments to pass to the API provider.

    Returns:
      list[dict[str, Any]]: list of completion choices
    """
    response = self.request(prompt, image, **kwargs)

    if dsp.settings.log_openai_usage:
      self.log_usage(response)

    choices = response["choices"]
    completed_choices = [c for c in choices if c["finish_reason"] != "length"]

    if only_completed and len(completed_choices):
      choices = completed_choices

    completions = [self._get_choice_content(c) for c in choices]
    if return_sorted and kwargs.get("n", 1) > 1:
      scored_completions = []

      for c in choices:
        tokens, logprobs = (
            c["logprobs"]["tokens"],
            c["logprobs"]["token_logprobs"],
        )

        if "<|endoftext|>" in tokens:
          index = tokens.index("<|endoftext|>") + 1
          tokens, logprobs = tokens[:index], logprobs[:index]

        avglog = sum(logprobs) / len(logprobs)
        scored_completions.append((avglog, self._get_choice_text(c)))

      scored_completions = sorted(scored_completions, reverse=True)
      completions = [c for _, c in scored_completions]

    return completions





@CacheMemory.cache
def cached_gpt4vision_chat_request(**kwargs) -> Any:
  if "stringify_request" in kwargs:
    kwargs = json.loads(kwargs["stringify_request"])
  return openai.ChatCompletion.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_gpt4vision_chat_request_wrapped(**kwargs) -> Any:
  return cached_gpt4vision_chat_request(**kwargs)


def chat_request(**kwargs) -> dict[str, Any]:
  return cached_gpt4vision_chat_request_wrapped(**kwargs)
