from concurrent.futures import ThreadPoolExecutor
import functools
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Type, Union
import ujson


from dspy import logger
from dspy.clients.finetune import FinetuneJob, TrainingMethod
from dspy.clients.self_hosted import (
    is_self_hosted_model,
    self_hosted_model_launch,
    self_hosted_model_kill,
)
from dspy.clients.anyscale import (
    is_anyscale_model,
    anyscale_model_launch,
    anyscale_model_kill,
)


DISK_CACHE_DIR = os.environ.get('DSPY_CACHEDIR') or os.path.join(Path.home(), '.dspy_cache')

    
try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
             os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        import litellm  
        litellm.telemetry = False

    from litellm.caching import Cache
    litellm.cache = Cache(disk_cache_dir=DISK_CACHE_DIR, type="disk")

except ImportError:
    class LitellmPlaceholder:
        def __getattr__(self, _): raise ImportError("The LiteLLM package is not installed. Run `pip install litellm`.")

    litellm = LitellmPlaceholder()


#-------------------------------------------------------------------------------
#    LiteLLM Client
#-------------------------------------------------------------------------------
        
class LM:
    def __init__(self, 
            model,
            model_type='chat', 
            temperature=0.0,
            max_tokens=1000,
            cache=True,
            launch_kwargs=None,
            **kwargs
        ):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.launch_kwargs = kwargs.pop("launch_kwargs", launch_kwargs or {})
        self.history = []

        # TODO: This is error prone!
        if "o1-" in model:
            assert max_tokens >= 5000 and temperature == 1.0, \
                "OpenAI's o1-* models require passing temperature=1.0 and max_tokens >= 5000 to `dspy.LM(...)`"

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Build the request.
        cache = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        # Make the request and handle LRU & disk caching.
        if self.model_type == "chat": completion = cached_litellm_completion if cache else litellm_completion
        else: completion = cached_litellm_text_completion if cache else litellm_text_completion

        response = completion(ujson.dumps(dict(model=self.model, messages=messages, **kwargs)))
        outputs = [c.message.content if hasattr(c, "message") else c["text"] for c in response["choices"]]

        # Logging, with removed api key & where `cost` is None on cache hit.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = dict(prompt=prompt, messages=messages, kwargs=kwargs, response=response)
        entry = dict(**entry, outputs=outputs, usage=dict(response["usage"]))
        entry = dict(**entry, cost=response.get("_hidden_params", {}).get("response_cost"))
        self.history.append(entry)

        return outputs
    
    def inspect_history(self, n: int = 1):
        _inspect_history(self, n)

    def launch(self):
        """Send a request to the provider to launch the model, if needed."""
        if is_self_hosted_model(self.model):
            self_hosted_model_launch(self.model, self.launch_kwargs)
        elif is_anyscale_model(self.model):
            anyscale_model_launch(self.model, self.launch_kwargs)
        logger.debug(f"`LM.launch()` is called for the auto-launched model {self.model} -- no action is taken.")

    def kill(self):
        """Send a request to the provider to kill the model, if needed."""
        if is_self_hosted_model(self.model):
            self_hosted_model_kill(self.model, self.launch_kwargs)
        elif is_anyscale_model(self.model):
            anyscale_model_kill(self.model, self.launch_kwargs)
        logger.debug(f"`LM.kill()` is called for the auto-launched model {self.model} -- no action is taken.")

    async def finetune(self,
            # message_completion_pairs: List[Dict[str, str]],
            method: TrainingMethod,
            train_path: str,
            eval_path: Optional[str],
            provider: str = "openai",
            train_kwargs: Optional[Dict[str, Any]]=None,
            launch_kwargs: Optional[Dict[str, Any]]=None,
            cache_finetune: bool = True,
        ) -> FinetuneJob:
        """Start model fine-tuning, if supported."""
        # Fine-tuning is experimental and requires the experimental flag
        from dspy import settings as settings
        err = "Fine-tuning is an experimental feature and requires `dspy.settings.experimental = True`."
        assert settings.experimental, err

        # Initialize the finetune job
        FinetuneJobClass = get_provider_finetune_job_class(provider=provider)
        finetune_job = FinetuneJobClass(
            model=self.model,
            train_path=train_path,
            eval_path=eval_path,
            train_kwargs=train_kwargs,
        )

        job = finetune_job.run_finetune()

        return job
    
    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        # model = kwargs.pop("model") or self.model
        init_kwargs = dict(model=self.model, model_type=self.model_type, cache=self.cache, temperature=self.kwargs["temperature"], max_tokens=self.kwargs["max_tokens"])
        init_kwargs = {**init_kwargs, **kwargs}
        return self.__class__(**init_kwargs)


@functools.lru_cache(maxsize=None)
def cached_litellm_completion(request):
    return litellm_completion(request, cache={"no-cache": False, "no-store": False})

def litellm_completion(request, cache={"no-cache": True, "no-store": True}):
    kwargs = ujson.loads(request)
    return litellm.completion(cache=cache, **kwargs)

@functools.lru_cache(maxsize=None)
def cached_litellm_text_completion(request):
    return litellm_text_completion(request, cache={"no-cache": False, "no-store": False})

def litellm_text_completion(request, cache={"no-cache": True, "no-store": True}):
    kwargs = ujson.loads(request)

    # Extract the provider and model from the model string.
    # TODO: Not all the models are in the format of "provider/model"
    model = kwargs.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the kwargs, or from the environment.
    api_key = kwargs.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = kwargs.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # Build the prompt from the messages.
    prompt = '\n\n'.join([x['content'] for x in kwargs.pop("messages")] + ['BEGIN RESPONSE:'])

    return litellm.text_completion(cache=cache, model=f'text-completion-openai/{model}', api_key=api_key,
                                   api_base=api_base, prompt=prompt, **kwargs)


def _green(text: str, end: str = "\n"):
    return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end

def _red(text: str, end: str = "\n"):
    return "\x1b[31m" + str(text) + "\x1b[0m" + end

def _inspect_history(lm, n: int = 1):
    """Prints the last n prompts and their completions."""

    for item in lm.history[-n:]:
        messages = item["messages"] or [{"role": "user", "content": item['prompt']}]
        outputs = item["outputs"]

        print("\n\n\n")
        for msg in messages:
            print(_red(f"{msg['role'].capitalize()} message:"))
            print(msg['content'].strip())
            print("\n")

        print(_red("Response:"))
        print(_green(outputs[0].strip()))

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs)-1} other completions)"
            print(_red(choices_text, end=""))
        
    print("\n\n\n")


#-------------------------------------------------------------------------------
#    Fine-tune support
#-------------------------------------------------------------------------------
# TODO: This part can be moved to a separate module

from dspy.clients.openai import (
    FinetuneJobOpenAI,
    is_openai_model,
    finetune_openai
)
from dspy.clients.anyscale import (
    FinetuneJobAnyScale,
    # is_anyscale_model,
    # finetune_anyscale
)


_PROVIDER_ANYSCALE = "anyscale"
_PROVIDER_OPENAI = "openai"
_SUPPORTED_FINETUNE_PROVIDERS = [
    _PROVIDER_ANYSCALE,
    _PROVIDER_OPENAI,
]

            
def _get_supported_finetune_provider(model: str) -> Union[str, ValueError]:
    """Return the finetuning provider for the given model.

    The provider must be in _SUPPORTED_FINETUNE_PROVIDERS. This function is not
    named `_get_provider` because it does not attempt to find the provider of a
    model if there is no DSPy fine-tuning support for it.
    """
    if is_self_hosted_model(model):
        return _PROVIDER_SELF_HOSTED

    if is_openai_model(model):
        return _PROVIDER_OPENAI

    if is_anyscale_model(model):
        return _PROVIDER_ANYSCALE
    
    return ValueError(f"DSPy does not have fine-tuning support for {model}")


def get_provider_finetune_job_class(provider: str) -> Type[FinetuneJob]:
    """Get the FinetuneJob class for the provider."""
    # Mapping from provider to finetune job type
    _PROVIDER_TO_FINETUNE_JOB_CLASS = {
        _PROVIDER_ANYSCALE: FinetuneJobAnyScale,
        _PROVIDER_OPENAI: FinetuneJobOpenAI,
    }

    # Get the FinetuneJob class for the provider
    _CLS = _PROVIDER_TO_FINETUNE_JOB_CLASS[provider]

    return _CLS


# def get_provider_finetune_function(provider: str) -> callable:
#     """Return the finetune function for the given model."""
#     # Mapping from provider to finetune function
#     _PROVIDER_TO_FINETUNE_FUNCTION = {
#         _PROVIDER_ANYSCALE: finetune_anyscale,
#         _PROVIDER_OPENAI: finetune_openai,
#     }

#     # Get the finetuning provider
#     finetune_function = _PROVIDER_TO_FINETUNE_FUNCTION[provider]

#     return finetune_function


# def execute_finetune_job(
#     job: FinetuneJob[Type[LM]],
#     launch_kwargs: Optional[Dict[str, Any]]=None,
#     cache_finetune: bool=True
# ):
#     """Execute the finetune job in a blocking manner."""
#     # Input validation
#     launch_kwargs = launch_kwargs or {}

#     # Execute finetune job
#     job_kwargs = job.get_kwargs()
#     if cache_finetune:
#         try:
#             model = cached_finetune(job=job, **job_kwargs)
#         except ValueError as err:
#             raise err
#     else:
#         model = finetune(job=job, **job_kwargs)

#     # Launch the LM
#     lm = LM(model=model, **launch_kwargs)

#     # Set the result of the finetuning job to the fine-tuned LM
#     job.set_result(lm)


# TODO: Perhaps we shouldn't directly cache the data
# TODO: Add DiskCache, ignore job
def cached_finetune(
    job,
    model: str,
    message_completion_pairs: List[Dict[str, str]],
    train_kwargs: Optional[Dict[str, Any]]=None,
) -> Union[str, ValueError]:
    return finetune(
        job=job,
        model=model,
        message_completion_pairs=message_completion_pairs,
        train_kwargs=train_kwargs,
    )


def finetune(
    job,
    model: str,
    message_completion_pairs: List[Dict[str, str]],
    train_kwargs: Optional[Dict[str, Any]]=None,
) -> Union[str, ValueError]:
    """Fine-tune a new model based on the given model."""
    # Get the fine-tuning provider
    try:
        provider = _get_supported_finetune_provider(model)
    except ValueError as err:
        raise err

    # Get the finetune function
    provider_finetune_function = get_provider_finetune_function(provider)

    # Fine-tune a new model based on the given model
    model = provider_finetune_function(
        job=job,
        model=model,
        message_completion_pairs=message_completion_pairs,
        train_kwargs=train_kwargs,
    )

    return model
