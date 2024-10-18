from typing import Any, Dict, List, Optional, Type, Union

from dspy.clients.anyscale import FinetuneJobAnyScale, finetune_anyscale
from dspy.clients.finetune import FinetuneJob, TrainingMethod
from dspy.clients.openai import FinetuneJobOpenAI, finetune_openai
from dspy.utils.logging import logger

_PROVIDER_ANYSCALE = "anyscale"
_PROVIDER_OPENAI = "openai"


def get_provider_finetune_job_class(provider: str) -> Type[FinetuneJob]:
    """Get the FinetuneJob class for the provider."""
    provider_to_job_class = {
        _PROVIDER_ANYSCALE: FinetuneJobAnyScale,
        _PROVIDER_OPENAI: FinetuneJobOpenAI,
    }
    return provider_to_job_class[provider]


def get_provider_finetune_function(provider: str) -> callable:
    """Return the finetune function for the given model."""
    provider_to_finetune_function = {
        _PROVIDER_ANYSCALE: finetune_anyscale,
        _PROVIDER_OPENAI: finetune_openai,
    }
    return provider_to_finetune_function[provider]


# Note: Type of LM should be LM. We aren't importing it here to avoid
# circular imports.
def execute_finetune_job(job: FinetuneJob, lm: Any, cache_finetune: bool = True):
    """Execute the finetune job in a blocking manner."""
    try:
        job_kwargs = job.get_kwargs()
        if cache_finetune:
            model = cached_finetune(job=job, **job_kwargs)
        else:
            model = finetune(job=job, **job_kwargs)
        lm = lm.copy(model=model)
        job.set_result(lm)
    except Exception as err:
        logger.error(err)
        job.set_result(err)


# TODO: Add DiskCache, ignore job
def cached_finetune(
    job,
    model: str,
    train_data: List[Dict[str, Any]],
    train_kwargs: Optional[Dict[str, Any]] = None,
    train_method: TrainingMethod = TrainingMethod.SFT,
    provider: str = "openai",
) -> Union[str, Exception]:
    return finetune(
        job=job,
        model=model,
        train_data=train_data,
        train_kwargs=train_kwargs,
        train_method=train_method,
        provider=provider,
    )


def finetune(
    job,
    model: str,
    train_data: List[Dict[str, Any]],
    train_kwargs: Optional[Dict[str, Any]] = None,
    train_method: TrainingMethod = TrainingMethod.SFT,
    provider: str = "openai",
) -> Union[str, Exception]:
    """Fine-tune a new model based on the given model."""
    # Get the fine-tuning provider
    try:
        # Get the finetune function
        provider_finetune_function = get_provider_finetune_function(provider)

        # Fine-tune a new model based on the given model
        model = provider_finetune_function(
            job=job,
            model=model,
            train_data=train_data,
            train_kwargs=train_kwargs,
            train_method=train_method,
        )
    except Exception as err:
        raise err

    return model
