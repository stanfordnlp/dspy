import re
import time
from typing import Any, Dict, List, Optional

from dspy import logger
from dspy.clients.finetune import (
    FinetuneJob,
    TrainingMethod,
    validate_training_data,
)


#-------------------------------------------------------------------------------
#    Variables
#-------------------------------------------------------------------------------

# List of training methods supported by OpenAI
TRAINING_METHODS_OPENAI = [
    TrainingMethod.SFT,
]

# TODO: Should we include the TTS/Dall-E models on this list?
# List of OpenAI model IDs
OPENAI_MODEL_IDS = model_ids = [
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime-preview-2024-10-01",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-instruct",
    "dall-e-3",
    "dall-e-2",
    "tts-1",
    "tts-1-hd",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
    "omni-moderation-latest",
    "omni-moderation-2024-09-26",
    "text-moderation-latest",
    "text-moderation-stable",
    "text-moderation-007",
    "babbage-002",
    "davinci-002"
]


#-------------------------------------------------------------------------------
#    Function and classes required for the fine-tune interface
#-------------------------------------------------------------------------------

class FinetuneJobOpenAI(FinetuneJob):

    def cancel(self):
        """Cancel the finetune job."""
        logger.info("[Finetune] Canceling the OpenAI finetune job")
        time.sleep(3)
        logger.info("[Finetune] Done")
        super().cancel()

    def status(self):
        """Get the status of the finetune job."""
        logger.info("[Finetune] Getting the status of the OpenAI finetune job")
        time.sleep(3)
        status = "Running"
        logger.info("[Finetune] Done")
        return status


def is_openai_model(model: str) -> bool:
    """Check if the model is an OpenAI model."""
    # Filter the "openai/" prefix, if exists
    if model.startswith("openai/"):
        model = model[len("openai/"):]

    # Check if the model is a base OpenAI model
    if model in OPENAI_MODEL_IDS:
        return True

    # Check if the model is a fine-tuned OpneAI model. Fine-tuned OpenAI models
    # have the prefix "ft:<BASE_MODEL_NAME>:", followed by a string specifying
    # the fine-tuned model. The following RegEx pattern is used to match the
    # base model name.
    # TODO: This part can be updated to match the actual fine-tuned model names
    # by making a call to the OpenAI API to be more exact, but this might
    # require an API key with the right permissions. 
    match = re.match(r"ft:([^:]+):", model)
    if match and match.group(1) in OPENAI_MODEL_IDS:
        return True

    # If the model is not a base OpenAI model or a fine-tuned OpenAI model, then
    # it is not an OpenAI model.
    return False


def finetune_openai(
        job: FinetuneJobOpenAI,
        model: str,
        message_completion_pairs: List[Dict[str, str]],
        train_kwargs: Optional[Dict[str, Any]]=None,
    ) -> str:
    """Fine-tune an OpenAI model."""
    # Fake fine-tuning
    train_kwargs = train_kwargs or {}

    try:
        # Validate the formatting of the fine-tuning data
        logger.info("[Finetune] Validating the formatting of the data")
        _validate_data(message_completion_pairs)
        logger.info("[Finetune] Done!")

        # Validate the formatting of the fine-tuning data
        logger.error("[Finetune] Saving the data to a file")
        # data_path = save_data(message_completion_pairs)
        time.sleep(1)
        logger.info("[Finetune] Done!")
    
        logger.error("[Finetune] Uploading the data to the cloud")
        time.sleep(1)
        logger.error("[Finetune] Done!")

        logger.error("[Finetune] Launch training")
        # We utilize model and train_kwargs here
        time.sleep(1)
        logger.error("[Finetune] Done!")

        logger.error("[Finetune] Wait for training to complete")
        time.sleep(1)
        logger.error("[Finetune] Done!")

        logger.error("[Finetune] Get trained model client")
        model = "ft:gpt-4o:2024-08-06:THIS_IS_A_REAL_MODEL!!!"  # Hardcoded
        time.sleep(1)
        logger.error("[Finetune] Done!")

        logger.error("[Finetune] Exiting finetune_openai")

        return model

    except Exception as e:
      logger.error(f"[Finetune] Error: {e}")
      raise e


#-------------------------------------------------------------------------------
#    Custom functions
#-------------------------------------------------------------------------------
def _validate_data(data: Dict[str, str]) -> Optional[Exception]:
    """Validate the formatting of the fine-tuning data."""
    # TODO: Hardcoded train method
    # TODO: Should we skip this validation since the server might have custom
    # requirements?
    training_method = TrainingMethod.SFT
    validate_training_data(data, training_method)
