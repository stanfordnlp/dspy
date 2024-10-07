import re
import time
from typing import List, Dict, Any

from dspy import logger
from dspy import LM
from dspy.clients.finetune import (
    FinetuneJob,
    TrainingMethod,
    validate_training_data
)


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


class FinetuneJobOpenAI(FinetuneJob):
    
    def cancel(self):
        """Cancel the finetuning job."""
        logger.info("[Finetune] Canceling the OpenAI finetuning job")
        time.sleep(3)
        logger.info("[Finetune] Done")
        super().cancel()

    def get_status(self):
        """Get the status of the finetuning job."""
        logger.info("[Finetune] Getting the status of the OpenAI finetuning job")
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


def finetune_openai(job: FinetuneJobOpenAI, model: str, message_completion_pairs: List[Dict[str, str]], config: Dict[str, Any]):
    """Fine-tune an OpenAI model."""

    # Fake fine-tuning
    logger.info("[Finetune] Fake fine-tuning of an OpenAI model")

    try:
      logger.info("[Finetune] Validate the formatting of the fine-tuning data")
      training_method = TrainingMethod.SFT  # Hardcoded
      validate_training_data(message_completion_pairs, training_method)
      time.sleep(5)
      logger.info("[Finetune] Done!")

      logger.info("[Finetune] Saving the data to a file")  # TODO: Get parent path, keep same across all finetuning jobs
      time.sleep(1)
      logger.info("[Finetune] Done!")

      logger.info("[Finetune] Uploading the data to the cloud")
      time.sleep(1)
      logger.info("[Finetune] Done!")

      logger.info("[Finetune] Wait for training to complete")
      time.sleep(1)
      logger.info("[Finetune] Done!")

      logger.info("[Finetune] Get trained model client")
      time.sleep(1)
      logger.info("[Finetune] Done!")
  
      logger.info("[Finetune] Create a DSPy LM object for the trained model")  # Will cause a circular import
      model_id = "ft:gpt-4o:2024-08-06:THIS_IS_A_REAL_MODEL!!!"  # Hardcoded
      lm = "FakeLM"  # Create an LM client
      time.sleep(1)
      logger.info("[Finetune] Done!")

      logger.info("Set the result of the FinetuningJob to the new lM")
      job.set_result(lm)
      logger.info("[Finetune] Done!")

    except Exception as e:
      logger.error(f"[Finetune] Error: {e}")
      raise e

    print()
    time.sleep(20)
