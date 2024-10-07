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


class FinetuneJobAnyScale(FinetuneJob):
    
    def cancel(self):
        """Cancel the finetuning job."""
        raise NotImplementedError("Method `cancel` is not implemented.")

    def get_status(self):
        """Return the status of the finetuning job."""
        raise NotImplementedError("Method `get_status` is not implemented.")


def is_anyscale_model(model: str) -> bool:
    """Check if the model is an AnyScale model."""
    logger.info("Is AnyScale model is not implemented, returning False as a default to not break lm.py")
    return False


def finetune_anyscale(job: FinetuneJobAnyScale, model: str, message_completion_pairs: List[Dict[str, str]], config: Dict[str, Any]):
    """Fine-tune an AnyScale model."""

    # Fake fine-tuning
    logger.info("[Finetune] Fake fine-tuning of an AnyScale model")

    try:
      logger.info("[Finetune] Verifying the formatting of the fine-tuning data")
      training_method = TrainingMethod.SFT  # Hardcoded
      logger.info("[Finetune] Validate the formatting of the fine-tuning data")
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
      model_id = "SOME_LOCAL_LM"  # Hardcoded
      lm = LM(model_id)
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
