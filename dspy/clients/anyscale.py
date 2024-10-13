import re
import time
from typing import Any, Dict, List, Optional

import dspy
from dspy import logger
from dspy.clients.finetune import (
    FinetuneJob,
    TrainingMethod,
    get_finetune_directory,
    validate_training_data
)

#-------------------------------------------------------------------------------
#    Variables
#-------------------------------------------------------------------------------

# List of training methods supported by AnyScale
TRAINING_METHODS_ANYSCALE = [
    TrainingMethod.SFT,
]

#-------------------------------------------------------------------------------
#    Launching and killing LMs
#-------------------------------------------------------------------------------

def anyscale_model_launch(model: str, launch_kwargs: Dict[str, Any]):
    """Launch an AnyScale model."""
    # TODO: Hardcode resources for launching a select server through docker
    raise NotImplementedError("Method `anyscale_model_launch` is not implemented.")


def anyscale_model_kill(model: str, launch_kwargs: Dict[str, Any]):
    # TODO: Hardcode resources for killing a select server through docker
    """Kill an AnyScale model."""
    raise NotImplementedError("Method `anyscale_model_kill` is not implemented.")


#-------------------------------------------------------------------------------
#    Function and classes required for the fine-tune interface
#-------------------------------------------------------------------------------

class FinetuneJobAnyScale(FinetuneJob):
    
    def cancel(self):
        """Cancel the finetune job."""
        raise NotImplementedError("Method `cancel` is not implemented.")
        # Call the super's cancel method after the custom cancellation logic, 
        # so that the future can be cancelled
        # super().cancel()

    def status(self):
        """Get the status of the finetune job."""
        raise NotImplementedError("Method `status` is not implemented.")


def is_anyscale_model(model: str) -> bool:
    """Check if the model is an AnyScale model."""
    logger.info("Is AnyScale model is not implemented, returning False as a default to not break lm.py")
    return False


def finetune_anyscale(
        job: FinetuneJobAnyScale,
        model: str,
        message_completion_pairs: List[Dict[str, str]],
        train_kwargs: Optional[Dict[str, Any]]=None,
    ) -> str:
    """Fine-tune with AnyScale."""
    # Fake fine-tuning
    logger.info("[Finetune] Fake fine-tuning")
    train_kwargs = train_kwargs or {}

    try:
      logger.info("[Finetune] Validate the formatting of the fine-tuning data")
      training_method = TrainingMethod.SFT  # Hardcoded
      validate_training_data(message_completion_pairs, training_method)
      time.sleep(5)
      logger.info("[Finetune] Done!")

      logger.info("[Finetune] Saving the data to a file")
      finetune_parent_dir = get_finetune_directory()
      # We utilize message_completion_pairs here
      time.sleep(1)
      logger.info("[Finetune] Done!")

      logger.info("[Finetune] Uploading the data to the cloud")
      time.sleep(1)
      logger.info("[Finetune] Done!")

      logger.info("[Finetune] Launch training")
      # We utilize model and train_kwargs here
      time.sleep(1)
      logger.info("[Finetune] Done!")

      logger.info("[Finetune] Wait for training to complete")
      time.sleep(1)
      logger.info("[Finetune] Done!")

      logger.info("[Finetune] Get trained model client")
      model = "anyscale_model"  # Hardcoded
      time.sleep(1)
      logger.info("[Finetune] Done!")

      return model

    except Exception as e:
      logger.error(f"[Finetune] Error: {e}")
      raise e
