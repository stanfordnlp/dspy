from typing import Dict, Any

from dspy import LM
from dspy import logger
from dspy.clients.openai import is_openai_model


#-------------------------------------------------------------------------------
#    Function and classes required for the fine-tune interface
#-------------------------------------------------------------------------------

# TODO: Missing FinetuneJobSelfHosted class and finetune_self_hosted function
def is_self_hosted_model(model: str) -> bool:
    """Check if the model is self-hosted."""

    # Filter the models names that don't start with the prefix "openai/", since
    # all the self-hosted models have this prefix.
    if not model.startswith("openai/"):
        return False
    
    return not is_openai_model(model)


#-------------------------------------------------------------------------------
#    Launching and killing self hosted LMs
#-------------------------------------------------------------------------------

def self_hosted_model_launch(model: str, launch_kwargs: Dict[str, Any]):
    """Launch a self-hosted model."""
    # TODO: Hardcode resources for launching a select server through docker
    raise NotImplementedError("Method `self_hosted_model_launch` is not implemented.")

def self_hosted_model_kill(model: str, launch_kwargs: Dict[str, Any]):
    # TODO: Hardcode resources for killing a select server through docker
    """Kill a self-hosted model."""
    raise NotImplementedError("Method `self_hosted_model_kill` is not implemented.")
