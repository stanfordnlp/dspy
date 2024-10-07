from dspy.clients.openai import is_openai_model


def is_self_hosted_model(model: str) -> bool:
    """Check if the model is self-hosted."""

    # Filter the models names that don't start with the prefix "openai/", since
    # all the self-hosted models have this prefix.
    if not model.startswith("openai/"):
        return False
    
    return not is_openai_model(model)
