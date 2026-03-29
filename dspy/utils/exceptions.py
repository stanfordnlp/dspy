
from dspy.signatures.signature import Signature


class ContextWindowExceededError(Exception):
    """Raised when the prompt exceeds the model's context window.

    Any `BaseLM` subclass should raise this error (or a subclass of it) when the
    request fails because the input is too long for the model. Adapters and some
    modules rely on catching this specific type to decide whether a fallback
    retry is appropriate.

    Args:
        model: The model identifier that rejected the request.
        message: Description of the error. Defaults to `"Context window exceeded"`.
    """

    def __init__(self, *, model: str | None = None, message: str = "Context window exceeded"):
        self.model = model
        msg = message
        prefix = f"[{model}] " if model else ""
        super().__init__(f"{prefix}{msg}")


class AdapterParseError(Exception):
    """Exception raised when adapter cannot parse the LM response."""

    def __init__(
        self,
        adapter_name: str,
        signature: Signature,
        lm_response: str,
        message: str | None = None,
        parsed_result: str | None = None,
    ):
        self.adapter_name = adapter_name
        self.signature = signature
        self.lm_response = lm_response
        self.parsed_result = parsed_result

        message = f"{message}\n\n" if message else ""
        message = (
            f"{message}"
            f"Adapter {adapter_name} failed to parse the LM response. \n\n"
            f"LM Response: {lm_response} \n\n"
            f"Expected to find output fields in the LM response: [{', '.join(signature.output_fields.keys())}] \n\n"
        )

        if parsed_result is not None:
            message += f"Actual output fields parsed from the LM response: [{', '.join(parsed_result.keys())}] \n\n"

        super().__init__(message)
