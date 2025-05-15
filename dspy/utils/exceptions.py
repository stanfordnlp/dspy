from typing import Optional

from dspy.signatures.signature import Signature


class AdapterParseError(Exception):
    """Exception raised when adapter cannot parse the LM response."""

    def __init__(
        self,
        adapter_name: str,
        signature: Signature,
        lm_response: str,
        message: Optional[str] = None,
        parsed_result: Optional[str] = None,
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
