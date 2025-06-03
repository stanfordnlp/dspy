
from typing import TYPE_CHECKING, Any, Optional, Type
import logging

from dspy.adapters.base import Adapter
from dspy.signatures.signature import Signature
from dspy.adapters.utils import create_signature_for_retry

if TYPE_CHECKING:
    from dspy.clients.lm import LM

logger = logging.getLogger(__name__)

_MAX_RETRY_ERROR = ValueError("Failed to parse LM outputs for maximum retries.")

class RetryAdapter(Adapter):
    """
    RetryAdapter is an adapter that retries the execution of another adapter for
    a specified number of times if it fails to parse completion outputs.
    """

    def __init__(self, main_adapter: Adapter, fallback_adapter: Optional[Adapter] = None, main_adapter_max_retries: int = 3):
        """
        Initializes the RetryAdapter.

        Args:
            main_adapter (Adapter): The main adapter to use.
            fallback_adapter (Optional[Adapter]): The fallback adapter to use if the main adapter fails.
            main_adapter_max_retries (int): The maximum number of retries. Defaults to 3.
        """
        self.main_adapter = main_adapter
        self.fallback_adapter = fallback_adapter
        self.main_adapter_max_retries = main_adapter_max_retries

    def __call__(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Execute main_adapter and fallback_adapter in the following procedure:
        1. Call the main_adapter.
        2. If the main_adapter fails, call the fallback_adapter.
        3. If the fallback_adapter fails, retry the main_adapter including previous response for `max_retries` times.

        Args:
            lm (LM): The dspy.LM to use.
            lm_kwargs (dict[str, Any]): Additional arguments for the lm.
            signature (Type[Signature]): The signature of the function.
            demos (list[dict[str, Any]]): A list of demo examples.
            inputs (dict[str, Any]): A list representating the user input.

        Returns:
            A list of parsed completions. The size of the list is equal to `n` argument. Defaults to 1.

        Raises:
            Exception: If fail to parse outputs after the maximum number of retries.
        """
        outputs = []
        max_retries = max(self.main_adapter_max_retries, 0)
        n_completion = lm_kwargs.get("n", 1)

        values, parse_failures = self._call_adapter(
            self.main_adapter,
            lm,
            lm_kwargs,
            signature,
            demos,
            inputs,
        )
        outputs.extend(values)

        if len(outputs) >= n_completion:
            return outputs
        
        lm_kwargs["n"] = n_completion - len(outputs)
        if self.fallback_adapter is not None:
            outputs.extend(self._call_adapter(
                self.fallback_adapter,
                lm,
                lm_kwargs,
                signature,
                demos,
                inputs,
            )[0])
            if len(outputs) >= n_completion:
                return outputs
        
        # Retry the main adapter with previous response for `max_retries` times
        lm_kwargs["n"] = 1
        signature = create_signature_for_retry(signature)
        if parse_failures:
            inputs["previous_response"] = parse_failures[0][0]
            inputs["error_message"] = str(parse_failures[0][1])
        for i in range(max_retries):
            values, parse_failures = self._call_adapter(
                self.main_adapter,
                lm,
                lm_kwargs,
                signature,
                demos,
                inputs,
            )
            outputs.extend(values)
            if len(outputs) == n_completion:
                return outputs
            logger.warning(f"Retry {i+1}/{max_retries} for {self.main_adapter.__class__.__name__} failed with error: {parse_failures[0][1]}")
            inputs["previous_response"] = parse_failures[0][0]
            inputs["error_message"] = str(parse_failures[0][1])
        
        # raise the last error
        if parse_failures:
            raise _MAX_RETRY_ERROR from parse_failures[0][1]
        raise _MAX_RETRY_ERROR
    
    def _call_adapter(
        self, 
        adapter: Adapter, 
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ):
        values = []
        parse_failures = []
        messages = adapter.format(signature=signature, demos=demos, inputs=inputs)
        outputs = lm(messages=messages, **lm_kwargs)
        for i, output in enumerate(outputs):
            try:
                output_logprobs = None

                if isinstance(output, dict):
                    output, output_logprobs = output["text"], output["logprobs"]

                value = adapter.parse(signature, output)

                if output_logprobs is not None:
                    value["logprobs"] = output_logprobs

                values.append(value)
            except ValueError as e:
                logger.warning(f"Failed to parse the {i+1}/{lm_kwargs.get('n', 1)} LM output with adapter {adapter.__class__.__name__}. Error: {e}")
                parse_failures.append((outputs[i], e))

        return values, parse_failures
