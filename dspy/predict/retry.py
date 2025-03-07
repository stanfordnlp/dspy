from dspy.primitives import Prediction, Module
from dspy.signatures import Signature
import dspy
from typing import Optional, List, Callable, Any, Tuple
from dspy.adapters.conversation_feedback_adapter import ConversationalFeedbackAdapter


class Retry(Module):
    """
    A module that Retries the output of a DSPy module using feedback from validator functions
    through a simulated conversation.

    This module enhances a DSPy module’s output by making multiple attempts to meet specified requirements.
    After each attempt, **validator functions**—callables that take a `dspy.Prediction` and return an
    optional error message—assess the output. If requirements are unmet, these error messages serve as
    *user* feedback in a simulated conversation, mimicking a user pointing out constraint violations.

    The feedback is structured as a chat history with:
    - **Assistant messages**: Previous model outputs (attempts).
    - **User messages**: Critiques based on validator errors.
    - A final prompt to try again, addressing the feedback.

    This process repeats until the output is valid or the maximum number of attempts is reached.

    **Example Usage:**
    ```python
    # Define validators
    def validate_length(pred: dspy.Prediction) -> Optional[str]:
        if len(pred.summary.split()) > 50:
            return "Summary too long. Keep it under 50 words."
        return None

    def validate_format(pred: dspy.Prediction) -> Optional[str]:
        if not pred.summary.startswith("TL;DR:"):
            return "Summary should start with 'TL;DR:'"
        return None

    # Create a Retryd module with validators
    Retryd_summarizer = dspy.Retry(
        signature=my_signature,
        use_cot=True,
        validators=[validate_length, validate_format],
        max_attempts=3,
        verbose=True
    )

    # Use the Retryd module
    result = Retryd_summarizer(text="Some text to process")
    """

    def __init__(
        self,
        signature: Signature,
        use_cot: bool = False,
        validators: Optional[List[Callable[[Any], Optional[str]]]] = None,
        max_attempts: int = 3,
        verbose: bool = False,
        fail_on_invalid: bool = False
    ) -> None:
        """
        Initialize the Retry module.

        Args:
            signature (dspy.Signature): The DSPy signature to use
            use_cot (bool): Whether to use Chain of Thought (True) or simple Predict (False)
            validators (Optional[List[Callable[[Any], Optional[str]]]]): List of functions that validate the module's output and return error messages.
                        Each validator should accept a dspy.Prediction and return None or an error string.
            max_attempts (int): Maximum number of Retry attempts (default: 3)
            verbose (bool): Whether to log detailed information about Retry attempts (default: False)
            fail_on_invalid (bool): Whether to raise an exception if Retry is unsuccessful after max attempts (default: False)
        """
        super().__init__()

        # Create the appropriate module based on use_cot flag
        if use_cot:
            self.module = dspy.ChainOfThought(signature)
        else:
            self.module = dspy.Predict(signature)

        self.validators = validators or []
        self.max_attempts = max(1, max_attempts)  # Ensure at least one attempt
        self.verbose = verbose
        self.fail_on_invalid = fail_on_invalid

    def _validate_prediction(self, prediction: Prediction) -> Tuple[bool, List[str]]:
        """
        Run all validators on a prediction and collect feedback.

        Args:
            prediction: The prediction to validate

        Returns:
            Tuple containing:
                - is_valid (bool): True if all validators pass, False otherwise
                - feedback_messages (List[str]): List of error messages from failed validators
        """
        feedback_messages: List[str] = []

        if not self.validators:
            return True, feedback_messages

        for i, validator in enumerate(self.validators):
            try:
                error_msg = validator(prediction)
                if error_msg:
                    feedback_messages.append(error_msg)
            except Exception as e:
                error = f"Validator {i + 1} raised an exception: {str(e)}"
                feedback_messages.append(error)

        return len(feedback_messages) == 0, feedback_messages

    def forward(self, **kwargs) -> Prediction:
        """
        Run the wrapped module with Retry based on validator feedback in a conversation format.

        Args:
            **kwargs: Inputs to pass to the wrapped module

        Returns:
            The Retryd output from the wrapped module (dspy.Prediction)

        Raises:
            ValueError: If fail_on_invalid is True and Retry is unsuccessful after max attempts
        """
        attempts_made: int = 0
        # Tracking variables for the refinement process
        prediction: Optional[Prediction] = None  # Current prediction being evaluated
        feedback_history: List[str] = []  # History of feedback messages from validators
        attempt_outputs: List[str] = []  # String representations of all prediction attempts
        best_prediction: Optional[Prediction] = None  # Prediction with the fewest validation issues found so far
        best_feedback_count: float = float('inf')  # Number of validation issues in the best prediction (lower is better)
        is_valid: bool = False

        if self.verbose:
            print(
                f"Starting Retry with {len(self.validators)} validators, max attempts: {self.max_attempts}")

        for attempt in range(self.max_attempts):
            attempts_made = attempt + 1

            if self.verbose:
                print(f"Attempt {attempts_made}/{self.max_attempts}")

            try:
                mod = self.module.deepcopy()

                # For subsequent attempts, use the conversational feedback
                if attempt > 0 and feedback_history:
                    if self.verbose:
                        print(
                            f"Using conversational feedback with {len(feedback_history)} previous attempts")
                    adapter = ConversationalFeedbackAdapter(feedback_history=feedback_history, attempt_outputs=attempt_outputs)
                    with dspy.context(adapter=adapter):
                        prediction = mod(**kwargs)
                else:
                    prediction = mod(**kwargs)

                if self.verbose:
                    print(f"Output: {prediction}")

                # Store the prediction output for conversation history
                if prediction is not None:
                    pred_str = str(prediction)
                    attempt_outputs.append(pred_str)

                # Validate the prediction
                is_valid, feedback_messages = self._validate_prediction(prediction)

                if self.verbose:
                    status = "VALID" if is_valid else f"INVALID ({len(feedback_messages)} issues)"
                    print(f"Validation result: {status}")
                    if not is_valid:
                        for i, msg in enumerate(feedback_messages):
                            print(f"  Issue {i + 1}: {msg}")

                # Track the best prediction so far (with the fewest issues)
                if prediction is not None:
                    current_feedback_count = len(feedback_messages)
                    if current_feedback_count < best_feedback_count:
                        best_feedback_count = current_feedback_count
                        best_prediction = prediction

                # If valid, exit the Retry loop
                if is_valid:
                    if self.verbose:
                        print(f"✓ Retry successful after {attempts_made} attempts")
                    break

                # Format feedback as bullet points for readability
                current_feedback = "\n".join([f"- {msg}" for msg in feedback_messages])
                feedback_history.append(current_feedback)

            except Exception as e:
                if self.verbose:
                    print(f"✗ Attempt {attempts_made} failed: {e}")

                # Store the actual error message in attempt_outputs
                error_details = f"[Error: {str(e)}]"
                attempt_outputs.append(error_details)
                # Provide instructive feedback for the next attempt
                error_feedback = f"Your response encountered this error: {str(e)}.\n Please modify your response to address this issue."

                # For common pattern of missing output keys, provide more specific guidance
                if "missing" in str(e).lower() and "key" in str(e).lower():
                    error_feedback += "\nMake sure your response includes all required output fields."

                feedback_history.append(error_feedback)
                continue

        # Use the best prediction if the final one isn't valid
        if prediction is None or (not is_valid and best_prediction is not None):
            prediction = best_prediction or Prediction()
            if self.verbose and best_prediction is not None:
                print("Using best prediction found (not fully valid)")

        # Add Retry metadata to the prediction
        if hasattr(prediction, "__dict__"):
            is_valid, _ = self._validate_prediction(prediction)
            metadata = {
                "attempts_made": attempts_made,
                "max_attempts": self.max_attempts,
                "Retry_successful": is_valid,
                "validators_count": len(self.validators),
                "feedback_history": feedback_history,
                "attempt_outputs": attempt_outputs
            }
            setattr(prediction, "_Retry_metadata", metadata)

        # Raise an error if configured to fail on invalid results
        if self.fail_on_invalid and not is_valid:
            raise ValueError(
                f"Retry failed after {attempts_made} attempts. Final issues: {feedback_history[-1] if feedback_history else 'Unknown'}")

        return prediction