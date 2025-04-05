import inspect
import textwrap
from typing import Any, Callable, Optional
import ujson
import dspy
from dspy.adapters.utils import get_field_description_string
from dspy.predict.predict import Prediction
from dspy.signatures import InputField, OutputField, Signature
from dspy.primitives import Module
import copy


class OfferFeedback(Signature):
    """
    In the discussion, assign blame to each module that contributed to the final reward being below the threshold, if
    any. Then, prescribe concrete advice of how the module should act on its future input when we retry the process, if
    it were to receive the same or similar inputs. If a module is not to blame, the advice should be N/A.
    The module will not see its own history, so it needs to rely on entirely concrete and actionable advice from you
    to avoid the same mistake on the same or similar inputs.
    """
    program_code: str = InputField(desc="The code of the program that we are analyzing")
    modules_defn: str = InputField(desc="The definition of each module in the program, including its I/O")
    program_inputs: str = InputField(desc="The inputs to the program that we are analyzing")
    program_trajectory: str = InputField(desc="The trajectory of the program's execution, showing each module's I/O")
    program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    reward_code: str = InputField(desc="The code of the reward function that we are analyzing")
    target_threshold: float = InputField(desc="The target threshold for the reward function")
    reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    module_names: list[str] = InputField(desc="The names of the modules in the program, for which we seek advice")
    discussion: str = OutputField(desc="Discussing blame of where each module went wrong, if it did")
    advice: dict[str, str] = OutputField(
        desc="For each module, describe very concretely, in this order: the specific scenarios in which it has made "
             "mistakes in the past and what each mistake was, followed by what it should do differently in that kind of"
             "scenario in the future. If the module is not to blame, write N/A."
    )


class Refine(Module):
    """
    Enhances prediction quality through a flexible refinement system that adapts based on the parameters provided.

    Using `reward_fn`, it leverages LLM-as-a-judge capabilities to numerically score outputs based on quality metrics,
    while `validators` employs programmatic functions that check for specific requirements or correctness properties
    in the generated content. When both parameters are present, validators act as gatekeepers that must be satisfied
    before optimization through the reward function occurs. If a `threshold` is specified alongside `reward_fn`,
    predictions scoring below this value trigger additional refinement attempts. The module will attempt up to `N`
    refinement iterations, with options for chain-of-thought reasoning (`use_cot`) and detailed printouts to stdout
    (via the `verbose` parameter). If all refinement attempts fail to produce satisfactory outputs, the
    `fail_on_invalid` parameter determines whether execution halts with an error or proceeds with the best available result.

    Example:
    >>> import dspy
    >>>
    >>> # Define a hard constraint: Answer must end with a period.
    >>> def validate_period(pred):
    ...     if not pred.answer.endswith("."):
    ...         return False, "Answer must end with a period."
    ...     return True, ""
    >>>
    >>> # Define a soft constraint: Reward factual accuracy (simulated).
    >>> # In a real scenario, this could involve an LLM call or external check.
    >>> def reward_factual(args, pred):
    ...     # Simplified simulation: high reward only for the correct capital.
    ...     if "Paris" in pred.answer and "France" in args.get("question", ""):
    ...         return 1.0
    ...     elif "Lyon" in pred.answer: # Penalize incorrect common answer
    ...          return 0.2
    ...     else:
    ...         return 0.5 # Neutral for other answers
    >>>
    >>> # Configure the base module and Refine
    >>> qa_module = dspy.ChainOfThought("question -> answer")
    >>> refined_qa = Refine(
    ...     module=qa_module,
    ...     validators=[validate_period],
    ...     reward_fn=reward_factual,
    ...     threshold=0.9,
    ...     N=3,
    ...     verbose=True
    ... )
    >>>
    >>> # question = "What is the capital of France?"
    >>>
    >>> # Hypothetical Run (output depends on the LLM and temperature settings):
    >>> # Attempt 1: LLM generates "Paris is the capital"
    >>> #   Validator Check: Fails ('validate_period' - missing period).
    >>> #   Refine Action: Retry with feedback "Answer must end with a period."
    >>>
    >>> # Attempt 2: LLM generates "The capital of France is Lyon."
    >>> #   Validator Check: Passes ('validate_period' - has period).
    >>> #   Reward Check: reward_factual returns 0.2 (incorrect). Threshold is 0.9. Fails soft constraint.
    >>> #   Refine Action: Retry with feedback (internally generated, aiming for factual accuracy).
    >>>
    >>> # Attempt 3: LLM generates "Paris is the capital of France."
    >>> #   Validator Check: Passes ('validate_period' - has period).
    >>> #   Reward Check: reward_factual returns 1.0 (correct). Threshold is 0.9. Passes soft constraint.
    >>> #   Refine Action: Success! Return this result.
    >>>
    >>> # Example illustrative final output after successful refinement:
    >>> # result = refined_qa(question="What is the capital of France")
    >>> # print(result.answer)
    >>> # "Paris is the capital of France."
    """
    def __init__(
        self,
        signature: Signature,
        reward_fn: Optional[Callable[[dict[str, Any], Prediction], float]] = None,
        threshold: Optional[float] = None,
        validators: Optional[list[Callable[[Prediction], tuple[bool, str]]]] = None,
        N: int = 3,
        use_cot: bool = False,
        verbose: bool = False,
        fail_on_invalid: bool = False
    ) -> None:
        """
        Initializes the Refine module.

        Args:
            signature (Signature): The DSPy signature that defines the input and output fields for the module being refined.
            reward_fn (Optional[Callable]): Function evaluating output quality (args, pred) -> float.
            threshold (Optional[float]): Minimum acceptable reward score.
            validators (Optional[List[Callable]]): Functions validating predictions (pred) -> (bool, str).
            N (int): Maximum refinement iterations. Defaults to 3.
            use_cot (bool): Use ChainOfThought for internal prediction steps. Defaults to False.
            verbose (bool): Print detailed logs during refinement. Defaults to False.
            fail_on_invalid (bool): Raise ValueError if constraints aren't met after N attempts. Defaults to False.
        """
        super().__init__()
        assert signature is not None, "A `signature` must be provided to Refine."
        self.reward_fn = reward_fn
        self.threshold = threshold
        self.validators = validators or []
        self.N = max(1, N)
        self.use_cot = use_cot
        self.verbose = verbose
        self.fail_on_invalid = fail_on_invalid
        self.signature = copy.deepcopy(signature)

        # Create internal modules based on the signature
        predictor_class = dspy.ChainOfThought if self.use_cot else dspy.Predict
        # Create the standard module for soft constraints (using original signature)
        self.soft_constraints_module = predictor_class(self.signature)
        # Create the validator module with added 'previous_attempts' field if needed
        if self.validators:
            # Add 'previous_attempts' field if hard constraints (validators) are used
            if "previous_attempts" not in self.signature.input_fields:
                validator_signature = self.signature.append(
                    "previous_attempts",
                    dspy.InputField(
                        desc="Review these previous attempts and feedback on their quality to improve your response."
                    )
                )
            else:
                validator_signature = self.signature
            # Create the module for validators
            self.hard_constraints_module = predictor_class(validator_signature)
        # Code Inspection
        self.soft_constraints_module_code = inspect.getsource(self.soft_constraints_module.__class__)
        try:
            self.reward_fn_code = inspect.getsource(reward_fn) if reward_fn else ""
        except (TypeError, OSError):
            self.reward_fn_code = ""

    def _get_temperatures(self) -> list[float]:
        """
        Generate a sequence of temperatures for multiple iterations.

        Returns:
            list: A list of temperature values to use for each iteration.
        """
        lm = dspy.settings.lm
        temps = [lm.kwargs["temperature"]] + [0.5 + i * (0.5 / self.N) for i in range(self.N)]
        temps = list(dict.fromkeys(temps))[: self.N]
        return temps

    def _validate_prediction(self, prediction: Prediction) -> tuple[bool, list[str]]:
        """
        Validates a prediction using the configured validators.

        Args:
            prediction (Prediction): The prediction to validate.

        Returns:
            tuple: (bool, list) indicating if all validators passed and feedback messages.
        """
        feedback_messages: list[str] = []
        if not self.validators:
            return True, feedback_messages
        for i, validator in enumerate(self.validators):
            try:
                is_valid, error_msg = validator(prediction)
                if not is_valid:
                    feedback_messages.append(error_msg)
            except Exception as e:
                feedback_messages.append(f"Validator {i + 1} raised an exception: {str(e)}")
        return len(feedback_messages) == 0, feedback_messages

    @staticmethod
    def _format_previous_attempts(attempts_log: list[dict[str, Any]]) -> str:
        """
        Formats the log of previous attempts into a string for inclusion in the module input.

        Args:
            attempts_log (list): List of dictionaries containing attempt information.

        Returns:
            str: Formatted string of previous attempts and their feedback.
        """
        if not attempts_log:
            return ""
        formatted_parts: list[str] = []
        for idx, attempt in enumerate(attempts_log):
            attempt_num = attempt.get("attempt_number", idx + 1)
            formatted_parts.append(f"ATTEMPT {attempt_num}:")
            if "output" in attempt:
                outputs = attempt.get("output", {})
                formatted_parts.append("Output:")
                for key, value in outputs.items():
                    formatted_parts.append(f"  {key}: {value}")
            if "error" in attempt:
                formatted_parts.append(f"Error: {attempt['error']}")
            formatted_parts.append("Feedback:")
            for msg in attempt.get("feedback", []):
                formatted_parts.append(f"  - {msg}")
            if "reward" in attempt:
                formatted_parts.append(f"Reward Score: {attempt.get('reward', 0)}")
            formatted_parts.append(f"All Validators Passed: {attempt.get('all_validators_passed', False)}")
            formatted_parts.append("")
        return "\n".join(formatted_parts)

    def _get_feedback(
        self,
        mod: Module,
        inputs: dict[str, Any],
        outputs: Prediction,
        reward_value: float,
        trace: list[Any]
    ) -> Optional[dict[str, str]]:
        """
        Generates feedback for module improvement using LLM.

        Args:
            mod (Module): The module instance.
            inputs (dict): Input values.
            outputs (Prediction): Output values.
            reward_value (float): Current reward value.
            trace (list): Execution trace.

        Returns:
            dict: Feedback for each module or None.
        """
        if self.reward_fn is None:
            return None
        predictor2name = {predictor: name for name, predictor in mod.named_predictors()}
        module_names = [name for name, _ in mod.named_predictors()]
        modules = dict(program_code=self.soft_constraints_module_code, modules_defn=inspect_modules(mod))
        trajectory = [dict(module_name=predictor2name.get(p, "unknown"), inputs=i, outputs=dict(o)) for p, i, o in trace]
        trajectory = dict(program_inputs=inputs, program_trajectory=trajectory, program_outputs=dict(outputs))
        reward = dict(reward_code=self.reward_fn_code, target_threshold=self.threshold, reward_value=reward_value)
        advise_kwargs = dict(**modules, **trajectory, **reward, module_names=module_names)
        advise_kwargs = {k: v if isinstance(v, str) else ujson.dumps(recursive_mask(v), indent=2) for k, v in advise_kwargs.items()}
        try:
            feedback = dspy.Predict(OfferFeedback)(**advise_kwargs).advice
            return feedback
        except Exception as e:
            if self.verbose:
                print(f"Feedback generation failed: {e}")
            return {name: "N/A" for name in module_names}

    def _evaluate_soft_constraints(
        self,
        mod: Module,
        kwargs: dict[str, Any],
        outputs: Prediction,
        trace: list[Any],
        generate_feedback: bool = False
    ) -> tuple[Optional[float], Optional[dict[str, str]]]:
        """
        Evaluates soft constraints and generates feedback if requested.

        Args:
            mod: The module instance.
            kwargs: Input arguments for the module.
            outputs: The prediction to evaluate.
            trace: Execution trace.
            generate_feedback: Whether to generate feedback for next iteration.

        Returns:
            A tuple containing (reward_value, feedback) where feedback is for the next iteration.
        """
        reward_value = None
        feedback = None
        if self.reward_fn:
            reward_value = self.reward_fn(kwargs, outputs)
            if self.verbose and reward_value is not None:
                print(f"Reward value: {reward_value} (threshold: {self.threshold})")
            if generate_feedback:
                feedback = self._get_feedback(mod, kwargs, outputs, reward_value, trace)
                if self.verbose and feedback:
                    print(f"Generated feedback: {feedback}")
        return reward_value, feedback

    def _evaluate_hard_constraints(
        self,
        outputs: Prediction,
        iteration: int,
        reward_value: Optional[float] = None
    ) -> tuple[bool, list[str], dict[str, Any]]:
        """
        Evaluates hard constraints and formats the attempt information.

        Args:
            outputs: The prediction to evaluate.
            iteration: Current iteration number.
            reward_value: Optional reward value from soft constraints.

        Returns:
            A tuple containing (validators_passed, feedback_messages, attempt_info).
        """
        validators_passed, feedback_messages = self._validate_prediction(outputs)
        if self.verbose:
            status = "VALID" if validators_passed else "INVALID"
            print(f"Validation status: {status}")
            if feedback_messages:
                print(f"Validation feedback ({len(feedback_messages)} messages):")
                for i, msg in enumerate(feedback_messages):
                    print(f"  {i + 1}: {msg}")
        attempt_info = {
            "attempt_number": iteration,
            "output": {k: getattr(outputs, k, "") for k in self.signature.output_fields}
            if self.signature else dict(outputs),
            "all_validators_passed": validators_passed,
            "feedback": feedback_messages
        }
        if reward_value is not None:
            attempt_info["reward"] = reward_value
        return validators_passed, feedback_messages, attempt_info

    def _is_current_better_than_previous(
        self,
        current_reward: Optional[float],
        current_validators_passed: bool,
        current_feedback_messages: list[str],
        previous_best_reward: float,
        previous_validators_passed: bool,
        previous_feedback_count: int
    ) -> bool:
        """
        Determines if the current prediction state is superior to the previous best state.

        Comparison Logic:
        1. Validator Passing: Passing validators is always better than failing.
        2. Reward Score (if applicable): If both states pass validators (or if no validators are used),
           the state with the higher reward score wins.
        3. Error Count (if applicable): If both states fail validators, the state with fewer
           validation errors wins.

        Args:
            current_reward: Reward score for the current prediction (None if reward_fn not used).
            current_validators_passed: Whether the current prediction passed all validators.
            current_feedback_messages: List of feedback messages from validators for the current prediction.
            previous_best_reward: Reward score of the previously identified best prediction.
            previous_validators_passed: Whether the previous best prediction passed all validators.
            previous_feedback_count: Number of validation errors for the previous best prediction.

        Returns:
            bool: True if the current prediction state is better than the previous best, False otherwise.
        """
        current_reward_val = current_reward if current_reward is not None else -float('inf')
        current_error_count = len(current_feedback_messages)
        # Case 1: No validators involved (reward-only mode)
        if not self.validators:
            return current_reward_val > previous_best_reward
        # Case 2: Current prediction passes validators, but previous best did not
        if current_validators_passed and not previous_validators_passed:
            return True
        # Case 3: Previous best passed validators, but current prediction does not
        if not current_validators_passed and previous_validators_passed:
            return False
        # Case 4: Both predictions pass validators
        if current_validators_passed and previous_validators_passed:
            # Higher reward wins
            return current_reward_val > previous_best_reward
        # Case 5: Neither prediction passes validators
        if not current_validators_passed and not previous_validators_passed:
            # Fewer validation errors win
            if current_error_count < previous_feedback_count:
                return True
            # If error counts are equal, the previous one is retained (current is not strictly better)
            return False

    def forward(self, **kwargs: Any) -> Prediction:
        """
        Runs the refinement process iteratively to improve prediction quality based on constraints.

        Args:
            **kwargs: Input arguments for the wrapped module.

        Returns:
            Prediction: The best prediction found during the refinement process.

        Raises:
            ValueError: If `fail_on_invalid` is True and no prediction meeting all
                        constraints is found within N iterations.
            RuntimeError: If no prediction could be generated at all after N attempts
                          (e.g., due to repeated exceptions).
        """
        lm = dspy.settings.lm
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        temps = self._get_temperatures()
        iterations_made = 0
        best_prediction: Optional[Prediction] = None
        best_reward: float = -float("inf")
        best_trace: Optional[list[Any]] = None
        best_feedback_count: float | int = float("inf")
        best_validators_passed: bool = False
        attempts_log: list[dict[str, Any]] = []
        programmatic_feedback_for_llm: Optional[dict[str, str]] = None
        has_reward_fn = self.reward_fn is not None
        has_validators = bool(self.validators)

        if self.verbose:
            mode_desc = []
            if has_reward_fn: mode_desc.append(f"soft constraints (reward threshold: {self.threshold})")
            if has_validators: mode_desc.append(f"hard constraints ({len(self.validators)} validators)")
            print(f"Starting Refine with N={self.N}, using: {', '.join(mode_desc) or 'base module execution'}")

        for idx, temp in enumerate(temps):
            iterations_made = idx + 1
            current_attempt_info: dict[str, Any] = {"attempt_number": iterations_made}
            passed_validators_current = True
            validator_feedback_messages = []
            if self.verbose:
                print(f"\nIteration {iterations_made}/{self.N} with temperature {temp}")
            current_kwargs = kwargs.copy()
            if has_validators:
                current_kwargs["previous_attempts"] = self._format_previous_attempts(attempts_log)
            lm_temp = lm.copy(temperature=temp)
            try:
                with dspy.context(lm=lm_temp, trace=[]):
                    if has_validators:
                        if not self.hard_constraints_module:
                            raise RuntimeError("Internal hard_constraints_module not initialized despite having validators.")
                        current_mod = self.hard_constraints_module
                        outputs: Prediction = current_mod(**current_kwargs)
                    elif has_reward_fn:
                        current_mod = self.soft_constraints_module
                        if programmatic_feedback_for_llm:
                            wrapped_adapter = apply_feedback(adapter, programmatic_feedback_for_llm, current_mod)
                            with dspy.context(adapter=wrapped_adapter):
                                outputs = current_mod(**current_kwargs)
                        else:
                            outputs = current_mod(**current_kwargs)
                    else:
                        current_mod = self.soft_constraints_module
                        outputs = current_mod(**current_kwargs)
                    trace = dspy.settings.trace

                # Evaluate Constraints
                current_reward = None
                # Hard Constraints (Validators)
                if has_validators:
                    passed_validators_current, validator_feedback_messages, attempt_info_hard = self._evaluate_hard_constraints(outputs, iterations_made)
                    current_attempt_info.update(attempt_info_hard)
                else:
                    current_attempt_info["all_validators_passed"] = True
                    current_attempt_info["feedback"] = []

                # Soft Constraints (Reward Function)
                programmatic_feedback_for_llm = None
                if has_reward_fn and passed_validators_current:
                    generate_llm_feedback = idx < self.N - 1  # Only generate if more attempts are possible
                    current_reward, programmatic_feedback_for_llm = self._evaluate_soft_constraints(
                        self.soft_constraints_module,
                        kwargs,
                        outputs,
                        trace,
                        generate_feedback=generate_llm_feedback
                    )
                    current_attempt_info["reward"] = current_reward
                    if self.verbose and current_reward is not None:
                        print(f"Reward: {current_reward} (Threshold: {self.threshold})")

                # Determine if Current Prediction is the New Best
                is_better = self._is_current_better_than_previous(
                    current_reward,
                    passed_validators_current,
                    validator_feedback_messages,
                    best_reward,
                    best_validators_passed,
                    best_feedback_count
                )
                if is_better:
                    if self.verbose:
                        print(
                            f"Found new best prediction (Reward: {current_reward}, Validators Passed: {passed_validators_current}, Errors: {len(validator_feedback_messages)})")
                    best_prediction = outputs
                    best_trace = trace
                    best_reward = current_reward if current_reward is not None else -float('inf')
                    best_validators_passed = passed_validators_current
                    best_feedback_count = len(validator_feedback_messages) if has_validators else 0

                # Check for Success (Met all constraints)
                reward_satisfied = not has_reward_fn or self.threshold is None or (current_reward is not None and current_reward >= self.threshold)
                validators_satisfied = not has_validators or passed_validators_current
                success = reward_satisfied and validators_satisfied
                attempts_log.append(current_attempt_info)
                if success:
                    if self.verbose:
                        print(f"✓ Success: Constraints met after {iterations_made} iterations.")
                    break

            except Exception as e:
                if self.verbose:
                    print(f"✗ Iteration {iterations_made} failed with exception: {e}")
                attempts_log.append({
                    "attempt_number": iterations_made,
                    "error": str(e),
                    "all_validators_passed": False,
                    "feedback": [f"Exception occurred: {str(e)}"]
                })
                # Reset LLM feedback if an exception occurred, as it might be misleading
                programmatic_feedback_for_llm = None

        if best_prediction is None:
            raise RuntimeError(f"Refine failed to produce any prediction after {self.N} attempts. See logs for errors.")

        final_reward_satisfied = not has_reward_fn or self.threshold is None or (best_reward is not None and best_reward >= (self.threshold or float('inf')))
        final_validators_satisfied = not has_validators or best_validators_passed
        refine_successful = final_reward_satisfied and final_validators_satisfied

        metadata = {
            "iterations_made": iterations_made,
            "N": self.N,
            "best_reward": best_reward if has_reward_fn else None,
            "target_threshold": self.threshold if has_reward_fn else None,
            "all_validators_passed": best_validators_passed if has_validators else True,
            "final_feedback_count": best_feedback_count if has_validators else 0,
            "refine_successful": refine_successful,
            "attempts_log": attempts_log
        }
        setattr(best_prediction, "Refine_metadata", metadata)

        if self.fail_on_invalid and not refine_successful:
            error_messages = [f"Refine failed to meet constraints after {iterations_made} iterations."]
            if has_reward_fn and not final_reward_satisfied:
                error_messages.append(f"- Best reward {best_reward} did not meet threshold {self.threshold}")
            if has_validators and not final_validators_satisfied:
                error_messages.append("- The best output failed the following mandatory constraints:")
                last_failed_validator_attempt = next(
                    (a for a in reversed(attempts_log) if not a.get("all_validators_passed", True) and "feedback" in a),
                    None)
                if last_failed_validator_attempt and last_failed_validator_attempt.get("feedback"):
                    for i, msg in enumerate(last_failed_validator_attempt["feedback"]):
                        error_messages.append(f"  - {i + 1}: {msg}")
                else:
                    error_messages.append(
                        "  - (No specific validator feedback captured in logs for the final best attempt)")
            raise ValueError("\n".join(error_messages))
        if best_trace:
            dspy.settings.trace.extend(best_trace)
        return best_prediction


def apply_feedback(adapter: Any, feedback: dict[str, str], mod: Module) -> Any:
    """
    Creates a wrapper adapter that applies feedback to module inputs.

    Args:
        adapter (Adapter): The original adapter.
        feedback (dict): Feedback for each module.
        mod (Module): The module instance.

    Returns:
        Adapter: A wrapped adapter that injects feedback.
    """
    signature2name = {predictor.signature: name for name, predictor in mod.named_predictors()}

    class WrapperAdapter(adapter.__class__):
        def __call__(self, lm: Any, lm_kwargs: dict[str, Any], signature: Signature,
                     demos: list[Any], inputs: dict[str, Any]) -> Any:
            if signature in signature2name:
                inputs["hint_"] = feedback
                signature = signature.append(
                    "hint_", InputField(desc="A hint to the module from an earlier run")
                )
            return adapter(lm, lm_kwargs, signature, demos, inputs)

    return WrapperAdapter()


def inspect_modules(program: Module) -> str:
    """
    Generate a formatted string representation of a module's structure.

    Args:
        program (Module): The module to inspect.

    Returns:
        str: A formatted string containing the module's structure.
    """
    separator = "-" * 80
    output = [separator]

    for idx, (name, predictor) in enumerate(program.named_predictors()):
        signature = predictor.signature
        instructions = textwrap.dedent(signature.instructions)
        instructions = ("\n" + "\t" * 2).join([""] + instructions.splitlines())

        output.append(f"Module {name}")
        output.append("\n\tInput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.input_fields).splitlines()))
        output.append("\tOutput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.output_fields).splitlines()))
        output.append(f"\tOriginal Instructions: {instructions}")
        output.append(separator)

    return "\n".join([o.strip("\n") for o in output])


def recursive_mask(o: Any) -> Any:
    """
    Recursively mask non-serializable objects to make them JSON-serializable.

    Args:
        o: Any Python object.

    Returns:
        A JSON-serializable version of the input object.
    """
    # If the object is already serializable, return it.
    try:
        ujson.dumps(o)
        return o
    except TypeError:
        pass
    # If it's a dictionary, apply recursively to its values.
    if isinstance(o, dict):
        return {k: recursive_mask(v) for k, v in o.items()}
    # If it's a list, apply recursively.
    elif isinstance(o, list):
        return [recursive_mask(v) for v in o]
    # If it's a tuple, apply recursively.
    elif isinstance(o, tuple):
        return tuple(recursive_mask(v) for v in o)
    # Otherwise, replace it with a placeholder string (or use repr(o)).
    else:
        return f"<non-serializable: {type(o).__name__}>"