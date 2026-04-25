from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from random import sample
from typing import Callable

from pydantic import BaseModel
from tqdm import tqdm

import dspy
from dspy.predict.avatar import ActionOutput
from dspy.teleprompt.teleprompt import Teleprompter

DEFAULT_MAX_EXAMPLES = 10


class EvalResult(BaseModel):
    example: dict
    score: float
    actions: list[ActionOutput] | None = None


class Comparator(dspy.Signature):
    """After executing the given actions on user inputs using the given instruction, some inputs have yielded
    good, results, while others have not. I'll provide you the inputs along with their, corresponding evaluation
    metrics:

    Task:
    (1) Firstly, identify and contrast the patterns of inputs that have achieved good results with those that have
    not.
    (2) Then, review the computational logic for any inconsistencies in the previous actions.
    (3) Lastly, specify the modification in tools used that can lead to improved performance on the negative
    inputs."""

    instruction: str = dspy.InputField(
        prefix="Instruction: ",
        desc="Instruction for the actor to execute the task",
    )
    actions: list[str] = dspy.InputField(
        prefix="Actions: ",
        desc="Actions actor can take to complete the task",
    )
    pos_input_with_metrics: list[EvalResult] = dspy.InputField(
        prefix="Positive Inputs: ",
        desc="Positive inputs along with their score on a evaluation metric and actions taken",
    )
    neg_input_with_metrics: list[EvalResult] = dspy.InputField(
        prefix="Negative Inputs: ",
        desc="Negative inputs along with their score on a evaluation metric and actions taken",
    )
    feedback: str = dspy.OutputField(
        prefix="Feedback: ",
        desc="Feedback for the actor to improve the performance of negative inputs",
    )


class FeedbackBasedInstruction(dspy.Signature):
    """There is a task that needs to be completed for which one can use multiple tools to achieve the desired
    outcome. A group's performance was evaluated on a dataset of inputs, the inputs that did well are positive
    inputs, and the inputs that did not do well are negative inputs. You received feedback on how they can better
    use the tools to improve your performance on the negative inputs. You have been provided with the previous
    instruction, that they followed to use tools to complete the task, and the feedback on your performance. Your
    task is to incorporate the feedback and generate a detailed instruction for the group to follow to improve
    their performance on the task. Make sure that the new instruction talks about how to use the tools effectively
    and should be no more than 3 paragraphs long. The previous instruction contains general guidelines that you
    must retain in the new instruction."""

    previous_instruction: str = dspy.InputField(
        prefix="Previous Instruction: ",
        desc="Previous instruction for the actor to execute the task",
    )
    feedback: str = dspy.InputField(
        prefix="Feedback: ",
        desc="Feedback for the actor to improve the performance of negative inputs",
    )
    new_instruction: str = dspy.OutputField(
        prefix="New Instruction: ",
        desc="New instruction for the actor to execute the task",
    )


class AvatarOptimizer(Teleprompter):
    """An optimizer that iteratively refines actor instructions based on performance feedback.

    AvatarOptimizer improves actor-based DSPy programs by analyzing their performance on positive
    and negative examples, generating feedback, and refining the instruction prompts. It uses a
    comparator to identify patterns in successful vs unsuccessful executions and updates instructions
    accordingly.

    The optimization process:
    1. Evaluates the actor on a training set
    2. Separates results into positive (score >= upper_bound) and negative (score <= lower_bound)
    3. Uses a comparator LM to analyze differences and generate improvement feedback
    4. Creates refined instructions based on the feedback
    5. Repeats for max_iters or until performance plateaus

    Args:
        metric: Callable evaluation function that takes (example, prediction) and returns a numeric
            score. Must not be None.
        max_iters: Maximum number of optimization iterations. Defaults to 10.
        lower_bound: Score threshold below which examples are considered negative. Defaults to 0.
        upper_bound: Score threshold above which examples are considered positive. Defaults to 1.
        max_positive_inputs: Maximum number of positive examples to sample for comparison. If None,
            uses DEFAULT_MAX_EXAMPLES (10).
        max_negative_inputs: Maximum number of negative examples to sample for comparison. If None,
            uses DEFAULT_MAX_EXAMPLES (10).
        optimize_for: Optimization direction, either "max" (maximize metric) or "min" (minimize metric).
            Defaults to "max".

    Examples:
        >>> import dspy
        >>> from dspy.teleprompt import AvatarOptimizer
        >>>
        >>> def accuracy_metric(example, prediction):
        ...     return int(example.answer == prediction.answer)
        >>>
        >>> optimizer = AvatarOptimizer(
        ...     metric=accuracy_metric,
        ...     max_iters=5,
        ...     upper_bound=1.0,
        ...     lower_bound=0.0
        ... )
        >>> optimized_actor = optimizer.compile(actor, trainset=train_examples)
    """

    def __init__(
        self,
        metric: Callable,
        max_iters: int = 10,
        lower_bound: int = 0,
        upper_bound: int = 1,
        max_positive_inputs: int | None = None,
        max_negative_inputs: int | None = None,
        optimize_for: str = "max",
    ):
        assert metric is not None, "`metric` argument cannot be None. Please provide a metric function."
        self.metric = metric
        self.optimize_for = optimize_for
        self.max_iters = max_iters
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_positive_inputs = max_positive_inputs or DEFAULT_MAX_EXAMPLES
        self.max_negative_inputs = max_negative_inputs or DEFAULT_MAX_EXAMPLES
        self.comparator = dspy.TypedPredictor(Comparator)
        self.feedback_instruction = dspy.Predict(FeedbackBasedInstruction)

    def process_example(self, actor, example, return_outputs):
        """Process a single example through the actor and compute its metric score.

        Args:
            actor: The DSPy actor module to evaluate.
            example: A dspy.Example instance containing inputs to process.
            return_outputs: If True, returns (example, prediction, score). If False, returns only score.

        Returns:
            If return_outputs is True: tuple of (example, prediction, score)
            If return_outputs is False: score (float)
            On exception: returns (example, None, 0) if return_outputs else 0
        """
        actor = deepcopy(actor)
        try:
            prediction = actor(**example.inputs().toDict())
            score = self.metric(example, prediction)
            if return_outputs:
                return example, prediction, score
            else:
                return score
        except Exception as e:
            print(e)
            if return_outputs:
                return example, None, 0
            else:
                return 0

    def thread_safe_evaluator(self, devset, actor, return_outputs=False, num_threads=None):
        """Evaluate an actor on a dataset using parallel thread execution.

        Args:
            devset: List of dspy.Example instances to evaluate.
            actor: The DSPy actor module to evaluate.
            return_outputs: If True, returns (avg_metric, results) where results is a list of
                (example, prediction, score) tuples. If False, returns only avg_metric.
            num_threads: Number of parallel threads. If None, uses dspy.settings.num_threads.

        Returns:
            If return_outputs is True: tuple of (avg_metric, results)
            If return_outputs is False: avg_metric (float)
        """
        total_score = 0
        total_examples = len(devset)
        results = []
        num_threads = num_threads or dspy.settings.num_threads

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.process_example, actor, example, return_outputs) for example in devset]
            for future in tqdm(futures, total=total_examples, desc="Processing examples"):
                result = future.result()
                if return_outputs:
                    example, prediction, score = result
                    total_score += score
                    results.append((example, prediction, score))
                else:
                    total_score += result

        avg_metric = total_score / total_examples
        if return_outputs:
            return avg_metric, results
        else:
            return avg_metric

    def _get_pos_neg_results(
        self, actor: dspy.Module, trainset: list[dspy.Example]
    ) -> tuple[float, list[EvalResult], list[EvalResult]]:
        """Separate training results into positive and negative examples based on score thresholds.

        Args:
            actor: The DSPy actor module to evaluate.
            trainset: List of training examples to evaluate.

        Returns:
            tuple of (avg_score, pos_inputs, neg_inputs) where:
                - avg_score: Average metric score across all examples
                - pos_inputs: List of EvalResult for examples with score >= upper_bound
                - neg_inputs: List of EvalResult for examples with score <= lower_bound

        Raises:
            ValueError: If no positive examples found (try lowering upper_bound) or no negative
                examples found (try raising lower_bound).
        """
        pos_inputs = []
        neg_inputs = []
        avg_score, results = self.thread_safe_evaluator(trainset, actor, return_outputs=True)
        print(f"Average Score: {avg_score}")

        for example, prediction, score in results:
            if score >= self.upper_bound:
                pos_inputs.append(
                    EvalResult(
                        example=example.inputs().toDict(),
                        score=score,
                        actions=prediction.actions if prediction else None
                    )
                )
            elif score <= self.lower_bound:
                neg_inputs.append(
                    EvalResult(
                        example=example.inputs().toDict(),
                        score=score,
                        actions=prediction.actions if prediction else None
                    )
                )

        if len(pos_inputs) == 0:
            raise ValueError("No positive examples found, try lowering the upper_bound or providing more training data")
        if len(neg_inputs) == 0:
            raise ValueError("No negative examples found, try raising the lower_bound or providing more training data")

        return (avg_score, pos_inputs, neg_inputs)

    def compile(self, student, *, trainset):
        """Optimize an actor by iteratively refining its instructions based on performance feedback.

        Args:
            student: The DSPy actor module to optimize. Must have an `actor` attribute with a
                `signature` that contains instructions, and a `tools` attribute.
            trainset: List of dspy.Example instances for training/evaluation.

        Returns:
            The optimized actor with refined instructions. The best-performing actor across all
            iterations is returned based on the optimization direction (max or min).

        Examples:
            >>> optimizer = AvatarOptimizer(metric=accuracy, max_iters=5)
            >>> optimized_actor = optimizer.compile(my_actor, trainset=training_data)
        """
        best_actor = deepcopy(student)
        best_score = -999 if self.optimize_for == "max" else 999

        for i in range(self.max_iters):
            print(20*"=")
            print(f"Iteration {i+1}/{self.max_iters}")
            score, pos_inputs, neg_inputs = self._get_pos_neg_results(best_actor, trainset)
            print(f"Positive examples: {len(pos_inputs)}")
            print(f"Negative examples: {len(neg_inputs)}")
            print(f"Sampling {self.max_positive_inputs} positive examples and {self.max_negative_inputs} negative examples")

            if self.max_positive_inputs and len(pos_inputs) > self.max_positive_inputs:
                pos_inputs = sample(pos_inputs, self.max_positive_inputs)
            if self.max_negative_inputs and len(neg_inputs) > self.max_negative_inputs:
                neg_inputs = sample(neg_inputs, self.max_negative_inputs)

            feedback = self.comparator(
                instruction=best_actor.actor.signature.instructions,
                actions=[str(tool) for tool in best_actor.tools],
                pos_input_with_metrics=pos_inputs,
                neg_input_with_metrics=neg_inputs
            ).feedback

            new_instruction = self.feedback_instruction(
                previous_instruction=best_actor.actor.signature.instructions,
                feedback=feedback
            ).new_instruction

            print(f"Generated new instruction: {new_instruction}")

            if (self.optimize_for == "max" and best_score < score) or (self.optimize_for == "min" and best_score > score):
                best_actor.actor.signature = best_actor.actor.signature.with_instructions(new_instruction)
                best_actor.actor_clone = deepcopy(best_actor.actor)
                best_score = score

        print(f"Best Actor: {best_actor}")
        return best_actor
