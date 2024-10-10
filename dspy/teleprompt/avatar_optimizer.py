import dspy

from tqdm import tqdm
from copy import deepcopy
from random import sample
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Tuple, Optional

from .teleprompt import Teleprompter
from dspy.predict.avatar import ActionOutput


DEFAULT_MAX_EXAMPLES = 10


class EvalResult(BaseModel):
    example: dict
    score: float
    actions: Optional[List[ActionOutput]] = None


class Comparator(dspy.Signature):
    """After executing the given actions on user inputs using the given instruction, some inputs have yielded good, results, while others have not. I'll provide you the inputs along with their, corresponding evaluation metrics:

Task:
(1) Firstly, identify and contrast the patterns of inputs that have achieved good results with those that have not.
(2) Then, review the computational logic for any inconsistencies in the previous actions.
(3) Lastly, specify the modification in tools used that can lead to improved performance on the negative inputs."""

    instruction: str = dspy.InputField(
        prefix="Instruction: ",
        desc="Instruction for the actor to execute the task",
    )
    actions: List[str] = dspy.InputField(
        prefix="Actions: ",
        desc="Actions actor can take to complete the task",
    )
    pos_input_with_metrics: List[EvalResult] = dspy.InputField(
        prefix="Positive Inputs: ",
        desc="Positive inputs along with their score on a evaluation metric and actions taken",
    )
    neg_input_with_metrics: List[EvalResult] = dspy.InputField(
        prefix="Negative Inputs: ",
        desc="Negative inputs along with their score on a evaluation metric and actions taken",
    )
    feedback: str = dspy.OutputField(
        prefix="Feedback: ",
        desc="Feedback for the actor to improve the performance of negative inputs",
    )


class FeedbackBasedInstruction(dspy.Signature):
    """There is a task that needs to be completed for which one can use multiple tools to achieve the desired outcome. A group's performance was evaluated on a dataset of inputs, the inputs that did well are positive inputs, and the inputs that did not do well are negative inputs.

You received feedback on how they can better use the tools to improve your performance on the negative inputs. You have been provided with the previous instruction, that they followed to use tools to complete the task, and the feedback on your performance.

Your task is to incorporate the feedback and generate a detailed instruction for the group to follow to improve their performance on the task.

Make sure that the new instruction talks about how to use the tools effectively and should be no more than 3 paragraphs long. The previous instruction contains general guidelines that you must retain in the new instruction."""

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
    def __init__(
        self,
        metric: Callable,
        max_iters: int = 10,
        lower_bound: int = 0,
        upper_bound: int = 1,
        max_positive_inputs: int = None,
        max_negative_inputs: int = None,
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


    def thread_safe_evaluator(self, devset, actor, return_outputs=False, num_threads=60):
        total_score = 0
        total_examples = len(devset)
        results = []

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
        self, 
        actor: dspy.Module, 
        trainset: List[dspy.Example]
    ) -> Tuple[float, List[EvalResult], List[EvalResult]]:
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
        best_actor = deepcopy(student)
        best_score = -999 if self.optimize_for == "max" else 999
        
        for i in range(self.max_iters):
            print(20*'=')
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
