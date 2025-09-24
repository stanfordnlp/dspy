import logging
import random
from typing import Any, Callable, Protocol, TypedDict

from gepa import EvaluationBatch, GEPAAdapter
from gepa.core.adapter import ProposalFn

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import History
from dspy.adapters.types.base_type import Type
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Prediction
from dspy.teleprompt.bootstrap_trace import TraceData

logger = logging.getLogger(__name__)

class LoggerAdapter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log(self, x: str):
        self.logger.info(x)

DSPyTrace = list[tuple[Any, dict[str, Any], Prediction]]


class ReflectiveExample(TypedDict):
    """
    Structure of individual examples in the reflective dataset.

    Each example contains the predictor inputs, generated outputs, and feedback from evaluation.
    """
    Inputs: dict[str, Any]                              # Predictor inputs (may include str, dspy.Image, etc.)
    Generated_Outputs: dict[str, Any] | str             # Success: dict with output fields, Failure: error message string
    Feedback: str                                       # Always a string - from metric function or parsing error message


class ScoreWithFeedback(Prediction):
    score: float
    feedback: str

class PredictorFeedbackFn(Protocol):
    def __call__(
        predictor_output: dict[str, Any],
        predictor_inputs: dict[str, Any],
        module_inputs: Example,
        module_outputs: Prediction,
        captured_trace: DSPyTrace,
    ) -> ScoreWithFeedback:
        """
        This function is used to provide feedback to a specific predictor.
        The function is called with the following arguments:
        - predictor_output: The output of the predictor.
        - predictor_inputs: The inputs to the predictor.
        - module_inputs: The inputs to the whole program --- `Example`.
        - module_outputs: The outputs of the whole program --- `Prediction`.
        - captured_trace: The trace of the module's execution.
        # Shape of trace is: [predictor_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
        # Each trace is a tuple of (Predictor, PredictorInputs, Prediction)

        The function should return a `ScoreWithFeedback` object.
        The feedback is a string that is used to guide the evolution of the predictor.
        """
        ...

class DspyAdapter(GEPAAdapter[Example, TraceData, Prediction]):
    def __init__(
        self,
        student_module,
        metric_fn: Callable,
        feedback_map: dict[str, Callable],
        failure_score=0.0,
        num_threads: int | None = None,
        add_format_failure_as_feedback: bool = False,
        rng: random.Random | None = None,
        reflection_lm=None,
        custom_instruction_proposer: "ProposalFn | None" = None,
        warn_on_score_mismatch: bool = True
    ):
        self.student = student_module
        self.metric_fn = metric_fn
        self.feedback_map = feedback_map
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.rng = rng or random.Random(0)
        self.reflection_lm = reflection_lm
        self.custom_instruction_proposer = custom_instruction_proposer
        self.warn_on_score_mismatch = warn_on_score_mismatch

        if self.custom_instruction_proposer is not None:
            # We are only overriding the propose_new_texts method when a custom
            # instruction proposer is provided. Otherwise, we use the GEPA
            # default propose_new_texts.

            def custom_propose_new_texts(
                candidate: dict[str, str],
                reflective_dataset: dict[str, list[dict[str, Any]]],
                components_to_update: list[str]
            ) -> dict[str, str]:
                if self.reflection_lm is not None:
                    with dspy.context(lm=self.reflection_lm):
                        return self.custom_instruction_proposer(
                            candidate=candidate,
                            reflective_dataset=reflective_dataset,
                            components_to_update=components_to_update
                        )
                else:
                    return self.custom_instruction_proposer(
                        candidate=candidate,
                        reflective_dataset=reflective_dataset,
                        components_to_update=components_to_update
                    )

            self.propose_new_texts = custom_propose_new_texts

        # Cache predictor names/signatures
        self.named_predictors = list(self.student.named_predictors())


    def build_program(self, candidate: dict[str, str]):
        new_prog = self.student.deepcopy()
        for name, pred in new_prog.named_predictors():
            if name in candidate:
                pred.signature = pred.signature.with_instructions(candidate[name])
        return new_prog

    def evaluate(self, batch, candidate, capture_traces=False):
        program = self.build_program(candidate)

        if capture_traces:
            # bootstrap_trace_data-like flow with trace capture
            from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module

            eval_callback_metadata = {"disable_logging": True}
            trajs = bootstrap_trace_module.bootstrap_trace_data(
                program=program,
                dataset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                raise_on_error=False,
                capture_failed_parses=True,
                failure_score=self.failure_score,
                format_failure_score=self.failure_score,
                callback_metadata=eval_callback_metadata,
            )
            scores = []
            outputs = []
            for t in trajs:
                outputs.append(t["prediction"])
                if hasattr(t["prediction"], "__class__") and t.get("score") is None:
                    scores.append(self.failure_score)
                else:
                    score = t["score"]
                    if hasattr(score, "score"):
                        score = score["score"]
                    scores.append(score)
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs)
        else:
            evaluator = Evaluate(
                devset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                return_all_scores=True,
                failure_score=self.failure_score,
                provide_traceback=True,
                max_errors=len(batch) * 100
            )
            res = evaluator(program)
            outputs = [r[1] for r in res.results]
            scores = [r[2] for r in res.results]
            scores = [s["score"] if hasattr(s, "score") else s for s in scores]
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update) -> dict[str, list[ReflectiveExample]]:
        from dspy.teleprompt.bootstrap_trace import FailedPrediction
        program = self.build_program(candidate)

        ret_d: dict[str, list[ReflectiveExample]] = {}
        for pred_name in components_to_update:
            module = None
            for name, m in program.named_predictors():
                if name == pred_name:
                    module = m
                    break
            assert module is not None

            items: list[ReflectiveExample] = []
            for data in eval_batch.trajectories or []:
                trace = data["trace"]
                example = data["example"]
                prediction = data["prediction"]
                module_score = data["score"]
                if hasattr(module_score, "score"):
                    module_score = module_score["score"]

                trace_instances = [t for t in trace if t[0].signature.equals(module.signature)]
                if not self.add_format_failure_as_feedback:
                    trace_instances = [t for t in trace_instances if not isinstance(t[2], FailedPrediction)]
                if len(trace_instances) == 0:
                    continue

                selected = None
                for t in trace_instances:
                    if isinstance(t[2], FailedPrediction):
                        selected = t
                        break

                if selected is None:
                    if isinstance(prediction, FailedPrediction):
                        continue
                    selected = self.rng.choice(trace_instances)

                inputs = selected[1]
                outputs = selected[2]

                new_inputs = {}
                new_outputs = {}

                contains_history = False
                history_key_name = None
                for input_key, input_val in inputs.items():
                    if isinstance(input_val, History):
                        contains_history = True
                        assert history_key_name is None
                        history_key_name = input_key

                if contains_history:
                    s = "```json\n"
                    for i, message in enumerate(inputs[history_key_name].messages):
                        s += f"  {i}: {message}\n"
                    s += "```"
                    new_inputs["Context"] = s

                for input_key, input_val in inputs.items():
                    if contains_history and input_key == history_key_name:
                        continue

                    if isinstance(input_val, Type) and self.custom_instruction_proposer is not None:
                        # Keep original object - will be properly formatted when sent to reflection LM
                        new_inputs[input_key] = input_val
                    else:
                        new_inputs[input_key] = str(input_val)

                if isinstance(outputs, FailedPrediction):
                    s = "Couldn't parse the output as per the expected output format. The model's raw response was:\n"
                    s += "```\n"
                    s += outputs.completion_text + "\n"
                    s += "```\n\n"
                    new_outputs = s
                else:
                    for output_key, output_val in outputs.items():
                        new_outputs[output_key] = str(output_val)

                d = {"Inputs": new_inputs, "Generated Outputs": new_outputs}
                if isinstance(outputs, FailedPrediction):
                    adapter = ChatAdapter()
                    structure_instruction = ""
                    for dd in adapter.format(module.signature, [], {}):
                        structure_instruction += dd["role"] + ": " + dd["content"] + "\n"
                    d["Feedback"] = "Your output failed to parse. Follow this structure:\n" + structure_instruction
                    # d['score'] = self.failure_score
                else:
                    feedback_fn = self.feedback_map[pred_name]
                    fb = feedback_fn(
                        predictor_output=outputs,
                        predictor_inputs=inputs,
                        module_inputs=example,
                        module_outputs=prediction,
                        captured_trace=trace,
                    )
                    d["Feedback"] = fb["feedback"]
                    if fb["score"] != module_score:
                        if self.warn_on_score_mismatch:
                            logger.warning("The score returned by the metric with pred_name is different from the overall metric score. This can indicate 2 things: Either the metric is non-deterministic (e.g., LLM-as-judge, Semantic score, etc.) or the metric returned a score specific to pred_name that differs from the module level score. Currently, GEPA does not support predictor level scoring (support coming soon), and only requires a feedback text to be provided, which can be specific to the predictor or program level. GEPA will ignore the differing score returned, and instead use module level score. You can safely ignore this warning if using a semantic metric, however, if this mismatch is caused due to predictor scoring, please return module-level scores. To disable this warning, set warn_on_score_mismatch=False.")
                            self.warn_on_score_mismatch = False
                        fb["score"] = module_score

                items.append(d)

            if len(items) == 0:
                # raise Exception(f"No valid predictions found for module {module.signature}.")
                continue
            ret_d[pred_name] = items

        if len(ret_d) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d

    # TODO: The current DSPyAdapter implementation uses the GEPA default propose_new_texts.
    # We can potentially override this, to use the instruction proposal similar to MIPROv2.

    # def propose_new_texts(
    #     self,
    #     candidate: Dict[str, str],
    #     reflective_dataset: Dict[str, List[Dict[str, Any]]],
    #     components_to_update: List[str]
    # ) -> Dict[str, str]:
    #     if self.adapter.propose_new_texts is not None:
    #         return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update)

    #     from .instruction_proposal import InstructionProposalSignature
    #     new_texts: Dict[str, str] = {}
    #     for name in components_to_update:
    #         base_instruction = candidate[name]
    #         dataset_with_feedback = reflective_dataset[name]
    #         new_texts[name] = InstructionProposalSignature.run(
    #             lm=self.reflection_lm,
    #             input_dict={
    #                 "current_instruction_doc": base_instruction,
    #                 "dataset_with_feedback": dataset_with_feedback
    #             }
    #         )['new_instruction']
    #     return new_texts
