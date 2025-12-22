import json
import logging
import random
from typing import Any, Callable, Protocol, TypedDict

from gepa import EvaluationBatch, GEPAAdapter
from gepa.core.adapter import ProposalFn
from gepa.strategies.instruction_proposal import InstructionProposalSignature

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import History
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.tool import Tool
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Module, Prediction
from dspy.teleprompt.bootstrap_trace import FailedPrediction, TraceData

logger = logging.getLogger(__name__)


# Constants for module optimization
TOOL_MODULE_PREFIX = "tool_module"


class LoggerAdapter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log(self, x: str):
        self.logger.info(x)


DSPyTrace = list[tuple[Any, dict[str, Any], Prediction]]

ReflectiveExample = TypedDict(
    "ReflectiveExample",
    {
        "Inputs": dict[str, Any],
        "Generated Outputs": dict[str, Any] | str,
        "Feedback": str,
    },
)

ReflectiveExample.__doc__ = """
Structure of individual examples in the reflective dataset.

Each example contains the predictor inputs, generated outputs, and feedback from evaluation.
"""


class ScoreWithFeedback(Prediction):
    score: float
    feedback: str


class PredictorFeedbackFn(Protocol):
    def __call__(
        self,
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
        warn_on_score_mismatch: bool = True,
        enable_tool_optimization: bool = False,
        reflection_minibatch_size: int | None = None,
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
        self.enable_tool_optimization = enable_tool_optimization
        self.reflection_minibatch_size = reflection_minibatch_size

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        reflection_lm = self.reflection_lm or dspy.settings.lm
        # If custom proposer provided, override everything with custom proposer
        if self.custom_instruction_proposer:
            with dspy.context(lm=reflection_lm):
                return self.custom_instruction_proposer(
                    candidate=candidate,
                    reflective_dataset=reflective_dataset,
                    components_to_update=components_to_update,
                )

        # Otherwise, route to appropriate proposers
        # Separate into two categories: tool-using modules (ReAct) vs regular instructions
        # TODO: Add generic tool module support when DSPy trace lineage is improved
        tool_components = []
        instruction_components = []

        for c in components_to_update:
            if c.startswith(TOOL_MODULE_PREFIX):
                tool_components.append(c)
            else:
                instruction_components.append(c)

        results: dict[str, str] = {}

        with dspy.context(lm=reflection_lm):
            # Handle regular instruction components
            if instruction_components:
                for name in instruction_components:
                    base_instruction = candidate[name]
                    dataset_with_feedback = reflective_dataset[name]
                    results[name] = InstructionProposalSignature.run(
                        lm=(lambda x: reflection_lm(x)[0]),
                        input_dict={
                            "current_instruction_doc": base_instruction,
                            "dataset_with_feedback": dataset_with_feedback,
                        },
                    )["new_instruction"]

            # Handle ReAct modules
            if tool_components:
                from dspy.teleprompt.gepa.instruction_proposal import ToolProposer

                tool_proposer = ToolProposer()
                results.update(
                    tool_proposer(
                        candidate=candidate,
                        reflective_dataset=reflective_dataset,
                        components_to_update=tool_components,
                    )
                )

        return results

    def build_program(self, candidate: dict[str, str]):
        new_prog = self.student.deepcopy()

        # Start with plain string instructions from candidate
        predictor_candidates = {k: v for k, v in candidate.items() if not k.startswith(TOOL_MODULE_PREFIX)}

        tool_candidates = {}
        if self.enable_tool_optimization:
            for key, value in candidate.items():
                if not key.startswith(TOOL_MODULE_PREFIX):
                    continue

                config = json.loads(value)

                for pred_name, instruction in config.items():
                    if isinstance(instruction, str):
                        predictor_candidates[pred_name] = instruction

                tool_candidates.update(config.get("tools", {}))

        # Update predictor instructions
        for name, pred in new_prog.named_predictors():
            if name in predictor_candidates:
                pred.signature = pred.signature.with_instructions(predictor_candidates[name])

        # Update tool descriptions
        if tool_candidates:
            self._update_tool_descriptions(new_prog, tool_candidates)

        return new_prog

    def _update_tool_descriptions(self, program: Module, tool_candidates: dict[str, Any]) -> None:
        all_tools = self._collect_tools(program)

        for tool_name, tool_config in tool_candidates.items():
            if tool_name not in all_tools:
                logger.warning(
                    f"Skipping updates for tool:'{tool_name}' because it cannot be detected on the student program."
                )
                continue

            tool = all_tools[tool_name]

            # Update tool description if present.
            if tool_config.get("desc"):
                tool.desc = tool_config["desc"]

            # Update arg descriptions if present.
            args_schema = tool_config.get("args") or {}
            for arg_name, arg_schema in args_schema.items():
                if arg_schema.get("description") is not None:
                    tool.args[arg_name]["description"] = arg_schema["description"]

    def _collect_tools(self, module: Module) -> dict[str, Tool]:
        """Recursively collect all Tool instances from a module and its sub-modules."""
        all_tools = {}
        visited = set()

        def _collect_from_attribute(attr_value):
            if isinstance(attr_value, Tool):
                all_tools[attr_value.name] = attr_value
            elif isinstance(attr_value, dspy.Module):
                _traverse(attr_value)
            elif isinstance(attr_value, list | dict):
                items = attr_value if isinstance(attr_value, list) else attr_value.values()
                for item in items:
                    if isinstance(item, Tool):
                        all_tools[item.name] = item

        def _traverse(current_module):
            if id(current_module) in visited or not hasattr(current_module, "__dict__"):
                return
            visited.add(id(current_module))

            for attr_value in current_module.__dict__.values():
                _collect_from_attribute(attr_value)

        _traverse(module)
        return all_tools

    def evaluate(self, batch, candidate, capture_traces=False):
        program = self.build_program(candidate)
        callback_metadata = (
            {"metric_key": "eval_full"}
            if self.reflection_minibatch_size is None or len(batch) > self.reflection_minibatch_size
            else {"disable_logging": True}
        )

        if capture_traces:
            # bootstrap_trace_data-like flow with trace capture
            from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module

            trajs = bootstrap_trace_module.bootstrap_trace_data(
                program=program,
                dataset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                raise_on_error=False,
                capture_failed_parses=True,
                failure_score=self.failure_score,
                format_failure_score=self.failure_score,
                callback_metadata=callback_metadata,
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
                max_errors=len(batch) * 100,
                callback_metadata=callback_metadata,
            )
            res = evaluator(program)
            outputs = [r[1] for r in res.results]
            scores = [r[2] for r in res.results]
            scores = [s["score"] if hasattr(s, "score") else s for s in scores]
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None)

    def make_reflective_dataset(
        self, candidate, eval_batch, components_to_update
    ) -> dict[str, list[ReflectiveExample]]:
        program = self.build_program(candidate)

        ret_d: dict[str, list[ReflectiveExample]] = {}

        for pred_name in components_to_update:
            # Extract predictor name from component key
            if pred_name.startswith(TOOL_MODULE_PREFIX):
                target_name = pred_name.removeprefix(f"{TOOL_MODULE_PREFIX}:")
            else:
                target_name = pred_name

            # Find the predictor object
            module = None
            for name, m in program.named_predictors():
                if name == target_name:
                    module = m
                    break
            assert module is not None, f"Predictor not found: {target_name}"

            # Create reflective examples from traces
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
                    # Use actual predictor name for feedback lookup
                    feedback_fn = self.feedback_map[target_name]
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
                            logger.warning(
                                "The score returned by the metric with pred_name is different from the overall metric score. This can indicate 2 things: Either the metric is non-deterministic (e.g., LLM-as-judge, Semantic score, etc.) or the metric returned a score specific to pred_name that differs from the module level score. Currently, GEPA does not support predictor level scoring (support coming soon), and only requires a feedback text to be provided, which can be specific to the predictor or program level. GEPA will ignore the differing score returned, and instead use module level score. You can safely ignore this warning if using a semantic metric, however, if this mismatch is caused due to predictor scoring, please return module-level scores. To disable this warning, set warn_on_score_mismatch=False."
                            )
                            self.warn_on_score_mismatch = False
                        fb["score"] = module_score

                items.append(d)

            if len(items) == 0:
                logger.warning(f"  No valid reflective examples found for {pred_name}")
                continue

            ret_d[pred_name] = items

        if len(ret_d) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d

    # TODO: Generic tool module optimization - pending DSPy trace lineage improvements
    # Currently only ReAct modules are supported for tool optimization.
    # Re-enable _update_candidate_tools when DSPy provides better tool→trace lineage.
    #
    # def _update_candidate_tools(self, candidate, program, trajectories) -> None:
    #     """Extract dspy.Tool objects from traces for tool modules and update candidate["tools"]."""
    #     ...

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
