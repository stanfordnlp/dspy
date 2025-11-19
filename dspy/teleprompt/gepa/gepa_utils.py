import json
import logging
import random
from collections.abc import Iterable
from typing import Any, Callable, Protocol, TypedDict

from gepa import EvaluationBatch, GEPAAdapter
from gepa.core.adapter import ProposalFn

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import History
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.tool import Tool
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Prediction
from dspy.teleprompt.bootstrap_trace import FailedPrediction, TraceData

logger = logging.getLogger(__name__)


# Constants for module optimization
REACT_MODULE_PREFIX = "react_module"
TOOL_MODULE_PREFIX = "tool_module"


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

    Inputs: dict[str, Any]  # Predictor inputs (may include str, dspy.Image, etc.)
    Generated_Outputs: dict[str, Any] | str  # Success: dict with output fields, Failure: error message string
    Feedback: str  # Always a string - from metric function or parsing error message


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

        self.propose_new_texts = self._build_propose_new_texts()
        self.reflection_minibatch_size = reflection_minibatch_size

    def _build_propose_new_texts(self):
        """Build proposal function that routes components to appropriate proposers."""
        # Init instruction proposer (custom or default)
        if self.custom_instruction_proposer is not None:
            instruction_proposer = self.custom_instruction_proposer
        else:
            from gepa.strategies.instruction_proposal import InstructionProposalSignature

            def default_instruction_proposer(
                candidate: dict[str, str],
                reflective_dataset: dict[str, list[dict[str, Any]]],
                components_to_update: list[str],
            ) -> dict[str, str]:
                lm = self.reflection_lm or dspy.settings.lm
                updated_components: dict[str, str] = {}
                for name in components_to_update:
                    base_instruction = candidate[name]
                    dataset_with_feedback = reflective_dataset[name]
                    updated_components[name] = InstructionProposalSignature.run(
                        lm=(lambda x: lm(x)[0]),
                        input_dict={
                            "current_instruction_doc": base_instruction,
                            "dataset_with_feedback": dataset_with_feedback,
                        },
                    )["new_instruction"]
                return updated_components

            instruction_proposer = default_instruction_proposer

        # Init tool module proposer if tool optimization is enabled
        tool_module_proposer = None
        if self.enable_tool_optimization:
            from .instruction_proposal import ToolModuleProposer

            tool_module_proposer = ToolModuleProposer()

        def propose_component_texts(
            candidate: dict[str, str],
            reflective_dataset: dict[str, list[dict[str, Any]]],
            components_to_update: list[str],
        ) -> dict[str, str]:
            # If custom proposer provided, override everything with custom proposer
            if self.custom_instruction_proposer:
                with dspy.context(lm=self.reflection_lm or dspy.settings.lm):
                    return instruction_proposer(
                        candidate=candidate,
                        reflective_dataset=reflective_dataset,
                        components_to_update=components_to_update,
                    )

            # Otherwise, route to appropriate proposers
            # Separate into two categories: components with tools vs regular instructions
            tool_module_components = []
            instruction_components = []

            for c in components_to_update:
                if c.startswith((REACT_MODULE_PREFIX, TOOL_MODULE_PREFIX)):
                    tool_module_components.append(c)
                else:
                    instruction_components.append(c)

            results: dict[str, str] = {}

            with dspy.context(lm=self.reflection_lm or dspy.settings.lm):
                # Handle regular instruction components
                if instruction_components:
                    results.update(
                        instruction_proposer(
                            candidate=candidate,
                            reflective_dataset=reflective_dataset,
                            components_to_update=instruction_components,
                        )
                    )

                # Handle components with tools (ReAct and Tool modules)
                if tool_module_components:
                    results.update(
                        tool_module_proposer(
                            candidate=candidate,
                            reflective_dataset=reflective_dataset,
                            components_to_update=tool_module_components,
                        )
                    )

            return results

        return propose_component_texts

    def build_program(self, candidate: dict[str, str]):
        new_prog = self.student.deepcopy()

        # Start with plain string instructions from candidate
        improved_predictors = {
            k: v for k, v in candidate.items()
            if not k.startswith((REACT_MODULE_PREFIX, TOOL_MODULE_PREFIX))
        }

        improved_tools = {}
        if self.enable_tool_optimization:
            for key, value in candidate.items():
                if not key.startswith((REACT_MODULE_PREFIX, TOOL_MODULE_PREFIX)):
                    continue

                config = json.loads(value)

                for pred_name, instruction in config.items():
                    if isinstance(instruction, str):
                        improved_predictors[pred_name] = instruction

                improved_tools.update(config.get("tools", {}))

        # Update predictor instructions
        for name, pred in new_prog.named_predictors():
            if name in improved_predictors:
                pred.signature = pred.signature.with_instructions(improved_predictors[name])

        # Update tool descriptions
        if improved_tools:
            def collect_tools(obj):
                all_tools = {}
                visited = set()

                def traverse(o):
                    if id(o) in visited or not hasattr(o, "__dict__"):
                        return
                    visited.add(id(o))

                    for attr_val in o.__dict__.values():
                        if isinstance(attr_val, Tool):
                            all_tools[attr_val.name] = attr_val
                        elif isinstance(attr_val, list):
                            for item in attr_val:
                                if isinstance(item, Tool):
                                    all_tools[item.name] = item
                        elif isinstance(attr_val, dict):
                            for item in attr_val.values():
                                if isinstance(item, Tool):
                                    all_tools[item.name] = item
                        elif isinstance(attr_val, dspy.Module):
                            traverse(attr_val)

                traverse(obj)
                return all_tools

            all_tools = collect_tools(new_prog)

            for tool_name, tool_config in improved_tools.items():
                if tool_name not in all_tools:
                    logger.warning(f"Skipping updates for tool:'{tool_name}' because it cannot be detected on the student program.")
                    continue

                tool = all_tools[tool_name]

                # Update tool description if present.
                if tool_config.get("desc") is not None:
                    tool.desc = tool_config["desc"]

                # Update arg descriptions if present.
                args_schema = tool_config.get("args") or {}
                for arg_name, arg_schema in args_schema.items():
                    if arg_schema.get("description") is not None:
                        tool.args[arg_name]["description"] = arg_schema["description"]

        return new_prog

    def evaluate(self, batch, candidate, capture_traces=False):
        program = self.build_program(candidate)
        callback_metadata = {"metric_key": "eval_full"} if self.reflection_minibatch_size is None or len(batch) > self.reflection_minibatch_size else {"disable_logging": True}

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

            if self.enable_tool_optimization:
                self._update_candidate_tools(candidate, program, trajs)

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
            if pred_name.startswith(REACT_MODULE_PREFIX):
                target_name = pred_name.removeprefix(f"{REACT_MODULE_PREFIX}:")

            elif pred_name.startswith(TOOL_MODULE_PREFIX):
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

                # TODO: Workaround for ReAct's multiple predictor calls with partial trajectories.
                # Using last trace ensures full aggregated trajectory (same as extract predictor).
                # After PR #8999 merges (https://github.com/stanfordnlp/dspy/pull/8999), test if we can
                # remove this and use extract predictor trace directly like other modules traces.
                if pred_name.startswith(REACT_MODULE_PREFIX):
                    selected = trace_instances[-1]

                else:
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
                            logger.warning("The score returned by the metric with pred_name is different from the overall metric score. This can indicate 2 things: Either the metric is non-deterministic (e.g., LLM-as-judge, Semantic score, etc.) or the metric returned a score specific to pred_name that differs from the module level score. Currently, GEPA does not support predictor level scoring (support coming soon), and only requires a feedback text to be provided, which can be specific to the predictor or program level. GEPA will ignore the differing score returned, and instead use module level score. You can safely ignore this warning if using a semantic metric, however, if this mismatch is caused due to predictor scoring, please return module-level scores. To disable this warning, set warn_on_score_mismatch=False.")
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

    def _update_candidate_tools(self, candidate, program, trajectories) -> None:
        """Extract dspy.Tool objects from traces for tool modules and update candidate["tools"]."""

        tools_by_predictor: dict[str, dict[str, Tool]] = {}

        def extract_tools_from_value(value: Any, tools_dict: dict[str, Tool]) -> None:
            """Recursively collect dspy.Tool instances from arbitrary input structures.
            Traverses nested containers (lists, dicts, etc.) to find all dspy.Tool objects passed as input arguments, populating the provided tools_dict.
            """

            if isinstance(value, Tool):
                tools_dict[value.name] = value
                return

            # For mappings, recurse over values only.
            if isinstance(value, dict):
                for v in value.values():
                    extract_tools_from_value(v, tools_dict)
                return

            # For other iterables (including list, tuple, set, dict_values, etc.), recurse over elements.
            # Skip strings/bytes to avoid treating them as iterables of characters.
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                for item in value:
                    extract_tools_from_value(item, tools_dict)

        # We iterate over all candidate keys to find tool modules
        for component_key in candidate.keys():
            if not component_key.startswith(TOOL_MODULE_PREFIX):
                continue

            target_name = component_key.removeprefix(f"{TOOL_MODULE_PREFIX}:")
            tools_by_predictor[component_key] = {}

            # Find the predictor object
            module = None
            for name, m in program.named_predictors():
                if name == target_name:
                    module = m
                    break
            if module is None:
                logger.warning(f"Predictor not found for tool module {target_name}")
                continue

            for data in trajectories or []:
                trace = data["trace"]

                trace_instances = [t for t in trace if t[0].signature.equals(module.signature)]
                if not self.add_format_failure_as_feedback:
                    trace_instances = [t for t in trace_instances if not isinstance(t[2], FailedPrediction)]

                if len(trace_instances) == 0:
                    continue

                for t in trace_instances:
                    trace_inputs = t[1]

                    for input_value in trace_inputs.values():
                        # Recursively collect dspy.Tool objects from input values
                        extract_tools_from_value(input_value, tools_by_predictor[component_key])

        # Update candidate["tools"] with tools found in traces
        for component_key, tools_dict in tools_by_predictor.items():
            if not tools_dict:
                logger.debug(f"No tools extracted from traces for {component_key} (eval_batch.trajectories may be missing tool calls)")
                continue

            config = json.loads(candidate[component_key])

            # Initialize tools dict from existing config if present, otherwise empty
            tools_config = config.get("tools", {})

            # Update with tools found in traces (this updates existing entries or adds new ones)
            for tool_name, tool in tools_dict.items():
                tools_config[tool_name] = {
                    "desc": tool.desc,
                    "args": tool.args,
                }
            config["tools"] = tools_config
            candidate[component_key] = json.dumps(config, indent=2)

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
