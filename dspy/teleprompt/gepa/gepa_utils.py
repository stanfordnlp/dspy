import logging
import random
from collections import defaultdict
from copy import deepcopy
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
        optimize_tool_descriptions: bool = False,
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
        self.optimize_tool_descriptions = optimize_tool_descriptions

        def build_propose_new_texts():
            instruction_proposer = None

            # Init Signature Proposer if custom proposer is provided.
            # Otherwise, use GEPA default proposer.
            if self.custom_instruction_proposer is not None:
                instruction_proposer = self.custom_instruction_proposer
            else:
                from gepa.strategies.instruction_proposal import InstructionProposalSignature

                def default_signature_proposer(
                    candidate: dict[str, str],
                    reflective_dataset: dict[str, list[dict[str, Any]]],
                    components_to_update: list[str],
                ) -> dict[str, str]:
                    lm = self.reflection_lm if self.reflection_lm is not None else dspy.settings.lm
                    sig_texts: dict[str, str] = {}
                    for name in components_to_update:
                        base_instruction = candidate[name]
                        dataset_with_feedback = reflective_dataset[name]
                        sig_texts[name] = InstructionProposalSignature.run(
                            lm=(lambda x: lm(x)[0]),
                            input_dict={
                                "current_instruction_doc": base_instruction,
                                "dataset_with_feedback": dataset_with_feedback,
                            },
                        )["new_instruction"]
                    return sig_texts

                instruction_proposer = default_signature_proposer

            # Init Tool Proposer if tool optimization is enabled.
            tool_proposer = None
            if self.optimize_tool_descriptions is not None:
                from .instruction_proposal import ToolProposer

                tool_proposer = ToolProposer()

            def propose_component_texts(
                candidate: dict[str, str],
                reflective_dataset: dict[str, list[dict[str, Any]]],
                components_to_update: list[str],
            ) -> dict[str, str]:
                tool_components = [c for c in components_to_update if c.startswith("tool:")]
                instruction_components = [c for c in components_to_update if not c.startswith("tool:")]

                results: dict[str, str] = {}

                # Handle signature components.
                if self.reflection_lm is not None:
                    with dspy.context(lm=self.reflection_lm):
                        results.update(
                            instruction_proposer(
                                candidate=candidate,
                                reflective_dataset=reflective_dataset,
                                components_to_update=instruction_components,
                            )
                        )
                else:
                    results.update(
                        instruction_proposer(
                            candidate=candidate,
                            reflective_dataset=reflective_dataset,
                            components_to_update=instruction_components,
                        )
                    )

                # Handle tool if tool proposer is provided.
                if tool_proposer is not None:
                    if self.reflection_lm is not None:
                        with dspy.context(lm=self.reflection_lm):
                            results.update(
                                tool_proposer(
                                    candidate=candidate,
                                    reflective_dataset=reflective_dataset,
                                    components_to_update=tool_components,
                                )
                            )
                    else:
                        results.update(
                            tool_proposer(
                                candidate=candidate,
                                reflective_dataset=reflective_dataset,
                                components_to_update=tool_components,
                            )
                        )

                return results

            return propose_component_texts

        self.propose_new_texts = build_propose_new_texts()

        # Cache predictor names/signatures
        self.named_predictors = list(self.student.named_predictors())

    def build_program(self, candidate: dict[str, str]):
        new_prog = self.student.deepcopy()
        for name, pred in new_prog.named_predictors():
            if name in candidate:
                pred.signature = pred.signature.with_instructions(candidate[name])

        if self.optimize_tool_descriptions:
            for _, module in new_prog.named_sub_modules():
                if hasattr(module, "tools"):
                    for tool_name, tool in module.tools.items():
                        tool_key = f"tool:{tool_name}"
                        if tool_key in candidate:
                            tool.desc = candidate[tool_key]

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
                max_errors=len(batch) * 100,
            )
            res = evaluator(program)
            outputs = [r[1] for r in res.results]
            scores = [r[2] for r in res.results]
            scores = [s["score"] if hasattr(s, "score") else s for s in scores]
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None)

    def make_reflective_dataset(
        self, candidate, eval_batch, components_to_update
    ) -> dict[str, list[ReflectiveExample]]:
        from dspy.teleprompt.bootstrap_trace import FailedPrediction

        program = self.build_program(candidate)

        ret_d: dict[str, list[ReflectiveExample]] = {}

        # First pass: Process all non-tool components (predictors)
        for pred_name in components_to_update:
            if pred_name.startswith("tool:"):
                continue  # Skip tools in first pass (tools are processed in the second pass)

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
                            self.warn_on_score_mismatch = False
                        fb["score"] = module_score

                items.append(d)

            if len(items) == 0:
                # raise Exception(f"No valid predictions found for module {module.signature}.")
                continue
            ret_d[pred_name] = items

        # Add tool examples to the reflective dataset
        tool_examples = defaultdict(list)

        if self.optimize_tool_descriptions:
            # Design Decision: Full ReAct Trajectory Sharing for Tools
            #
            # Each tool receives the COMPLETE ReAct trajectory (all thoughts, actions, observations)
            # rather than only the segments where that tool was used. This trades token efficiency
            # for richer optimization context.
            #
            # Rationale:
            # 1. Tools are interdependent: search results inform calculator usage, API responses
            #    guide follow-up queries. Full trajectory shows these dependencies.
            # 2. Reflection LM needs context to understand tool SELECTION patterns:
            #    - Why did the agent choose this tool over alternatives?
            #    - When in the reasoning process is this tool most useful?
            #    - What prior information typically triggers this tool's usage?
            # 3. Goal is descriptions that guide "when to use" not just "what it does"
            #
            # Trade-offs:
            # - Cost: N tools = N copies of same trajectory (5 tools = 5x duplication)
            # - Benefit: Descriptions capture tool's role in multi-step workflows
            #   Example: "Use after search when numerical analysis is needed" vs "Does math"
            #
            for module_path, sub_module in program.named_sub_modules():
                # Walk each sub-module to locate its tools and remember the predictor scope
                # so we can share those reflections with the tool descriptions below
                tools = getattr(sub_module, "tools", None)
                if not tools:
                    continue

                prefix = module_path.removeprefix("self.") if module_path != "self" else ""

                tool_entries = list(tools.items())

                for child_name, _ in sub_module.named_predictors():
                    predictor_key = child_name if not prefix else f"{prefix}.{child_name}"
                    reflections = ret_d.get(predictor_key)
                    if not reflections:
                        continue

                    # Share the FULL ReAct trajectory with each tool
                    for tool_name, _ in tool_entries:
                        tool_key = f"tool:{tool_name}"
                        for item in reflections:
                            annotated = deepcopy(item)
                            annotated["Feedback"] = f"[Tool '{tool_name}' from '{predictor_key}'] {item['Feedback']}"
                            tool_examples[tool_key].append(annotated)

        # Merge tool examples into main dataset (shared tools get examples from all predictors)
        ret_d.update(tool_examples)

        if len(ret_d) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d

    # Future Work: Joint Tool Optimization with ReAct for Token Efficiency
    # ===========================================================
    # Current approach duplicates the same trajectory N times for N tools in a ReAct module.
    # For multi-tool agents, we could optimize all tools simultaneously to reduce token usage.
    #
    # Assumption:
    # - ReAct module is the only module that uses the tools
    # - When optimizing tool descriptions of ReAct, reflection LM would capture general pattern of tools and ReAct's decision making process
    # - It's probably better to holistically optimize all tools and ReAct together

    # Proposed Architecture:
    # 1. During reflective dataset construction, group tools by their parent ReAct module:
    #    - Walk program.named_sub_modules() to find ReAct predictors
    #    - Extract tools from each ReAct module via getattr(module, "tools", None)
    #    - Build mapping: {module_path: [tool_name1, tool_name2, ...]}
    #    - Detect when a module has multiple tools
    #
    # 2. For multi-tool ReAct modules, choose architectural approach:
    #
    #    Option A: Separate tool-specific proposer signature
    #    - Create custom signature extending GenerateImprovedToolDescriptionFromFeedback
    #    - Use dspy.Signature.append_field() to add one output field per tool
    #    - Example: For 3 tools, add fields "improved_search_desc", "improved_calc_desc", "improved_api_desc"
    #    - Pro: Clean separation between instruction and tool optimization
    #    - Con: Separate LM call from ReAct instruction optimization
    #
    #    Option B: Extend ReAct instruction proposer directly
    #    - Append tool description fields to existing ReAct instruction proposer
    #    - Update proposer instructions/docstring to include tool optimization guidance
    #    - Use dspy.Signature's helper functions to add output fields for each tool
    #    - Aggregate all tools' input/output fields expected to be updated from that ReAct module
    #    - Pro: Single LM call optimizes ReAct instructions AND tool descriptions together
    #    - Pro: Reflection LM sees relationship between instructions and tools holistically
    #    - Con: More complex signature modification, harder to maintain separation of concerns
    #
    # 3. Pass the ReAct trajectory ONCE to generate all tool descriptions and ReAct instruction simultaneously:
    #    - Single LM call with multi-field output instead of N separate calls
    #    - Proposer prompt instructs LM to consider tool interactions
    #
    # 4. Parse the multi-field output and update each tool's description:
    #    - Extract each field from the prediction
    #    - Map back to tool names using the grouping from step 1
    #    - Handle parsing errors with fallback to current one-at-a-time approach
    #
    # Benefits:
    # - Eliminates trajectory duplication: 1x token cost instead of Nx
    # - Reflection LM sees all tools holistically, can coordinate descriptions
    # - Tool descriptions can complement each other ("use search before calculator")
    # - Scales better for agents with 10+ tools
    #
    # Challenges:
    # - Signature modification at runtime requires careful field naming/parsing
    # - More output fields → higher chance of LM parsing errors
    # - Need robust fallback when multi-field output fails
    # - Requires refactoring GEPA's "one component at a time" architecture
    # - Tool proposer prompt becomes more complex with multiple tools
    #
    # Implementation Notes:
    # - Start with simple case: all tools from one ReAct module
    # - Add retry logic for malformed multi-field outputs
    # - Consider hybrid approach: joint optimization for <5 tools, separate for more
    # - May need different proposer prompt template for joint vs. individual optimization

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
