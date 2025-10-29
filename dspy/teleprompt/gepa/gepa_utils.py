import json
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
from dspy.predict.react import ReAct
from dspy.primitives import Example, Prediction
from dspy.teleprompt.bootstrap_trace import TraceData

logger = logging.getLogger(__name__)


# Constants for ReAct module optimization
REACT_MODULE_PREFIX = "react_module"


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
        optimize_react_components: bool = False,
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
        self.optimize_react_components = optimize_react_components

        def build_propose_new_texts():
            instruction_proposer = None

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
                    lm = self.reflection_lm if self.reflection_lm is not None else dspy.settings.lm
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

            # Init ReAct module proposer if tool optimization is enabled
            react_module_proposer = None
            if self.optimize_react_components:
                from .instruction_proposal import ReActModuleProposer

                react_module_proposer = ReActModuleProposer()

            def propose_component_texts(
                candidate: dict[str, str],
                reflective_dataset: dict[str, list[dict[str, Any]]],
                components_to_update: list[str],
            ) -> dict[str, str]:
                # If custom proposer provided, override everything with custom proposer
                if self.custom_instruction_proposer:
                    if self.reflection_lm is not None:
                        with dspy.context(lm=self.reflection_lm):
                            return instruction_proposer(
                                candidate=candidate,
                                reflective_dataset=reflective_dataset,
                                components_to_update=components_to_update,
                            )
                    else:
                        return instruction_proposer(
                            candidate=candidate,
                            reflective_dataset=reflective_dataset,
                            components_to_update=components_to_update,
                        )

                # Otherwise, route to appropriate proposers
                # Separate react_module components from regular instruction components
                react_module_components = [c for c in components_to_update if c.startswith(REACT_MODULE_PREFIX)]
                instruction_components = [c for c in components_to_update if not c.startswith(REACT_MODULE_PREFIX)]

                results: dict[str, str] = {}

                # Handle regular instruction components
                logger.debug(f"Routing {len(instruction_components)} instruction components to instruction_proposer")
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

                # Handle ReAct module components
                if react_module_components:
                    logger.debug(f"Routing {len(react_module_components)} react_module components to react_module_proposer")
                    if self.reflection_lm is not None:
                        with dspy.context(lm=self.reflection_lm):
                            results.update(
                                react_module_proposer(
                                    candidate=candidate,
                                    reflective_dataset=reflective_dataset,
                                    components_to_update=react_module_components,
                                )
                            )
                    else:
                        results.update(
                            react_module_proposer(
                                candidate=candidate,
                                reflective_dataset=reflective_dataset,
                                components_to_update=react_module_components,
                            )
                        )

                return results

            return propose_component_texts

        self.propose_new_texts = build_propose_new_texts()

        # Cache predictor names/signatures
        self.named_predictors = list(self.student.named_predictors())

    def build_program(self, candidate: dict[str, str]):
        new_prog = self.student.deepcopy()

        # Apply regular predictor instructions
        for name, pred in new_prog.named_predictors():
            if name in candidate:
                pred.signature = pred.signature.with_instructions(candidate[name])

        # Apply ReAct module updates (JSON configs for ReAct modules: react, extract, tools)
        if self.optimize_react_components:

            for module_path, module in new_prog.named_sub_modules():
                # Only process ReAct modules
                if not isinstance(module, ReAct):
                    continue

                # Build module key
                normalized_path = module_path.removeprefix("self.") if module_path != "self" else ""
                module_key = REACT_MODULE_PREFIX if normalized_path == "" else f"{REACT_MODULE_PREFIX}:{normalized_path}"

                # Check if this module was optimized
                if module_key not in candidate:
                    continue

                # Deserialize JSON containing optimized module configuration
                try:
                    module_config = json.loads(candidate[module_key])
                    logger.debug(f"Applying optimized module config to {module_key}")

                    # Apply react instruction
                    if "react" in module_config:
                        module.react.signature = module.react.signature.with_instructions(module_config["react"])
                        logger.debug("  Updated react instruction")

                    # Apply extract instruction
                    if "extract" in module_config:
                        module.extract.predict.signature = module.extract.predict.signature.with_instructions(module_config["extract"])
                        logger.debug("  Updated extract instruction")

                    # Apply tool descriptions
                    if "tools" in module_config:
                        for tool_name, tool_config in module_config["tools"].items():
                            tool = module.tools[tool_name]

                            # Update tool description
                            if tool_config.get("desc"):
                                tool.desc = tool_config["desc"]
                                logger.debug(f"  Updated tool '{tool_name}' description")

                            # Update tool arg descriptions
                            arg_desc = tool_config.get("arg_desc")
                            if arg_desc:
                                tool.arg_desc = tool.arg_desc or {}
                                tool.arg_desc.update(arg_desc)
                                # Propagate to tool.args
                                for arg_name, description in arg_desc.items():
                                    if arg_name in tool.args:
                                        tool.args[arg_name]["description"] = description
                                logger.debug(f"  Updated tool '{tool_name}' arg descriptions: {list(arg_desc.keys())}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON config for {module_key}: {e}")
                    raise

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

        # Debug: Log what components we're trying to update
        logger.info(f"make_reflective_dataset called with components_to_update: {components_to_update}")

        for pred_name in components_to_update:
            logger.info(f"Processing component: {pred_name}")

            # Handle ReAct module components - use extract predictor for final outputs
            if pred_name.startswith(REACT_MODULE_PREFIX):
                # Extract the target path from the key
                target_path = pred_name.removeprefix(f"{REACT_MODULE_PREFIX}:") if ":" in pred_name else ""

                # Find the ReAct module by traversing program structure (same as regular predictors)
                react_module = None
                for module_path, m in program.named_sub_modules():
                    if not isinstance(m, ReAct):
                        continue

                    # Normalize path (same pattern as build_program)
                    normalized_path = module_path.removeprefix("self.") if module_path != "self" else ""
                    if normalized_path == target_path:
                        react_module = m
                        break

                if react_module is None:
                    logger.warning(f"ReAct module not found for key: {pred_name}")
                    continue

                module = react_module.extract.predict
                logger.debug(f"  ReAct module detected: using {target_path or 'top-level'}.extract for final outputs")

            # Regular predictor - find by name
            else:
                module = None
                for name, m in program.named_predictors():
                    if name == pred_name:
                        module = m
                        break
                assert module is not None
                logger.debug(f"  Regular predictor: {pred_name}")

            items: list[ReflectiveExample] = []
            for data in eval_batch.trajectories or []:
                trace = data["trace"]
                example = data["example"]
                prediction = data["prediction"]
                module_score = data["score"]
                if hasattr(module_score, "score"):
                    module_score = module_score["score"]

                logger.debug(f"  Processing trace with {len(trace)} entries for example: {example}")
                trace_instances = [t for t in trace if t[0].signature.equals(module.signature)]
                logger.debug(f"    Found {len(trace_instances)} matching trace instances for signature: {module.signature}")
                if not self.add_format_failure_as_feedback:
                    trace_instances = [t for t in trace_instances if not isinstance(t[2], FailedPrediction)]
                    logger.debug(f"    After filtering FailedPrediction: {len(trace_instances)} instances")
                if len(trace_instances) == 0:
                    logger.debug("    Skipping example - no matching trace instances")
                    continue

                # For ReAct modules, use LAST extract invocation (has trajectory + final outputs)
                if pred_name.startswith(REACT_MODULE_PREFIX):
                    selected = trace_instances[-1]
                    logger.debug(f"  Using LAST extract call ({len(trace_instances)} total) with trajectory + final outputs")
                    if "trajectory" in selected[1]:
                        traj_preview = str(selected[1]["trajectory"])[:100]
                        logger.debug(f"  Trajectory preview: {traj_preview}...")
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
                    # Map react_module component keys to their react predictor names for feedback lookup
                    if pred_name.startswith(REACT_MODULE_PREFIX):
                        # "react_module" → "react", "react_module:salary_agent" → "salary_agent.react"
                        actual_pred_name = pred_name.split(":", 1)[1] + ".react" if ":" in pred_name else "react"
                    else:
                        actual_pred_name = pred_name

                    feedback_fn = self.feedback_map[actual_pred_name]
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

                # Log exact reflective example that reflection LM will see
                if pred_name.startswith(REACT_MODULE_PREFIX) and len(items) == 1:
                    logger.info(f"  First reflective example for {pred_name}:")
                    logger.info(f"    Inputs: {list(d['Inputs'].keys())}")
                    if "trajectory" in d["Inputs"]:
                        traj = d["Inputs"]["trajectory"]
                        logger.info(f"    Trajectory length: {len(traj)} chars")
                        logger.info(f"    Trajectory sample:\n{traj[:300]}...")
                    logger.info(f"    Outputs: {list(d['Generated Outputs'].keys()) if isinstance(d['Generated Outputs'], dict) else '<string>'}")
                    logger.info(f"    Feedback: {d['Feedback'][:100]}...")

            if len(items) == 0:
                logger.warning(f"  No valid reflective examples found for {pred_name}")
                continue

            ret_d[pred_name] = items
            logger.info(f"  Created {len(items)} reflective examples for {pred_name}")

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
