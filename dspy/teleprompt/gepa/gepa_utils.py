import logging
import random
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Callable, Protocol, TypedDict

from gepa import EvaluationBatch, GEPAAdapter
from gepa.core.adapter import ProposalFn
from gepa.strategies.instruction_proposal import InstructionProposalSignature

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import History
from dspy.adapters.types.base_type import Type
from dspy.dsp.utils.settings import thread_local_overrides
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Prediction
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.teleprompt.bootstrap_trace import FailedPrediction, TraceData
from dspy.teleprompt.gepa.gepa_flex_utils import (
    code_reflective_records,
    enumerate_flex_submodules,
    evaluate_with_trace,
    flex_task_context,
    propose_code,
    rebind_flex_code,
)
from dspy.utils.callback_context import _bind_active_call_id

logger = logging.getLogger(__name__)


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


def _stripped_lm_outputs(lm, x: str) -> list[str]:
    raw_outputs = lm(x)
    outputs = []
    for raw_output in raw_outputs:
        if type(raw_output) == str:
            outputs.append(raw_output)
        elif type(raw_output) == dict:
            if "text" not in raw_output:
                raise KeyError("Missing 'text' field in the output from the base LM!")
            outputs.append(raw_output["text"])
        else:
            raise TypeError("Unexpected output type from the base LM! Expected str or dict")
    return outputs


class TrackedReflectionLM:
    """Reflection callable handed to `gepa.optimize` that exposes real cost totals.

    gepa's `max_reflection_cost` stopper reads `total_cost` from the reflection LM;
    plain callables get wrapped in a gepa `TrackingLM` that always reports 0.0, so
    the stopper would never fire. Totals accumulate from the underlying
    `dspy.LM.history` — entries are tallied once (keyed by their `uuid`) on every
    call through this wrapper or `DspyAdapter.stripped_lm_call`, and on every
    property read — so spend survives history eviction at
    `settings.max_history_size`, and reflection calls made by the instruction and
    code proposers that never go through this callable are still counted. Requires
    LM history to be enabled (it is unless `dspy.settings.disable_history` is set);
    entries without a `uuid` (real dspy history entries always have one) are
    ignored, as they cannot be safely de-duplicated.

    Totals are scoped to this instance's lifetime: entries already in `lm.history`
    at construction are marked as counted without contributing, so reusing a
    `reflection_lm` across multiple `compile()` calls doesn't carry over spend from
    a previous run (`DspyAdapter.__init__` constructs a fresh `TrackedReflectionLM`
    per compile). Two residual caveats: sharing the reflection LM instance as both
    the task LM and the reflection LM counts task spend toward the reflection
    budget, and a custom proposer that makes more direct LM calls than
    `settings.max_history_size` between two syncs can have the excess evicted
    before it is tallied.
    """

    def __init__(self, lm):
        self.lm = lm
        self._counted_uuids: set[str] = set()
        self._total_cost = 0.0
        self._total_tokens_in = 0
        self._total_tokens_out = 0
        self._lock = threading.Lock()
        with self._lock:
            for entry in self._entries():
                uid = entry.get("uuid")
                if uid is not None:
                    self._counted_uuids.add(uid)

    def __call__(self, x: str) -> str:
        outputs = _stripped_lm_outputs(self.lm, x)
        self._sync()
        return outputs[0]

    def _entries(self) -> list:
        return [entry for entry in self.lm.history if isinstance(entry, Mapping)]

    def _sync(self) -> None:
        """Tally history entries not yet counted, then prune uuids evicted from the window.

        Evicted uuids can never reappear (uuid4), so the set stays bounded by
        `settings.max_history_size` without ever double-counting.
        """
        with self._lock:
            window_uuids = set()
            for entry in self._entries():
                uid = entry.get("uuid")
                if uid is None:
                    continue
                window_uuids.add(uid)
                if uid in self._counted_uuids:
                    continue
                self._counted_uuids.add(uid)
                self._total_cost += entry.get("cost") or 0.0
                usage = entry.get("usage") or {}
                self._total_tokens_in += usage.get("prompt_tokens") or 0
                self._total_tokens_out += usage.get("completion_tokens") or 0
            self._counted_uuids &= window_uuids

    @property
    def total_cost(self) -> float:
        self._sync()
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        self._sync()
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        self._sync()
        return self._total_tokens_out


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
        self.tracked_reflection_lm = TrackedReflectionLM(reflection_lm) if reflection_lm is not None else None
        self.custom_instruction_proposer = custom_instruction_proposer
        self.warn_on_score_mismatch = warn_on_score_mismatch
        self.reflection_minibatch_size = reflection_minibatch_size
        self._warned_custom_proposer_skips_code = False

        # dspy.Flex code components are keyed by the submodule's parameter path.
        self._flex_paths = set(enumerate_flex_submodules(student_module))
        # Task description + available context shown to the dspy.Flex code proposer, per submodule.
        self._flex_task_descriptions, self._flex_context_blurbs = flex_task_context(student_module)

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        reflection_lm = self.reflection_lm or dspy.settings.lm

        # dspy.Flex code components are rewritten by the code proposer.
        results: dict[str, str] = {}
        code_keys = [c for c in components_to_update if c in self._flex_paths]
        if code_keys:
            results.update(
                propose_code(
                    code_keys, candidate, reflective_dataset,
                    self._flex_task_descriptions, self._flex_context_blurbs, reflection_lm,
                )
            )
        components_to_update = [c for c in components_to_update if c not in self._flex_paths]
        if not components_to_update:
            return results

        # A custom proposer overrides the default *instruction* proposer only.
        if self.custom_instruction_proposer:
            if code_keys and not self._warned_custom_proposer_skips_code:
                logger.warning(
                    "A custom instruction_proposer is set, but %d dspy.Flex code component(s) are "
                    "being optimized. The custom proposer handles instruction components only; the "
                    "built-in code proposer rewrites the flex source.",
                    len(code_keys),
                )
                self._warned_custom_proposer_skips_code = True
            with dspy.context(lm=reflection_lm):
                results.update(
                    self.custom_instruction_proposer(
                        candidate=candidate,
                        reflective_dataset=reflective_dataset,
                        components_to_update=components_to_update,
                    )
                )
                return results

        with dspy.context(lm=reflection_lm):
            for name in components_to_update:
                base_instruction = candidate[name]
                dataset_with_feedback = reflective_dataset[name]
                results[name] = InstructionProposalSignature.run(
                    lm=(lambda x: self.stripped_lm_call(x)[0]),
                    input_dict={
                        "current_instruction_doc": base_instruction,
                        "dataset_with_feedback": dataset_with_feedback,
                    },
                )["new_instruction"]

        return results

    def build_program(self, candidate: dict[str, str]):
        new_prog = self.student.deepcopy()

        # Rebind code for any dspy.Flex submodules in the candidate (see gepa_flex_utils). A
        # candidate that doesn't parse raises here (evaluate() catches it and scores the batch as
        # a failure); runtime breakage surfaces per example when the code first runs at forward.
        rebind_flex_code(new_prog, candidate)

        # Apply instruction updates. Code components are keyed by a Flex's parameter path, which
        # never names a predictor, and predictors inside a Flex were excluded from the candidate,
        # so both are left untouched.
        for name, pred in new_prog.named_predictors():
            if name in candidate:
                pred.signature = pred.signature.with_instructions(candidate[name])

        return new_prog

    def evaluate(self, batch, candidate, capture_traces=False):
        try:
            program = self.build_program(candidate)
        except (SyntaxError, CodeInterpreterError) as e:
            # A code candidate that fails to bind scores as a failure rather than
            # crashing the run. Only source-level failures are caught. Anything else
            # (e.g., an LM provider or rate-limit error) is not the candidate's fault and must
            # propagate.
            logger.warning("Candidate failed to build (%s); scoring the batch as failures.", e)
            return EvaluationBatch(
                outputs=[None] * len(batch),
                scores=[self.failure_score] * len(batch),
                trajectories=[] if capture_traces else None,
            )
        callback_metadata = (
            {"metric_key": "eval_full"}
            if self.reflection_minibatch_size is None or len(batch) > self.reflection_minibatch_size
            else {"disable_logging": True}
        )

        # When a dspy.Flex submodule is present, capture the execution trace at scoring time so a
        # metric that declares a `program_trace` parameter can score against it (e.g. an LM-call
        # penalty).
        if self._flex_paths:
            return evaluate_with_trace(
                program,
                batch,
                metric_fn=self.metric_fn,
                num_threads=self.num_threads,
                failure_score=self.failure_score,
                callback_metadata=callback_metadata,
                capture_traces=capture_traces,
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

    def batch_evaluate(self, items):
        """Evaluate multiple (candidate, batch) pairs concurrently.

        gepa's multi-proposal iterations (`sampling_strategy`) evaluate several
        candidates per step; its fallback runs them one at a time, and each
        `evaluate` call only parallelizes across the (small) minibatch. Candidates
        are independent, so run them on worker threads. dspy settings overrides and
        the active callback call ID are thread-local and must be copied into each
        worker, or callbacks emitted there would start new root traces.
        """
        if len(items) <= 1:
            return [self.evaluate(batch, candidate, capture_traces=True) for candidate, batch in items]

        parent_overrides = thread_local_overrides.get().copy()

        def _evaluate_pair(pair):
            candidate, batch = pair
            new_overrides = {**thread_local_overrides.get(), **parent_overrides}
            if new_overrides.get("usage_tracker"):
                # Mirror dspy's ParallelExecutor: deep-copy the usage tracker per worker so each
                # thread tracks its own usage instead of contending over one shared tracker.
                new_overrides["usage_tracker"] = deepcopy(new_overrides["usage_tracker"])
            token = thread_local_overrides.set(new_overrides)
            try:
                return self.evaluate(batch, candidate, capture_traces=True)
            finally:
                thread_local_overrides.reset(token)

        with ThreadPoolExecutor(max_workers=min(len(items), 8)) as executor:
            return list(executor.map(_bind_active_call_id(_evaluate_pair), items))

    def get_adapter_state(self) -> dict[str, Any]:
        """Snapshot adapter state into gepa's checkpoint (gepa persists it via pickle)."""
        return {"rng_state": self.rng.getstate()}

    def set_adapter_state(self, state: dict[str, Any]) -> None:
        """Restore adapter state on checkpoint resume."""
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)

    def make_reflective_dataset(
        self, candidate, eval_batch, components_to_update
    ) -> dict[str, list[ReflectiveExample]]:
        program = self.build_program(candidate)

        ret_d: dict[str, list[ReflectiveExample]] = {}

        for pred_name in components_to_update:
            # Code components (dspy.Flex submodules) reflect on whole-program I/O, not per-predictor
            # traces.
            if pred_name in self._flex_paths:
                recs = code_reflective_records(eval_batch)
                if recs:
                    ret_d[pred_name] = recs
                continue

            # Find the predictor object
            module = None
            for name, m in program.named_predictors():
                if name == pred_name:
                    module = m
                    break
            assert module is not None, f"Predictor not found: {pred_name}"

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

    # Always return strings from the LM outputs
    # Even when it returns a dict with e.g., "text" and "reasoning" fields
    def stripped_lm_call(self, x: str) -> list[str]:
        outputs = _stripped_lm_outputs(self.reflection_lm, x)
        # Tally the call into the reflection budget while its history entry is
        # guaranteed to still be inside the (possibly small) retention window.
        if self.tracked_reflection_lm is not None:
            self.tracked_reflection_lm._sync()
        return outputs

