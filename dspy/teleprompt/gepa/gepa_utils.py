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
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Prediction
from dspy.teleprompt.bootstrap_trace import FailedPrediction, TraceData

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


# ---------------------------------------------------------------------------
# Code optimization for vibe-marked (dspy.Flex) submodules.
#
# A vibe-marked submodule is optimized by rewriting its *source* — the candidate
# carries a single `<submodule_path>::code` key — rather than by tuning a predictor's
# instructions. The whole module (its PREDICTORS dict AND its forward function) is one
# component: the two are coupled (forward references predictor names), so they MUST be
# rewritten together — evolving them independently produces broken, mismatched code.
# The "::code" suffix never collides with the dotted/bracketed names produced by
# named_parameters()/named_sub_modules(), so code keys and instruction keys coexist.
# ---------------------------------------------------------------------------

_CODE_KEY_SUFFIX = "::code"


def _is_code_key(key: str) -> bool:
    return key.endswith(_CODE_KEY_SUFFIX)


def make_code_key(path: str) -> str:
    return f"{path}{_CODE_KEY_SUFFIX}"


def _code_key_path(key: str) -> str:
    return key[: -len(_CODE_KEY_SUFFIX)]


def join_module_code(predictors_src: str, forward_src: str) -> str:
    """Combine the two code artifacts into one source string (PREDICTORS then forward)."""
    return f"{(predictors_src or '').strip()}\n\n{(forward_src or '').strip()}"


def split_module_code(combined: str) -> tuple[str, str]:
    """Inverse of :func:`join_module_code`: split on the top-level ``def forward``.

    Robust to the LM dropping any delimiter — everything before the first ``def forward``
    is the PREDICTORS block, the rest is the forward function.
    """
    idx = combined.find("def forward")
    if idx == -1:
        return combined.strip(), ""
    return combined[:idx].strip(), combined[idx:].strip()


def enumerate_flex_submodules(root) -> dict[str, Any]:
    """Map sub-module path -> module for every vibe-marked (code-optimizable) submodule.

    Duck-typed (no import of dspy.Flex): a submodule qualifies if it exposes the
    ``_code_optimizable`` marker and a ``_bind_code`` method. Paths use the
    ``named_sub_modules()`` naming ("self" for a top-level Flex student, "self.extract"
    for a nested one) and are stable across deepcopy, so the same key identifies the
    submodule at seed time and at build_program time.
    """
    out: dict[str, Any] = {}
    for name, sub in root.named_sub_modules():
        if getattr(sub, "_code_optimizable", False) and hasattr(sub, "_bind_code"):
            out[name] = sub
    return out


def flex_internal_predictor_ids(flex_submodules: dict[str, Any]) -> set[int]:
    """Object ids of predictors owned by vibe-marked submodules.

    Their instructions are governed by the submodule's code (rewritten wholesale), so
    they must be excluded from the instruction-optimization candidate. We key on object
    identity because named_sub_modules() and named_predictors() use different path
    formats ("self.flex" vs "flex.rlm.generate_action").
    """
    ids: set[int] = set()
    for flex in flex_submodules.values():
        for _, pred in flex.named_predictors():
            ids.add(id(pred))
    return ids


def _short_repr(value: Any, max_len: int = 800) -> str:
    s = str(value)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _render_prediction(prediction: Any) -> Any:
    if isinstance(prediction, FailedPrediction):
        return f"FailedPrediction: {prediction.completion_text[:1000]}"
    if isinstance(prediction, Prediction):
        try:
            return {k: _short_repr(v) for k, v in prediction.items()}
        except Exception:
            pass
    return _short_repr(prediction)


def _format_failures(records: list[dict[str, Any]]) -> str:
    if not records:
        return "(no failing examples available)"
    chunks: list[str] = []
    for i, rec in enumerate(records):
        chunks.append(
            f"=== Example {i} ===\n"
            f"Inputs:\n{rec.get('Inputs')!r}\n"
            f"Generated Outputs:\n{rec.get('Generated Outputs')!r}\n"
            f"Feedback:\n{rec.get('Feedback')}"
        )
    return "\n\n".join(chunks)


class CodeProposalSignature(dspy.Signature):
    """Revise the full source code of a vibe-marked dspy.Flex submodule.

    You receive the submodule's task description, the catalog of allowed primitives plus a
    good/bad-behavior knowledge base, the module's current source, and a batch of failing
    examples with feedback. Produce a revised source that fixes the observed failures and
    follows the catalog.

    The source has two coupled parts and you MUST output BOTH, consistent with each other:
      1. a module-scope ``PREDICTORS = {...}`` dict (use ``PREDICTORS = {}`` if no LM is
         needed), then
      2. a ``def forward(self, **inputs):`` that references those predictors as
         ``self.<name>`` and returns ``dspy.Prediction(<output fields>=...)``.
    Because ``forward`` calls predictors by name, never rename a predictor in one place
    without updating the other — emit the entire, internally-consistent module.
    """

    task_description: str = dspy.InputField(
        desc="The submodule's Signature: name, objective, input and output fields."
    )
    primitives_catalog: str = dspy.InputField(
        desc="Catalog of allowed primitives plus good/bad-behavior guidance to follow."
    )
    current_source: str = dspy.InputField(
        desc="The module's current full source: the PREDICTORS dict followed by the forward function."
    )
    failures: str = dspy.InputField(
        desc="A batch of failing examples and feedback. Diagnose them and revise the module to fix them."
    )
    revised_source: str = dspy.OutputField(
        desc="The full revised module source: a `PREDICTORS = {...}` dict, then a `def forward(self, **inputs):`."
    )


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
        self.custom_instruction_proposer = custom_instruction_proposer
        self.warn_on_score_mismatch = warn_on_score_mismatch
        self.reflection_minibatch_size = reflection_minibatch_size

        # Task descriptions for any vibe-marked (Flex) submodules, used when proposing
        # revised code. Keyed by the same path used in the code candidate keys.
        self._flex_task_descriptions: dict[str, str] = {}
        for path, sub in enumerate_flex_submodules(student_module).items():
            ctx = getattr(sub, "_flex_ctx", None)
            if ctx is not None and hasattr(ctx, "render_signature_spec"):
                self._flex_task_descriptions[path] = ctx.render_signature_spec()

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        reflection_lm = self.reflection_lm or dspy.settings.lm

        instr_keys = [c for c in components_to_update if not _is_code_key(c)]
        code_keys = [c for c in components_to_update if _is_code_key(c)]

        results: dict[str, str] = {}

        # --- instruction components (default GEPA behavior) ------------------
        if instr_keys:
            # A custom instruction proposer overrides only the instruction components;
            # code components always use the code proposer below.
            if self.custom_instruction_proposer:
                # Pass the FULL candidate/reflective_dataset (a custom proposer may read
                # sibling components for context); only scope which components to update.
                # With no code components this is identical to the original GEPA call.
                with dspy.context(lm=reflection_lm):
                    results.update(
                        self.custom_instruction_proposer(
                            candidate=candidate,
                            reflective_dataset=reflective_dataset,
                            components_to_update=instr_keys,
                        )
                    )
            else:
                with dspy.context(lm=reflection_lm):
                    for name in instr_keys:
                        results[name] = InstructionProposalSignature.run(
                            lm=(lambda x: self.stripped_lm_call(x)[0]),
                            input_dict={
                                "current_instruction_doc": candidate[name],
                                "dataset_with_feedback": reflective_dataset[name],
                            },
                        )["new_instruction"]

        # --- code components (vibe-marked dspy.Flex submodules) --------------
        if code_keys:
            from dspy.flex.codegen import _strip_code_fences
            from dspy.flex.primitives_doc import KNOWLEDGE_BASE, PRIMITIVES_CATALOG

            catalog = PRIMITIVES_CATALOG + "\n\n" + KNOWLEDGE_BASE
            proposer = dspy.Predict(CodeProposalSignature)
            with dspy.context(lm=reflection_lm):
                for ckey in code_keys:
                    path = _code_key_path(ckey)
                    try:
                        out = proposer(
                            task_description=self._flex_task_descriptions.get(path, path),
                            primitives_catalog=catalog,
                            current_source=candidate[ckey],
                            failures=_format_failures(reflective_dataset.get(ckey, [])),
                        )
                        results[ckey] = _strip_code_fences(out.revised_source)
                    except Exception as e:
                        logger.warning("Code proposer failed on %s (%s); keeping original.", ckey, e)
                        results[ckey] = candidate[ckey]

        return results

    def build_program(self, candidate: dict[str, str]):
        new_prog = self.student.deepcopy()

        # Rebind code for vibe-marked (Flex) submodules whose source is in the candidate.
        for path, flex in enumerate_flex_submodules(new_prog).items():
            key = make_code_key(path)
            if key in candidate:
                predictors_src, forward_src = split_module_code(candidate[key])
                flex._auto_repair = False  # a broken proposed candidate must raise, not self-repair
                flex._bind_code(predictors_src, forward_src)

        # Apply instruction updates to predictors. Code keys contain "::" so never match
        # here; predictors *inside* a Flex were excluded from the candidate (their
        # instructions are owned by the Flex's code), so they are left untouched.
        for name, pred in new_prog.named_predictors():
            if name in candidate:
                pred.signature = pred.signature.with_instructions(candidate[name])

        return new_prog

    def _code_reflective_records(self, eval_batch) -> list[dict[str, Any]]:
        """Module-level reflective records for a code component (whole-program I/O).

        Unlike instruction components (per-predictor traces), a Flex submodule's code is
        reflected on using the whole program's inputs/outputs/feedback per example.
        """
        records: list[dict[str, Any]] = []
        for data in eval_batch.trajectories or []:
            example = data["example"]
            prediction = data["prediction"]
            raw_score = data.get("score")
            if hasattr(raw_score, "score"):
                feedback = raw_score.get("feedback") or ""
                score_val = raw_score["score"]
            else:
                feedback = ""
                score_val = raw_score
            records.append(
                {
                    "Inputs": {k: _short_repr(v) for k, v in example.inputs().items()},
                    "Generated Outputs": _render_prediction(prediction),
                    "Feedback": feedback or f"This trajectory got a score of {score_val}.",
                }
            )
        return records

    def evaluate(self, batch, candidate, capture_traces=False):
        try:
            program = self.build_program(candidate)
        except Exception as e:
            # A proposed code candidate that fails to bind (broken source) must score as a
            # failure, not crash the whole optimization run. Instruction candidates always
            # build, so this only guards the vibe-marked code-optimization path.
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
            # Code components (vibe-marked Flex submodules) reflect on whole-program I/O,
            # not per-predictor traces.
            if _is_code_key(pred_name):
                recs = self._code_reflective_records(eval_batch)
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
        raw_outputs = self.reflection_lm(x)
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

