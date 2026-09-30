"""Rewrite a Flex's code into a decomposition of decisions, calibrating each rewrite before comparing it.

A generative proposer, run as a `dspy.RLM`, reads the training examples with the current program's
calibrated outputs, metric scores, and the decision evidence behind every predictor call, and
writes new source for the Flex. It can run a draft on chosen training examples to see the
probabilities its questions get before submitting. Each proposal is then calibrated like the
original program and kept only when it scores higher on the selection set, or the same with fewer
predictor calls. Proposals that fail to run are recorded, with their error, for the next round.
"""

import json
import logging
from typing import Any, Callable

import dspy
from dspy.adapters.decision import record_evidence
from dspy.predict.flex import Flex
from dspy.predict.flex.bridge import FLEX_ORIGIN, _encode_decisions, _jsonable
from dspy.predict.flex.ctx import DECISION_NOTE, _strip_code_fences
from dspy.predict.flex.primitives_doc import PRIMITIVES_CATALOG
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.utils.exceptions import LMError
from dspy.utils.parallelizer import ParallelExecutor

logger = logging.getLogger(__name__)

MAX_TRIAL_EXAMPLES = 25  # training examples one `run_code` call may run

SYSTEM_ONE_NOTE = """\
Every predictor in this module runs on a System One model (for example TypeSafe's Jev). It does not
generate text: it answers declared decision questions with probabilities. So every predictor output must
be a decision type (`Noul[...]` or `bool`, `Score[...]`, `Choice[...]`), and each needs a question, set in
`__init__` as `self.<name>.fields["<output>"] = {"instructions": "...", "criteria": ...}` (criteria is
optional: for a Noul a {"true": ..., "false": ...} description of each outcome). The model reads the
predictor's signature instructions, its input fields, and their values as one state, and answers every
output of that predictor from it in a single request.

Cost: a request is billed per input token (state plus questions); output is free. All outputs of ONE
predictor share one request, so asking five narrow questions about the same inputs in one predictor costs
about the same as asking one. Separate predictors over the same inputs pay for the state again. Pass a
predictor only the inputs its questions need, and skip a call entirely when plain Python already knows the
answer. Aim for no more requests per example than the current code."""

GENERATIVE_NOTE = """\
Predictors run on a generative LM. Decision outputs (`Noul[...]`, `bool`, `Score[...]`, `Choice[...]`)
are decided from probabilities the LM states for each option, and need a question: an output description or
`self.<name>.fields["<output>"] = {"instructions": "..."}`. Every predictor call is a separately billed
request; prefer fewer calls, and do in plain Python whatever does not need the LM."""


class DecomposeSignature(dspy.Signature):
    """Rewrite a dspy.Flex module so the calibrated program scores higher on the metric, at no higher cost.

    `examples` holds every training example: its inputs, the expected outputs, what the current code
    returned after calibration, the metric score, and the evidence of every predictor call (for each
    decision output, the probability or distribution the model gave, and the value that was decided).
    `attempts` lists the codes tried so far with their scores, requests per example, fitted parameters,
    and errors; the first is the current best.

    Work like an analyst. Find the examples the current code gets wrong and why: which questions have
    probabilities that do not separate the classes, which cases a broad question conflates. Then decompose
    the judgment into narrow, atomic decisions whose answers plain Python combines into the outputs, for
    example a Noul per distinct reason an email needs a reply, gated by a Noul that it is automated or
    fraudulent. Write each question and its criteria from what the data shows, never from single examples,
    and never hardcode example inputs or outputs.

    Every threshold, Score cut, and Choice weight is calibrated against the metric after you submit, so a
    question is good when its probability RANKS examples well, even if its 0.5 cut is wrong. Combine decided
    values (`bool(out.x)`, `out.y.level`, `out.z.value`) in code; do not threshold probabilities yourself.

    Use `run_code(module_src, indexes)` to run a draft on up to 25 training examples before submitting: it
    returns each example's outputs, score, errors, and the probabilities your questions got at the default
    thresholds. Check that the code runs and that the new questions separate the examples you target.
    Submit the full source of one dspy.Module subclass that follows the catalog.
    """

    task: str = dspy.InputField(desc="The Flex's signature, and the tools and notes available to its code.")
    backend: str = dspy.InputField(desc="What the predictors run on, what they can answer, and what a call costs.")
    catalog: str = dspy.InputField(desc="The primitives the code may use and the conventions it must follow.")
    attempts: list[dict] = dspy.InputField(desc="Codes tried so far, best first, with scores and errors.")
    examples: list[dict] = dspy.InputField(desc="Training examples with the current code's calibrated results.")
    module_src: str = dspy.OutputField(desc="The full revised source: one dspy.Module subclass.")


def _names(program) -> dict[int, str]:
    """Plain predictors' names by id, for attributing evidence outside any Flex."""
    return {id(p): name for name, p in program.named_predictors()}


def _caller_name(caller, names: dict[int, str]) -> str:
    origin = getattr(caller, FLEX_ORIGIN, None)
    return origin[1] if origin is not None else names.get(id(caller), type(caller).__name__)


def trace(program, examples: list, metric: Callable, num_threads: int | None = None) -> list[dict[str, Any]]:
    """Run the program on each example and record its outputs, score, errors, and decision evidence.

    A failing example is recorded with its error and a score of 0; LM infrastructure errors propagate.
    """
    from dspy.teleprompt.reanchor.calibrate import metric_value

    names = _names(program)

    def one(item):
        index, example = item
        record = {"index": index, "inputs": example.inputs().toDict(), "expected": example.labels().toDict()}
        with dspy.context(trace=[]), record_evidence() as log:
            try:
                pred = program(**example.inputs())
                record["outputs"] = {k: _jsonable(_encode_decisions(v)) for k, v in pred.items()}
                record["score"] = metric_value(metric, example, pred)
            except LMError:
                raise
            except Exception as e:
                record.update(outputs=None, score=0.0, error=f"{type(e).__name__}: {e}")
        calls = []
        for caller, field, evidence in log:
            calls.append({"predictor": _caller_name(caller, names), "output": field, "evidence": evidence})
        record["decisions"] = calls
        return record

    executor = ParallelExecutor(num_threads=num_threads, max_errors=len(examples) + 1, disable_progress_bar=True)
    records = executor.execute(one, list(enumerate(examples)))
    for record in records:
        if isinstance(record, BaseException):  # The executor returns what `one` raised: LM errors only.
            raise record
    return records


def backend_note(flex: Flex) -> str:
    lm = flex.lm or dspy.settings.lm
    return SYSTEM_ONE_NOTE if getattr(lm, "supports_decision_requests", False) is True else GENERATIVE_NOTE


def propose(
    flex: Flex,
    proposer,
    attempts: list[dict],
    records: list[dict],
    trainset: list,
    metric: Callable,
    max_iters: int = 20,
    num_threads: int | None = None,
) -> str:
    """One proposal for the Flex's new source, from an RLM driven by `proposer`."""
    # The RLM runs with the proposer as the configured LM; drafts must still call the Flex's own.
    predictor_lm = flex.lm or dspy.settings.lm
    backend = backend_note(flex)

    def run_code(module_src: str, indexes: list[int]) -> str:
        """Run a draft module_src on the training examples at `indexes` (at most 25) with default thresholds.

        Returns JSON: each example's outputs, expected outputs, score, error, and decision evidence.
        """
        chosen = [i for i in indexes if isinstance(i, int) and 0 <= i < len(trainset)][:MAX_TRIAL_EXAMPLES]
        trial = flex.deepcopy()
        trial.lm = predictor_lm
        try:
            trial._bind_code(_strip_code_fences(module_src))
        except (SyntaxError, CodeInterpreterError) as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
        results = trace(trial, [trainset[i] for i in chosen], metric, num_threads)
        for result, i in zip(results, chosen, strict=True):
            result["index"] = i
        return json.dumps(results, default=str)

    task = flex._flex_ctx.render_signature_spec() + "\n" + flex._flex_ctx.render_context_blurb(sandboxed=True)
    rlm = dspy.RLM(DecomposeSignature, max_iters=max_iters, tools=[run_code], sub_lm=proposer)
    with dspy.context(lm=proposer):
        result = rlm(
            task=task,
            backend=backend,
            catalog=PRIMITIVES_CATALOG + "\n\n" + DECISION_NOTE,
            attempts=attempts,
            examples=json.loads(json.dumps(records, default=str)),
        )
    return _strip_code_fences(result.module_src)
