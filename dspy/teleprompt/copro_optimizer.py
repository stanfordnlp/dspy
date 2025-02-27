import logging
from collections import defaultdict

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.signatures import Signature
from dspy.teleprompt.teleprompt import Teleprompter

logger = logging.getLogger(__name__)

"""
USAGE SUGGESTIONS:

The following code can be used to compile a optimized signature teleprompter, and evaluate it on an end task:

teleprompter = COPRO(prompt_model=prompt_model, metric=metric, breadth=BREADTH, depth=DEPTH, init_temperature=INIT_TEMPERATURE)
kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
compiled_prompt_opt = teleprompter.compile(program.deepcopy(), trainset=trainset[:DEV_NUM], eval_kwargs=kwargs)
eval_score = evaluate(compiled_prompt_opt, devset=evalset[:EVAL_NUM], **kwargs)

Note that this teleprompter takes in the following parameters:

* prompt_model: The model used for prompt generation. When unspecified, defaults to the model set in settings (ie. dspy.settings.configure(lm=task_model)).
* metric: The task metric used for optimization.
* breadth: The number of new prompts to generate at each iteration. Default=10.
* depth: The number of times we should ask our prompt model to generate new prompts, with the history of the past prompts as input. Default=3.
* init_temperature: The temperature used to generate new prompts. Higher roughly equals more creative. Default=1.4.
* track_stats: Tells the method whether or not to track statistics about the optimization process.
                If True, the method will track the following statistics:
                    * results_best: The min,max,avg,stddev of top 10 scores for each predictor at each depth.
                    * results_latest: The min,max,avg,stddev of newest prompt scores for each predictor at each depth.
                    * total_calls: The total number of calls to the task metric.
                These statistics will be returned as attributes of the best program.
"""


class BasicGenerateInstruction(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(
        desc="The string at the end of the prompt, which will help the model start solving the task",
    )


class GenerateInstructionGivenAttempts(dspy.Signature):
    """You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.

    Your task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."""

    attempted_instructions = dspy.InputField()
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(
        desc="The string at the end of the prompt, which will help the model start solving the task",
    )


class COPRO(Teleprompter):
    def __init__(
        self,
        prompt_model=None,
        metric=None,
        breadth=10,
        depth=3,
        init_temperature=1.4,
        track_stats=False,
        **_kwargs,
    ):
        if breadth <= 1:
            raise ValueError("Breadth must be greater than 1")
        self.metric = metric
        self.breadth = breadth
        self.depth = depth
        self.init_temperature = init_temperature
        self.prompt_model = prompt_model
        self.track_stats = track_stats

    def _check_candidates_equal(self, candidate1, candidate2):
        for p1, p2 in zip(candidate1["program"].predictors(), candidate2["program"].predictors()):
            if self._get_signature(p1).instructions != self._get_signature(p2).instructions:
                return False
            *_, p1_last_field = self._get_signature(p1).fields.values()
            *_, p2_last_field = self._get_signature(p2).fields.values()
            if p1_last_field != p2_last_field:
                return False
        return True

    def _drop_duplicates(self, candidates):
        final_candidates = []
        last_batch = []
        last_batch_score = -1
        for c in candidates:
            repeat = False
            if c["score"] == last_batch_score:
                for c2 in last_batch:
                    if self._check_candidates_equal(c, c2):
                        repeat = True
                        break
                if not repeat:
                    last_batch.append(c)
            else:
                last_batch = [c]
                last_batch_score = c["score"]
            if not repeat:
                final_candidates.append(c)
        return final_candidates

    def _print_signature(self, predictor):
        signature = self._get_signature(predictor)

        logger.debug(f"i: {signature.instructions}")
        logger.debug(f"p: {list(signature.fields.values())[-1].json_schema_extra['prefix']}")

    def _get_signature(self, predictor):
        assert hasattr(predictor, "signature")
        return predictor.signature

    def _set_signature(self, predictor, updated_signature):
        assert hasattr(predictor, "signature")
        predictor.signature = updated_signature

    def compile(self, student, *, trainset, eval_kwargs):
        """
        optimizes `signature` of `student` program - note that it may be zero-shot or already pre-optimized (demos already chosen - `demos != []`)

        parameters:
        student: program to optimize and left modified.
        trainset: iterable of `Example`s
        eval_kwargs: optional, dict
           Additional keywords to go into `Evaluate` for the metric.

        Returns optimized version of `student`.
        """
        module = student.deepcopy()
        evaluate = Evaluate(devset=trainset, metric=self.metric, **eval_kwargs)
        total_calls = 0
        results_best = {
            id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []} for p in module.predictors()
        }
        results_latest = {
            id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []} for p in module.predictors()
        }

        if self.track_stats:
            import numpy as np

        candidates = {}
        evaluated_candidates = defaultdict(dict)

        # Seed the prompt optimizer zero shot with just the instruction, generate BREADTH new prompts
        for predictor in module.predictors():
            basic_instruction = None
            basic_prefix = None
            *_, last_key = self._get_signature(predictor).fields.keys()
            basic_instruction = self._get_signature(predictor).instructions
            basic_prefix = self._get_signature(predictor).fields[last_key].json_schema_extra["prefix"]
            if self.prompt_model:
                with dspy.settings.context(lm=self.prompt_model):
                    instruct = dspy.Predict(
                        BasicGenerateInstruction,
                        n=self.breadth - 1,
                        temperature=self.init_temperature,
                    )(basic_instruction=basic_instruction)
            else:
                instruct = dspy.Predict(
                    BasicGenerateInstruction,
                    n=self.breadth - 1,
                    temperature=self.init_temperature,
                )(basic_instruction=basic_instruction)
            # Add in our initial prompt as a candidate as well
            instruct.completions.proposed_instruction.append(basic_instruction)
            instruct.completions.proposed_prefix_for_output_field.append(basic_prefix)
            candidates[id(predictor)] = instruct.completions
            evaluated_candidates[id(predictor)] = {}

        if self.prompt_model:
            logger.debug(f"{self.prompt_model.inspect_history(n=1)}")

        latest_candidates = candidates
        all_candidates = candidates

        module_clone = module.deepcopy()

        # For each iteration in depth...
        for d in range(
            self.depth,
        ):  # TODO: fix this so that we eval the new batch of predictors with the new best following predictors
            logger.info(f"Iteration Depth: {d+1}/{self.depth}.")

            latest_scores = []

            # Go through our module's predictors
            for p_i, (p_old, p_new) in enumerate(zip(module.predictors(), module_clone.predictors())):
                candidates_ = latest_candidates[id(p_old)]  # Use the most recently generated candidates for evaluation
                if len(module.predictors()) > 1:
                    # Unless our program has multiple predictors, in which case we need to reevaluate all prompts with
                    # the new prompt(s) for the other predictor(s).
                    candidates_ = all_candidates[
                        id(p_old)
                    ]

                # For each candidate
                for c_i, c in enumerate(candidates_):
                    # Get the candidate instruction and prefix
                    instruction, prefix = (
                        c.proposed_instruction.strip('"').strip(),
                        c.proposed_prefix_for_output_field.strip('"').strip(),
                    )

                    # Set this new module with our instruction / prefix
                    *_, last_key = self._get_signature(p_new).fields.keys()
                    updated_signature = (
                        self._get_signature(p_new)
                        .with_instructions(instruction)
                        .with_updated_fields(last_key, prefix=prefix)
                    )
                    self._set_signature(p_new, updated_signature)

                    # Score the instruction / prefix
                    for i, predictor in enumerate(module_clone.predictors()):
                        logger.debug(f"Predictor {i+1}")
                        self._print_signature(predictor)
                    logger.info(
                        f"At Depth {d+1}/{self.depth}, Evaluating Prompt Candidate #{c_i+1}/{len(candidates_)} for "
                        f"Predictor {p_i+1} of {len(module.predictors())}.",
                    )
                    score = evaluate(module_clone, devset=trainset, **eval_kwargs)
                    if self.prompt_model:
                        logger.debug(f"prompt_model.inspect_history(n=1) {self.prompt_model.inspect_history(n=1)}")
                    total_calls += 1

                    replace_entry = True
                    logger.debug(f"(instruction, prefix) {(instruction, prefix)}")
                    if (instruction, prefix) in evaluated_candidates[id(p_old)]:
                        if evaluated_candidates[id(p_old)][(instruction, prefix)]["score"] >= score:
                            replace_entry = False

                    if replace_entry:
                        # Add it to our evaluated candidates list
                        evaluated_candidates[id(p_old)][(instruction, prefix)] = {
                            "score": score,
                            "program": module_clone.deepcopy(),
                            "instruction": instruction,
                            "prefix": prefix,
                            "depth": d,
                        }

                    if len(candidates_) - self.breadth <= c_i:
                        latest_scores.append(score)

                if self.track_stats:
                    results_latest[id(p_old)]["depth"].append(d)
                    results_latest[id(p_old)]["max"].append(max(latest_scores))
                    results_latest[id(p_old)]["average"].append(sum(latest_scores) / len(latest_scores))
                    results_latest[id(p_old)]["min"].append(min(latest_scores))
                    results_latest[id(p_old)]["std"].append(np.std(latest_scores))

                # Now that we've evaluated the candidates, set this predictor to the best performing version
                # to ensure the next round of scores reflect the best possible version
                best_candidate = max(evaluated_candidates[id(p_old)].values(), key=lambda candidate: candidate["score"])
                *_, last_key = self._get_signature(p_old).fields.keys()
                updated_signature = (
                    self._get_signature(p_new)
                    .with_instructions(best_candidate["instruction"])
                    .with_updated_fields(last_key, prefix=best_candidate["prefix"])
                )
                self._set_signature(p_new, updated_signature)

                logger.debug(
                    f"Updating Predictor {id(p_old)} to:\ni: {best_candidate['instruction']}\n"
                    f"p: {best_candidate['prefix']}",
                )
                logger.debug("Full predictor with update: ")
                for i, predictor in enumerate(module_clone.predictors()):
                    logger.debug(f"Predictor {i}")
                    self._print_signature(predictor)

            if d == self.depth - 1:
                break

            new_candidates = {}
            for p_base in module.predictors():
                # Build Few-Shot Example of Optimized Prompts
                attempts = []
                shortest_len = self.breadth
                shortest_len = min(len(evaluated_candidates[id(p_base)]), shortest_len)
                best_predictors = list(evaluated_candidates[id(p_base)].values())

                # best_predictors = evaluated_candidates[id(p_base)].values()[:]
                best_predictors.sort(key=lambda x: x["score"], reverse=True)

                if self.track_stats:
                    scores = [x["score"] for x in best_predictors][:10]
                    results_best[id(p_base)]["depth"].append(d)
                    results_best[id(p_base)]["max"].append(max(scores))
                    results_best[id(p_base)]["average"].append(sum(scores) / len(scores))
                    results_best[id(p_base)]["min"].append(min(scores))
                    results_best[id(p_base)]["std"].append(np.std(scores))

                for i in range(shortest_len - 1, -1, -1):
                    # breakpoint()
                    attempts.append(f'Instruction #{shortest_len-i}: {best_predictors[i]["instruction"]}')
                    attempts.append(f'Prefix #{shortest_len-i}: {best_predictors[i]["prefix"]}')
                    attempts.append(f'Resulting Score #{shortest_len-i}: {best_predictors[i]["score"]}')

                # Generate next batch of potential prompts to optimize, with previous attempts as input
                if self.prompt_model:
                    with dspy.settings.context(lm=self.prompt_model):
                        instr = dspy.Predict(
                            GenerateInstructionGivenAttempts,
                            n=self.breadth,
                            temperature=self.init_temperature,
                        )(attempted_instructions=attempts)
                else:
                    instr = dspy.Predict(
                        GenerateInstructionGivenAttempts,
                        n=self.breadth,
                        temperature=self.init_temperature,
                    )(attempted_instructions=attempts)

                if self.prompt_model:
                    logger.debug(
                        f"(self.prompt_model.inspect_history(n=1)) {self.prompt_model.inspect_history(n=1)}"
                    )
                # Get candidates for each predictor
                new_candidates[id(p_base)] = instr.completions
                all_candidates[id(p_base)].proposed_instruction.extend(instr.completions.proposed_instruction)
                all_candidates[id(p_base)].proposed_prefix_for_output_field.extend(
                    instr.completions.proposed_prefix_for_output_field,
                )

            if self.prompt_model:
                logger.debug(f"{self.prompt_model.inspect_history(n=1)}")
            latest_candidates = new_candidates

        candidates = []
        for predictor in module.predictors():
            candidates.extend(list(evaluated_candidates[id(predictor)].values()))

            if self.track_stats:
                best_predictors = list(evaluated_candidates[id(predictor)].values())
                best_predictors.sort(key=lambda x: x["score"], reverse=True)

                scores = [x["score"] for x in best_predictors][:10]
                results_best[id(predictor)]["depth"].append(d)
                results_best[id(predictor)]["max"].append(max(scores))
                results_best[id(predictor)]["average"].append(sum(scores) / len(scores))
                results_best[id(predictor)]["min"].append(min(scores))
                results_best[id(predictor)]["std"].append(np.std(scores))

        candidates.sort(key=lambda x: x["score"], reverse=True)

        candidates = self._drop_duplicates(candidates)

        best_program = candidates[0]["program"]
        best_program.candidate_programs = candidates
        best_program.total_calls = total_calls
        if self.track_stats:
            best_program.results_best = results_best
            best_program.results_latest = results_latest

        return best_program
