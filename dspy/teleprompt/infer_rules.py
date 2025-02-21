import logging
import random

import numpy as np

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

logger = logging.getLogger(__name__)


class InferRules(BootstrapFewShot):
    def __init__(self, num_candidates=10, num_rules=10, num_threads=8, teacher_settings=None, **kwargs):
        super().__init__(teacher_settings=teacher_settings, **kwargs)

        self.num_candidates = num_candidates
        self.num_rules = num_rules
        self.num_threads = num_threads
        self.rules_induction_program = RulesInductionProgram(num_rules, teacher_settings=teacher_settings)
        self.metric = kwargs.get("metric")
        self.max_errors = kwargs.get("max_errors", 10)

    def compile(self, student, *, teacher=None, trainset, valset=None):
        if valset is None:
            train_size = int(0.5 * len(trainset))
            trainset, valset = trainset[:train_size], trainset[train_size:]

        super().compile(student, teacher=teacher, trainset=trainset)

        original_program = self.student.deepcopy()
        all_predictors = [p for p in original_program.predictors() if hasattr(p, "signature")]
        instructions_list = [p.signature.instructions for p in all_predictors]

        best_score = -np.inf
        best_program = None

        for candidate_idx in range(self.num_candidates):
            candidate_program = original_program.deepcopy()
            candidate_predictors = [p for p in candidate_program.predictors() if hasattr(p, "signature")]

            for i, predictor in enumerate(candidate_predictors):
                predictor.signature.instructions = instructions_list[i]

            for i, predictor in enumerate(candidate_predictors):
                rules = self.induce_natural_language_rules(predictor, trainset)
                predictor.signature.instructions = instructions_list[i]
                self.update_program_instructions(predictor, rules)

            score = self.evaluate_program(candidate_program, valset)

            if score > best_score:
                best_score = score
                best_program = candidate_program

            logger.info(f"Evaluated Candidate {candidate_idx + 1} with score {score}. Current best score: {best_score}")

        logger.info(f"Final best score: {best_score}")

        return best_program

    def induce_natural_language_rules(self, predictor, trainset):
        demos = self.get_predictor_demos(trainset, predictor)
        signature = predictor.signature
        while True:
            examples_text = self.format_examples(demos, signature)
            try:
                return self.rules_induction_program(examples_text)
            except Exception as e:
                assert (
                    isinstance(e, ValueError)
                    or e.__class__.__name__ == "BadRequestError"
                    or "ContextWindowExceededError" in str(e)
                )
                if len(demos) > 1:
                    demos = demos[:-1]
                else:
                    raise RuntimeError(
                        "Failed to generate natural language rules since a single example couldn't fit in the model's "
                        "context window."
                    ) from e

    def update_program_instructions(self, predictor, natural_language_rules):
        predictor.signature.instructions = (
            f"{predictor.signature.instructions}\n\n"
            f"Please adhere to the following rules when making your prediction:\n{natural_language_rules}"
        )

    def format_examples(self, demos, signature):
        examples_text = ""
        for demo in demos:
            input_fields = {k: v for k, v in demo.items() if k in signature.input_fields}
            output_fields = {k: v for k, v in demo.items() if k in signature.output_fields}
            input_text = "\n".join(f"{k}: {v}" for k, v in input_fields.items())
            output_text = "\n".join(f"{k}: {v}" for k, v in output_fields.items())
            examples_text += f"Input Fields:\n{input_text}\n\n=========\nOutput Fields:\n{output_text}\n\n"
        return examples_text

    def get_predictor_demos(self, trainset, predictor):
        # TODO: Consider how this handled "incomplete" demos.
        signature = predictor.signature
        return [
            {
                key: value
                for key, value in example.items()
                if key in signature.input_fields or key in signature.output_fields
            }
            for example in trainset
        ]

    def evaluate_program(self, program, dataset):
        evaluate = Evaluate(
            devset=dataset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=self.max_errors,
            display_table=False,
            display_progress=True,
            return_all_scores=True,
        )
        score, _ = evaluate(program, metric=self.metric)
        return score


class RulesInductionProgram(dspy.Module):
    def __init__(self, num_rules, teacher_settings=None):
        super().__init__()

        class CustomRulesInduction(dspy.Signature):
            __doc__ = (
                f"Given a set of examples, extract a list of {num_rules} concise and non-redundant natural language "
                "rules that provide clear guidance for performing the task. All rules should be actionable for a "
                "well-specified scope of examples of this general kind of task."
            )
            examples_text = dspy.InputField(desc="Text containing examples")
            natural_language_rules = dspy.OutputField(desc="Induced natural language rules")

        self.rules_induction = dspy.ChainOfThought(CustomRulesInduction)
        self.teacher_settings = teacher_settings or {}
        self.rng = random.Random(0)

    def forward(self, examples_text):
        with dspy.settings.context(**self.teacher_settings):
            lm = dspy.settings.lm.copy(temperature=self.rng.uniform(0.9, 1.0))
            with dspy.settings.context(lm=lm):
                rules = self.rules_induction(examples_text=examples_text).natural_language_rules

        return rules.strip()
