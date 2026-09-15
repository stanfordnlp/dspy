import pytest

import dspy
from dspy.predict import Predict
from dspy.propose.grounded_proposer import GroundedProposer
from dspy.utils.dummies import DummyLM


@pytest.mark.parametrize(
    "demo_candidates",
    [
        None,
        [[[dspy.Example(question="What is the capital of France?", answer="Paris")]]],
    ],
)
def test_propose_instructions_for_program(demo_candidates):
    # Set large number here so that lm always returns the same response
    prompt_model = DummyLM([{"proposed_instruction": "instruction"}] * 10)
    program = Predict("question -> answer")
    trainset = []

    proposer = GroundedProposer(prompt_model=prompt_model, program=program, trainset=trainset, verbose=False)
    result = proposer.propose_instructions_for_program(
        trainset=trainset, program=program, demo_candidates=demo_candidates, trial_logs={}, N=1
    )
    assert isinstance(result, dict)
    assert len(result) == len(program.predictors())
    for pred_instructions in result.values():
        assert pred_instructions == ["instruction"]


@pytest.mark.parametrize(
    "demo_candidates",
    [
        None,
        [[[dspy.Example(question="What is the capital of France?", answer="Paris")]]],
    ],
)
def test_propose_instruction_for_predictor(demo_candidates):
    class TrackingDummyLM(DummyLM):
        def copy(self, **kwargs):
            self.last_copy_kwargs = kwargs
            return super().copy(**kwargs)

    prompt_model = TrackingDummyLM([{"proposed_instruction": "instruction"}] * 10)
    program = Predict("question -> answer")

    proposer = GroundedProposer(
        prompt_model=prompt_model,
        program=program,
        trainset=[],
        verbose=False,
        init_temperature=0.7,
    )
    result = proposer.propose_instruction_for_predictor(
        program=program,
        predictor=None,
        pred_i=0,
        demo_candidates=demo_candidates,
        demo_set_i=0,
        trial_logs={},
        tip=None,
    )
    assert result == "instruction"
    assert prompt_model.last_copy_kwargs["temperature"] == 0.7


def test_labeled_demos_included_in_task_demos():
    """Regression test: labeled (non-augmented) demos must appear in task_demos context.

    gather_examples_from_sets previously required the "augmented" key, which only
    bootstrapped demos carry. Hand-labeled examples from the training set never have
    this key, so they were silently excluded. When max_bootstrapped_demos=0 and only
    labeled demos exist, task_demos always collapsed to "No task demos provided.",
    giving the instruction proposer no context about the task.
    """

    class SharedHistoryDummyLM(DummyLM):
        """DummyLM whose copies share the parent's history list for post-call inspection."""

        def copy(self, **kwargs):
            clone = super().copy(**kwargs)
            clone.history = self.history  # share list reference
            return clone

    prompt_model = SharedHistoryDummyLM([{"proposed_instruction": "instruction"}] * 20)
    program = Predict("question -> answer")

    # A labeled example WITHOUT the "augmented" key, as produced from a training set.
    labeled_example = dspy.Example(question="What is 2+2?", answer="4")
    assert "augmented" not in labeled_example.keys(), "setup: labeled_example must not have 'augmented'"

    # Two demo candidate sets so demo_set_i=1 avoids the demo_set_i==0 shortcircuit.
    demo_candidates = [
        [
            [labeled_example],  # set 0 — adjacent set used when demo_set_i=1
            [labeled_example],  # set 1 — current set
        ]
    ]

    proposer = GroundedProposer(
        prompt_model=prompt_model,
        program=program,
        trainset=[],
        use_dataset_summary=False,
        program_aware=False,
        use_task_demos=True,
        use_instruct_history=False,
        verbose=False,
    )

    proposer.propose_instruction_for_predictor(
        program=program,
        predictor=None,
        pred_i=0,
        demo_candidates=demo_candidates,
        demo_set_i=1,
        trial_logs={},
        tip=None,
    )

    assert len(prompt_model.history) > 0, "LM must have been called"
    full_prompt = " ".join(
        str(m.get("content", "")) for m in prompt_model.history[-1]["messages"]
    )
    assert "What is 2+2?" in full_prompt, (
        "Labeled demo ('What is 2+2?') must appear in the task_demos context sent to the "
        "instruction proposer. If missing, gather_examples_from_sets is still filtering out "
        "non-augmented examples."
    )
    assert "No task demos provided." not in full_prompt, (
        "task_demos must contain the labeled example rather than the fallback message."
    )
