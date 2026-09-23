import pytest

import dspy
from dspy.predict.chain_of_density import (
    ChainOfDensity,
    ChainOfDensityWithPreference,
    DensificationStep,
    InitialSummary,
)
from dspy.utils import DummyLM

ARTICLE = "Apple Inc. announced the iPhone 14 in Cupertino. CEO Tim Cook presented it."


def make_lm(num_steps=3, reasoning=False):
    responses = [{"summary": "a company released a phone"}]
    entities = [["Apple"], ["iPhone 14"], ["Tim Cook"], ["Cupertino"]]
    for i in range(num_steps):
        responses.append({"missing_entities": entities[i], "denser_summary": f"dense summary {i + 1} Apple"})
    if reasoning:
        responses = [{"reasoning": "think", **response} for response in responses]
    return DummyLM(responses)


def expected_summaries(num_steps=3):
    return ["a company released a phone"] + [f"dense summary {i + 1} Apple" for i in range(num_steps)]


def test_exported_from_dspy():
    assert dspy.ChainOfDensity is ChainOfDensity
    assert dspy.ChainOfDensityWithPreference is ChainOfDensityWithPreference


def test_signatures():
    assert set(InitialSummary.input_fields) == {"article", "target_length"}
    assert set(InitialSummary.output_fields) == {"summary"}
    assert set(DensificationStep.input_fields) == {"article", "previous_summary", "target_length"}
    assert set(DensificationStep.output_fields) == {"missing_entities", "denser_summary"}


def test_invalid_num_steps():
    with pytest.raises(ValueError, match="num_steps"):
        ChainOfDensity(num_steps=-1)


def test_chain_of_density_sync():
    lm = make_lm(reasoning=True)
    with dspy.context(lm=lm):
        result = ChainOfDensity(num_steps=3, target_length=50)(article=ARTICLE)

    assert result.summaries == expected_summaries()
    assert result.final_summary == "dense summary 3 Apple"
    assert result.entities_added == [["Apple"], ["iPhone 14"], ["Tim Cook"]]
    assert len(result.entity_density) == 4
    assert result.entity_density[0] == {"word_count": 5, "potential_entities": 0, "density": 0.0}
    assert result.entity_density[-1] == {"word_count": 4, "potential_entities": 1, "density": 0.25}

    # Each densification step is fed the previous summary.
    for step, previous in enumerate(expected_summaries()[:-1], start=1):
        assert previous in lm.history[step]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_chain_of_density_async():
    with dspy.context(lm=make_lm()):
        result = await ChainOfDensity(num_steps=3, use_chain_of_thought=False).acall(article=ARTICLE)

    assert result.summaries == expected_summaries()
    assert result.final_summary == "dense summary 3 Apple"
    assert result.entities_added == [["Apple"], ["iPhone 14"], ["Tim Cook"]]


def test_zero_steps_returns_initial_summary():
    with dspy.context(lm=make_lm(num_steps=0)):
        result = ChainOfDensity(num_steps=0, use_chain_of_thought=False)(article=ARTICLE)

    assert result.summaries == ["a company released a phone"]
    assert result.final_summary == "a company released a phone"
    assert result.entities_added == []


def test_return_intermediate_false():
    with dspy.context(lm=make_lm()):
        result = ChainOfDensity(num_steps=3, return_intermediate=False, use_chain_of_thought=False)(article=ARTICLE)

    assert result.summaries == ["dense summary 3 Apple"]
    assert result.final_summary == "dense summary 3 Apple"
    assert len(result.entity_density) == 1
    assert result.entities_added == [["Apple"], ["iPhone 14"], ["Tim Cook"]]


@pytest.mark.parametrize("return_intermediate", [True, False])
def test_preferred_step(return_intermediate):
    cod = ChainOfDensityWithPreference(
        num_steps=3, preferred_step=2, return_intermediate=return_intermediate, use_chain_of_thought=False
    )
    with dspy.context(lm=make_lm()):
        result = cod(article=ARTICLE)

    assert result.preferred_step == 2
    assert result.final_summary == "dense summary 2 Apple"
    if return_intermediate:
        assert result.summaries == expected_summaries()
    else:
        assert result.summaries == ["dense summary 2 Apple"]


def test_preferred_step_is_clamped():
    cod = ChainOfDensityWithPreference(num_steps=3, preferred_step=10, use_chain_of_thought=False)
    with dspy.context(lm=make_lm()):
        result = cod(article=ARTICLE)

    assert result.preferred_step == 3
    assert result.final_summary == "dense summary 3 Apple"


def test_default_preference_is_densest():
    with dspy.context(lm=make_lm()):
        result = ChainOfDensityWithPreference(num_steps=3, use_chain_of_thought=False)(article=ARTICLE)

    assert result.preferred_step == 3
    assert result.final_summary == "dense summary 3 Apple"


def test_invalid_preferred_step():
    with pytest.raises(ValueError, match="preferred_step"):
        ChainOfDensityWithPreference(preferred_step=-1)


@pytest.mark.parametrize("return_intermediate", [True, False])
def test_selection_fn_receives_full_chain(return_intermediate):
    calls = []

    def select(summaries, entities_added):
        calls.append((summaries, entities_added))
        return 1

    cod = ChainOfDensityWithPreference(
        num_steps=3,
        preferred_step=3,
        selection_fn=select,
        return_intermediate=return_intermediate,
        use_chain_of_thought=False,
    )
    with dspy.context(lm=make_lm()):
        result = cod(article=ARTICLE)

    assert calls == [(expected_summaries(), [["Apple"], ["iPhone 14"], ["Tim Cook"]])]
    assert result.preferred_step == 1
    assert result.final_summary == "dense summary 1 Apple"


def test_selection_fn_out_of_range():
    cod = ChainOfDensityWithPreference(
        num_steps=3, selection_fn=lambda summaries, _: len(summaries), use_chain_of_thought=False
    )
    with dspy.context(lm=make_lm()):
        with pytest.raises(ValueError, match="Selected step"):
            cod(article=ARTICLE)


@pytest.mark.asyncio
@pytest.mark.parametrize("return_intermediate", [True, False])
async def test_preference_applies_async(return_intermediate):
    cod = ChainOfDensityWithPreference(
        num_steps=3, preferred_step=1, return_intermediate=return_intermediate, use_chain_of_thought=False
    )
    with dspy.context(lm=make_lm()):
        result = await cod.acall(article=ARTICLE)

    assert result.preferred_step == 1
    assert result.final_summary == "dense summary 1 Apple"


@pytest.mark.asyncio
async def test_selection_fn_applies_async():
    cod = ChainOfDensityWithPreference(num_steps=3, selection_fn=lambda summaries, _: 0, use_chain_of_thought=False)
    with dspy.context(lm=make_lm()):
        result = await cod.acall(article=ARTICLE)

    assert result.preferred_step == 0
    assert result.final_summary == "a company released a phone"
