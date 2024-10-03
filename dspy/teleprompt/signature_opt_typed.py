import textwrap
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import pydantic

import dspy
from dspy import BaseModel
from dspy.functional.functional import TypedChainOfThought, TypedPredictor
from dspy.signatures import Signature
from dspy.signatures.field import InputField, OutputField

# TODO:
# - Parallelize the generation of new signatures when we have multiple predictors
# - Consider generating multiple new signatures at once, which we can test in parallel
# - Consider using the prompt optimizer to optimize the prompt optimizer :O


def make_info(signature: type[Signature]) -> BaseModel:
    """Creates a SignatureInfo pydantic type, that describes the Signature.

    Returns an instnce of this type, with the instructions and field descriptions of the input type.
    """
    # First, create the SignatureInfo type
    fields = {
        "instructions": (str, pydantic.Field(description="The instructions for the task")),
    }
    for name in signature.fields:
        fields[name + "_prefix"] = (str, pydantic.Field(description=f"The prefix for {name}"))
        fields[name + "_desc"] = (str, pydantic.Field(description=f"The description for {name}"))
    SignatureInfo = pydantic.create_model(  # noqa: N806
        f"SignatureInfo[{signature.__name__}]",
        **fields,
    )

    # Add a method to convert the SignatureInfo back into a Signature
    def to_signature(info):
        new_signature = signature.with_instructions(info.instructions)
        for name in signature.fields:
            new_signature = new_signature.with_updated_fields(
                name,
                prefix=getattr(info, name + "_prefix"),
                desc=getattr(info, name + "_desc"),
            )
        return new_signature

    SignatureInfo.to_signature = to_signature

    # Finally, make an instance of the SignatureInfo type with the signature's
    # default instructions and field descriptions
    values = {"instructions": signature.instructions}
    for name, field in signature.fields.items():
        values[name + "_prefix"] = field.json_schema_extra["prefix"]
        values[name + "_desc"] = field.json_schema_extra["desc"]
    return SignatureInfo(**values)


T = TypeVar("T", bound=BaseModel)


# Note: This function wouldn't be necessary if we could make the number of prompts a generic parameter of the class,
# but alas it seems like this isn't possible in Python right now. The main reason being that types and generics only
# live inside the type system, and can't be used to generate code at runtime.
def make_initial_signature(n_prompts: int) -> type[Signature]:
    """Creates a GenerateInstructionInitial signature with the given number of initial prompts."""

    class GenerateInstructionInitial(Signature, Generic[T]):
        # TODO: Can we make textwrap default/automatic in all signatures?
        __doc__ = textwrap.dedent(
            """\
        You are a creative instruction optimizer for large language models.

        I will give you a ``signature`` of fields (inputs and outputs) in English.
        Your task is to propose variations of the signature that will lead a good language model.

        Be very creative and think out of the box.
        You can use as long instructions as you want.
        Consider using inspiration such as:
        Openers:
        - You are as smart as ChatGPT.
        - You are highly intelligent.
        - You are an expert mathematician.
        - You are a professor of mathematics.
        Task Descriptions:
        - Be consise in your answer.
        - Be as clear as possible.
        - Use lots of creativity.
        Closers:
        - This will be fun!
        - Take a deep breath and think carefully.
        - I really need your help!
        """
        )

        basic_signature: T = InputField()
        proposed_signatures: list[T] = OutputField(
            desc=f"A list of {n_prompts} very different variations of the basic signature",
            min_items=n_prompts,
            max_items=n_prompts,
        )

    return GenerateInstructionInitial


def generate_with_avoidance(signatures_to_avoid: list[BaseModel]) -> type[Signature]:
    class GenerateSignature(dspy.Signature, Generic[T]):
        __doc__ = textwrap.dedent(
            """\
        You are an instruction optimizer for large language models.

        I will give some task instructions I've tried, along with their corresponding validation scores.
        - The instructions are arranged in order based on their scores, where higher scores indicate better quality.
        - Your task is to propose a new instruction that will lead a good language model to perform the task even better.
        - Be creative, and think out of the box.
        - Don't repeat instructions, descriptions and prefixes that have already been attempted.
        """
        )

        analysis: str = OutputField(desc="Consider what made the previous instructions good or bad.")
        proposed_signature: T = OutputField(desc="A signature that will likely lead to a high score.")
        score: float = OutputField(
            desc="The expected score for the new signature. Don't write anything after this number.",
        )

        @pydantic.field_validator("proposed_signature")
        @classmethod
        def check_signature_not_attempted(cls, s: T) -> T:
            if s in signatures_to_avoid:
                raise ValueError("Never propose a signature already in the list above.")
            return s

    return GenerateSignature


@dataclass
class OptimizerResult:
    program: dspy.Program
    signatures: list[dict[str, Signature]]
    scores: list[float]


def optimize_signature(
    student,
    evaluator,
    n_iterations=10,
    sorted_order: Literal["increasing", "decreasing", "chronological"] = "increasing",
    strategy: Literal["last", "best"] = "best",
    max_examples=20,
    # Formerly part of the constructor
    prompt_model=None,
    initial_prompts=2,
    verbose=False,
    max_retries: int = 3,
) -> dspy.Program:
    """Create a new program that is optimized for the given task.

    `student` is a program that needs to be optimized,
    note that it may be zero-shot or already pre-optimized for demos != [].

    Parameters
    ----------
    student : dspy.Program
        The program to optimize.
    evaluator : dspy.Evaluator
        The evaluator to use to score the program.
    n_iterations : int, optional
        The number of iterations to run, by default 10
    max_examples : int, optional
        The maximum number of examples to use for the evaluator, by default 20
    sorted_order : Literal["increasing", "decreasing", "chronological"], optional
        The order in which to sort the scores, by default "increasing"
    strategy : Literal["last", "best"], optional
        The strategy to use to select the final program, by default "best"
    prompt_model : dspy.LanguageModel, optional
        The language model to use to generate prompts, by default None
    initial_prompts : int, optional
        The number of initial prompts to generate, by default 2.
        Note that we also use the "plain" signature as a prompt, so the total number of prompts is initial_prompts + 1.
    verbose : bool, optional
        Whether to print debug information, by default False
    max_retries : int
        The number of retries to use when generating new signatures, by default 3.

    Notes:
    -----
    We don't support temperatures, since it tends to break the typed generation.
    """
    if n_iterations < 1 + initial_prompts:
        raise ValueError("n_iterations must be at least 1 + initial_prompts")

    prompt_model = prompt_model or dspy.settings.lm
    MyGenerateInstructionInitial = make_initial_signature(initial_prompts)  # noqa: N806

    module = student.deepcopy()
    # In contrast to the original implementation, we don't want the Predict's, but the TypedPredictor's.
    # This is because TypedPredictor changes the signature before it runs forward. So changing the signature
    # on the Predicts won't work.
    named_predictors = [
        (name, module)
        for name, module in module.named_sub_modules()
        if isinstance(module, TypedPredictor) and not getattr(module, "_compiled", False)
    ]
    if not named_predictors:
        raise ValueError("No unfrozen/uncompiled TypedPredictors found in the module.")
    if verbose:
        print(f"Found {len(named_predictors)} typed predictors to optimize.")

    candidates = {}
    scores = []

    # First round, just use initial prompts
    for name, p in named_predictors:
        candidates[name] = [make_info(p.signature)]

    # Make some initial candidates
    with dspy.settings.context(lm=prompt_model):
        # TODO: Parallelize this
        for name, _p in named_predictors:
            if verbose:
                print(f"Generating {initial_prompts} initial signatures for {name}...")
            info = candidates[name][0]  # Use initial info, to make sure types are identical
            generator = TypedChainOfThought(MyGenerateInstructionInitial[type(info)], max_retries=max_retries)
            candidates[name] += generator(
                basic_signature=info,
            ).proposed_signatures
            assert len(candidates[name]) == initial_prompts + 1  # Basic signature + initial prompts

    # Main loop of scoring + generating new candidates
    for i in range(n_iterations):
        if verbose:
            print("\n" + "=" * 80)
            print(f"Running eval iteration {i}...")

        # Install signatures
        for name, p in named_predictors:
            p.signature = candidates[name][i].to_signature()

        # Run evaluator given by user
        score = evaluator(module)
        if isinstance(score, tuple):
            scores.append(score[0])
        else:
            scores.append(score)

        # If we are still testing initial prompts, continue
        if i + 1 < len(next(iter(candidates.values()))):
            continue

        # If we are done, there's no need to generate new candidates
        if i + 1 == n_iterations:
            break

        # Otherwise generate the next candidate
        with dspy.settings.context(lm=prompt_model):
            # TODO: Parallelize this
            for name, _p in named_predictors:
                SignatureInfo = type(candidates[name][0])  # noqa: N806

                demos = [dspy.Example(proposed_signature=info, score=sc) for info, sc in zip(candidates[name], scores)]
                if sorted_order == "chronological":
                    demos = demos[-max_examples:]
                elif sorted_order == "increasing":
                    demos.sort(key=(lambda x: x.score), reverse=False)
                    demos = demos[-max_examples:]
                elif sorted_order == "decreasing":
                    demos.sort(key=(lambda x: x.score), reverse=True)
                    demos = demos[:max_examples]
                else:
                    raise ValueError(f"Invalid sorted_order: {sorted_order}")

                # We can only tell the LM to avoid the signatures we are actually giving it as demos.
                avoid = [ex.proposed_signature for ex in demos]
                generator = TypedPredictor(generate_with_avoidance(avoid)[SignatureInfo], max_retries=max_retries)
                generator.predictor.demos = demos

                if verbose:
                    print(f"Generating new signature for {name}...")
                new_signature = generator().proposed_signature
                candidates[name].append(new_signature)

    if strategy == "last":
        pass
    elif strategy == "best":
        i = scores.index(max(scores))
        if verbose:
            print(f"Best signature: {i} with score: {scores[i]}")
        for name, p in named_predictors:
            p.signature = candidates[name][i].to_signature()
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    return OptimizerResult(
        program=module,
        signatures=[{name: sigs[i].to_signature() for name, sigs in candidates.items()} for i in range(n_iterations)],
        scores=scores,
    )
