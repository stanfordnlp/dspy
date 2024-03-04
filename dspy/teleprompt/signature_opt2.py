import random
from typing import Generic, Literal, TypeVar, Type

import pydantic
import dspy
from dspy.functional.functional import TypedChainOfThought
from dspy.signatures import Signature
from dspy import BaseModel

# TODO: Consider using the prompt optimizer to optimize the prompt optimizer :O


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
def make_initial_signature(n_prompts: int) -> Type[Signature]:
    """Creates a GenerateInstructionInitial signature with the given number of initial prompts."""

    class GenerateInstructionInitial(Signature, Generic[T]):
        """You are a creative instruction optimizer for large language models.

        I will give you a ``signature`` of fields (inputs and outputs) in English.
        Your task is to propose variations of the signature that will lead a good language model.

        Be very creative and think out of the box. Consider using inspiration such as:
        Openers:
        # You are as smart as ChatGPT.
        # You are highly intelligent.
        # You are an expert mathematician.
        # You are a professor of mathematics.
        Task Descriptions:
        # Solve the following math question.
        # Answer the following math question.
        Closers:
        # This will be fun!
        # Take a deep breath and think carefully.
        # I really need your help!
        """

        basic_signature: T = dspy.InputField()
        proposed_signatures: list[T] = dspy.OutputField(
            desc=f"A list of {n_prompts} very different variations of the basic signature",
            min_items=n_prompts,
            max_items=n_prompts,
        )

    return GenerateInstructionInitial


class ScoredSignature(BaseModel, Generic[T]):
    signature: T
    score: float = dspy.Field(gt=0, lt=100)


class GenerateInstructionGivenAttempts(dspy.Signature, Generic[T]):
    """You are an instruction optimizer for large language models.

    I will give some task instructions I've tried, along with their corresponding validation scores.
    - The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.
    - Your task is to propose a new instruction that will lead a good language model to perform the task even better.
    - Be creative, and think out of the box.
    - Don't repeat instructions, descriptions and prefixes that have already been attempted.
    """

    attempted_signatures: list[ScoredSignature[T]] = dspy.InputField()
    proposed_signature: T = dspy.OutputField(desc="Next signature to try")
    # expected_score: float = dspy.OutputField(desc="The expected score for the new signature")


def optimize_signature(
    student,
    evaluator,
    n_iterations=10,
    strategy: Literal["best", "last"] = "best",
    # Formerly part of the constructor
    prompt_model=None,
    initial_prompts=2,
    temperature=1.4,
    verbose=False,
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
    strategy : Literal["best", "last"], optional
        The strategy to use to select the final program, by default "best"
    prompt_model : dspy.LanguageModel, optional
        The language model to use to generate prompts, by default None
    initial_prompts : int, optional
        The number of initial prompts to generate, by default 2.
        Note that we also use the "plain" signature as a prompt, so the total number of prompts is initial_prompts + 1.
    temperature : float, optional
        The temperature to use when generating new prompts, by default 1.4
    verbose : bool, optional
        Whether to print debug information, by default False
    """
    prompt_model = prompt_model or dspy.settings.lm
    MyGenerateInstructionInitial = make_initial_signature(initial_prompts)  # noqa: N806

    module = student.deepcopy()
    # For some reason named_predictors sometimes returns an empty list, so we use named_parameters instead
    named_predictors = module.named_parameters()
    if verbose:
        print("All predictors:")
        print(f"{named_predictors=}")

    candidates = {}
    scores = []

    # First round, just use initial prompts
    for name, p in named_predictors:
        candidates[name] = [make_info(p.signature)]

    # Make some initial candidates
    with dspy.settings.context(lm=prompt_model):
        # TODO: Parallelize this
        for name, p in named_predictors:
            if verbose:
                print(f"Generating new signature for {p}...")
            info = candidates[name][0]  # Use initial info, to make sure types are identical
            generator = TypedChainOfThought(MyGenerateInstructionInitial[type(info)])
            candidates[name] += generator(
                basic_signature=info,
                config={"temperature": temperature},
            ).proposed_signatures
            assert len(candidates[name]) == initial_prompts + 1  # Basic signature + initial prompts

            candidates[name] = [
                info.model_copy(update={"instructions": info.instructions + f"({i})"})
                for i, info in enumerate(candidates[name])
            ]

            for i, c in enumerate(candidates[name]):
                print(f"Generated candidate {i}:")
                print(c.to_signature())

    # Main loop of scoring + generating new candidates
    for i in range(n_iterations):
        if verbose:
            print("\n" + "=" * 80)
            print(f"Running eval iteration {i}...")

        # Test candidate i
        for p in module.predictors():
            print(f"Installing signature {i}: ")
            print(candidates[name][i].to_signature())
            p.signature = candidates[name][i].to_signature()

        score = evaluator(module)
        score += random.random() * 10
        scores.append(score)

        if verbose:
            print(f"Scores for iteration {i}: {score}")

        # If we are still testing initial prompts, continue
        if i + 1 < len(next(iter(candidates.values()))):
            continue

        # If we are done, there's no need to generate new candidates
        if i + 1 == n_iterations:
            break

        # Otherwise generate the next candidate
        with dspy.settings.context(lm=prompt_model):
            # TODO: Parallelize this
            for name, p in named_predictors:
                SignatureInfo = type(candidates[name][0])  # noqa: N806
                generator = TypedChainOfThought(GenerateInstructionGivenAttempts[SignatureInfo])
                attempted_signatures = [
                    ScoredSignature[SignatureInfo](signature=info, score=sc)
                    for info, sc in zip(candidates[name], scores)
                ]
                attempted_signatures.sort(key=lambda x: x.score)
                if verbose:
                    print(
                        f"Generating new signature for {name} based on {len(attempted_signatures)} previous signatures..."
                    )
                new_signature = generator(
                    attempted_signatures=attempted_signatures,
                    config={"temperature": temperature},
                ).proposed_signature
                if verbose:
                    print("Generated candidate:")
                    print(new_signature.to_signature())
                candidates[name].append(new_signature)

    if strategy == "last":
        return module

    if strategy == "best":
        i = scores.index(max(scores))
        for name, p in named_predictors:
            p.signature = candidates[name][i].to_signature()
        return module

    raise ValueError(f"Invalid strategy: {strategy}")
