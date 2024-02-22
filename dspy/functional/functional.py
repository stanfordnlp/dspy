import inspect, os, openai, dspy, functools, typing, pydantic, re
from typing import Annotated
from dsp.templates import passages2text, format_answers
import copy


MAX_RETRIES = 3


def predictor(func):
    return _FunctionalPredict(func, cot=False)


def cot(func):
    return _FunctionalPredict(func, cot=True)


class FunctionalModule(dspy.Module):
    def __init__(self):
        super().__init__()
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, _FunctionalPredict):
                self.__dict__[name] = attr.copy()


class _FunctionalPredict(dspy.Module):
    def __init__(self, func, cot=False):
        super().__init__()
        self.func = func
        self.cot = cot
        (
            self.signature,
            self.parser,
            self.wrapped,
            self.output_key,
        ) = self._func_to_fields(func, cot=cot)
        self.predictor = dspy.Predict(self.signature)

    def copy(self):
        return _FunctionalPredict(self.func, self.cot)

    def forward(self, **kwargs):
        modified_kwargs = kwargs.copy()
        try:
            for _ in range(MAX_RETRIES):
                result = getattr(self.predictor(**modified_kwargs), self.output_key)
                try:
                    parsed = self.parser(result)
                    if self.wrapped:
                        parsed = parsed.value
                    return parsed
                except (pydantic.ValidationError, ValueError) as e:
                    # Add a new field explaining the error
                    modified_kwargs["error"] = str(e)
                    self.predictor.signature = self.signature.prepend(
                        "error",
                        dspy.InputField(
                            prefix="Past Error:", desc="An error to avoid in the future"
                        ),
                    )
            print("Warning: Too many retries")
        finally:
            # Restore the original signature, so we don't show the error field in the next call
            self.predictor.signature = self.signature

    def _func_to_fields(self, func, cot=False):
        sig = inspect.signature(func)
        annotations = typing.get_type_hints(func)
        output_key = func.__name__
        instructions = func.__doc__
        fields = {}

        # Input fields
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            # We default to str as the type of the input
            annotation = annotations.get(param.name, str)
            kwargs = {}
            # Check if annotation is an Annotated[type, desc]
            if hasattr(annotation, "__metadata__"):
                kwargs["desc"] = annotation.__metadata__
                annotation = annotation.__origin__
            if annotation == list[str]:
                kwargs["format"] = passages2text
            else:
                # TODO: This may not be the best way to handle pydantic input types
                # should we use .model_dump_json() for pydantic inputs instead?
                kwargs["format"] = lambda x: x if isinstance(x, str) else str(x)
            fields[param.name] = dspy.InputField(**kwargs)

        # Chain of Thought
        if cot:
            fields["reasoning"] = dspy.OutputField(
                prefix="Reasoning: Let's think step by step in order to",
                desc="${produce the " + output_key + "}. We ...",
            )

        # Output field
        kwargs = {}
        annotation = annotations.get("return", str)
        wrapped = False  # Whether the output is wrapped in an "Output" pydantic model
        if hasattr(annotation, "__metadata__"):
            kwargs["desc"] = annotation.__metadata__
            annotation = annotation.__origin__
        else:
            kwargs["desc"] = ""
        if annotation in (str, int, float, bool):
            kwargs["desc"] = desc = f"a single {annotation.__name__} value"
            kwargs["format"] = lambda x: x if isinstance(x, str) else str(x)
            parser = annotation  # For simple types the parser is the type itself
        # TODO: I'm not sure how format_answers really works
        # elif annotation == list[str]:
        #     kwargs["desc"] = desc = "a list of strings"
        #     kwargs["format"] = format_answers
        #     parser = lambda x: x
        else:
            # Anything else we wrap in a pydantic object
            if not inspect.isclass(annotation) or not issubclass(
                annotation, pydantic.BaseModel
            ):
                annotation = pydantic.create_model(
                    "Output", value=(annotation, ...), __base__=pydantic.BaseModel
                )
                wrapped = True

            kwargs["desc"] += (
                f" Respond with a single JSON object using the schema"
                f" {annotation.model_json_schema()}"
            )
            kwargs["format"] = lambda x: x if isinstance(x, str) else x.json()
            parser = lambda x: annotation.model_validate_json(self._unwrap_json(x))

        fields[output_key] = dspy.OutputField(**kwargs)
        signature = dspy.Signature(fields, instructions)
        return signature, parser, wrapped, output_key

    def _unwrap_json(self, output):
        output = output.strip()
        if output.startswith("```"):
            if not output.startswith("```json"):
                raise ValueError("json output should start with ```json")
            if not output.endswith("```"):
                raise ValueError("json output should end with ```")
            output = output[7:-3].strip()
        if not output.startswith("{") or not output.endswith("}"):
            raise ValueError("json output should start and end with { and }")
        return output


################################################################################
# Example usage
################################################################################


def main():
    class Answer(pydantic.BaseModel):
        value: float
        certainty: float
        comments: list[str] = pydantic.Field(
            description="At least two comments about the answer"
        )

    class QA(dspy.Module):
        @predictor
        def hard_question(self, topic: str) -> str:
            """Think of a hard factual question about a topic. It should be answerable with a number."""

        @cot
        def answer(self, question: Annotated[str, "Question to answer"]) -> Answer:
            pass

        def forward(self, **kwargs):
            question = self.hard_question(**kwargs)
            return (question, self.answer(question=question))

    openai.api_key = os.getenv("OPENAI_API_KEY")
    lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=4000)
    # lm = dspy.OpenAI(model="gpt-4", max_tokens=4000)
    # lm = dspy.OpenAI(model="gpt-4-preview-1106", max_tokens=4000)
    with dspy.context(lm=lm):
        qa = QA()
        question, answer = qa(topic="Physics")
        # lm.inspect_history(n=5)

        print("Question:", question)
        print("Answer:", answer)


################################################################################
# HotpotQA example with SimpleBaleen
################################################################################


def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred):
        return False
    if not dspy.evaluate.answer_passage_match(example, pred):
        return False

    hops = [example.question] + [
        outputs.query for *_, outputs in trace if "query" in outputs
    ]

    if max([len(h) for h in hops]) > 100:
        return False
    if any(
        dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8)
        for idx in range(2, len(hops))
    ):
        return False

    return True


def gold_passages_retrieved(example, pred, trace=None):
    gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
    found_titles = set(
        map(dspy.evaluate.normalize_text, [c.split(" | ")[0] for c in pred.context])
    )

    return gold_titles.issubset(found_titles)


def hotpot():
    from dsp.utils import deduplicate
    import dspy.evaluate
    from dspy.datasets import HotPotQA
    from dspy.evaluate.evaluate import Evaluate
    from dspy.teleprompt.bootstrap import BootstrapFewShot

    print("Load the dataset.")
    dataset = HotPotQA(
        train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0
    )
    trainset = [x.with_inputs("question") for x in dataset.train]
    devset = [x.with_inputs("question") for x in dataset.dev]
    print("Done")

    class SimplifiedBaleen(FunctionalModule):
        def __init__(self, passages_per_hop=1, max_hops=2):
            super().__init__()
            self.retrieve = dspy.Retrieve(k=passages_per_hop)
            self.max_hops = max_hops

        @cot
        def generate_query(self, context: list[str], question) -> str:
            """Write a simple search query that will help answer a complex question."""
            pass

        @cot
        def generate_answer(self, context: list[str], question) -> str:
            """Answer questions with short factoid answers."""
            pass

        def forward(self, question):
            context = []

            for hop in range(self.max_hops):
                query = self.generate_query(context=context, question=question)
                passages = self.retrieve(query).passages
                context = deduplicate(context + passages)

            answer = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=answer)

    openai.api_key = os.getenv("OPENAI_API_KEY")
    rm = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
    lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=4000)
    dspy.settings.configure(lm=lm, rm=rm, trace=[])

    evaluate_on_hotpotqa = Evaluate(
        devset=devset, num_threads=10, display_progress=True, display_table=5
    )

    # uncompiled (i.e., zero-shot) program
    uncompiled_baleen = SimplifiedBaleen()
    print(
        "Uncompiled Baleen retrieval score:",
        evaluate_on_hotpotqa(uncompiled_baleen, metric=gold_passages_retrieved),
    )

    # compiled (i.e., few-shot) program
    compiled_baleen = BootstrapFewShot(
        metric=validate_context_and_answer_and_hops
    ).compile(
        SimplifiedBaleen(),
        teacher=SimplifiedBaleen(passages_per_hop=2),
        trainset=trainset,
    )
    print(
        "Compiled Baleen retrieval score:",
        evaluate_on_hotpotqa(compiled_baleen, metric=gold_passages_retrieved),
    )
    lm.inspect_history(n=5)


if __name__ == "__main__":
    # main()
    hotpot()
