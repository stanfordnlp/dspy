import inspect
import os
import openai
import dspy
import typing
import pydantic
from typing import Annotated, List, Tuple  # noqa: UP035
from dsp.templates import passages2text
import json

from dspy.signatures.signature import ensure_signature


MAX_RETRIES = 3


def predictor(func) -> dspy.Module:
    """Decorator that creates a predictor module based on the provided function."""
    signature = _func_to_signature(func)
    *_, output_key = signature.output_fields.keys()
    return _StripOutput(TypedPredictor(signature), output_key)


def cot(func) -> dspy.Module:
    """Decorator that creates a chain of thought module based on the provided function."""
    signature = _func_to_signature(func)
    *_, output_key = signature.output_fields.keys()
    return _StripOutput(TypedChainOfThought(signature), output_key)


class _StripOutput(dspy.Module):
    def __init__(self, predictor, output_key):
        super().__init__()
        self.predictor = predictor
        self.output_key = output_key

    def copy(self):
        return _StripOutput(self.predictor.copy(), self.output_key)

    def forward(self, **kwargs):
        prediction = self.predictor(**kwargs)
        return prediction[self.output_key]


class FunctionalModule(dspy.Module):
    """To use the @cot and @predictor decorators, your module needs to inheret form this class."""

    def __init__(self):
        super().__init__()
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, dspy.Module):
                self.__dict__[name] = attr.copy()


def TypedChainOfThought(signature) -> dspy.Module:  # noqa: N802
    """Just like TypedPredictor, but adds a ChainOfThought OutputField."""
    signature = ensure_signature(signature)
    output_keys = ", ".join(signature.output_fields.keys())
    return TypedPredictor(
        signature.prepend(
            "reasoning",
            dspy.OutputField(
                prefix="Reasoning: Let's think step by step in order to",
                desc="${produce the " + output_keys + "}. We ...",
            ),
        ),
    )


class TypedPredictor(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)

    def copy(self) -> "TypedPredictor":
        return TypedPredictor(self.signature)

    @staticmethod
    def _make_example(type_) -> str:
        # Note: DSPy will cache this call so we only pay the first time TypedPredictor is called.
        json_object = dspy.Predict(
            dspy.Signature(
                "json_schema -> json_object",
                "Make a very succinct json object that validates with the following schema",
            ),
        )(json_schema=json.dumps(type_.model_json_schema())).json_object
        # We use the model_validate_json method to make sure the example is valid
        try:
            type_.model_validate_json(_unwrap_json(json_object))
        except (pydantic.ValidationError, ValueError):
            return ""  # Unable to make an example
        return json_object
        # TODO: Another fun idea is to only (but automatically) do this if the output fails.
        # We could also have a more general "suggest solution" prompt that tries to fix the output
        # More directly.
        # TODO: Instead of using a language model to create the example, we can also just use a
        # library like https://pypi.org/project/polyfactory/ that's made exactly to do this.

    def _prepare_signature(self) -> dspy.Signature:
        """Add formats and parsers to the signature fields, based on the type
        annotations of the fields.
        """
        signature = self.signature
        for name, field in self.signature.fields.items():
            is_output = field.json_schema_extra["__dspy_field_type"] == "output"
            type_ = field.annotation
            if is_output:
                if type_ in (str, int, float, bool):
                    signature = signature.with_updated_fields(
                        name,
                        desc=field.json_schema_extra.get("desc", "")
                        + (f" (Respond with a single {type_.__name__} value)" if type_ != str else ""),
                        format=lambda x: x if isinstance(x, str) else str(x),
                        parser=type_,
                    )
                else:
                    # Anything else we wrap in a pydantic object
                    to_json = lambda x: x.model_dump_json()
                    from_json = lambda x, type_=type_: type_.model_validate_json(x)
                    if not (inspect.isclass(type_) and issubclass(type_, pydantic.BaseModel)):
                        type_ = pydantic.create_model("Output", value=(type_, ...), __base__=pydantic.BaseModel)
                        to_json = lambda x, type_=type_: type_(value=x).model_dump_json()
                        from_json = lambda x, type_=type_: type_.model_validate_json(x).value
                    signature = signature.with_updated_fields(
                        name,
                        desc=field.json_schema_extra.get("desc", "")
                        + (
                            ". Respond with a single JSON object. JSON Schema: "
                            + json.dumps(type_.model_json_schema())
                        ),
                        format=lambda x, to_json=to_json: (x if isinstance(x, str) else to_json(x)),
                        parser=lambda x, from_json=from_json: from_json(_unwrap_json(x)),
                        type_=type_,
                    )
            else:  # If input field
                format_ = lambda x: x if isinstance(x, str) else str(x)
                if type_ in (List[str], list[str], Tuple[str], tuple[str]):
                    format_ = passages2text
                elif inspect.isclass(type_) and issubclass(type_, pydantic.BaseModel):
                    format_ = lambda x: x if isinstance(x, str) else x.model_dump_json()
                signature = signature.with_updated_fields(name, format=format_)

        return signature

    def forward(self, **kwargs) -> dspy.Prediction:
        modified_kwargs = kwargs.copy()
        # We have to re-prepare the signature on every forward call, because the base
        # signature might have been modified by an optimizer or something like that.
        signature = self._prepare_signature()
        for try_i in range(MAX_RETRIES):
            result = self.predictor(**modified_kwargs, new_signature=signature)
            errors = {}
            parsed_results = {}
            # Parse the outputs
            for name, field in signature.output_fields.items():
                try:
                    value = getattr(result, name)
                    parser = field.json_schema_extra.get("parser", lambda x: x)
                    parsed_results[name] = parser(value)
                except (pydantic.ValidationError, ValueError) as e:
                    errors[name] = _format_error(e)
                    # If we can, we add an example to the error message
                    current_desc = field.json_schema_extra.get("desc", "")
                    i = current_desc.find("JSON Schema: ")
                    if i == -1:
                        continue  # Only add examples to JSON objects
                    suffix, current_desc = current_desc[i:], current_desc[:i]
                    prefix = "You MUST use this format: "
                    if try_i + 1 < MAX_RETRIES \
                            and prefix not in current_desc \
                            and (example := self._make_example(field.annotation)):
                        signature = signature.with_updated_fields(
                            name, desc=current_desc + "\n" + prefix + example + "\n" + suffix,
                        )
            if errors:
                # Add new fields for each error
                for name, error in errors.items():
                    modified_kwargs[f"error_{name}_{try_i}"] = error
                    signature = signature.append(
                        f"error_{name}_{try_i}",
                        dspy.InputField(
                            prefix="Past Error " + (f"({name}):" if try_i == 0 else f"({name}, {try_i+1}):"),
                            desc="An error to avoid in the future",
                        ),
                    )
            else:
                # If there are no errors, we return the parsed results
                for name, value in parsed_results.items():
                    setattr(result, name, value)
                return result
        raise ValueError(
            "Too many retries trying to get the correct output format. " + "Try simplifying the requirements.", errors,
        )


def _format_error(error: Exception):
    if isinstance(error, pydantic.ValidationError):
        errors = []
        for e in error.errors():
            fields = ", ".join(e["loc"])
            errors.append(f"{e['msg']}: {fields} (error type: {e['type']})")
        return "; ".join(errors)
    else:
        return repr(error)


def _func_to_signature(func):
    """Make a dspy.Signature based on a function definition."""
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
        if typing.get_origin(annotation) is Annotated:
            annotation, kwargs["desc"] = typing.get_args(annotation)
        fields[param.name] = (annotation, dspy.InputField(**kwargs))

    # Output field
    kwargs = {}
    annotation = annotations.get("return", str)
    if typing.get_origin(annotation) is Annotated:
        annotation, kwargs["desc"] = typing.get_args(annotation)
    fields[output_key] = (annotation, dspy.OutputField(**kwargs))

    return dspy.Signature(fields, instructions)


def _unwrap_json(output):
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


def main() -> None:
    class Answer(pydantic.BaseModel):
        value: float
        certainty: float
        comments: list[str] = pydantic.Field(description="At least two comments about the answer")

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

        print("Question:", question)  # noqa: T201
        print("Answer:", answer)  # noqa: T201


################################################################################
# HotpotQA example with SimpleBaleen
################################################################################


def validate_context_and_answer_and_hops(example, pred, trace=None) -> bool:
    if not dspy.evaluate.answer_exact_match(example, pred):
        return False
    if not dspy.evaluate.answer_passage_match(example, pred):
        return False

    hops = [example.question] + [outputs.query for *_, outputs in trace if "query" in outputs]

    if max([len(h) for h in hops]) > 100:
        return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))):
        return False

    return True


def gold_passages_retrieved(example, pred, _trace=None) -> bool:
    gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
    found_titles = set(map(dspy.evaluate.normalize_text, [c.split(" | ")[0] for c in pred.context]))

    return gold_titles.issubset(found_titles)


def hotpot() -> None:
    from dsp.utils import deduplicate
    import dspy.evaluate
    from dspy.datasets import HotPotQA
    from dspy.evaluate.evaluate import Evaluate
    from dspy.teleprompt.bootstrap import BootstrapFewShot

    print("Load the dataset.")  # noqa: T201
    dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
    trainset = [x.with_inputs("question") for x in dataset.train]
    devset = [x.with_inputs("question") for x in dataset.dev]
    print("Done")  # noqa: T201

    class SimplifiedBaleen(FunctionalModule):
        def __init__(self, passages_per_hop=3, max_hops=1):
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

            for _ in range(self.max_hops):
                query = self.generate_query(context=context, question=question)
                passages = self.retrieve(query).passages
                context = deduplicate(context + passages)

            answer = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=answer)

    openai.api_key = os.getenv("OPENAI_API_KEY")
    rm = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
    lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=4000)
    dspy.settings.configure(lm=lm, rm=rm, trace=[])

    evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=10, display_progress=True, display_table=5)

    # uncompiled (i.e., zero-shot) program
    uncompiled_baleen = SimplifiedBaleen()
    print(  # noqa: T201
        "Uncompiled Baleen retrieval score:",
        evaluate_on_hotpotqa(uncompiled_baleen, metric=gold_passages_retrieved),
    )

    # compiled (i.e., few-shot) program
    compiled_baleen = BootstrapFewShot(metric=validate_context_and_answer_and_hops).compile(
        SimplifiedBaleen(),
        teacher=SimplifiedBaleen(passages_per_hop=2),
        trainset=trainset,
    )
    print(  # noqa: T201
        "Compiled Baleen retrieval score:",
        evaluate_on_hotpotqa(compiled_baleen, metric=gold_passages_retrieved),
    )
    # lm.inspect_history(n=5)


if __name__ == "__main__":
    # main()
    hotpot()
