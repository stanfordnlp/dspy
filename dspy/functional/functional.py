import inspect, os, openai, dspy, functools, typing, pydantic, re
from typing import Annotated

MAX_RETRIES = 3


def predictor(func):
    return _FunctionalModule(func, cot=False)


def cot(func):
    return _FunctionalModule(func, cot=True)


class _FunctionalModule(dspy.Module):
    def __init__(self, func, cot=False):
        super().__init__()
        (
            self.signature,
            self.parser,
            self.wrapped,
            self.output_key,
        ) = self._func_to_fields(func)
        self.predictor = dspy.Predict(self.signature)

    def forward(self, *args, **kwargs):
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

    def _func_to_fields(self, func):
        sig = inspect.signature(func)
        annotations = typing.get_type_hints(func)
        output_key = func.__name__
        instructions = func.__doc__
        fields = {}

        # Input fields
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            annotation = annotations.get(param.name, str)
            kwargs = {}
            if hasattr(annotation, "__origin__"):
                kwargs["desc"] = annotation.__metadata__
                annotation = annotation.__origin__
            kwargs["format"] = lambda x: x if isinstance(x, str) else str(x)
            fields[param.name] = dspy.InputField(**kwargs)

        # Chain of Thought
        if cot:
            # TODO: If the function already has a field called reasoning, this will
            # be overwritten.
            fields["reasoning"] = dspy.OutputField(
                prefix="Reasoning: Let's think step by step in order to",
                desc="${produce the " + output_key + "}. We ...",
            )

        # Output field
        kwargs = {}
        annotation = annotations.get("return", str)
        parser = annotation
        wrapped = False  # Whether the output is wrapped in an "Output" pydantic model
        if hasattr(annotation, "__metadata__"):
            kwargs["desc"] = annotation.__metadata__
            annotation = annotation.__origin__
        else:
            kwargs["desc"] = ""
        if annotation in (str, int, float, bool):
            # kwargs["prefix"] = output_key + " = "
            kwargs["desc"] = desc = f"a single {annotation.__name__} value"
            kwargs["format"] = lambda x: x if isinstance(x, str) else str(x)
        # Anything else we wrap in a pydantic object
        else:
            if not inspect.isclass(annotation) or not issubclass(annotation, pydantic.BaseModel):
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
        #lm.inspect_history(n=5)

        print("Question:", question)
        print("Answer:", answer)


if __name__ == "__main__":
    main()
