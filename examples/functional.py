import inspect, os, openai, dspy, functools, typing, pydantic
from typing import Annotated

openai.api_key = os.getenv("OPENAI_API_KEY")
lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=2000)

def predictor(func):
    return _dspy_method(func, cot=False)

def cot(func):
    return _dspy_method(func, cot=True)

def _dspy_method(func, cot=False):
    sig = inspect.signature(func)
    annotations = typing.get_type_hints(func)
    inputs = ", ".join(list(sig.parameters.keys())[1:])
    output = func.__name__
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
        # TODO: If the function already has a field called rationale, this will
        # be overwritten.
        fields["rationale"] = dspy.OutputField(
            prefix="Reasoning: Let's think step by step in order to",
            desc="${produce the " + output + "}. We ...",
        )

    # Output field
    kwargs = {}
    annotation = annotations.get('return', str)
    parser = annotation
    if hasattr(annotation, "__origin__"):
        kwargs["desc"] = annotation.__metadata__
        annotation = annotation.__origin__
    else:
        kwargs["desc"] = ""
    if issubclass(annotation, pydantic.BaseModel):
        kwargs["prefix"] = output + " = {"
        kwargs["desc"] += f" Respond with a single JSON object using the schema {annotation.schema()}"
        kwargs["format"] = lambda x: x if isinstance(x, str) else x.json()
        parser = lambda x: annotation.parse_raw("{" + x)
    else:
        kwargs["prefix"] = output + " = "
        kwargs["desc"] = desc=f"a single {annotation.__name__} value"
        kwargs["format"] = lambda x: x if isinstance(x, str) else str(x)
    fields[output] = dspy.OutputField(**kwargs)

    def inner(self, **kwargs):
        modified_kwargs = kwargs.copy()
        signature = dspy.Signature(fields, instructions)
        print(signature)
        for _ in range(3):
            predictor = dspy.Predict(signature)
            result = getattr(predictor(**modified_kwargs), output)
            try:
                parsed = parser(result)
                return parsed
            except (pydantic.ValidationError, ValueError) as e:
                # TODO: Use free field name
                fields['error'] = dspy.InputField(
                    prefix="Past Error:",
                    desc="An error to avoid in the future"
                )
                modified_kwargs['error'] = str(e)
                signature = dspy.Signature(fields, instructions)
    return inner

class QA(dspy.Module):
    @predictor
    def hard_question(self, topic: str) -> str:
        """Think of a hard factual question about a topic. It should be answerale with a number."""

    @cot
    def answer(self, question: Annotated[str, "Question to ask"]) -> float:
        pass

    # @predictor
    # def hard_question(self,
    #            topic: str = InputField("Topic of question"),
    #            question: str = OutputField("Question to ask"),
    #            ):
    #     """Think of a hard factual question about a topic. It should be answerale with a number."""
    #     assert "fire" in question.lower(), "The question should be about fire"

    def forward(self, **kwargs):
        question = self.hard_question(**kwargs)
        return (question, self.answer(question=question))

qa = QA()
with dspy.context(lm=lm):
    question, answer = qa(topic="Physics")
    print(question)
    print(answer)
    lm.inspect_history(n=5)
