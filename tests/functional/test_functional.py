import datetime
import textwrap
from typing import Annotated, Any, Generic, List, Literal, Optional, TypeVar

import pydantic
import pytest
from pydantic import AfterValidator, BaseModel, Field, ValidationError, field_validator, model_validator

import dspy
from dspy.functional import FunctionalModule, TypedChainOfThought, TypedPredictor, cot, predictor
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.vanilla import LabeledFewShot
from dspy.utils.dummies import DummyLM


def test_simple():
    @predictor
    def hard_question(topic: str) -> str:
        """Think of a hard factual question about a topic."""

    expected = "What is the speed of light?"
    lm = DummyLM([{"hard_question": expected}])
    dspy.settings.configure(lm=lm)

    question = hard_question(topic="Physics")
    lm.inspect_history(n=2)

    assert question == expected


def test_list_output():
    @predictor
    def hard_questions(topics: List[str]) -> List[str]:
        pass

    expected = ["What is the speed of light?", "What is the speed of sound?"]
    lm = DummyLM([{"hard_questions": '["What is the speed of light?", "What is the speed of sound?"]'}])
    dspy.settings.configure(lm=lm)

    question = hard_questions(topics=["Physics", "Music"])
    lm.inspect_history(n=2)

    assert question == expected


def test_simple_type():
    class Question(pydantic.BaseModel):
        value: str

    @predictor
    def hard_question(topic: str) -> Question:
        """Think of a hard factual question about a topic."""

    expected = "What is the speed of light?"
    lm = DummyLM([{"hard_question": f'{{"value": "{expected}"}}'}])
    dspy.settings.configure(lm=lm)

    question = hard_question(topic="Physics")

    assert isinstance(question, Question)
    assert question.value == expected


def test_simple_type_input():
    class Question(pydantic.BaseModel):
        value: str

    class Answer(pydantic.BaseModel):
        value: str

    @predictor
    def answer(question: Question) -> Answer:
        pass

    question = Question(value="What is the speed of light?")
    lm = DummyLM([{"answer": '{"value": "3e8"}'}])
    dspy.settings.configure(lm=lm)

    result = answer(question=question)

    assert result == Answer(value="3e8")


def test_simple_class():
    class Answer(pydantic.BaseModel):
        value: float
        certainty: float
        comments: List[str] = pydantic.Field(description="At least two comments about the answer")

    class QA(FunctionalModule):
        @predictor
        def hard_question(self, topic: str) -> str:
            """Think of a hard factual question about a topic. It should be answerable with a number."""

        @cot
        def answer(self, question: Annotated[str, "Question to answer"]) -> Answer:
            pass

        def forward(self, **kwargs):
            question = self.hard_question(**kwargs)
            return (question, self.answer(question=question))

    expected = Answer(
        value=3e8,
        certainty=0.9,
        comments=["It is the speed of light", "It is a constant"],
    )

    lm = DummyLM(
        [
            {"hard_question": "What is the speed of light?"},
            {"reasoning": "Some bad reasoning, 3e8 m/s.", "answer": "3e8"},  # Bad answer 1
            {"json_object": "{...}"},  # Model is asked to create an example
            {"reasoning": "Some good reasoning, 3e8 m/s.", "answer": f"{expected.model_dump_json()}"},  # Good answer
        ]
    )
    dspy.settings.configure(lm=lm)

    qa = QA()
    assert isinstance(qa, FunctionalModule)
    assert isinstance(qa.answer, dspy.Module)

    question, answer = qa(topic="Physics")

    print(qa.answer)

    assert question == "What is the speed of light?"
    assert answer == expected


def test_simple_oop():
    class Question(pydantic.BaseModel):
        value: str

    class MySignature(dspy.Signature):
        topic: str = dspy.InputField()
        output: Question = dspy.OutputField()

    # Run the signature
    program = TypedPredictor(MySignature)
    expected = "What is the speed of light?"
    lm = DummyLM(
        [
            {"output": f"{Question(value=expected).model_dump_json()}"},
        ]
    )
    dspy.settings.configure(lm=lm)

    question = program(topic="Physics").output

    assert isinstance(question, Question)
    assert question.value == expected


def test_equivalent_signatures():
    class ClassSignature(dspy.Signature):
        input: str = dspy.InputField()
        output: str = dspy.OutputField()

    @predictor
    def output(input: str) -> str:
        pass

    function_signature = output.predictor.signature

    simple_signature = dspy.Signature("input -> output")

    assert ClassSignature.equals(function_signature)
    assert ClassSignature.equals(simple_signature)


def test_named_params():
    class QA(FunctionalModule):
        @predictor
        def hard_question(self, topic: str) -> str:
            """Think of a hard factual question about a topic. It should be answerable with a number."""

        @cot
        def answer(self, question: str) -> str:
            pass

    qa = QA()
    named_predictors = list(qa.named_predictors())
    assert len(named_predictors) == 2
    names, _ = zip(*qa.named_predictors())
    assert set(names) == {
        "hard_question.predictor.predictor",
        "answer.predictor.predictor",
    }


def test_bootstrap_effectiveness():
    class SimpleModule(FunctionalModule):
        @predictor
        def output(self, input: str) -> str:
            pass

        def forward(self, **kwargs):
            return self.output(**kwargs)

    def simple_metric(example, prediction, trace=None):
        return example.output == prediction.output

    examples = [
        ex.with_inputs("input")
        for ex in (
            Example(input="What is the color of the sky?", output="blue"),
            Example(
                input="What does the fox say?",
                output="Ring-ding-ding-ding-dingeringeding!",
            ),
        )
    ]
    trainset = [examples[0]]
    valset = [examples[1]]

    # This test verifies if the bootstrapping process improves the student's predictions
    student = SimpleModule()
    teacher = SimpleModule()
    assert student.output.predictor.signature.equals(teacher.output.predictor.signature)

    lm = DummyLM(["blue", "Ring-ding-ding-ding-dingeringeding!"], follow_examples=True)
    dspy.settings.configure(lm=lm, trace=[])

    bootstrap = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    compiled_student = bootstrap.compile(student, teacher=teacher, trainset=trainset)

    # Check that the compiled student has the correct demos
    _, predict = next(compiled_student.named_sub_modules(Predict, skip_compiled=False))
    demos = predict.demos
    assert len(demos) == 1
    assert demos[0].input == trainset[0].input
    assert demos[0].output == trainset[0].output

    # Test the compiled student's prediction.
    # We are using a DummyLM with follow_examples=True, which means that
    # even though it would normally reply with "Ring-ding-ding-ding-dingeringeding!"
    # on the second output, if it seems an example that perfectly matches the
    # prompt, it will use that instead. That is why we expect "blue" here.
    prediction = compiled_student(input=trainset[0].input)
    assert prediction == trainset[0].output


def test_regex():
    class TravelInformation(BaseModel):
        origin: str = Field(pattern=r"^[A-Z]{3}$")
        destination: str = Field(pattern=r"^[A-Z]{3}$")
        date: datetime.date

    @predictor
    def flight_information(email: str) -> TravelInformation:
        pass

    email = textwrap.dedent(
        """\
        We're excited to welcome you aboard your upcoming flight from
        John F. Kennedy International Airport (JFK) to Los Angeles International Airport (LAX)
        on December 25, 2022. Here's everything you need to know before you take off: ...
    """
    )
    lm = DummyLM(
        [
            # Example with a bad origin code.
            {"flight_information": '{"origin": "JF0", "destination": "LAX", "date": "2022-12-25"}'},
            # Example to help the model understand
            {"json_object": "{...}"},
            # Fixed
            {"flight_information": '{"origin": "JFK", "destination": "LAX", "date": "2022-12-25"}'},
        ]
    )
    dspy.settings.configure(lm=lm)

    assert flight_information(email=email) == TravelInformation(
        origin="JFK", destination="LAX", date=datetime.date(2022, 12, 25)
    )


def test_custom_model_validate_json():
    class Airport(BaseModel):
        code: str = Field(pattern=r"^[A-Z]{3}$")
        lat: float
        lon: float

    class TravelInformation(BaseModel):
        origin: Airport
        destination: Airport
        date: datetime.date

        @classmethod
        def model_validate_json(
            cls, json_data: str, *, strict: Optional[bool] = None, context: Optional[dict[str, Any]] = None
        ) -> "TravelInformation":
            try:
                __tracebackhide__ = True
                return cls.__pydantic_validator__.validate_json(json_data, strict=strict, context=context)
            except ValidationError:
                for substring_length in range(len(json_data), 1, -1):
                    for start in range(len(json_data) - substring_length + 1):
                        substring = json_data[start : start + substring_length]
                        try:
                            __tracebackhide__ = True
                            res = cls.__pydantic_validator__.validate_json(substring, strict=strict, context=context)
                            return res
                        except ValidationError as exc:
                            last_exc = exc
                            pass
            raise ValueError("Could not find valid json") from last_exc

    @predictor
    def flight_information(email: str) -> TravelInformation:
        pass

    email = textwrap.dedent(
        """\
        We're excited to welcome you aboard your upcoming flight from
        John F. Kennedy International Airport (JFK) to Los Angeles International Airport (LAX)
        on December 25, 2022. Here's everything you need to know before you take off: ...
    """
    )
    lm = DummyLM(
        [
            # Example with a bad origin code.
            (
                {
                    "flight_information": "Here is your json: {"
                    '"origin": {"code":"JFK", "lat":40.6446, "lon":-73.7797}, '
                    '"destination": {"code":"LAX", "lat":33.942791, "lon":-118.410042}, '
                    '"date": "2022-12-25"}'
                }
            ),
        ]
    )
    dspy.settings.configure(lm=lm)

    assert flight_information(email=email) == TravelInformation(
        origin={"code": "JFK", "lat": 40.6446, "lon": -73.7797},
        destination={"code": "LAX", "lat": 33.942791, "lon": -118.410042},
        date=datetime.date(2022, 12, 25),
    )


def test_raises():
    class TravelInformation(BaseModel):
        origin: str = Field(pattern=r"^[A-Z]{3}$")
        destination: str = Field(pattern=r"^[A-Z]{3}$")
        date: datetime.date

    @predictor
    def flight_information(email: str) -> TravelInformation:
        pass

    lm = DummyLM(
        [
            {"flight_information": '{"origin": "JF0", "destination": "LAX", "date": "2022-12-25"}'},
            {"flight_information": '{"origin": "JFK", "destination": "LAX", "date": "bad date"}'},
        ]
    )
    dspy.settings.configure(lm=lm)

    with pytest.raises(ValueError):
        flight_information(email="Some email")


def test_multi_errors():
    class TravelInformation(BaseModel):
        origin: str = Field(pattern=r"^[A-Z]{3}$")
        destination: str = Field(pattern=r"^[A-Z]{3}$")
        date: datetime.date

    @predictor
    def flight_information(email: str) -> TravelInformation:
        pass

    lm = DummyLM(
        [
            # First origin is wrong, then destination, then all is good
            {"flight_information": '{"origin": "JF0", "destination": "LAX", "date": "2022-12-25"}'},
            {"json_object": "{...}"},  # Example to help the model understand
            {"flight_information": '{"origin": "JFK", "destination": "LA0", "date": "2022-12-25"}'},
            {"json_object": "{...}"},  # Example to help the model understand
            {"flight_information": '{"origin": "JFK", "destination": "LAX", "date": "2022-12-25"}'},
        ]
    )
    dspy.settings.configure(lm=lm)

    assert flight_information(email="Some email") == TravelInformation(
        origin="JFK", destination="LAX", date=datetime.date(2022, 12, 25)
    )


def test_field_validator():
    class UserDetails(BaseModel):
        name: str
        age: int

        @field_validator("name")
        @classmethod
        def validate_name(cls, v):
            if v.upper() != v:
                raise ValueError("Name must be in uppercase.")
            return v

    @predictor
    def get_user_details() -> UserDetails:
        pass

    # Keep making the mistake (lower case name) until we run
    # out of retries.
    lm = DummyLM([{"get_user_details": '{"name": "lower case name", "age": 25}'}] * 10)
    dspy.settings.configure(lm=lm)

    with pytest.raises(ValueError):
        get_user_details()


def test_annotated_field():
    @predictor
    def test(input: Annotated[str, Field(description="description")]) -> Annotated[float, Field(gt=0, lt=1)]:
        pass

    # First try 0, which fails, then try 0.5, which passes
    lm = DummyLM([{"test": "0"}, {"test": "0.5"}])
    dspy.settings.configure(lm=lm)

    output = test(input="input")

    assert output == 0.5


def test_multiple_outputs():
    lm = DummyLM([{"output": f"{i}"} for i in range(100)])
    dspy.settings.configure(lm=lm)

    test = TypedPredictor("input -> output")
    output = test(input="input", config=dict(n=3)).completions.output
    assert output == ["0", "1", "2"]


def test_multiple_outputs_int():
    lm = DummyLM([{"output": f"{i}"} for i in range(100)])
    dspy.settings.configure(lm=lm)

    class TestSignature(dspy.Signature):
        input: int = dspy.InputField()
        output: int = dspy.OutputField()

    test = TypedPredictor(TestSignature)

    output = test(input=8, config=dict(n=3)).completions.output
    assert output == [0, 1, 2]


def test_multiple_outputs_int_cot():
    # Note: Multiple outputs only work when the language model "speculatively" generates all the outputs in one go.
    lm = DummyLM(
        [
            {"reasoning": "thoughts 0", "output": "0"},
            {"reasoning": "thoughts 1", "output": "1"},
            {"reasoning": "thoughts 2", "output": "2"},
        ]
    )
    dspy.settings.configure(lm=lm)

    test = TypedChainOfThought("input:str -> output:int")

    output = test(input="8", config=dict(n=3)).completions.output
    assert output == [0, 1, 2]


def test_parse_type_string():
    lm = DummyLM([{"output": f"{i}"} for i in range(100)])
    dspy.settings.configure(lm=lm)

    test = TypedPredictor("input:int -> output:int")

    output = test(input=8, config=dict(n=3)).completions.output
    assert output == [0, 1, 2]


def test_literal():
    lm = DummyLM([{"f": '"2"'}, {"f": '"3"'}])
    dspy.settings.configure(lm=lm)

    @predictor
    def f() -> Literal["2", "3"]:
        pass

    assert f() == "2"


def test_literal_missmatch():
    lm = DummyLM([{"f": f"{i}"} for i in range(5, 100)])
    dspy.settings.configure(lm=lm)

    @predictor(max_retries=1)
    def f() -> Literal["2", "3"]:
        pass

    with pytest.raises(Exception) as e_info:
        f()

    assert e_info.value.args[1]["f"] == "Input should be '2' or '3':  (error type: literal_error)"


def test_literal_int():
    lm = DummyLM([{"f": "2"}, {"f": "3"}])
    dspy.settings.configure(lm=lm)

    @predictor
    def f() -> Literal[2, 3]:
        pass

    assert f() == 2


def test_literal_int_missmatch():
    lm = DummyLM([{"f": f"{i}"} for i in range(5, 100)])
    dspy.settings.configure(lm=lm)

    @predictor(max_retries=1)
    def f() -> Literal[2, 3]:
        pass

    with pytest.raises(Exception) as e_info:
        f()

    assert e_info.value.args[1]["f"] == "Input should be 2 or 3:  (error type: literal_error)"


def test_fields_on_base_signature():
    class SimpleOutput(dspy.Signature):
        output: float = dspy.OutputField(gt=0, lt=1)

    lm = DummyLM(
        [
            {"output": "2.1"},  # Bad output
            {"output": "0.5"},  # Good output
        ]
    )
    dspy.settings.configure(lm=lm)

    predictor = TypedPredictor(SimpleOutput)

    assert predictor().output == 0.5


def test_synthetic_data_gen():
    class SyntheticFact(BaseModel):
        fact: str = Field(..., description="a statement")
        varacity: bool = Field(..., description="is the statement true or false")

    class ExampleSignature(dspy.Signature):
        """Generate an example of a synthetic fact."""

        fact: SyntheticFact = dspy.OutputField()

    lm = DummyLM(
        [
            {"fact": '{"fact": "The sky is blue", "varacity": true}'},
            {"fact": '{"fact": "The sky is green", "varacity": false}'},
            {"fact": '{"fact": "The sky is red", "varacity": true}'},
            {"fact": '{"fact": "The earth is flat", "varacity": false}'},
            {"fact": '{"fact": "The earth is round", "varacity": true}'},
            {"fact": '{"fact": "The earth is a cube", "varacity": false}'},
        ]
    )
    dspy.settings.configure(lm=lm)

    generator = TypedPredictor(ExampleSignature)
    examples = generator(config=dict(n=3))
    for ex in examples.completions.fact:
        assert isinstance(ex, SyntheticFact)
    assert examples.completions.fact[0] == SyntheticFact(fact="The sky is blue", varacity=True)

    # If you have examples and want more
    existing_examples = [
        dspy.Example(fact="The sky is blue", varacity=True),
        dspy.Example(fact="The sky is green", varacity=False),
    ]
    trained = LabeledFewShot().compile(student=generator, trainset=existing_examples)

    augmented_examples = trained(config=dict(n=3))
    for ex in augmented_examples.completions.fact:
        assert isinstance(ex, SyntheticFact)


def test_list_input2():
    # Inspired by the Signature Optimizer

    class ScoredString(pydantic.BaseModel):
        string: str
        score: float

    class ScoredSignature(dspy.Signature):
        attempted_signatures: list[ScoredString] = dspy.InputField()
        proposed_signature: str = dspy.OutputField()

    program = TypedChainOfThought(ScoredSignature)

    lm = DummyLM([{"reasoning": "Thoughts", "proposed_signature": "Output"}])
    dspy.settings.configure(lm=lm)

    output = program(
        attempted_signatures=[
            ScoredString(string="string 1", score=0.5),
            ScoredString(string="string 2", score=0.4),
            ScoredString(string="string 3", score=0.3),
        ]
    ).proposed_signature

    print(lm.get_convo(-1))

    assert output == "Output"


def test_custom_reasoning_field():
    class Question(pydantic.BaseModel):
        value: str

    class QuestionSignature(dspy.Signature):
        topic: str = dspy.InputField()
        question: Question = dspy.OutputField()

    reasoning = dspy.OutputField(
        prefix="Custom Reasoning: Let's break this down. To generate a question about",
        desc="${topic}, we should ...",
    )

    program = TypedChainOfThought(QuestionSignature, reasoning=reasoning)

    expected = "What is the speed of light?"
    lm = DummyLM([{"reasoning": "Thoughts", "question": f'{{"value": "{expected}"}}'}])
    dspy.settings.configure(lm=lm)

    output = program(topic="Physics")

    assert isinstance(output.question, Question)
    assert output.question.value == expected


def test_generic_signature():
    T = TypeVar("T")

    class GenericSignature(dspy.Signature, Generic[T]):
        """My signature"""

        output: T = dspy.OutputField()

    predictor = TypedPredictor(GenericSignature[int])
    assert predictor.signature.instructions == "My signature"

    lm = DummyLM([{"output": "23"}])
    dspy.settings.configure(lm=lm)

    assert predictor().output == 23


def test_field_validator_in_signature():
    class ValidatedSignature(dspy.Signature):
        a: str = dspy.OutputField()

        @pydantic.field_validator("a")
        @classmethod
        def space_in_a(cls, a: str) -> str:
            if " " not in a:
                raise ValueError("a must contain a space")
            return a

    with pytest.raises(pydantic.ValidationError):
        _ = ValidatedSignature(a="no-space")

    _ = ValidatedSignature(a="with space")


def test_lm_as_validator():
    @predictor
    def is_square(n: int) -> bool:
        """Is n a square number?"""

    def check_square(n):
        assert is_square(n=n)
        return n

    @predictor
    def next_square(n: int) -> Annotated[int, AfterValidator(check_square)]:
        """What is the next square number after n?"""

    lm = DummyLM(
        [
            {"next_square": "3"},
            {"is_square": "False"},
            {"next_square": "4"},
            {"is_square": "True"},
        ]
    )
    dspy.settings.configure(lm=lm)

    m = next_square(n=2)

    assert m == 4


def test_annotated_validator():
    def is_square(n: int) -> int:
        root = n**0.5
        if not root.is_integer():
            raise ValueError(f"{n} is not a square")
        return n

    class MySignature(dspy.Signature):
        """What is the next square number after n?"""

        n: int = dspy.InputField()
        next_square: Annotated[int, AfterValidator(is_square)] = dspy.OutputField()

    lm = DummyLM([{"next_square": "3"}, {"next_square": "4"}])
    dspy.settings.configure(lm=lm)

    m = TypedPredictor(MySignature)(n=2).next_square
    lm.inspect_history(n=2)

    assert m == 4


def test_annotated_validator_functional():
    def is_square(n: int) -> int:
        if not (n**0.5).is_integer():
            raise ValueError(f"{n} is not a square")
        return n

    @predictor
    def next_square(n: int) -> Annotated[int, AfterValidator(is_square)]:
        """What is the next square number after n?"""

    lm = DummyLM([{"next_square": "3"}, {"next_square": "4"}])
    dspy.settings.configure(lm=lm)

    m = next_square(n=2)
    lm.inspect_history(n=2)

    assert m == 4


def test_demos():
    demos = [
        dspy.Example(input="What is the speed of light?", output="3e8"),
    ]
    program = LabeledFewShot(k=len(demos)).compile(
        student=dspy.TypedPredictor("input -> output"),
        trainset=[ex.with_inputs("input") for ex in demos],
    )

    lm = DummyLM([{"output": "Paris"}])
    dspy.settings.configure(lm=lm)

    assert program(input="What is the capital of France?").output == "Paris"


def test_demos_missing_input_in_demo():
    demos = [dspy.Example(input="What is the speed of light?", output="3e8")]
    program = LabeledFewShot(k=len(demos)).compile(
        student=dspy.TypedPredictor("input -> output, thoughts"),
        trainset=[ex.with_inputs("input") for ex in demos],
    )
    lm = DummyLM([{"thoughts": "My thoughts", "output": "Paris"}])
    dspy.settings.configure(lm=lm)
    assert program(input="What is the capital of France?").output == "Paris"


def test_conlist():
    dspy.settings.configure(
        lm=DummyLM(
            [
                {"make_numbers": "[]"},
                {"make_numbers": "[1]"},
                {"make_numbers": "[1, 2]"},
                {"make_numbers": "[1, 2, 3]"},
            ]
        )
    )

    @predictor
    def make_numbers(input: str) -> Annotated[list[int], Field(min_items=2)]:
        pass

    assert make_numbers(input="What are the first two numbers?") == [1, 2]


def test_conlist2():
    dspy.settings.configure(
        lm=DummyLM(
            [
                {"output": "[]"},
                {"output": "[1]"},
                {"output": "[1, 2]"},
                {"output": "[1, 2, 3]"},
            ]
        )
    )

    make_numbers = TypedPredictor("input:str -> output:Annotated[List[int], Field(min_items=2)]")
    assert make_numbers(input="What are the first two numbers?").output == [1, 2]


def test_model_validator():
    class MySignature(dspy.Signature):
        input_data: str = dspy.InputField()
        allowed_categories: list[str] = dspy.InputField()
        category: str = dspy.OutputField()

        @model_validator(mode="after")
        def check_cateogry(self):
            if self.category not in self.allowed_categories:
                raise ValueError(f"category not in {self.allowed_categories}")
            return self

    lm = DummyLM([{"category": "horse"}, {"category": "dog"}])
    dspy.settings.configure(lm=lm)
    predictor = TypedPredictor(MySignature)

    pred = predictor(input_data="What is the best animal?", allowed_categories=["cat", "dog"])
    assert pred.category == "dog"
