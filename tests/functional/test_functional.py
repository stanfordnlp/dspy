import datetime
import textwrap
import pydantic
from pydantic import Field, BaseModel, field_validator
from typing import Annotated
from typing import List

import pytest

import dspy
from dspy.functional import predictor, cot, FunctionalModule, TypedPredictor, functional
from dspy.primitives.example import Example
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.utils.dummies import DummyLM


def test_simple():
    @predictor
    def hard_question(topic: str) -> str:
        """Think of a hard factual question about a topic."""

    expected = "What is the speed of light?"
    lm = DummyLM([expected])
    dspy.settings.configure(lm=lm)

    question = hard_question(topic="Physics")
    lm.inspect_history(n=2)

    assert question == expected


def test_list_output():
    @predictor
    def hard_questions(topics: List[str]) -> List[str]:
        pass

    expected = ["What is the speed of light?", "What is the speed of sound?"]
    lm = DummyLM(
        ['{"value": ["What is the speed of light?", "What is the speed of sound?"]}']
    )
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
    lm = DummyLM([f'{{"value": "{expected}"}}'])
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
    lm = DummyLM([f'{{"value": "3e8"}}'])
    dspy.settings.configure(lm=lm)

    result = answer(question=question)

    assert result == Answer(value="3e8")


def test_simple_class():
    class Answer(pydantic.BaseModel):
        value: float
        certainty: float
        comments: List[str] = pydantic.Field(
            description="At least two comments about the answer"
        )

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
            "What is the speed of light?",
            "Some bad reasoning, 3e8 m/s.",
            "3e8",  # Bad answer 1
            "{...}",  # Model is asked to create an example
            "Some good reasoning...",
            expected.model_dump_json(),  # Good answer
        ]
    )
    dspy.settings.configure(lm=lm)

    qa = QA()
    assert isinstance(qa, FunctionalModule)
    assert isinstance(qa.answer, functional._StripOutput)

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
            Question(value=expected).model_dump_json(),
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

    bootstrap = BootstrapFewShot(
        metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1
    )
    compiled_student = bootstrap.compile(student, teacher=teacher, trainset=trainset)

    lm.inspect_history(n=2)

    # Check that the compiled student has the correct demos
    demos = compiled_student.predictors()[0].demos
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

    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields `input`, produce the fields `output`.

        ---

        Follow the following format.

        Input: ${input}
        Output: ${output}. Respond with a single str value

        ---

        Input: What is the color of the sky?
        Output: blue

        ---

        Input: What is the color of the sky?
        Output: blue"""
    )


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
            '{"origin": "JF0", "destination": "LAX", "date": "2022-12-25"}',
            # Example to help the model understand
            '{...}',
            # Fixed
            '{"origin": "JFK", "destination": "LAX", "date": "2022-12-25"}',
        ]
    )
    dspy.settings.configure(lm=lm)

    assert flight_information(email=email) == TravelInformation(
        origin="JFK", destination="LAX", date=datetime.date(2022, 12, 25)
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
            "A list of bad inputs",
            '{"origin": "JF0", "destination": "LAX", "date": "2022-12-25"}',
            '{"origin": "JFK", "destination": "LAX", "date": "bad date"}',
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
            '{"origin": "JF0", "destination": "LAX", "date": "2022-12-25"}',
            '{...}', # Example to help the model understand
            '{"origin": "JFK", "destination": "LA0", "date": "2022-12-25"}',
            '{...}', # Example to help the model understand
            '{"origin": "JFK", "destination": "LAX", "date": "2022-12-25"}',
        ]
    )
    dspy.settings.configure(lm=lm)

    assert flight_information(email="Some email") == TravelInformation(
        origin="JFK", destination="LAX", date=datetime.date(2022, 12, 25)
    )
    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields `email`, produce the fields `flight_information`.

        ---

        Follow the following format.

        Email: ${email}

        Past Error (flight_information): An error to avoid in the future

        Past Error (flight_information, 2): An error to avoid in the future

        Flight Information: ${flight_information}. Respond with a single JSON object. JSON Schema: {"properties": {"origin": {"pattern": "^[A-Z]{3}$", "title": "Origin", "type": "string"}, "destination": {"pattern": "^[A-Z]{3}$", "title": "Destination", "type": "string"}, "date": {"format": "date", "title": "Date", "type": "string"}}, "required": ["origin", "destination", "date"], "title": "TravelInformation", "type": "object"}

        ---

        Email: Some email

        Past Error (flight_information): String should match pattern '^[A-Z]{3}$': origin (error type: string_pattern_mismatch)

        Past Error (flight_information, 2): String should match pattern '^[A-Z]{3}$': destination (error type: string_pattern_mismatch)

        Flight Information: {"origin": "JFK", "destination": "LAX", "date": "2022-12-25"}"""
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
    lm = DummyLM(
        [
            '{"name": "lower case name", "age": 25}',
        ]
        * 10
    )
    dspy.settings.configure(lm=lm)

    with pytest.raises(ValueError):
        get_user_details()

    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields , produce the fields `get_user_details`.

        ---

        Follow the following format.

        Past Error (get_user_details): An error to avoid in the future
        Past Error (get_user_details, 2): An error to avoid in the future
        Get User Details: ${get_user_details}. Respond with a single JSON object. JSON Schema: {"properties": {"name": {"title": "Name", "type": "string"}, "age": {"title": "Age", "type": "integer"}}, "required": ["name", "age"], "title": "UserDetails", "type": "object"}

        ---

        Past Error (get_user_details): Value error, Name must be in uppercase.: name (error type: value_error)
        Past Error (get_user_details, 2): Value error, Name must be in uppercase.: name (error type: value_error)
        Get User Details: {"name": "lower case name", "age": 25}"""
    )


def test_annotated_field():
    # Since we don't currently validate fields on the main signature,
    # the annotated fields are also not validated.
    # But at least it should not crash.

    @predictor
    def test(input: Annotated[str, Field(description="description")]) -> Annotated[float, Field(gt=0, lt=1)]:
        pass

    lm = DummyLM(["0.5"])
    dspy.settings.configure(lm=lm)

    output = test(input="input")

    assert output == 0.5
