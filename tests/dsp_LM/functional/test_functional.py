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
from dspy.utils.dummies import DSPDummyLM


def test_simple():
    @predictor
    def hard_question(topic: str) -> str:
        """Think of a hard factual question about a topic."""

    expected = "What is the speed of light?"
    lm = DSPDummyLM([expected])
    dspy.settings.configure(lm=lm)

    question = hard_question(topic="Physics")
    lm.inspect_history(n=2)

    assert question == expected


def test_list_output():
    @predictor
    def hard_questions(topics: List[str]) -> List[str]:
        pass

    expected = ["What is the speed of light?", "What is the speed of sound?"]
    lm = DSPDummyLM(['["What is the speed of light?", "What is the speed of sound?"]'])
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
    lm = DSPDummyLM([f'{{"value": "{expected}"}}'])
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
    lm = DSPDummyLM([f'{{"value": "3e8"}}'])
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

    lm = DSPDummyLM(
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
    lm = DSPDummyLM(
        [
            Question(value=expected).model_dump_json(),
        ]
    )
    dspy.settings.configure(lm=lm)

    question = program(topic="Physics").output

    assert isinstance(question, Question)
    assert question.value == expected


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

    lm = DSPDummyLM(["blue", "Ring-ding-ding-ding-dingeringeding!"], follow_examples=True)
    dspy.settings.configure(lm=lm, trace=[])

    bootstrap = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    compiled_student = bootstrap.compile(student, teacher=teacher, trainset=trainset)

    lm.inspect_history(n=2)

    # Check that the compiled student has the correct demos
    _, predict = next(compiled_student.named_sub_modules(Predict, skip_compiled=False))
    demos = predict.demos
    assert len(demos) == 1
    assert demos[0].input == trainset[0].input
    assert demos[0].output == trainset[0].output

    # Test the compiled student's prediction.
    # We are using a DSPDummyLM with follow_examples=True, which means that
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
        Output: ${output}

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
    lm = DSPDummyLM(
        [
            # Example with a bad origin code.
            '{"origin": "JF0", "destination": "LAX", "date": "2022-12-25"}',
            # Example to help the model understand
            "{...}",
            # Fixed
            '{"origin": "JFK", "destination": "LAX", "date": "2022-12-25"}',
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
    lm = DSPDummyLM(
        [
            # Example with a bad origin code.
            (
                "Here is your json: "
                "{"
                '"origin": {"code":"JFK", "lat":40.6446, "lon":-73.7797}, '
                '"destination": {"code":"LAX", "lat":33.942791, "lon":-118.410042}, '
                '"date": "2022-12-25"}'
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

    lm = DSPDummyLM(
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

    lm = DSPDummyLM(
        [
            # First origin is wrong, then destination, then all is good
            '{"origin": "JF0", "destination": "LAX", "date": "2022-12-25"}',
            "{...}",  # Example to help the model understand
            '{"origin": "JFK", "destination": "LA0", "date": "2022-12-25"}',
            "{...}",  # Example to help the model understand
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

        Past Error in Flight Information: An error to avoid in the future

        Past Error (2) in Flight Information: An error to avoid in the future

        Flight Information: ${flight_information}. Respond with a single JSON object. JSON Schema: {"properties": {"origin": {"pattern": "^[A-Z]{3}$", "title": "Origin", "type": "string"}, "destination": {"pattern": "^[A-Z]{3}$", "title": "Destination", "type": "string"}, "date": {"format": "date", "title": "Date", "type": "string"}}, "required": ["origin", "destination", "date"], "title": "TravelInformation", "type": "object"}

        ---

        Email: Some email

        Past Error in Flight Information: String should match pattern '^[A-Z]{3}$': origin (error type: string_pattern_mismatch)

        Past Error (2) in Flight Information: String should match pattern '^[A-Z]{3}$': destination (error type: string_pattern_mismatch)

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
    lm = DSPDummyLM(
        [
            '{"name": "lower case name", "age": 25}',
        ]
        * 10
    )
    dspy.settings.configure(lm=lm)

    with pytest.raises(ValueError):
        get_user_details()

    print(lm.get_convo(-1))
    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields , produce the fields `get_user_details`.

        ---

        Follow the following format.

        Past Error in Get User Details: An error to avoid in the future
        Past Error (2) in Get User Details: An error to avoid in the future
        Get User Details: ${get_user_details}. Respond with a single JSON object. JSON Schema: {"properties": {"name": {"title": "Name", "type": "string"}, "age": {"title": "Age", "type": "integer"}}, "required": ["name", "age"], "title": "UserDetails", "type": "object"}

        ---

        Past Error in Get User Details: Value error, Name must be in uppercase.: name (error type: value_error)
        Past Error (2) in Get User Details: Value error, Name must be in uppercase.: name (error type: value_error)
        Get User Details: {"name": "lower case name", "age": 25}"""
    )


def test_annotated_field():
    @predictor
    def test(input: Annotated[str, Field(description="description")]) -> Annotated[float, Field(gt=0, lt=1)]:
        pass

    # First try 0, which fails, then try 0.5, which passes
    lm = DSPDummyLM(["0", "0.5"])
    dspy.settings.configure(lm=lm)

    output = test(input="input")

    assert output == 0.5


def test_multiple_outputs():
    lm = DSPDummyLM([str(i) for i in range(100)])
    dspy.settings.configure(lm=lm)

    test = TypedPredictor("input -> output")
    output = test(input="input", config=dict(n=3)).completions.output
    assert output == ["0", "1", "2"]


def test_multiple_outputs_int():
    lm = DSPDummyLM([str(i) for i in range(100)])
    dspy.settings.configure(lm=lm)

    class TestSignature(dspy.Signature):
        input: int = dspy.InputField()
        output: int = dspy.OutputField()

    test = TypedPredictor(TestSignature)

    output = test(input=8, config=dict(n=3)).completions.output
    assert output == [0, 1, 2]


def test_multiple_outputs_int_cot():
    # Note: Multiple outputs only work when the language model "speculatively" generates all the outputs in one go.
    lm = DSPDummyLM(
        [
            "thoughts 0\nOutput: 0\n",
            "thoughts 1\nOutput: 1\n",
            "thoughts 2\nOutput: 2\n",
        ]
    )
    dspy.settings.configure(lm=lm)

    test = TypedChainOfThought("input:str -> output:int")

    output = test(input="8", config=dict(n=3)).completions.output
    assert output == [0, 1, 2]


def test_parse_type_string():
    lm = DSPDummyLM([str(i) for i in range(100)])
    dspy.settings.configure(lm=lm)

    test = TypedPredictor("input:int -> output:int")

    output = test(input=8, config=dict(n=3)).completions.output
    assert output == [0, 1, 2]


def test_literal():
    lm = DSPDummyLM(['"2"', '"3"'])
    dspy.settings.configure(lm=lm)

    @predictor
    def f() -> Literal["2", "3"]:
        pass

    assert f() == "2"


def test_literal_missmatch():
    lm = DSPDummyLM([f'"{i}"' for i in range(5, 100)])
    dspy.settings.configure(lm=lm)

    @predictor(max_retries=1)
    def f() -> Literal["2", "3"]:
        pass

    with pytest.raises(Exception) as e_info:
        f()

    assert e_info.value.args[1]["f"] == "Input should be '2' or '3':  (error type: literal_error)"


def test_literal_int():
    lm = DSPDummyLM(["2", "3"])
    dspy.settings.configure(lm=lm)

    @predictor
    def f() -> Literal[2, 3]:
        pass

    assert f() == 2


def test_literal_int_missmatch():
    lm = DSPDummyLM([f"{i}" for i in range(5, 100)])
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

    lm = DSPDummyLM(
        [
            "2.1",  # Bad output
            "0.5",  # Good output
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

    lm = DSPDummyLM(
        [
            '{"fact": "The sky is blue", "varacity": true}',
            '{"fact": "The sky is green", "varacity": false}',
            '{"fact": "The sky is red", "varacity": true}',
            '{"fact": "The earth is flat", "varacity": false}',
            '{"fact": "The earth is round", "varacity": true}',
            '{"fact": "The earth is a cube", "varacity": false}',
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

    lm = DSPDummyLM(["Thoughts", "Output"])
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

    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields `attempted_signatures`, produce the fields `proposed_signature`.

        ---

        Follow the following format.

        Attempted Signatures: ${attempted_signatures}
        Reasoning: Let's think step by step in order to ${produce the proposed_signature}. We ...
        Proposed Signature: ${proposed_signature}

        ---

        Attempted Signatures: [{"string":"string 1","score":0.5},{"string":"string 2","score":0.4},{"string":"string 3","score":0.3}]
        Reasoning: Let's think step by step in order to Thoughts
        Proposed Signature: Output"""
    )


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
    lm = DSPDummyLM(["Thoughts", f'{{"value": "{expected}"}}'])
    dspy.settings.configure(lm=lm)

    output = program(topic="Physics")

    assert isinstance(output.question, Question)
    assert output.question.value == expected

    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields `topic`, produce the fields `question`.

        ---

        Follow the following format.

        Topic: ${topic}
        Custom Reasoning: Let's break this down. To generate a question about ${topic}, we should ...
        Question: ${question}. Respond with a single JSON object. JSON Schema: {"properties": {"value": {"title": "Value", "type": "string"}}, "required": ["value"], "title": "Question", "type": "object"}

        ---

        Topic: Physics
        Custom Reasoning: Let's break this down. To generate a question about Thoughts
        Question: {"value": "What is the speed of light?"}"""
    )


def test_generic_signature():
    T = TypeVar("T")

    class GenericSignature(dspy.Signature, Generic[T]):
        """My signature"""

        output: T = dspy.OutputField()

    predictor = TypedPredictor(GenericSignature[int])
    assert predictor.signature.instructions == "My signature"

    lm = DSPDummyLM(["23"])
    dspy.settings.configure(lm=lm)

    assert predictor().output == 23


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

    lm = DSPDummyLM(["3", "False", "4", "True"])
    dspy.settings.configure(lm=lm)

    m = next_square(n=2)
    lm.inspect_history(n=2)

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

    lm = DSPDummyLM(["3", "4"])
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

    lm = DSPDummyLM(["3", "4"])
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

    lm = DSPDummyLM(["Paris"])
    dspy.settings.configure(lm=lm)

    assert program(input="What is the capital of France?").output == "Paris"

    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields `input`, produce the fields `output`.

        ---

        Follow the following format.

        Input: ${input}
        Output: ${output}

        ---

        Input: What is the speed of light?
        Output: 3e8

        ---

        Input: What is the capital of France?
        Output: Paris"""
    )


def _test_demos_missing_input():
    demos = [dspy.Example(input="What is the speed of light?", output="3e8")]
    program = LabeledFewShot(k=len(demos)).compile(
        student=dspy.TypedPredictor("input -> output, thoughts"),
        trainset=[ex.with_inputs("input") for ex in demos],
    )
    dspy.settings.configure(lm=DSPDummyLM(["My thoughts", "Paris"]))
    assert program(input="What is the capital of France?").output == "Paris"

    assert dspy.settings.lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields `input`, produce the fields `output`.

        ---

        Follow the following format.

        Input: ${input}
        Thoughts: ${thoughts}
        Output: ${output}

        ---

        Input: What is the speed of light?
        Output: 3e8

        ---

        Input: What is the capital of France?
        Thoughts: My thoughts
        Output: Paris"""
    )


def test_conlist():
    dspy.settings.configure(lm=DSPDummyLM(["[]", "[1]", "[1, 2]", "[1, 2, 3]"]))

    @predictor
    def make_numbers(input: str) -> Annotated[list[int], Field(min_items=2)]:
        pass

    assert make_numbers(input="What are the first two numbers?") == [1, 2]


def test_conlist2():
    dspy.settings.configure(lm=DSPDummyLM(["[]", "[1]", "[1, 2]", "[1, 2, 3]"]))

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

    lm = DSPDummyLM(["horse", "dog"])
    dspy.settings.configure(lm=lm)
    predictor = TypedPredictor(MySignature)

    pred = predictor(input_data="What is the best animal?", allowed_categories=["cat", "dog"])
    assert pred.category == "dog"
