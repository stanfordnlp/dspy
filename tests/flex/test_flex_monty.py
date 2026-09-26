"""Real Monty execution of unchanged Flex programs; no host exec or fallback."""

import json
import textwrap
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pydantic
import pytest

import dspy
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import LMRateLimitError
from tests.flex.test_flex_interpreter import (
    CODEACT_MODULE,
    COT_GLUE_MODULE,
    LM_FAILURE_MODULE,
    REACT_MODULE,
    RECOVERED_LM_FAILURE_MODULE,
    RUNAWAY_MODULE,
    TOOL_GLUE_MODULE,
    Doubler,
    ShoutSig,
    _evil_read_module,
    _RateLimitedLM,
    deno_required,
    shout,
)

pytest.importorskip("pydantic_monty")


@pytest.fixture(params=[dspy.MontyInterpreter, pytest.param(dspy.PythonInterpreter, marks=deno_required)])
def backend(request):
    return request.param


def program(source, signature=Doubler, interpreter_factory=dspy.MontyInterpreter, **kwargs):
    flex = dspy.Flex(signature, interpreter_factory=interpreter_factory, **kwargs)
    flex._bind_code(textwrap.dedent(source).strip())
    return flex


def test_cot_and_original_source_survive_save_load(tmp_path):
    flex = program(COT_GLUE_MODULE)
    path = tmp_path / "flex.json"
    flex.save(path)
    assert json.loads(path.read_text())["module_src"] == COT_GLUE_MODULE
    restored = dspy.Flex(Doubler, interpreter_factory=lambda: dspy.MontyInterpreter())
    restored.load(path)
    with dspy.context(lm=DummyLM([{"reasoning": "double", "result": "the answer is 42"}])):
        assert restored(value=21).result == 42
    assert restored.module_src == COT_GLUE_MODULE


@pytest.mark.parametrize(
    "source,responses",
    [
        (TOOL_GLUE_MODULE, []),
        (
            CODEACT_MODULE,
            [
                {"generated_code": "print(shout('hello'))", "finished": True},
                {"reasoning": "done", "out": "HELLO"},
            ],
        ),
        (
            REACT_MODULE,
            [
                {"next_thought": "tool", "next_tool_name": "shout", "next_tool_args": {"text": "hello"}},
                {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
                {"reasoning": "done", "out": "HELLO"},
            ],
        ),
    ],
)
def test_existing_tool_programs(source, responses):
    flex = program(source, ShoutSig, tools=[shout])
    with dspy.context(lm=DummyLM(responses)):
        assert flex(text="hello").out == "HELLO"


def test_configured_backend_runs_default_tool_rlm_baseline():
    flex = dspy.Flex(ShoutSig, tools=[shout])
    with dspy.context(
        interpreter_factory=dspy.MontyInterpreter,
        lm=DummyLM([{"reasoning": "use tool", "code": "SUBMIT(out=shout('hello'))"}]),
    ):
        assert flex(text="hello").out == "HELLO"


def test_computed_signatures_prediction_fields_and_helpers(backend):
    flex = program(
        """
        class M(dspy.Module):
            def __init__(self):
                super().__init__()
                field = "number"
                sig = dspy.Signature("value: int -> " + field + ": int", "Find the number")
                self.solve = dspy.ChainOfThought(sig)

            def forward(self, value):
                def unpack(prediction):
                    assert prediction.reasoning == "computed"
                    assert prediction["number"] == prediction.number
                    assert getattr(prediction, "missing", 19) == 19
                    return prediction.number
                predictions = [self.solve(value=value)]
                fields = {"result": unpack(predictions[0]) + len([1, 2, 3, 4][1::2])}
                return dspy.Prediction(**fields)
    """,
        interpreter_factory=backend,
    )
    with dspy.context(lm=DummyLM([{"reasoning": "computed", "number": 37}])):
        assert flex(value=11).result == 39


def test_assignment_order_and_state_are_preserved(backend):
    events = []

    def record(value: int) -> int:
        events.append(value)
        return value

    flex = program(
        """
        class M(dspy.Module):
            def __init__(self):
                super().__init__(record(1))
                self.left = self.right = record(2)
                self.a, (self.b, *self.c) = [3, [5, 7, 11]]
                self.calls = 0

            def forward(self, value):
                self.calls += 1
                def helper(x):
                    return self.a + self.b + sum(self.c) + self.left + self.right + x
                return dspy.Prediction(result=helper(self.calls))
    """,
        tools=[record],
        interpreter_factory=backend,
    )
    assert flex(value=0).result == 31
    assert flex(value=0).result == 31
    assert events == [1, 2, 1, 2]


@pytest.mark.parametrize(
    "source,error,match",
    [
        (LM_FAILURE_MODULE, LMRateLimitError, "429"),
        (RECOVERED_LM_FAILURE_MODULE, dspy.CodeExecutionError, "oops"),
    ],
)
def test_lm_error_identity_and_recovery(source, error, match):
    with dspy.context(lm=_RateLimitedLM([])):
        with pytest.raises(error, match=match):
            program(source)(value=2)


def test_predictor_call_budget():
    with dspy.context(lm=DummyLM([{"reasoning": "r", "result": "1"}] * 4)):
        with pytest.raises(dspy.CodeExecutionError, match="budget"):
            program(RUNAWAY_MODULE, max_predictor_calls=2)(value=3)


def test_host_files_are_not_exposed(tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("host-only secret")
    flex = program(_evil_read_module(str(secret)), "value: int -> result: str")
    result = flex(value=1).result
    assert result.startswith("BLOCKED")
    assert "host-only secret" not in result


def test_nested_rlm_sessions_and_retry_keep_correct_state():
    workers = []

    class TrackedMonty(dspy.MontyInterpreter):
        def start(self):
            super().start()
            if self._session.worker_pid not in workers:
                workers.append(self._session.worker_pid)

    def spawn_inner(task: str) -> str:
        return dspy.RLM("task -> out")(task=task).out

    flex = dspy.Flex(ShoutSig, tools=[spawn_inner])
    flex._bind_code(
        textwrap.dedent("""
        class M(dspy.Module):
            def __init__(self):
                super().__init__()
                self.agent = dspy.RLM("text -> out: int", tools=[spawn_inner])
            def forward(self, **inputs):
                n = 100
                r = self.agent(**inputs)
                return dspy.Prediction(out=str(r.out) + ":" + str(n))
    """).strip()
    )
    responses = [
        {"reasoning": "initialize", "code": "n = 7\nprint(spawn_inner(task='go'))\nSUBMIT(out='invalid')"},
        {"reasoning": "inner", "code": "n = 23\nSUBMIT(out=str(n))"},
        {"reasoning": "fix output", "code": "SUBMIT(out=n + 4)"},
    ]
    with dspy.context(interpreter_factory=TrackedMonty, lm=DummyLM(responses)):
        assert flex(text="hello").out == "11:100"
    assert len(workers) == 3


def test_parallel_forwards_have_fresh_instances():
    flex = program("""
        class M(dspy.Module):
            def __init__(self):
                super().__init__()
                self.values = []
            def forward(self, value):
                self.values.append(value)
                return dspy.Prediction(result=sum(self.values))
    """)
    with ThreadPoolExecutor(max_workers=3) as executor:
        assert list(executor.map(lambda value: flex(value=value).result, [3, 11, 29])) == [3, 11, 29]


def test_typed_outputs_and_custom_input_identity():
    class Person(pydantic.BaseModel):
        name: str
        age: int

    class Signature(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        person: Person = dspy.OutputField()
        count: int = dspy.OutputField(default=17)

    image = dspy.Image(url="https://example.com/image.png")
    seen = []

    def inspect_image(image: dspy.Image) -> str:
        seen.append(image)
        return "Ada, age 36"

    flex = program(
        """
        class M(dspy.Module):
            def __init__(self):
                super().__init__()
                self.extract = dspy.Predict("text -> person: Person")
            def forward(self, image):
                result = self.extract(text=inspect_image(image))
                return dspy.Prediction(person=result.person)
    """,
        Signature,
        tools=[inspect_image],
    )
    with dspy.context(lm=DummyLM([{"person": {"name": "Ada", "age": 36}}])):
        result = flex(image=image)
    assert result.person == Person(name="Ada", age=36)
    assert result.count == 17
    assert seen == [image]
    assert seen[0] is image


def test_authored_functions_cannot_become_host_tools():
    flex = program("""
        class M(dspy.Module):
            def __init__(self):
                super().__init__()
                def local_tool(x):
                    return x + 1
                self.agent = dspy.RLM("value -> result", tools=[local_tool])
            def forward(self, value):
                return self.agent(value=value)
    """)
    with pytest.raises(dspy.CodeExecutionError, match="Only provided tools"):
        flex(value=5)


@pytest.mark.parametrize(
    "imports,base,initializer,binding",
    [
        ("", "dspy.Module", "super().__init__()", 'setattr(self, "solve", pending)'),
        (
            "import dspy as ds\nBase = ds.Module",
            "Base",
            "dspy.Module.__init__(self)",
            "for self.solve in [pending]:\n            pass",
        ),
        ("from dspy import Module as Base", "Base", "super().__init__()", "self.solve = pending"),
    ],
)
def test_compiler_binding_variations(backend, imports, base, initializer, binding):
    source = f"""{imports}
class M({base}):
    def __init__(self):
        {initializer}
        pending = dspy.Predict("value: int -> result: int")
        {binding}
    def forward(self, value):
        return self.solve(value=value)
"""
    with dspy.context(lm=DummyLM([{"result": 37}])):
        assert program(source, interpreter_factory=backend)(value=11).result == 37


def test_prediction_mutation_shadowing_and_method_closures(backend):
    flex = program(
        """
        class M(dspy.Module):
            def forward(self, value):
                p = dspy.Prediction(y=3)
                p._fields["y"] = 7
                assert p.y == p["y"] == 7
                p.y += 4
                assert p.y == 11 and p["y"] == 7
                def getattr(obj, name):
                    return 13
                def helper(other):
                    assert isinstance(other, dspy.Module)
                    assert type(other) is M
                    return getattr(p, "y")
                return dspy.Prediction(result=helper(self) + p.y + p["y"])
    """,
        interpreter_factory=backend,
    )
    assert flex(value=0).result == 31


def test_augmented_stores_preserve_receiver_index_and_rhs_order(backend):
    events = []

    def record(value: int) -> int:
        events.append(value)
        return value

    flex = program(
        """
        class M(dspy.Module):
            def forward(self, value):
                self.values = [3, 5, 7]
                original = self.values
                def receiver():
                    record(11)
                    return self
                receiver().values += [record(13)]
                assert original is self.values
                receiver().values[record(1)] += record(17)
                self.left, (self.right, *self.rest) = [19, [23, 29, 31]]
                return dspy.Prediction(result=sum(original) + self.left + self.right + sum(self.rest))
    """,
        tools=[record],
        interpreter_factory=backend,
    )
    assert flex(value=0).result == 147
    assert events == [11, 13, 11, 1, 17]


@pytest.mark.parametrize(
    "body,match",
    [
        ("x = locals()", "execution scope"),
        ("del self.x", "Delete"),
        ("self.__private = 1", "name-mangled"),
        ("f = lambda: super()", "only direct super"),
        ("parent = super", "only direct super"),
        ("cls = __class__", "only direct super"),
        ("super = lambda: self", "only direct super"),
    ],
)
def test_unsupported_compilation_fails_explicitly(body, match):
    flex = program(f"class M(dspy.Module):\n    def forward(self, value):\n        {body}")
    with pytest.raises(dspy.CodeExecutionError, match="Unsupported Monty syntax at sandbox:3:.*" + match):
        flex(value=0)


def test_compiled_failure_reports_original_source_statement():
    flex = program("""
        class M(dspy.Module):
            def forward(self, value):
                def helper(x):
                    return 17 // x
                return dspy.Prediction(result=helper(value))
    """)
    with pytest.raises(dspy.CodeExecutionError, match="sandbox:4: return 17 // x"):
        flex(value=0)


@pytest.mark.parametrize(
    "methods,match",
    [
        (
            "def __init__(self):\n    def helper(obj):\n        super().__init__()\n    helper(self)",
            "only direct super",
        ),
        ("def __call__(self):\n    return self.forward(value=0)", "custom special methods"),
    ],
)
def test_removed_object_model_features_fail_validation(methods, match):
    from dspy.primitives._monty import compile_source

    source = "class M(dspy.Module):\n" + textwrap.indent(methods, "    ")
    with pytest.raises(dspy.CodeExecutionError, match=match):
        compile_source(source)


def test_failed_unpack_preserves_partial_writes(backend):
    flex = program(
        """
        class M(dspy.Module):
            def __init__(self):
                self.a = 2
                self.b = 3
                try:
                    self.a, (self.b, self.c) = [11, [17]]
                except ValueError:
                    pass
                assert self.a == 11 and self.b == 3
                assert not hasattr(self, "c")
            def forward(self, value):
                return dspy.Prediction(result=self.a + self.b)
    """,
        interpreter_factory=backend,
    )
    assert flex(value=0).result == 14


@pytest.mark.parametrize("expression", ['" Hello ".strip', "[].append", 'f"{value}".lower'])
def test_native_method_lint_runs_before_any_guest_execution(expression):
    from dspy.primitives._monty import compile_source

    source = f"class M(dspy.Module):\n    def forward(self, value):\n        fn = {expression}"
    with pytest.raises(dspy.CodeExecutionError, match=r"sandbox:3: native method.*named helper"):
        compile_source(source)


def test_named_helpers_and_flex_method_values_are_allowed(backend):
    flex = program(
        """
        class M(dspy.Module):
            def clean(self, text):
                return text.strip().lower()
            def forward(self, value):
                normalize = self.clean
                def helper(text):
                    return text.strip()
                direct = len(" Ab ".strip())
                return dspy.Prediction(result=direct + len(normalize(" CDE ")) + len(helper(" FGHI ")))
    """,
        interpreter_factory=backend,
    )
    assert flex(value=0).result == 9


def test_unknown_receiver_has_actionable_runtime_diagnostic():
    flex = program("""
        class M(dspy.Module):
            def forward(self, value):
                text = str(value)
                normalize = text.strip
                return dspy.Prediction(result=len(normalize()))
    """)
    with pytest.raises(dspy.CodeExecutionError, match="named helper") as error:
        flex(value=13)
    assert "sandbox:4:" in str(error.value)


@deno_required
def test_pyodide_does_not_receive_monty_lint():
    flex = program(
        """
        class M(dspy.Module):
            def forward(self, value):
                normalize = " Hello ".strip
                return dspy.Prediction(result=len(normalize()))
    """,
        interpreter_factory=dspy.PythonInterpreter,
    )
    assert flex(value=0).result == 5


def test_monty_authoring_context_reads_metadata_without_calling_factories():
    from dspy.teleprompt.gepa.gepa_flex_utils import flex_task_context

    class TrackedMonty(dspy.MontyInterpreter):
        def __init__(self):
            pytest.fail("reading authoring instructions must not invoke a factory")

    class OtherBackend(dspy.PythonInterpreter):
        pass

    default = dspy.Flex(Doubler)
    opaque = dspy.Flex(Doubler, interpreter_factory=lambda: TrackedMonty())
    configured = dspy.Flex(Doubler, interpreter_factory=partial(TrackedMonty, limits={"max_memory": 100_000}))
    other = dspy.Flex(Doubler, interpreter_factory=OtherBackend)
    assert "Monty Flex" not in flex_task_context(default)[1]["self"]
    with dspy.context(interpreter_factory=TrackedMonty):
        assert "Monty Flex" in flex_task_context(default)[1]["self"]
        assert "Monty Flex" not in flex_task_context(other)[1]["self"]
    assert "Monty Flex" in flex_task_context(configured)[1]["self"]
    assert "Monty Flex" not in flex_task_context(opaque)[1]["self"]


def test_monty_diagnostic_reaches_reflection_and_authoring_rules_reach_prompt():
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    source = """class M(dspy.Module):
    def forward(self, value):
        normalize = " Hello ".strip
        return dspy.Prediction(result=len(normalize()))
"""
    student = dspy.Flex(Doubler, interpreter_factory=dspy.MontyInterpreter)
    lm = DummyLM([{"revised_source": student.module_src}])
    adapter = DspyAdapter(
        student_module=student, metric_fn=lambda *args, **kwargs: 1.0, feedback_map={}, reflection_lm=lm
    )
    candidate = {"self": source}
    example = dspy.Example(value=0, result=5).with_inputs("value")
    batch = adapter.evaluate([example], candidate, capture_traces=True)
    assert batch.scores == [0.0]
    records = adapter.make_reflective_dataset(candidate, batch, ["self"])
    assert "sandbox:3:" in records["self"][0]["Generated Outputs"]
    adapter.propose_new_texts(candidate, records, ["self"])
    messages = str(lm.history[-1]["messages"])
    assert "Monty Flex authoring rules" in messages
    assert "named helper" in messages
    assert "sandbox:3:" in messages
