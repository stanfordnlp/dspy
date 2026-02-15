import pytest

pd = pytest.importorskip("pandas")

import dspy
from dspy.adapters.types.dataframe import DataFrame, _is_dataframe


# -- Construction and validation --


def test_wraps_pandas_dataframe():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    wrapped = DataFrame(df)
    assert wrapped.data is df


def test_wraps_dspy_dataframe():
    df = pd.DataFrame({"x": [10]})
    inner = DataFrame(df)
    outer = DataFrame(inner)
    assert outer.data is df


def test_rejects_non_dataframe():
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        DataFrame([1, 2, 3])


def test_pydantic_auto_wrap():
    df = pd.DataFrame({"col": [1]})
    wrapped = DataFrame.model_validate(df)
    assert _is_dataframe(wrapped.data)


# -- Attribute proxy --


def test_proxies_shape():
    df = pd.DataFrame({"a": [1, 2, 3]})
    wrapped = DataFrame(df)
    assert wrapped.shape == (3, 1)


def test_proxies_head():
    df = pd.DataFrame({"a": range(10)})
    wrapped = DataFrame(df)
    assert len(wrapped.head(3)) == 3


# -- format / serialize --


def test_format_returns_repr():
    df = pd.DataFrame({"a": [1]})
    wrapped = DataFrame(df)
    assert "a" in wrapped.format()


def test_serialize_to_records():
    df = pd.DataFrame({"a": [1, 2]})
    wrapped = DataFrame(df)
    result = wrapped.model_dump()
    assert isinstance(result, list)
    assert result == [{"a": 1}, {"a": 2}]


# -- RLM sandbox support --


def test_sandbox_setup():
    df = pd.DataFrame({"a": [1]})
    wrapped = DataFrame(df)
    setup = wrapped.sandbox_setup()
    assert "import pandas" in setup
    assert "import pyarrow" in setup
    assert "import base64" in setup
    assert "import io" in setup


def test_to_sandbox_returns_parquet():
    df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
    wrapped = DataFrame(df)
    payload = wrapped.to_sandbox()

    assert isinstance(payload, bytes)

    # Payload is base64-encoded parquet
    import base64
    import io
    decoded = base64.b64decode(payload)
    result = pd.read_parquet(io.BytesIO(decoded))
    assert len(result) == 2
    assert list(result["x"]) == [1, 2]
    assert list(result["y"]) == [3.0, 4.0]


def test_sandbox_assignment():
    df = pd.DataFrame({"x": [1]})
    wrapped = DataFrame(df)
    code = wrapped.sandbox_assignment("my_df", "open('/tmp/data').read()")
    assert "my_df" in code
    assert "pd.read_parquet" in code
    assert "base64.b64decode" in code


def test_to_sandbox_preserves_dtypes():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01", "2024-06-15"]),
        "cat": pd.Categorical(["a", "b"]),
        "num": [1, 2],
        "flag": [True, False],
    })
    wrapped = DataFrame(df)
    payload = wrapped.to_sandbox()

    import base64
    import io
    result = pd.read_parquet(io.BytesIO(base64.b64decode(payload)))
    assert result["ts"].dtype == "datetime64[us]"
    assert result["cat"].dtype.name == "category"
    assert result["num"].dtype == "int64"
    assert result["flag"].dtype == "bool"


def test_to_sandbox_handles_nulls():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    wrapped = DataFrame(df)
    payload = wrapped.to_sandbox()

    import base64
    import io
    result = pd.read_parquet(io.BytesIO(base64.b64decode(payload)))
    assert pd.isna(result["a"].iloc[1])
    assert result["a"].iloc[0] == 1.0
    assert result["a"].iloc[2] == 3.0


# -- rlm_preview --


def test_rlm_preview_basic():
    df = pd.DataFrame({"name": ["alice", "bob"], "score": [90, 85]})
    wrapped = DataFrame(df)
    preview = wrapped.rlm_preview()
    assert "2 rows" in preview
    assert "2 columns" in preview
    assert "name" in preview
    assert "score" in preview


def test_rlm_preview_truncates():
    df = pd.DataFrame({f"col_{i}": range(100) for i in range(20)})
    wrapped = DataFrame(df)
    preview = wrapped.rlm_preview(max_chars=200)
    assert len(preview) <= 203  # 200 + "..."
    assert preview.endswith("...")


def test_rlm_preview_shows_nulls():
    df = pd.DataFrame({"a": [1, None, 3]})
    wrapped = DataFrame(df)
    preview = wrapped.rlm_preview()
    assert "1 nulls" in preview


# -- _is_dataframe helper --


def test_is_dataframe_true():
    assert _is_dataframe(pd.DataFrame())


def test_is_dataframe_false():
    assert not _is_dataframe([1, 2, 3])
    assert not _is_dataframe({"a": 1})


# -- Integration: DataFrame in a Signature --


def test_dataframe_in_signature():
    class AnalyzeSig(dspy.Signature):
        data: dspy.DataFrame = dspy.InputField()
        result: str = dspy.OutputField()

    assert "data" in AnalyzeSig.input_fields


# -- Integration: PythonInterpreter injection --


@pytest.mark.integration
def test_interpreter_injects_dataframe():
    from dspy.primitives.python_interpreter import PythonInterpreter

    df = pd.DataFrame({"x": [10, 20, 30]})
    wrapped = DataFrame(df)

    with PythonInterpreter() as interp:
        result = interp.execute(
            "print(sum(data['x']))",
            variables={"data": wrapped},
        )
        assert "60" in result


@pytest.mark.integration
def test_interpreter_rejects_raw_dataframe():
    """Raw pandas DataFrames are not auto-wrapped; they raise an error."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    df = pd.DataFrame({"x": [1, 2, 3]})

    with PythonInterpreter() as interp:
        with pytest.raises(Exception):
            interp.execute("print(data)", variables={"data": df})


# -- Integration: RLM auto-wrapping --


def test_rlm_auto_wraps_raw_dataframe():
    """RLM._wrap_rlm_inputs auto-wraps raw pd.DataFrame into dspy.DataFrame."""
    from dspy.predict.rlm import RLM

    class AnalyzeSig(dspy.Signature):
        data: dspy.DataFrame = dspy.InputField()
        result: str = dspy.OutputField()

    rlm = RLM(AnalyzeSig)
    raw_df = pd.DataFrame({"x": [1, 2, 3]})

    wrapped = rlm._wrap_rlm_inputs({"data": raw_df})
    assert isinstance(wrapped["data"], DataFrame)
    assert list(wrapped["data"].data["x"]) == [1, 2, 3]


def test_rlm_passthrough_already_wrapped():
    """RLM._wrap_rlm_inputs passes through already-wrapped dspy.DataFrame."""
    from dspy.predict.rlm import RLM

    class AnalyzeSig(dspy.Signature):
        data: dspy.DataFrame = dspy.InputField()
        result: str = dspy.OutputField()

    rlm = RLM(AnalyzeSig)
    raw_df = pd.DataFrame({"x": [1, 2, 3]})
    already_wrapped = DataFrame(raw_df)

    wrapped = rlm._wrap_rlm_inputs({"data": already_wrapped})
    assert wrapped["data"] is already_wrapped


def test_rlm_passthrough_non_type_fields():
    """RLM._wrap_rlm_inputs passes through plain types like str."""
    from dspy.predict.rlm import RLM

    class Sig(dspy.Signature):
        query: str = dspy.InputField()
        result: str = dspy.OutputField()

    rlm = RLM(Sig)
    wrapped = rlm._wrap_rlm_inputs({"query": "hello"})
    assert wrapped["query"] == "hello"
