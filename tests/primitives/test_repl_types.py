"""Tests for REPL data types."""

import pytest

from dspy.primitives.repl_types import REPLEntry, REPLHistory, REPLVariable

# Check if pandas is available for DataFrame tests
try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TestREPLVariableBasic:
    """Tests for basic REPLVariable functionality."""

    def test_from_value_string(self):
        """Test REPLVariable creation from string value."""
        var = REPLVariable.from_value("my_var", "hello world")

        assert var.name == "my_var"
        assert var.type_name == "str"
        assert "hello world" in var.preview

    def test_from_value_list(self):
        """Test REPLVariable creation from list value."""
        var = REPLVariable.from_value("numbers", [1, 2, 3, 4, 5])

        assert var.name == "numbers"
        assert var.type_name == "list"
        assert "1" in var.preview

    def test_from_value_dict(self):
        """Test REPLVariable creation from dict value."""
        var = REPLVariable.from_value("data", {"key": "value", "count": 42})

        assert var.name == "data"
        assert var.type_name == "dict"
        assert "key" in var.preview

    def test_format_includes_name(self):
        """Test that format() includes the variable name."""
        var = REPLVariable.from_value("test_var", "test value")
        formatted = var.format()

        assert "test_var" in formatted
        assert "str" in formatted

    def test_truncation_for_long_values(self):
        """Test that long values are truncated in preview."""
        long_value = "x" * 1000
        var = REPLVariable.from_value("long_var", long_value, preview_chars=100)

        assert len(var.preview) < len(long_value)
        assert "..." in var.preview


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
class TestREPLVariableDataFrame:
    """Tests for REPLVariable DataFrame support."""

    def test_from_dataframe_basic(self):
        """Test REPLVariable creation from DataFrame."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        })

        var = REPLVariable.from_value("my_df", df)

        assert var.name == "my_df"
        assert var.type_name == "DataFrame"
        assert var.total_length == 3  # row count
        assert "col1" in var.preview
        assert "col2" in var.preview

    def test_from_dataframe_shows_shape(self):
        """Test that DataFrame preview shows shape."""
        df = pd.DataFrame({
            "a": range(10),
            "b": range(10),
            "c": range(10),
        })

        var = REPLVariable.from_value("df", df)

        assert "10" in var.preview  # row count
        assert "3" in var.preview  # column count
        assert "rows" in var.preview.lower()
        assert "columns" in var.preview.lower()

    def test_from_dataframe_shows_dtypes(self):
        """Test that DataFrame preview shows column dtypes."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        })

        var = REPLVariable.from_value("df", df)

        assert "int_col" in var.preview
        assert "float_col" in var.preview
        assert "str_col" in var.preview
        # Check that dtypes are shown
        assert "int" in var.preview.lower() or "int64" in var.preview

    def test_from_dataframe_shows_null_counts(self):
        """Test that DataFrame preview shows null counts."""
        df = pd.DataFrame({
            "values": [1, np.nan, 3, np.nan, 5],
        })

        var = REPLVariable.from_value("df", df)

        assert "2" in var.preview  # 2 nulls
        assert "null" in var.preview.lower()

    def test_from_dataframe_shows_first_rows(self):
        """Test that DataFrame preview shows first rows."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [100, 90, 85, 95, 88],
        })

        var = REPLVariable.from_value("df", df)

        assert "Alice" in var.preview
        assert "Bob" in var.preview
        assert "Charlie" in var.preview

    def test_from_dataframe_shows_last_rows_for_large_df(self):
        """Test that DataFrame preview shows last rows for large DataFrames."""
        df = pd.DataFrame({
            "value": range(100),
        })

        var = REPLVariable.from_value("df", df)

        # Should show both first and last rows
        assert "First" in var.preview or "first" in var.preview
        assert "Last" in var.preview or "last" in var.preview

    def test_format_shows_total_rows_for_dataframe(self):
        """Test that format() shows 'Total rows' instead of 'Total length' for DataFrames."""
        df = pd.DataFrame({"x": [1, 2, 3]})

        var = REPLVariable.from_value("df", df)
        formatted = var.format()

        assert "Total rows" in formatted
        assert "Total length" not in formatted

    def test_format_shows_total_length_for_non_dataframe(self):
        """Test that format() shows 'Total length' for non-DataFrame values."""
        var = REPLVariable.from_value("text", "hello world")
        formatted = var.format()

        assert "Total length" in formatted
        assert "Total rows" not in formatted

    def test_from_dataframe_with_datetime(self):
        """Test REPLVariable creation from DataFrame with datetime."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "value": [1, 2, 3],
        })

        var = REPLVariable.from_value("df", df)

        assert "datetime64" in var.preview

    def test_from_dataframe_with_categorical(self):
        """Test REPLVariable creation from DataFrame with categorical."""
        df = pd.DataFrame({
            "category": pd.Categorical(["a", "b", "a"]),
        })

        var = REPLVariable.from_value("df", df)

        assert "category" in var.preview.lower()


class TestREPLEntry:
    """Tests for REPLEntry."""

    def test_format_basic(self):
        """Test basic formatting of REPLEntry."""
        entry = REPLEntry(
            reasoning="I need to calculate the sum",
            code="result = 1 + 2",
            output="3"
        )

        formatted = entry.format(index=0)

        assert "Step 1" in formatted
        assert "I need to calculate" in formatted
        assert "result = 1 + 2" in formatted
        assert "3" in formatted

    def test_format_truncates_long_output(self):
        """Test that long output is truncated."""
        long_output = "x" * 10000
        entry = REPLEntry(
            reasoning="",
            code="print('x' * 10000)",
            output=long_output
        )

        formatted = entry.format(index=0, max_output_chars=100)

        assert len(formatted) < len(long_output)
        assert "truncated" in formatted


class TestREPLHistory:
    """Tests for REPLHistory."""

    def test_empty_history_format(self):
        """Test formatting of empty history."""
        history = REPLHistory()
        formatted = history.format()

        assert "not interacted" in formatted.lower()

    def test_append_creates_new_instance(self):
        """Test that append returns a new instance (immutability)."""
        history1 = REPLHistory()
        history2 = history1.append(code="print(1)", output="1")

        assert len(history1) == 0
        assert len(history2) == 1

    def test_multiple_entries(self):
        """Test history with multiple entries."""
        history = REPLHistory()
        history = history.append(code="x = 1", output="")
        history = history.append(code="print(x)", output="1")

        assert len(history) == 2

        formatted = history.format()
        assert "Step 1" in formatted
        assert "Step 2" in formatted


# =============================================================================
# dspy.DataFrame Type Annotation Tests
# =============================================================================


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
class TestDataFrameTypeAnnotation:
    """Tests for dspy.DataFrame Pydantic type annotation."""

    def test_dataframe_type_in_signature(self):
        """Test dspy.DataFrame works as type annotation in Signature."""
        import warnings

        import dspy
        from dspy.primitives.repl_types import DataFrame

        # Suppress the expected warning during test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            class DataSig(dspy.Signature):
                """Process data."""
                data: DataFrame = dspy.InputField()
                result: str = dspy.OutputField()

            # Verify the signature was created successfully
            assert "data" in DataSig.model_fields

    def test_dataframe_type_warns_on_serialization(self):
        """Test warning is raised when dspy.DataFrame is serialized (non-RLM usage)."""
        import warnings

        import dspy
        from dspy.primitives.repl_types import DataFrame

        # Create signature without warning (schema creation doesn't warn)
        class WarningSig(dspy.Signature):
            """Test signature."""
            df: DataFrame = dspy.InputField()
            output: str = dspy.OutputField()

        df = pd.DataFrame({"a": [1, 2, 3]})
        instance = WarningSig.model_validate({"df": df, "output": "test"})

        # Warning should be raised during serialization (non-RLM path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            instance.model_dump()

            # Check that our warning was raised
            dataframe_warnings = [
                warning for warning in w
                if "dspy.DataFrame is being serialized to JSON" in str(warning.message)
            ]
            assert len(dataframe_warnings) >= 1, "Expected warning about JSON serialization"

    def test_dataframe_type_validates_dataframe(self):
        """Test dspy.DataFrame rejects non-DataFrame values."""
        import pydantic

        import dspy
        from dspy.primitives.repl_types import DataFrame

        class ValidateSig(dspy.Signature):
            """Test signature."""
            df: DataFrame = dspy.InputField()
            output: str = dspy.OutputField()

        # Valid DataFrame should work
        df = pd.DataFrame({"a": [1, 2, 3]})
        # This tests the validator accepts DataFrame
        validated = ValidateSig.model_validate({"df": df, "output": "test"})
        assert validated.df is df

        # Invalid type should fail
        with pytest.raises(pydantic.ValidationError):
            ValidateSig.model_validate({"df": "not a dataframe", "output": "test"})

    def test_dataframe_type_serializes_to_records(self):
        """Test dspy.DataFrame serializes to list of records."""
        import warnings

        import dspy
        from dspy.primitives.repl_types import DataFrame

        class SerializeSig(dspy.Signature):
            """Test signature."""
            df: DataFrame = dspy.InputField()
            output: str = dspy.OutputField()

        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        instance = SerializeSig.model_validate({"df": df, "output": "test"})

        # Serialize to dict - DataFrame becomes list of records (ignore serialization warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            serialized = instance.model_dump()
        expected = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
        assert serialized["df"] == expected
