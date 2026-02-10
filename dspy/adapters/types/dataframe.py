"""DataFrame type for DSPy signatures with RLM sandbox support."""

from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def _is_dataframe(value: Any) -> bool:
    """Check if value is a pandas DataFrame without requiring pandas import."""
    type_module = getattr(type(value), "__module__", "")
    type_name = type(value).__name__
    return type_module.startswith("pandas") and type_name == "DataFrame"


class DataFrame(Type):
    """DataFrame type for DSPy signatures with RLM sandbox support.

    Wraps a pandas DataFrame as a dspy.Type. When used with dspy.RLM, the
    DataFrame is injected into the sandbox and available for code execution.
    Other modules (ChainOfThought, Predict, etc.) only see a string summary.

    Example::

        class AnalyzeData(dspy.Signature):
            data: dspy.DataFrame = dspy.InputField()
            result: str = dspy.OutputField()

        rlm = dspy.RLM(AnalyzeData)
        result = rlm(data=dspy.DataFrame(my_pandas_df))
    """

    data: Any  # pd.DataFrame at runtime

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    def __init__(self, data: Any = None, **kwargs):
        """Create a DataFrame wrapper.

        Args:
            data: A pandas DataFrame or dspy.DataFrame instance

        Raises:
            TypeError: If data is not a pandas DataFrame
        """
        if data is not None and "data" not in kwargs:
            if _is_dataframe(data):
                kwargs["data"] = data
            elif isinstance(data, DataFrame):
                # Already wrapped - extract underlying DataFrame
                kwargs["data"] = data.data
            else:
                raise TypeError(
                    f"Expected pandas DataFrame, got {type(data).__name__}. "
                    f"Install pandas with: pip install pandas"
                )

        super().__init__(**kwargs)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_dataframe(cls, value: Any) -> Any:
        """Auto-wrap pandas DataFrames for Pydantic validation.

        This allows users to pass raw pandas DataFrames directly to signatures.
        """
        # If it's already a dict (from model_validate), pass through
        if isinstance(value, dict):
            return value
        # If it's a raw pandas DataFrame, wrap it
        if _is_dataframe(value):
            return {"data": value}
        # If it's already a DataFrame instance, extract the data field
        if isinstance(value, DataFrame):
            return {"data": value.data}
        return value

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying DataFrame.

        Allows df.shape, df.head(), etc. without accessing df.data explicitly.
        """
        if name.startswith("_") or name in ("data", "model_fields"):
            raise AttributeError(name)
        return getattr(self.data, name)

    def format(self) -> str:
        """Format for display purposes (returns string representation)."""
        return repr(self.data)

    @pydantic.model_serializer()
    def _serialize(self) -> list:
        """Serialize DataFrame to list of records for JSON serialization."""
        import warnings

        warnings.warn(
            "dspy.DataFrame is being serialized to JSON. "
            "Only dspy.RLM preserves DataFrame data; other modules receive a string representation.",
            UserWarning,
            stacklevel=6,
        )
        return self.data.to_dict(orient="records")

    # RLM Sandbox Support

    def sandbox_setup(self) -> str:
        return "import pandas as pd\nimport json"

    def to_sandbox(self, var_name: str) -> tuple[str, bytes | None]:
        """Serialize DataFrame for sandbox injection using pandas' built-in JSON."""
        if not PANDAS_AVAILABLE:
            return None, None

        json_bytes = self.data.to_json(orient="records", date_format="iso").encode("utf-8")
        assignment_code = (
            f'{var_name} = pd.DataFrame(json.loads('
            f'open("/tmp/dspy_vars/{var_name}.json").read()))'
        )
        return assignment_code, json_bytes

    @classmethod
    def from_sandbox(cls, data: Any) -> "DataFrame | None":
        """Reconstruct DataFrame from sandbox output."""
        if not PANDAS_AVAILABLE:
            return None
        if _is_dataframe(data):
            return cls(data=data)
        if isinstance(data, (list, dict)):
            return cls(data=pd.DataFrame(data))
        return None

    def rlm_preview(self, max_chars: int = 500) -> str:
        """Generate LLM-friendly preview of DataFrame contents."""
        df = self.data
        lines = [
            f"DataFrame: {df.shape[0]:,} rows x {df.shape[1]} columns",
            "",
            "Columns:",
        ]

        for col in list(df.columns)[:10]:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isna().sum())
            null_info = f" ({null_count:,} nulls)" if null_count > 0 else ""
            lines.append(f"  {col}: {dtype}{null_info}")

        if len(df.columns) > 10:
            lines.append(f"  ... and {len(df.columns) - 10} more columns")

        if len(df) > 0:
            lines.extend(["", "Sample (first 3 rows):", df.head(3).to_string()])

        preview = "\n".join(lines)
        return preview[:max_chars] + "..." if len(preview) > max_chars else preview
