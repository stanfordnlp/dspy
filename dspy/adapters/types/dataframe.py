"""DataFrame type for DSPy signatures with RLM support."""

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
    """DataFrame type for DSPy signatures.

    Wraps pandas DataFrames for use in DSPy signatures. Supports auto-wrapping
    and attribute proxying for ergonomic usage.

    WARNING: dspy.DataFrame should only be used with dspy.RLM, which provides
    a Python sandbox where the DataFrame is available for code execution.
    Other modules (ChainOfThought, Predict, etc.) will only see a string
    representation of the DataFrame, not the actual data.

    Example:
        ```python
        class AnalyzeData(dspy.Signature):
            data: dspy.DataFrame = dspy.InputField()
            result: str = dspy.OutputField()

        # Pass pandas DataFrame directly (auto-wraps)
        rlm = dspy.RLM(AnalyzeData)
        result = rlm(data=my_pandas_dataframe)
        ```
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
