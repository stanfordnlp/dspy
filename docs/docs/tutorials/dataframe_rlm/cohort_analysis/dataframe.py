"""Standalone DataFrame wrapper with RLM sandbox support.

This module provides a DataFrame class that implements dspy.SandboxSerializable,
allowing pandas DataFrames to be injected into the RLM sandbox for code execution.

Usage with RLM:

    import dspy
    from dataframe import DataFrame

    rlm = dspy.RLM("data, query -> answer")
    result = rlm(data=DataFrame(my_pandas_df), query="What is the total revenue?")

The DataFrame is serialized as base64-encoded Parquet for efficient, type-preserving
transfer into the Deno/Pyodide sandbox.
"""

import base64
from typing import Any

from dspy import SandboxSerializable


def _is_dataframe(value: Any) -> bool:
    """Check if value is a pandas DataFrame without requiring pandas import."""
    type_module = getattr(type(value), "__module__", "")
    type_name = type(value).__name__
    return type_module.startswith("pandas") and type_name == "DataFrame"


class DataFrame(SandboxSerializable):
    """DataFrame wrapper with RLM sandbox support.

    Wraps a pandas DataFrame and implements the SandboxSerializable protocol
    so it can be injected into the RLM sandbox for code execution.

    Example:

        import pandas as pd
        from dataframe import DataFrame

        df = DataFrame(pd.DataFrame({"x": [1, 2, 3]}))
        # Use with RLM:
        rlm = dspy.RLM("data -> answer")
        result = rlm(data=df)
    """

    def __init__(self, data: Any):
        if _is_dataframe(data):
            self.data = data
        elif isinstance(data, DataFrame):
            self.data = data.data
        else:
            raise TypeError(
                f"Expected pandas DataFrame, got {type(data).__name__}. "
                f"Install pandas with: pip install pandas"
            )

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying DataFrame."""
        if name.startswith("_") or name == "data":
            raise AttributeError(name)
        return getattr(self.data, name)

    # SandboxSerializable protocol

    def sandbox_setup(self) -> str:
        return "import pandas as pd\nimport pyarrow\nimport base64\nimport io"

    def to_sandbox(self) -> bytes:
        """Serialize DataFrame as base64-encoded Parquet."""
        return base64.b64encode(self.data.to_parquet(index=False))

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = pd.read_parquet(io.BytesIO(base64.b64decode({data_expr})))"

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

    def __repr__(self) -> str:
        return f"DataFrame({self.data.shape[0]} rows x {self.data.shape[1]} cols)"
