# dspy.DataFrame

A Pydantic-compatible type annotation for pandas DataFrames, designed for use with `dspy.RLM`.

## Overview

`dspy.DataFrame` allows you to use pandas DataFrames as input fields in DSPy Signatures. When used with `dspy.RLM`, the DataFrame is:

1. Serialized to Parquet format (preserving dtypes)
2. Injected into the Python sandbox
3. Available for pandas operations in generated code

> **Warning**: `dspy.DataFrame` should only be used with `dspy.RLM`. Other modules like `ChainOfThought` or `Predict` will only see a string representation of the DataFrame, not the actual data.

## Usage

```python
import dspy
import pandas as pd

class DataAnalysis(dspy.Signature):
    """Analyze the provided data."""

    data: dspy.DataFrame = dspy.InputField(desc="The dataset to analyze")
    summary: str = dspy.OutputField(desc="Analysis summary")
    row_count: int = dspy.OutputField(desc="Number of rows")

# Create a DataFrame
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "score": [85, 92, 78]
})

# Use with RLM
analyzer = dspy.RLM(DataAnalysis, max_iterations=10)
result = analyzer(data=df)

print(result.summary)
print(f"Rows: {result.row_count}")
```

## How It Works

### Serialization

DataFrames are serialized using Apache Parquet format via PyArrow. This preserves:

- Integer types (int8, int16, int32, int64)
- Float types (float32, float64)
- Boolean values
- String/object columns
- Datetime columns
- Categorical columns
- Nullable integer types

### Metadata in Prompts

The LLM receives rich metadata about each DataFrame:

- **Shape**: Number of rows and columns
- **Memory usage**: Approximate memory footprint
- **Column info**: Name, dtype, and null count for each column
- **Sample data**: First 3 and last 3 rows

### Sandbox Access

In the RLM sandbox, the DataFrame is loaded via `pd.read_parquet()` and available as a variable with its original name:

```python
# In the sandbox, if your input field is named 'data':
result = data.groupby('category')['value'].mean()
SUBMIT(result.to_dict())
```

## Retrieving Variables After Execution

After RLM execution, you can retrieve variables from the sandbox using `get_variable()`:

```python
# Pass your own interpreter to RLM (optional, but required if you want to use get_variable)
interpreter = dspy.PythonInterpreter()
analyzer = dspy.RLM(DataAnalysis, interpreter=interpreter)
result = analyzer(data=df)

# Now use the interpreter directly to retrieve variables
chart_data = interpreter.get_variable("chart_base64")
intermediate_df = interpreter.get_variable("filtered_data")
```

This is useful for retrieving:
- Base64-encoded images from matplotlib charts
- Intermediate DataFrames created during analysis
- Any other computed values not included in SUBMIT

## Limitations

- Maximum DataFrame size: 500MB (serialized)
- Warning issued for DataFrames over 1 million rows
- Only works with `dspy.RLM` - other modules receive string representation
- Requires pandas and pyarrow to be installed

## See Also

- [DataFrames with RLM Tutorial](../../tutorials/dataframes_rlm/index.ipynb)
- [dspy.RLM](../modules/RLM.md)
- [dspy.PythonInterpreter](../tools/PythonInterpreter.md)
