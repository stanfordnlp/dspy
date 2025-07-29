# DSPy Minimal

A minimal implementation of DSPy for AWS Lambda deployment. This package includes only the essential components needed for basic DSPy functionality while significantly reducing the package size.

## What's Included

- `Module` - Base module class
- `Predict` - Basic prediction functionality
- `ReAct` - Simplified reasoning and acting
- `Tool` - Tool definition
- `Signature`, `InputField`, `OutputField` - Signature system
- `BootstrapFewShot` - Simplified few-shot learning
- `Example` - Example data structure
- `LM` - Language model interface (OpenAI only)
- `configure` - Configuration function

## What's Removed

- Heavy dependencies like `numpy`, `litellm`, `optuna`, `tqdm`
- Advanced features like caching, complex adapters, streaming
- Multiple provider support (only OpenAI)
- Complex teleprompting optimizers
- Evaluation and metrics
- Retrieval systems

## Installation

```bash
pip install dspy-minimal
```

## Usage

```python
from dspy_minimal import Module, Predict, Signature, InputField, OutputField, LM, configure

# Configure the LM
lm = LM("gpt-4o-mini")
configure(lm=lm)

# Define a signature
class MySignature(Signature):
    input_text: str = InputField(desc="Input text to process")
    output_summary: str = OutputField(desc="Summary of the input")

# Create a predictor
predictor = Predict(MySignature)

# Use it
result = predictor(input_text="This is a long text that needs summarization")
print(result["output_summary"])
```

## Package Size

The minimal package is approximately **~5MB** compared to the full DSPy package which is **~350MB**.

## Limitations

- Only supports OpenAI API
- Simplified prediction logic
- No caching or advanced features
- Limited signature parsing
- Basic error handling

## License

MIT License 