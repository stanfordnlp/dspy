---
sidebar_position: 2
---

# Executing DSPy Signatures

So far we've understood what signatures are and how we can use them to craft our prompt. Let's look at how to execute them.

## Configuring the Language Model

To execute signatures, we require DSPy modules which depend on a connection to a language model (LM). DSPy 2.5 provides unified access to LM APIs and local model hosting through the `dspy.LM` class.

```python
# Configure a language model (e.g., GPT-3.5)
lm = dspy.LM('openai/gpt-3.5-turbo')
dspy.configure(lm=lm)
```

## Executing Signatures

Let's use the simplest module in DSPy - the `Predict` module that takes a signature as input to construct the prompt and generates a response.

```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Define the predictor
predictor = dspy.Predict(BasicQA)

# Call the predictor on a particular input
pred = predictor(question="Are both Cangzhou and Qionghai in the Hebei province of China?")

# Print the prediction
print(f"Question: {pred.question}")
print(f"Predicted Answer: {pred.answer}")
```

The `Predict` module generates a response via the configured LM and executes the prompt crafted by the signature. The output is returned as a `Prediction` object whose fields can be accessed via the dot operator.

## Inspecting Output

You can inspect how DSPy uses your signature to build the prompt through the LM's history. Every LM object maintains the history of its interactions, including inputs, outputs, token usage, and metadata.

```python
# Check the number of calls made to the LM
len(lm.history)  

# Access the last call's metadata
lm.history[-1].keys()  # Returns dict_keys(['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost'])
```

## How Predict Works

The `Predict` module works through several steps:

1. When you call the predictor, it executes the `forward` method of the `Predict` class.
2. In the `forward` method, DSPy:
   - Initializes the signature
   - Sets up LM call parameters
   - Prepares any few-shot examples if provided
3. The `_generate` method then:
   - Formats the examples to match the signature
   - Uses the configured LM to generate output
   - Returns a `Prediction` object

The prompt construction is handled by DSPy's Adapter system, which is responsible for formatting the signature I/O fields, instructions, and examples, as well as generating and parsing the outputs.

## Adapters and Type Enforcement

In DSPy 2.5, Adapters serve as a layer between Signatures and LMs, handling:
- Formatting of signature fields
- Incorporation of instructions and examples
- Output generation and parsing
- Type enforcement

You can configure the adapter behavior when setting up DSPy:

```python
dspy.configure(lm=lm, experimental=True)  # Enables enhanced type enforcement and formatting
```

While `Predict` provides a straightforward pipeline for executing signatures, you can build more sophisticated pipelines by creating custom Modules and Adapters to suit your specific needs.


# Defining Custom Adapters

Warning
Adapters are low-level features that change the way input and output is handled by DSPy, it's not recommended to build and use custom Adapters unless you are sure of what you are doing.

Adapters are a powerful feature in DSPy, allowing you to define custom behavior for your Signatures.

For example, you could define an Adapter that processes numerical data and enforces statistical analysis formatting. This is a practical example that shows how you can create custom Adapters that handle data analysis tasks.

You'll need to inherit the base Adapter class and implement two methods to create a usable custom Adapter:

format: This method is responsible for formatting the input for the LM. This method takes signature, demos and inputs as input parameters. Demos are in-context examples set manually or through example. The output of this function can be a string prompt supported by completions function, a list of message dictionaries or any format that the LM you are using supports.

parse: This method is responsible for parsing the output of the LM. This method takes signature, completions and _parse_values as input parameters.

```python
from dspy.adapters.base import Adapter
from typing import List, Dict
import re

class DataAnalysisAdapter(Adapter):
    def __init__(self):
        super().__init__()
        self.numeric_pattern = r'-?\d*\.?\d+'

    def format(self, signature, demos, inputs):
        system_prompt = signature.instructions
        all_fields = signature.model_fields
        all_field_data = [(all_fields[f].json_schema_extra["prefix"], all_fields[f].json_schema_extra["desc"]) for f in all_fields]

        all_field_data_str = "\n".join([f"{p}: {d}" for p, d in all_field_data])
        format_instruction_prompt = "="*20 + f"""\n\nAnalysis Format Required:\n\n{all_field_data_str}\n\n""" + "="*20

        all_input_fields = signature.input_fields
        input_fields_data = [(all_input_fields[f].json_schema_extra["prefix"], inputs[f]) for f in all_input_fields]

        # Add statistical context to the input
        input_fields_str = "\n".join([
            f"{p}: {v}\nPlease include numerical analysis and statistics where applicable."
            for p, v in input_fields_data
        ])

        return system_prompt + format_instruction_prompt + input_fields_str

    def parse(self, signature, completions, _parse_values=None):
        output_fields = signature.output_fields
        output_dict = {}

        for field in output_fields:
            field_info = output_fields[field]
            prefix = field_info.json_schema_extra["prefix"]

            # Extract the field content
            field_text = completions.split(prefix + ":")[-1].split("\n")[0].strip()
            
            # Process numerical values if present
            numbers = re.findall(self.numeric_pattern, field_text)
            if numbers:
                field_text = f"{field_text} (Numerical values found: {', '.join(numbers)})"
            
            output_dict[field] = field_text

        return output_dict
```

Let's understand the DataAnalysisAdapter class. The format method takes signature, demos, and inputs as input parameters. It constructs a prompt by combining the system prompt, format instruction prompt, and input fields, while adding context for statistical analysis.

The parse method takes signature, completions, and _parse_values as input parameters. It extracts output fields from the completions, processes any numerical values found, and returns them as a dictionary.

Once you have defined your custom Adapter, you can use it in your Signatures by passing it as an argument to the dspy.configure method.

```python
dspy.configure(adapter=DataAnalysisAdapter())
```

Now, when you run an inference over a Signature, the input will be processed with statistical context before being passed to the LM. The output will be parsed as a dictionary with numerical analysis.

```python
lm = dspy.LM('openai/gpt-4')
dspy.configure(lm=lm, adapter=DataAnalysisAdapter())

# Define a signature for data analysis
class DataAnalysis(dspy.Signature):
    """Analyze numerical data and provide statistics."""
    
    data = dspy.InputField()
    analysis = dspy.OutputField(desc="statistical analysis of the data")

# Use the adapter
predictor = dspy.Predict(DataAnalysis)
response = predictor(data="Sales increased from 1000 to 1500 over three months")
print(response.analysis)
```

Output:
```
The sales showed a 50% increase (Numerical values found: 1000, 1500) over the quarterly period, with an average monthly growth of approximately 166.67 units.
```

Let's see how the prompt after Adapter looks like!

```python
lm.inspect_history()
```

Output:

User message:
```
Analyze numerical data and provide statistics.

===================
Analysis Format Required:

data: ${data}
analysis: statistical analysis of the data

===================
data: Sales increased from 1000 to 1500 over three months
Please include numerical analysis and statistics where applicable.

Response:

Data: Sales increased from 1000 to 1500 over three months
Analysis: The sales showed a 50% increase over the quarterly period, with an average monthly growth of approximately 166.67 units.
```

The above example is a practical Adapter that processes numerical data and enforces statistical analysis formatting. You can define more complex Adapters based on your requirements.

Overriding __call__ method

To gain control over usage of format and parse and even more fine-grained control over the flow of input from signature to outputs you can override __call__ method and implement your custom flow. Although for most cases only implementing parse and format function will be fine.