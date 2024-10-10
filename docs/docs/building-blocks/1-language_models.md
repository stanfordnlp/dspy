---
sidebar_position: 2
---

# Language Models

The most powerful features in DSPy revolve around algorithmically optimizing the prompts (or weights) of LMs, especially when you're building programs that use the LMs within a pipeline.

Let's first make sure you can set up your language model. DSPy support clients for many remote and local LMs.

## Using `dspy.LM`

:::warning
Earlier versions of DSPy involved tons of clients for different LM providers, e.g. `dspy.OpenAI`, `dspy.GoogleVertexAI`, and `dspy.HFClientTGI`, etc. These are now deprecated and will be removed in DSPy 2.6.

Instead, use `dspy.LM` to access any LM endpoint for local and remote models. This relies on [LiteLLM](https://github.com/BerriAI/litellm) to translate the different client APIs into an OpenAI-compatible interface.

Any [provider supported in LiteLLM](https://docs.litellm.ai/docs/providers) should work with `dspy.LM`.
:::

### Setting up the LM client

In DSPy 2.5, we use the `dspy.LM` class to set up language models. This replaces the previous client-specific classes. Then, use `dspy.configure` to declare this as the default LM.

For example, to use OpenAI language models, you can do it as follows.

```python
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

### Directly calling the LM

You can simply call the LM with a string to give it a raw prompt, i.e. a string.

```python
lm("hello! this is a raw prompt to GPT-4o-mini")
```

**Output:**
```text
["Hello! It looks like you're trying to interact with a model. How can I assist you today?"]
```

For chat LMs, you can pass a list of messages.

```python
lm(messages=[{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "What is 2+2?"}])
```

**Output:**
```text
['2 + 2 equals 4.']
```

This is almost never the recommended way to interact with LMs in DSPy, but it is allowed.


### Using the LM with DSPy signatures

You can also use the LM via DSPy [`signature` (input/output spec)](https://dspy-docs.vercel.app/docs/building-blocks/signatures) and [`modules`](https://dspy-docs.vercel.app/docs/building-blocks/modules), which we discuss in more depth in the remaining guides.

```python
# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought('question -> answer')

# Run with the default LM configured with `dspy.configure` above.
response = qa(question="How many floors are in the castle David Gregory inherited?")
print(response.answer)
```
**Output:**
```text
The castle David Gregory inherited has 7 floors.
```

### Using multiple LMs at once.

The default LM above is GPT-3.5, `gpt3_turbo`. What if I want to run a piece of code with, say, GPT-4 or LLama-2?

Instead of changing the default LM, you can just change it inside a block of code.

:::tip
Using `dspy.configure` and `dspy.context` is thread-safe!
:::

```python
# Run with the default LM configured above, i.e. GPT-3.5
response = qa(question="How many floors are in the castle David Gregory inherited?")
print('GPT-3.5:', response.answer)

gpt4_turbo = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=300)

# Run with GPT-4 instead
with dspy.context(lm=gpt4_turbo):
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print('GPT-4-turbo:', response.answer)
```
**Output:**
```text
GPT-3.5: The castle David Gregory inherited has 7 floors.
GPT-4-turbo: The number of floors in the castle David Gregory inherited cannot be determined with the information provided.
```

### Configuring LM attributes

For any LM, you can configure any of the following attributes at initialization or per call.

```python
gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)
```

By default LMs in DSPy are cached. If you repeat the same call, you will get the same outputs. But you can turn of caching by setting `cache=False` while declaring `dspy.LM` object.

### Using locally hosted LMs

Any OpenAI-compatible endpoint is easy to set up with an `openai/` prefix as well. This works great for open LMs from HuggingFace hosted locally with SGLang, VLLM, or HF Text-Generation-Inference.

```python
sglang_port = 7501
sglang_url = f"http://localhost:{sglang_port}/v1"
sglang_llama = dspy.LM("openai/meta-llama/Meta-Llama-3-8B-Instruct", api_base=sglang_url)

# You could also use text mode, in which the prompts are *not* formatted as messages.
sglang_llama_text = dspy.LM("openai/meta-llama/Meta-Llama-3-8B-Instruct", api_base=sglang_url, model_type='text')
```

### Inspecting output and usage metadata

Every LM object maintains the history of its interactions, including inputs, outputs, token usage (and $$$ cost), and metadata.

```python
len(lm.history)  # e.g., 3 calls to the LM

lm.history[-1].keys()  # access the last call to the LM, with all metadata
```

**Output:**
```text
dict_keys(['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost'])
```

## Creating Custom LM Class

Creating custom LM class is quite straightforward in DSPy. You can inherit from the `dspy.LM` class or create a new class with a similar interface. You'll need to implement/override the three methods:

* `__init__`: Initialize the LM with the given `model` and other keyword arguments.
* `__call__`: Call the LM with the given input prompt and return a list of string outputs.
* `inspect_history`: The history of interactions with the LM. This is optional but is needed by some optimizers in DSPy.

:::tip
If there is not much overlap in features between your LM and LiteLLM it's better to not inherit and implement all methods from ground up.
:::

Let's create an LM for Gemini using `google-generativeai` package from scratch:

```python
import os
import dspy
import google.generativeai as genai

class GeminiLM(dspy.LM):
    def __init__(self, model, api_key=None, endpoint=None, **kwargs):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"] or api_key)

        self.endpoint = endpoint
        self.history = []

        super().__init__(model, **kwargs)
        self.model = genai.GenerativeModel(model)

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Custom chat model working for text completion model
        prompt = '\n\n'.join([x['content'] for x in messages] + ['BEGIN RESPONSE:'])

        completions = self.model.generate_content(prompt)
        self.history.append({"prompt": prompt, "completions": completions})
        
        # Must return a list of strings
        return [completions.candidates[0].content.parts[0].text]

    def inspect_history(self):
        for interaction in self.history:
            print(f"Prompt: {interaction['prompt']} -> Completions: {interaction['completions']}")

lm = GeminiLM("gemini-1.5-flash", temperature=0)
dspy.configure(lm=lm)

qa = dspy.ChainOfThought("question->answer")
qa(question="What is the capital of France?")
```

**Output:**
```text
Prediction(
    reasoning='France is a country in Western Europe. Its capital city is Paris.',
    answer='Paris'
)
```

The above example is the simplest form of LM. You can add more options to tweak generation config and even control the generated output based on your requirement.


## Structured LM output with Adapters

Prompt optimizers in DSPy generate and tune the _instructions_ or the _examples_ in the prompts corresponding to your Signatures. DSPy 2.5 introduces **Adapters** as a layer between Signatures and LMs, responsible for formatting these pieces (Signature I/O fields, instructions, and examples) as well as generating and parsing the outputs. 

In DSPy 2.5, the default Adapters are now more natively aware of chat LMs and are responsible for enforcing types, building on earlier experimental features from `TypedPredictor` and `configure(experimental=True)`. In our experience, this change tends to deliver more consistent pre-optimization quality from various LMs, especially for sophisticated Signatures.

```python
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm, experimental=True)

fact_checking = dspy.ChainOfThought('claims -> verdicts')
fact_checking(claims=["Python was released in 1991.", "Python is a compiled language."])
```

**Output:**
```text
Prediction(
    reasoning='The first claim states that "Python was released in 1991," which is true. Python was indeed first released by Guido van Rossum in February 1991. The second claim states that "Python is a compiled language." This is false; Python is primarily an interpreted language, although it can be compiled to bytecode, it is not considered a compiled language in the traditional sense like C or Java.',
    verdicts=[True, False]
)
```

### Defining Custom Adapters

:::warning
Adapters are low level feature that change the way input and output is handled by DSPy, it's not recommended to build and use custom Adapters unless you are sure of what you are doing.
:::

Adapters are a powerful feature in DSPy, allowing you to define custom behavior for your Signatures. 

For example, you could define an Adapter that automatically converts the input to uppercase before passing it to the LM. This is a simple example, but it shows how you can create custom Adapters that modify the inputs or outputs of your LMs.

You'll need to inherit the base `Adapter` class and implement two method to create a usable custom Adapter:

* `format`: This method is responsible for formatting the input for the LM. This method takes `signature`, `demos` and `inputs` as input parameters. Demos are in-context examples set manually or through example. The output of this function can be a string prompt supported by completions function, list of message dictionary or any format that the LM you are using supports.

* `parse`: This method is responsible for parsing the output of the LM. This method takes `signature`, `completions` and `_parse_values` as input parameters.

```python
from dspy.adapters.base import Adapter
from typing import List, Dict

class UpperCaseAdapter(Adapter):
    def __init__(self):
        super().__init__()

    def format(self, signature, demos, inputs):
        system_prompt = signature.instructions
        all_fields = signature.model_fields
        all_field_data = [(all_fields[f].json_schema_extra["prefix"], all_fields[f].json_schema_extra["desc"]) for f in all_fields]

        all_field_data_str = "\n".join([f"{p}: {d}" for p, d in all_field_data])
        format_instruction_prompt = "="*20 + f"""\n\nOutput Format:\n\n{all_field_data_str}\n\n""" + "="*20

        all_input_fields = signature.input_fields
        input_fields_data = [(all_input_fields[f].json_schema_extra["prefix"], inputs[f]) for f in all_input_fields]

        input_fields_str = "\n".join([f"{p}: {v}" for p, v in input_fields_data])

        # Convert to uppercase
        return (system_prompt + format_instruction_prompt + input_fields_str).upper()

    def parse(self, signature, completions, _parse_values=None):
        output_fields = signature.output_fields
        
        output_dict = {}
        for field in output_fields:
            field_info = output_fields[field]
            prefix = field_info.json_schema_extra["prefix"]

            field_completion = completions.split(prefix.upper())[-1].split("\n")[0].strip(": ")
            output_dict[field] = field_completion

        return output_dict
```

Let's understand the `UpperCaseAdapter` class. The `format` method takes `signature`, `demos`, and `inputs` as input parameters. It then constructs a prompt by combining the system prompt, format instruction prompt, and input fields. It then converts the prompt to uppercase. 

The `parse` method takes `signature`, `completions`, and `_parse_values` as input parameters. It then extracts the output fields from the completions and returns them as a dictionary.

Once you have defined your custom Adapter, you can use it in your Signatures by passing it as an argument to the `dspy.configure` method.

```python
dspy.configure(adapter=UpperCaseAdapter())
```

Now, when you run an inference over a Signature, the input will be converted to uppercase before being passed to the LM. The output will be parsed as a dictionary.

```python
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm, adapter=UpperCaseAdapter())

qa = dspy.ChainOfThought('question -> answer')

response = qa(question="How many floors are in the castle David Gregory inherited?")
response
```

**Output:**
```text
Prediction(
    reasoning='determine the number of floors in the castle that David Gregory inherited. This information typically comes from a specific source or context, such as a book, movie, or historical reference. Without that context, I cannot provide an exact number.',
    answer='I do not have the specific information regarding the number of floors in the castle David Gregory inherited.'
)
```

Now let's see how the prompt after Adapter looks like!

```python
lm.inspect_history()
```

**Output:**
```text
User message:

GIVEN THE FIELDS `QUESTION`, PRODUCE THE FIELDS `ANSWER`.====================

OUTPUT FORMAT:

QUESTION:: ${QUESTION}
REASONING: LET'S THINK STEP BY STEP IN ORDER TO: ${REASONING}
ANSWER:: ${ANSWER}

====================QUESTION:: HOW MANY FLOORS ARE IN THE CASTLE DAVID GREGORY INHERITED?


Response:

QUESTION:: HOW MANY FLOORS ARE IN THE CASTLE DAVID GREGORY INHERITED?  
REASONING: LET'S THINK STEP BY STEP IN ORDER TO: determine the number of floors in the castle that David Gregory inherited. This information typically comes from a specific source or context, such as a book, movie, or historical reference. Without that context, I cannot provide an exact number.  
ANSWER:: I do not have the specific information regarding the number of floors in the castle David Gregory inherited.
```

The above example is a simple Adapter that converts the input to uppercase before passing it to the LM. You can define more complex Adapters based on your requirements.

### Overriding `__call__` method

To gain control over usage of format and parse and even more fine-grained control over the flow of input from signature to outputs you can override `__call__` method and implement your custom flow. Although for most cases only implementing `parse` and `format` function will be fine.
