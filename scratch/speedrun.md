# Intro to DSPy Speedrun

DSPy is a framework for programming language models through structured declarations rather than hand-tuned prompt strings. You describe what you want using signatures, choose a module that defines how the model should reason, and let DSPy handle prompt construction, output parsing, and execution. When you need better performance, you write a metric and run an optimizer to automatically improve your program's instructions.

This guide covers the core primitives you need to go from zero to a compiled, optimized DSPy program. It assumes you have a Python environment ready and an API key for your preferred language model provider.

## 1. Install and Configure

DSPy installs from PyPI and requires Python 3.9 or newer. It communicates with language model providers through LiteLLM, which means you can use virtually any commercial or self-hosted endpoint without changing your application code. You simply swap the model string.

The first step in any DSPy project is creating an LM instance and configuring it as the default. Once configured, every module in your process will use that model unless you override it explicitly. You can also set a global default and then switch to a different model for specific calls using a context manager, which is useful when you want a cheap model for most work and a powerful model for a specific judgment step.

```python
import dspy

lm = dspy.LM("openai/gpt-5-nano", api_key="YOUR_KEY")
dspy.configure(lm=lm)
```

## 2. Signatures Define the Contract

A signature is the central declaration in DSPy. It specifies the input fields your program will receive and the output fields it must produce. The simplest signatures are strings with an arrow separating inputs from outputs. DSPy reads the field names and uses them to construct prompts and parse responses, which means naming matters significantly. A field named `research_request` will yield better results than one named `x` because the model infers intent from your identifiers.

You can declare multiple inputs and outputs by separating field names with commas. You can also add inline types using the `name: type` syntax. DSPy will attempt to coerce model outputs into these types and warn you when parsing fails or when inputs violate the declared schema. Using types turns a class of silent prompt failures into explicit errors that you can catch during development.

String signatures are great for quick prototyping. Class-based signatures give you more control: a docstring becomes task instructions, and each field can carry a description that adds nuance beyond what the field name conveys. Class-based signatures also support richer Python types like Pydantic models, `TypedDict`, `Literal` enums, and dataclasses, letting you enforce structure at the boundary between your code and the language model.

```python
# String signature
bot = dspy.Predict("location, mood -> haiku")

# Typed fields
bot = dspy.Predict("location, mood, contains_pun: bool -> haiku")

# Class-based signature
class HaikuBot(dspy.Signature):
    """Write a classical haiku given the provided inputs."""
    location: str = dspy.InputField(desc="The setting of the poem")
    mood: str = dspy.InputField()
    haiku: str = dspy.OutputField()

bot = dspy.Predict(HaikuBot)
result = bot(location="a quiet library", mood="mysterious")
```

## 3. Modules Implement Strategies

If a signature is the spec, a module is the implementation. Modules determine the test-time strategy used to fulfill the signature. The most basic module is `Predict`, which builds a prompt from the signature and inputs, sends it to the configured LM, parses the response into the declared output fields, and returns a `dspy.Prediction` object. You access outputs as attributes on that object.

`ChainOfThought` extends `Predict` by asking the model to produce a reasoning step before its final answer. This often improves quality on tasks that benefit from explicit intermediate thinking. The reasoning text is available on the prediction result.

`ReAct` is an agentic module that gives the model access to tools. It runs a loop where the model reasons, chooses a tool to call, observes the result, and repeats until it decides it has enough information to finish. You can cap the loop with the `max_iters` parameter.

The important thing is that your signature stays the same across modules. You can start with `Predict`, swap in `ChainOfThought` for better reasoning, or upgrade to `ReAct` for tool use without rewriting your interface.

```python
# Basic inference
bot = dspy.Predict("subject -> haiku")

# Reasoning before output
bot = dspy.ChainOfThought(HaikuBot)
result = bot(location="Bodega Bay", mood="mysterious", season="autumn")
print(result.reasoning)

# Tool-using agent
bot = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page], max_iters=5)
```

## 4. Tools and ReAct

Tools in DSPy are standard Python functions with type hints and docstrings. DSPy reads the function name, parameter types, and docstring to present the tool to the language model. Good naming matters here too: a function called `wikipedia_search` with a parameter called `query` is easier for the model to use correctly than vague names.

When you pass tools to `ReAct`, DSPy automatically adds a special `finish` tool that the model calls when it believes the task is complete. The module then runs a synthesis step to produce the final output fields. Each tool call and observation is recorded in a trajectory dictionary on the result, which you should inspect when an agent behaves unexpectedly.

```python
import wikipedia

def wikipedia_search(query: str) -> list[str]:
    """Search Wikipedia and return page titles."""
    return wikipedia.search(query)

def get_wikipedia_page(title: str) -> str:
    """Get the content of a Wikipedia page."""
    return wikipedia.page(title).content

bot = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page], max_iters=5)
result = bot(location="Camp Meeker", mood="pensive", season="summer")

# Inspect the agent's trajectory
for step, value in result.trajectory.items():
    print(f"{step}: {value}")
```

## 5. Composing Custom Modules

For complex tasks, you subclass `dspy.Module` and compose smaller modules together. You define submodules inside `__init__` and wire their execution inside `forward`. This pattern lets you isolate context, reuse components across projects, route expensive calls to cheaper models, implement custom control flow, and inspect each step independently.

A common pattern is an ensemble: generate multiple candidates with one module, then judge them with another. You can use `dspy.context` to temporarily switch to a different language model for a specific submodule call. For example, you might use a fast cheap model to generate haikus and a large frontier model to pick the best one.

```python
class HaikuEnsemble(dspy.Module):
    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n
        self.writer = dspy.ReAct(
            "location, season, mood, num_haikus: int -> haikus: list[str]",
            tools=[wikipedia_search, get_wikipedia_page],
            max_iters=5
        )
        self.judge = dspy.ChainOfThought(
            "location, season, mood, candidates: list[str] -> most_evocative_index: int"
        )

    def forward(self, location: str, season: str, mood: str):
        candidates = self.writer(
            location=location, season=season, mood=mood, num_haikus=self.n
        ).haikus

        with dspy.context(lm=dspy.LM("openai/gpt-5.4")):
            verdict = self.judge(
                location=location, season=season, mood=mood, candidates=candidates
            )

        return dspy.Prediction(
            haiku=candidates[verdict.most_evocative_index],
            candidates=candidates,
            reasoning=verdict.reasoning,
        )
```

## 6. Metrics and Evaluation

Before optimizing, you need a metric: a Python function that scores a single prediction, typically returning a float between 0.0 and 1.0. Metrics can compare against labeled data, apply rule-based checks, or call another language model as a judge. The optimizer uses this score to decide which instruction variations to keep.

You prepare evaluation data as `dspy.Example` objects and mark which fields are inputs using `with_inputs`. This lets DSPy pass only the relevant fields to your program while holding back gold-standard answers for the metric.

```python
def haiku_score(example, prediction) -> float:
    text = prediction.haiku.lower()
    if example.season.strip().lower() in text:
        return 0.0
    return 1.0

example = dspy.Example(
    location="Bodega Bay", season="autumn", mood="inspired"
).with_inputs("location", "season", "mood")

evaluate = dspy.Evaluate(devset=val_set, metric=haiku_score)
score = evaluate(bot)
```

## 7. Optimization with GEPA

Prompt optimization replaces hand-tuning. You provide a training set, a validation set, and a metric. The optimizer generates variations of your instructions, runs your program against examples, scores the results, and keeps the best candidates. GEPA specifically uses a reflection language model to analyze failures and rewrite instructions based on metric feedback.

The `reflection_lm` is usually a larger, more capable model than the one running your program, since it is only called a handful of times during optimization. The `auto` parameter controls budget: `light` tries around six prompts, while `medium` and `heavy` go further. Behind the scenes, compile runs an iterative loop of execute, reflect, and retry until the budget is exhausted.

```python
optimizer = dspy.GEPA(
    metric=haiku_metric,
    reflection_lm=dspy.LM("openai/gpt-5.4"),
    auto="light",
    num_threads=2,
)

optimized = optimizer.compile(bot, trainset=train, valset=val)
```

## 8. Save and Load

DSPy offers two save modes. Saving to a JSON file stores only the learned state: optimized instructions and few-shot demonstrations. This file is small, human-readable, safe for version control, and requires you to reinstantiate the program structure in code before loading. Saving to a directory with `save_program=True` pickles the entire module including its structure. Use this when the loader will not have your Python class definitions, but treat it as executable code and only load from trusted sources.

Saved state does not include LM configuration. You must reconfigure your model after loading, which means the same optimized program can target different providers or model sizes without modification.

```python
# State only
optimized.save("haiku_bot.json")

fresh = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page])
fresh.load("haiku_bot.json")

# Full program
optimized.save("haiku_bot/", save_program=True)
loaded = dspy.load("haiku_bot/")
```

## 9. Debug

Use `inspect_history` to view the exact prompts and responses for recent calls. This is the first tool to reach for when outputs are unexpected.

```python
dspy.inspect_history(n=1)
```
