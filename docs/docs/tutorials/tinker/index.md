# Running DSPy Programs on Tinker Models

[Tinker](https://thinkingmachines.ai/tinker) hosts open-weight models and the
LoRA checkpoints you train on them. This tutorial runs a question-answering
program and a ReAct agent on one of them with `dspy.TinkerLM`.

## Prerequisites

- Python 3.11 or later, and `pip install "dspy[tinker]"`. The extra pulls in
  PyTorch and Transformers through `tinker-cookbook`, so the install is large.
- A Tinker API key in the `TINKER_API_KEY` environment variable.
- Optionally, a checkpoint you trained, as a `tinker://` sampler path in
  `TINKER_MODEL_PATH`. Without it the example uses the `Qwen/Qwen3-8B` base model.

Each run makes a few sampling calls, which Tinker bills to your account.

## The example

Save this as `tinker_example.py` and run `python tinker_example.py`:

```python
import asyncio
import os

import dspy

model_path = os.environ.get("TINKER_MODEL_PATH")
if model_path:
    lm = dspy.TinkerLM(model_path=model_path, max_tokens=2048)
else:
    lm = dspy.TinkerLM("Qwen/Qwen3-8B", max_tokens=2048)
dspy.configure(lm=lm)

qa = dspy.Predict("question -> answer")
print(qa(question="What is the largest planet in the solar system?").answer)


def population(city: str) -> str:
    """Return the population of a city."""
    return {"paris": "2.1 million", "lyon": "0.5 million"}.get(city.lower(), "unknown")


agent = dspy.ReAct("question -> answer", tools=[population], max_iters=4)
print(agent(question="How many more people live in Paris than in Lyon?").answer)


async def main():
    questions = ["What is 12 squared?", "Name a prime number above 100."]
    results = await asyncio.gather(*(qa.acall(question=q) for q in questions))
    for question, result in zip(questions, results):
        print(question, "->", result.answer)


asyncio.run(main())

qa.set_lm(lm)
qa.save("tinker_qa.json")
```

The model's tool calls arrive as native tool calls, so the ReAct agent runs
`population` and passes the results back in the model's own tool-response
format. The async calls share one sampling client.

`qa.save` writes the model, checkpoint path and generation options, but not the
API key. Load it in a process with `TINKER_API_KEY` set:

```python
restored = dspy.Predict("question -> answer")
restored.load("tinker_qa.json", allow_unsafe_lm_state=True)  # Trusted files only.
```

Qwen3's default renderer lets the model think before it answers. The thinking
is returned as a reasoning part, apart from the output fields, and counts toward
`max_tokens`. Pass `renderer="qwen3_disable_thinking"` for shorter replies. See
the [`dspy.TinkerLM` reference](../../api/models/TinkerLM.md) for the supported
options and what is refused.
