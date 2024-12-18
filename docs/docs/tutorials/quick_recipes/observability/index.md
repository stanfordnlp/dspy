# Tutorial: Debugging and Observability in DSPy

This guide demonstrates how to debug problems and improve observability in DSPy. Modern AI programs often involve multiple components, such as language models, retrievers, and tools. DSPy allows you to build nad optimize such complex AI systems in a clean and modular way.

However, as systems grow more sophisticated, the ability to **understand what your system is doing** becomes critical. Without transparency, the prediction process can easily become a black box, making failures or quality issues difficult to diagnose and production maintenance challenging.

By the end of this tutorial, you'll understand how to debug an issue and improve observability using [MLflow Tracing](#mlflow-tracing). You'll also explore how to build a custom logging solution using callbacks.



## Define a Program

We'll start by creating a simple ReAct agent that uses ColBERTv2's Wikipedia dataset as a retrieval source. You can replace this with a more sophisticated program.

```python
import dspy
from dspy.datasets import HotPotQA

lm = dspy.LM('openai/gpt-4o-mini')
colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(lm=lm, rm=colbert)

agent = dspy.ReAct("question -> answer", tools=[dspy.Retrieve(k=1)])
```

Now, let's ask the agent a simple question:

```python
prediction = agent(question="Which baseball team does Shohei Ohtani play for?")
print(prediction.answer)
```

```
Shohei Ohtani plays for the Los Angeles Angels.
```

Oh, this is incorrect. He no longer plays for the Angels; he moved to the Dodgers and won the World Series in 2024! Let's debug the program and explore potential fixes.

## Using ``inspect_history``

DSPy provides the `inspect_history()` utility, which prints out all LLM invocations made so far:

```python
# Print out 5 LLM calls
dspy.inspect_history(n=5)
```

```
[2024-12-01T10:23:29.144257]

System message:

Your input fields are:
1. `question` (str)

...

Response:

[[ ## Thought_5 ## ]]
The search results continue to be unhelpful and do not provide the current team for Shohei Ohtani in Major League Baseball. I need to conclude that he plays for the Los Angeles Angels based on prior knowledge, as the searches have not yielded updated information.

[[ ## Action_5 ## ]]
Finish[Los Angeles Angels] 

[[ ## completed ## ]]
```
The log reveals that the agent could not retrieve helpful information from the search tool. However, what exactly did the retriever return? While useful, `inspect_history` has some limitations:

* In real-world systems, other components like retrievers, tools, and custom modules play significant roles, but `inspect_history` only logs LLM calls.
* DSPy programs often make multiple LLM calls within a single prediction. Monolith log history makes it hard to organize logs, especially when handling multiple questions.
* Metadata such as parameters, latency, and the relationship between modules are not captured.

**Tracing** addresses these limitations and provides a more comprehensive solution.

## Tracing

[MLflow](https://mlflow.org/docs/latest/llms/tracing/index.html) is an end-to-end machine learning platform that is integrated seamlessly with DSPy to support best practices in LLMOps. Using MLflow's automatic tracing capability with DSPy is straightforward; **No sign up for services or an API key is required**. You just need to install MLflow and call `mlflow.dspy.autolog()` in your notebook or script.

```bash
pip install -U mlflow>=2.18.0
```

```python
import mlflow

mlflow.dspy.autolog()

# This is optional. Create an MLflow Experiment to store and organize your traces.
mlflow.set_experiment("DSPy")
```

Now you're all set! Let's run your agent again:

```python
agent(question="Which baseball team does Shohei Ohtani play for?")
```

MLflow automatically generates a **trace** for the prediction and records it in the experiment. To explore traces visually, launch the MLflow UI by the following command and access it in your browser:

```bash
mlflow ui --port 5000
```

![DSPy MLflow Tracing](./dspy-tracing.gif)

From the retriever step output, you can observe that it returned outdated information; indicating Shohei Ohtani was still playing in the Japanese league and the final answer was based on the LLM's prior knowledge! We should update the dataset or add additional tools to ensure access to the latest information.

!!! info Learn more about MLflow

    MLflow is an end-to-end LLMOps platform that offers extensive features like experiment tracking, evaluation, and deployment. To learn more about DSPy and MLflow integration, visit [this tutorial](../deployment/#deploying-with-mlflow).

For example, we can add a web search capability to the agent, using the [Tavily](https://tavily.com/) web search API.

```python
from dspy.predict.react import Tool
from tavily import TavilyClient

search_client = TavilyClient(api_key="<YOUR_TAVILY_API_KEY>")

def web_search(query: str) -> list[str]:
    """Run a web search and return the content from the top 5 search results"""
    response = search_client.search(query)
    return [r["content"] for r in response["results"]]

agent = dspy.ReAct("question -> answer", tools=[Tool(web_search)])

prediction = agent(question="Which baseball team does Shohei Ohtani play for?")
print(agent.answer)
```

```
Los Angeles Dodgers
```


## Building a Custom Logging Solution

Sometimes, you may want to implement a custom logging solution. For instance, you might need to log specific events triggered by a particular module. DSPy's **callback** mechanism supports such use cases. The ``BaseCallback`` class provides several handlers for customizing logging behavior:

|Handlers|Description|
|:--|:--|
|`on_module_start` / `on_module_end` | Triggered when a `dspy.Module` subclass is invoked. |
|`on_lm_start` / `on_lm_end` | Triggered when a `dspy.LM` subclass is invoked. |
|`on_adapter_format_start` / `on_adapter_format_end`| Triggered when a `dspy.Adapter` subclass formats the input prompt. |
|`on_adapter_parse_start` / `on_adapter_parse_end`| Triggered when a `dspy.Adapter` subclass postprocess the output text from an LM. |

Here’s an example of custom callback that logs the intermediate steps of a ReAct agent:

```python
import dspy
from dspy.utils.callback import BaseCallback

# 1. Define a custom callback class that extends BaseCallback class
class AgentLoggingCallback(BaseCallback):

    # 2. Implement on_module_end handler to run a custom logging code.
    def on_module_end(self, call_id, outputs, exception):
        step = "Reasoning" if self._is_reasoning_output(outputs) else "Acting"
        print(f"== {step} Step ===")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\n")

    def _is_reasoning_output(self, outputs):
        return any(k.startswith("Thought") for k in outputs.keys())

# 3. Set the callback to DSPy setting so it will be applied to program execution
dspy.configure(callbacks=[AgentLoggingCallback()])
```


```
== Reasoning Step ===
  Thought_1: I need to find the current team that Shohei Ohtani plays for in Major League Baseball.
  Action_1: Search[Shohei Ohtani current team 2023]

== Acting Step ===
  passages: ["Shohei Otani ..."]

...
```

!!! info Handling Inputs and Outputs in Callbacks

    Be cautious when working with input or output data in callbacks. Mutating them in-place can modify the original data passed to the program, potentially leading to unexpected behavior. To avoid this, it’s strongly recommended to create a copy of the data before performing any operations that may alter it.
