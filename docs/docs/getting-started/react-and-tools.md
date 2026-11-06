# Give the program tools with `dspy.ReAct`

## Grounding our program with tools

Our haiku writer is competent, but it could use more information about the locations we name. When asked for a haiku set in Bodega Bay in autumn, our LM makes plausible guesses about the weather, but sometimes gets them wrong. Giving the program tools allows our program to research and ground its references before writing its verse.

## Tools are just Python functions

A DSPy tool is a standard Python function, with type-hinted parameters and a docstring. DSPy reads the name, parameters, and docstring of a function to assemble the instructions it sends to an LM.

For example, let’s define a tool that lets an agent search Wikipedia using the [`wikipedia`](https://pypi.org/project/wikipedia/) library (run `pip install wikipedia`, if you’re following along):

(You should also set a user agent with contact information as [required by Wikipedia's API guidelines](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy), otherwise these requests may fail due to rate limiting of the default user agent set by the library.)

```py
import wikipedia

wikipedia.set_user_agent("research-assistant/0.1 (yourname@example.com)")

def wikipedia_search(query: str) -> list[str]:
    """Search Wikipedia for the given query and return a list of page titles."""
    return wikipedia.search(query)
```

The `dspy.ReAct` module presents the model with a tool schema containing the name `wikipedia_search`, its docstring, and a required string argument named `query`. The active adapter renders this schema as native function-calling metadata or formatted prompt text.

Note how DSPy presents the function name, docstring, and parameters to the model. Writing tools, like signatures, requires being mindful about naming. `wikipedia_search` and the parameter `query` are helpful names, that clearly describe their function and role.

ReAct also adds an internal `submit` tool whose arguments are the signature's typed output fields. The model calls it with the final outputs when it is done; there is no separate extraction pass.

## Defining our Wikipedia-wielding haiku bot

In addition to searching Wikipedia, we’ll need a tool that lets our agent fetch a given page:

```py
def get_wikipedia_page(title: str) -> str:
    """Get the content of a Wikipedia page given its title."""
    return wikipedia.page(title).content
```

With these two tools and our existing `HaikuBot` signature, we have all we need to define our agent:

```py
haiku_bot = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page])
result = haiku_bot(location="Camp Meeker", mood="pensive", season="summer")
print(result.haiku)
```

When we called `haiku_bot` it:

1. Searched Wikipedia for “Camp Meeker”, which yielded nine relevant results, including “Camp Meeker, California”.  
2. Fetched the page content for “Camp Meeker, California”.  
3. Called `submit` with its final `haiku` output.

The code above yields:

```
Redwood shadows breathe  
Summer creek whispers old tales  
Quiet mind lingers
```

ReAct returns the output fields declared by `HaikuBot`, along with `history` and `termination_reason`. If you also want a final explanation, declare it as an output field in your signature; ReAct does not add an extractor-generated `reasoning` field.

## ReAct manages an agentic loop

ReAct is a test-time (or inference time) loop strategy. We hand the model a set of tools and a task. The `dspy.ReAct` module instructs the model to reason then act using its tools. Each model turn can request multiple tools, which DSPy executes sequentially. A successful `submit` ends the loop with typed output fields.

The model decides how many loops to run, but we can cap the number with the `max_iters` parameter, like so:

```
haiku_bot = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page], max_iters=4)
```

`max_iters` limits model turns, not individual tool calls. If the budget runs out before submission, ReAct makes a final call requesting `submit`. If that fails, it raises instead of returning incomplete outputs. For async tools, run the same agent with `await haiku_bot.acall(...)`.

## Inspecting ReAct history

ReAct’s returned `Prediction` carries a `history` field: a `dspy.History` containing structured events with thoughts, tool calls, and results. When an agent does something surprising, history is the first thing to read.

We can print it like so:

```py
for event in result.history.messages:
    print(event)
```

For tool authoring patterns and MCP integration, see [Tools and MCP](../diving-deeper/tools.md). For the loop's execution model, continuation history, and debugging, see [ReAct](../diving-deeper/react.md).

!!! note "Upgrading to DSPy 3.5"
    Structured-history execution is now canonical `dspy.ReAct`. The old trajectory, extractor, and truncation APIs are gone, and old optimized state must be recompiled for the new predictor schema. `dspy.ReActV2` is a deprecated compatibility subclass that warns on construction and will be removed in 3.6. See [Migration to DSPy 3.5](../diving-deeper/react.md#migration-to-dspy-35).

---

**Next:** [Composing modules →](composing-modules.md)
