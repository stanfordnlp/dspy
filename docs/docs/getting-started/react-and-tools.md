# Give the program tools with `dspy.ReAct`

## Grounding our program with tools

Our haiku writer is competent, but it could use more information about the locations we name. When asked for a haiku set in Bodega Bay in autumn, our LM makes plausible guesses about the weather, but sometimes gets them wrong. Giving the program tools allows our program to research and ground its references before writing its verse.

## Tools are just Python functions

A DSPy tool is a standard Python function, with type-hinted parameters and a docstring. DSPy reads the name, parameters, and docstring of a function to assemble the instructions it sends to an LM.

For example, let’s define a tool that lets an agent search Wikipedia using the [`wikipedia`](https://pypi.org/project/wikipedia/) library (run `pip install wikipedia`, if you’re following along):

```py
import wikipedia

def wikipedia_search(query: str) -> list[str]:
    """Search Wikipedia for the given query and return a list of page titles."""
    return wikipedia.search(query)
```

The `dspy.ReAct` module presents it like so:

```
When selecting the next_tool_name and its next_tool_args, the tool must be one of:
        
(1) wikipedia_search, whose description is <desc>Search Wikipedia for the given query and return a list of page titles.</desc>. It takes arguments {'query': {'type': 'string'}}.
(2) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `haiku`, are now available to be extracted.</desc>. It takes arguments {}.
```

Note how DSPy presents the function name, docstring, and parameters to the model. Writing tools, like signatures, requires being mindful about naming. `wikipedia_search` and the parameter `query` are helpful names, that clearly describe their function and role.

Note that there’s a tool in the mix that we didn’t define. `finish` is a special tool used by `dspy.ReAct` that the model calls when it’s done. `dspy.ReAct` populates that one for us.

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
3. Called `finish`, then synthesized its outputs.

The code above yields:

```
Redwood shadows breathe  
Summer creek whispers old tales  
Quiet mind lingers
```

Printing `result.reasoning` produces:

```
I gathered context about Camp Meeker, California—its redwood forest, Dutch Bill Creek, and the quiet summer atmosphere—from the Wikipedia page. I focused on a pensive mood and the summer season, weaving in imagery of towering trees and a gently flowing creek. The haiku follows the traditional 5‑7‑5 syllable structure, evoking the stillness and reflection of a summer day among the redwoods.
```

## ReAct manages an agentic loop

ReAct is a test-time (or inference time) loop strategy. We hand the model a set of tools and a task. The `dspy.ReAct` module instructs the model to reason then act using its tools. When the model calls `finish`, DSPy stops the loop and runs one last synthesis pass to produce the declared output fields.

The model decides how many loops to run, but we can cap the number with the `max_iters` parameter, like so:

```
haiku_bot = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page], max_iters=4)
```

While testing this program, we saw runs where the model made as many as 8 tool calls (investigating Vermont syrup farming) and runs where it skipped research entirely.

## Inspecting the ReAct trajectory

ReAct’s returned `Prediction` instance carries a `trajectory` field: a dictionary that records each thought, tool call, and observation (what the tool returned) in order. When an agent does something surprising, the trajectory is the first thing to read.

We can print it like so:

```py
for step, value in result.trajectory.items():
    print(f"{step}: {value}")
```

For tool authoring patterns, MCP integration, and trajectory debugging, see [Tools, ReAct, and MCP](../diving-deeper/tools-react-and-mcp.md).

---

**Next:** [Composing modules →](composing-modules.md)
