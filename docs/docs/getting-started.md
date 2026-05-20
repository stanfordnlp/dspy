# Getting Started with DSPy

## Program, don’t prompt

DSPy is a declarative way to build with LLMs. We describe each step as structured inputs and outputs, then compose those steps into programs where each piece stays independently inspectable, swappable, and tunable. DSPy handles prompt construction, context management, and optimization that tunes each step to improve the whole program.

When a step needs to act on the world (to search, fetch, or compute), we give it tools as plain Python functions. Functions also guide DSPy in optimizing your steps: you define a metric to grade the output, and DSPy generates a better prompt accordingly. DSPy programs don't just run against any LM; the optimizer tunes the prompt for the specific model you pick, so a small, cheap model can often match or beat a hand-prompted frontier one.

DSPy helps us program LLMs, rather than prompting them, creating modular, maintainable and optimizable AI software.

### What we'll learn today

In this tutorial we'll build a haiku-writing program that starts with four lines of Python and grows into a tool-using, prompt-optimized agent. Along the way we'll touch each of DSPy's core components. We'll learn:

- How to install DSPy, configure a **language model** and write a simple DSPy program.  
- What a **Signature** is, and why DSPy uses signatures instead of hand-written prompt strings.  
- What a **Module** is, how `Predict`, `ChainOfThought`, and `ReAct` differ, and when to reach for each.  
- How to compose a custom `dspy.Module` that decomposes a task into named, independent stages.  
- How to write **metrics** and use **optimizers** to compile better versions of our program.  
- How to save optimized programs and reload them.

---

## 1\. Installing DSPy and connecting to a language model

Install DSPy with `pip install dspy` or run `uv add dspy` to add DSPy to a virtual environment. DSPy works in Python 3.10+ environments.

### Connecting to a language model

DSPy connects to language models with the `dspy.LM` class. To set up a language model, we provide a `"provider/model"` format string and an API key:

```py
import dspy

# Pass the key explicitly...
lm = dspy.LM("openai/gpt-5-nano", api_key="YOUR_OPENAI_API_KEY")

dspy.configure(lm=lm)
```

Behind the scenes, DSPy uses the [LiteLLM](https://docs.litellm.ai/docs/#litellm-python-sdk) library to normalize inference providers into a single format. This allows you to provide a LiteLLM model string and connect to nearly any model and its provider. [Click here to search for the full list](https://models.litellm.ai/).

For this tutorial, we could replace the `"openai/gpt-5-nano"` model string and our OpenAI API key with options from Anthropic, Google, OpenRouter, and more. All the examples below will work with any LiteLLM model string, without code changes.

Once we have an `LM`, calling `dspy.configure(lm=lm)` sets our `LM` as the default provider for every DSPy program in the process. This sets our `LM` globally, but we can selectively override this with [`dspy.context`](https://dspy.ai/api/utils/context/?h=dspy+context) when more granular control is needed.

Let's ensure everything works by manually calling the model:

```py
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]

print(lm(messages = messages))

```

Which returns:

```
['The capital of France is Paris.']
```

With an `LM` configured, we can proceed to writing our first program.  
---

## 2\. Writing our first DSPy program

In this tutorial we're building a haiku-generating program, extending it in each section to introduce new DSPy concepts and capabilities.

Let’s start by writing the simplest version of our program and run it, then walk through everything DSPy is doing behind the scenes:

```py
haiku_signature = "subject -> haiku"
haiku_generator = dspy.Predict(haiku_signature)
result = haiku_generator(subject="computer science")
print(result.haiku)
```

The first line specifies our `Signature`. Signature is a core concept in DSPy, and it’s how we define our task. Similar to function signatures in programming, a DSPy Signature describes the inputs a function accepts and the outputs it returns. 

The simplest way to define a DSPy Signature is with a string of the form `"inputs -> outputs"`. In our case, we want to provide a `subject` and get back a `haiku`.

Because we are programming with language models, the names of our variables matter. They both define the interface for our program and give the model a hint at our intent. If we change `haiku` to `limerick`, the model would note our cue and produce a limerick instead. Additionally, our program’s output would be accessible as `result.limerick`, rather than `result.haiku`.

To turn our Signature into a callable function, we use `dspy.Predict`. Predict is a kind of DSPy `Module`. If Signatures specify *what* we want, Modules define *how* we aim to achieve it. They implement a call-time strategy, manage the control flow, tools, and more. 

`Predict` is the foundational Module. Let’s look at what happens when we call `dspy.Predict(haiku_signature)`:

1. The string, `“subject -> haiku”` is parsed into a Signature class, with input and output fields defaulting to type `str`.  
2. A default instruction string is generated for the `Signature` instance, in this case: "Given the fields `subject`, produce the fields `haiku`."  
3. A `Predict` module is instantiated with this `Signature`.

`haiku_generator` is now callable. Calling `haiku_generator(subject=”computer science”)` kicks off the following process:

1. The DSPy settings are checked to ensure an `LM` is configured.  
2. An `Adapter` is used to render the `Signature` and its inputs into messages the `LM` can consume. By default, this is the `ChatAdapter`, but there are JSON, XML, and other variants to format your messages suitable for a given `LM`.  
3. The `ChatAdapter` builds the prompt, which includes the `Signature` instructions, the field schema describing the inputs and outputs, formatting instructions, and the provided input (in this case, "computer science").   
4. The messages are sent to the `LM`. Caching is enabled by default so identical calls return cached responses.  
5. A response is returned, which the `ChatAdapter` parses to extract the output fields.  
6. The `Predict` module returns a `Prediction` object with accessible output fields. Calling `result.haiku` returns the generated haiku.  
7. The call is recorded in `LM` history, which can be inspected later with `dspy.inspect_history()`.

Running `print(result.haiku)` produces:

```
Silent code unfolds
Logic threads through hidden paths
Bugs bloom, then resolve
```

In just four lines we built an AI-powered program that reads like software and acts like a function.

DSPy manages all the prompting and templating. Call `dspy.inspect_history(n=1)` to take a look at the formatted prompt our program produced and the string the `lm` returned.

First, DSPy generated the system instructions:

```
Your input fields are:
1. `subject` (str):
Your output fields are:
1. `haiku` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## subject ## ]]
{subject}

[[ ## haiku ## ]]
{haiku}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
    Given the fields `subject`, produce the fields `haiku`.
```

The brackets and hashes style is how the default `ChatAdapter` structures the prompt, demarcating inputs and sections. 

Next, the adapter templated our input into a user message:

```
[[ ## subject ## ]]
computer science

Respond with the corresponding output fields, starting with the field `[[ ## haiku ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.
```

DSPy sent both to the `lm`, which produced this response:

```
[[ ## haiku ## ]]
Silent code unfolds
Logic threads through hidden paths
Bugs bloom, then resolve

[[ ## completed ## ]]
```

In DSPy, to produce a prompt you compose a signature and the messages are written for you. 

---

## 3\. Expanding your signature: more inputs and adding types

### Adding additional inputs & outputs

Adding more fields to signature strings is as easy as separating field names with commas. For example, let's update our program to accept two inputs, a `location` and `mood`:

```py
haiku_bot = dspy.Predict("location, mood -> haiku")
result = haiku_bot(location="a quiet library", mood="mysterious")
print(result.haiku)
```

This yields:

```
Books whisper in hush  
Ink curls in soft dim corners  
Night reads in quiet
```

Defining additional outputs works identically:

```py
haiku_bot = dspy.Predict("location, mood -> haiku, haiku_title")
result = haiku_bot(location="a quiet library", mood="mysterious")
print(result.haiku_title)
print("- - -")
print(result.haiku)
```

Yielding:

```
Where the Pages Breath
- - -
Books whisper in hush  
Ink curls in soft dim corners  
Night reads in quiet
```

### Hone your signature by mindfully naming your fields

The field names we choose aren't just for our own readability. Unlike traditional programming, where variable names are purely identifiers, the LM reads them too, and uses them to infer what each input and output means.

If we replaced `"location, mood -> haiku"` with `"a, b -> c"`, the LM would be lost. Let's try it:

```py
haiku_bot = dspy.Predict("a, b -> c")
result = haiku_bot(a="a quiet library", b="mysterious")
print(result.c)
```

This produces:

```
In the quiet library, a mysterious presence seemed to linger between the shelves, as if it had been waiting for someone to notice.
```

Our model doesn't know we want a haiku, so it just makes a guess.

Naming is the cheapest optimization in DSPy. A field called `research_request` will produce better completions than one called `request`, with no other changes. Signatures are easy to edit; take advantage.

### Typing your fields yields more reliable programs

We can add more specificity to our task by *typing* our fields using the format `name: type`. For example, the signature `"location, mood, contains_pun: bool -> haiku"` accepts a boolean to indicate whether we want our poem to include a pun:

```py
haiku_bot = dspy.Predict("location, mood, contains_pun: bool -> haiku")
result = haiku_bot(location="a quiet library", mood="mysterious", contains_pun=True)
print(result.haiku)
```

Which yields:

```
Dusty hush of books  
Spine turns to hush of riddles  
Novel jokes echo
```

Inline types instruct DSPy to coerce the LM's output into the types we ask for, and surface clear warnings when they can't. This catches a class of silent failures that prompt-only systems hide.

Types also let us communicate structural details that are easier to express in code than in natural language. Richer types – like Pydantic models, `TypedDicts`, or `dataclasses` – can pack plenty of details that help LMs correctly complete a task.

This is especially helpful when typing output fields. For example, if we wanted to modify our program to generate several haikus we could make our output field name plural and type it as a `list[str]`:

```
haiku_bot = dspy.Predict("location, mood -> haikus:list[str]")
result = haiku_bot(location="a sunny beach", mood="relaxed")
print(f"Generated {len(result.haikus)} haikus, here's the first:")
print(result.haikus[0])
```

Which yields:

```
Generated 3 haikus, here's the first:
Sun on sandy toes,
waves breathe slow, worries drift away—
soft light, easy air.
```

Once a program accrues several fields or we want to add nuanced instructions, it's likely time to graduate to a class-based signature.

---

## 4\. Writing a class-based signature

### Adding instructional nuance with class-based signatures

A class-based signature details the same structure a string signature can, but adds a few levers for adding additional nuance. Here's our haiku writer string signature rephrased as a class-based signature:

```py
class HaikuBot(dspy.Signature):
    """
    Write a classical haiku given the provided inputs.
    """
    location: str = dspy.InputField(desc="The setting of the poem")
    mood: str = dspy.InputField()
    haiku: str = dspy.OutputField()
```

We have the same fields (`location`, `mood`, and the `haiku` output) typed as strings, but we now have the ability to add descriptions to each. Field descriptions allow us to add nuance that might not fit within a field name.

Class-based signatures also let us write a docstring, the string at the start of the class, which DSPy uses as task instructions when preparing prompts.

We pass class-based signatures to modules just like we passed string signatures (it is the same object after all, just another way of making it):

```py
haiku_bot = dspy.Predict(HaikuBot)
result = haiku_bot(location="a quiet library", mood="mysterious")
print(result.haiku)
```

This call renders and sends similar instructions to the LM, with two exceptions. The docstring and any field descriptions are used when building the system instructions, like so:

```
Your input fields are:
1. `location` (str): The setting of the poem
2. `mood` (str):
Your output fields are:
1. `haiku` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## location ## ]]
{location}

[[ ## mood ## ]]
{mood}

[[ ## haiku ## ]]
{haiku}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
    Write a classical haiku given the provided inputs.
```

Signature docstrings and field descriptions are optional, but they are handy levers when field names don't provide sufficient context for a task. However, resist the urge to restate what the signature already says or write prescriptive tutorials. Expansive rules, watch-outs, and guidance are what optimizers are for (more on that later...). 

Though it’s worth noting: field descriptions are not touched by the optimizers, so mind your naming. A poorly chosen field name can’t be adjusted by optimizers.

### Tightening signature fields with richer types

Sometimes a plain `str` is too loose. When a value should come from a small fixed set, we'd rather pin it down so the LM (and the caller) can't drift outside it. This is the unit-test framing from Section 3 made stricter: not just *some string*, but *one of these specific strings*.

We can use `typing`, from Python’s standard lib, to add richer types.

Typing `season` as `Literal["spring", "summer", "autumn", "winter"]` does exactly that. DSPy now accepts only those four values, both at call time and when parsing the LM's response.

```py
from typing import Literal

Season = Literal[
    "spring", "summer", "autumn", "winter",
]

class HaikuBot(dspy.Signature):
    """
    Write a classical haiku given the provided inputs.
    """
    location: str = dspy.InputField()
    mood: str = dspy.InputField()
    season: Season = dspy.InputField()
    haiku: str = dspy.OutputField()

haiku_bot = dspy.Predict(HaikuBot)
result = haiku_bot(location="Bodega Bay", mood="mysterious", season="autumn")
print(result.haiku)
```

This yields:

```
Fog drifts over waves,  
Crimson leaves swirl by the pier—  
Night whispers secrets.
```

But if we pass `season="fall"`, we get a warning explaining the mismatch:

```
WARNING dspy.predict.predict: Type mismatch for field 'season': expected Literal['spring', 'summer', 'autumn', 'winter'] based on given Signature, but the provided value is incompatible: fall.
```

## 5\. Change inference strategies by changing the module

In our previous examples, we used the `Predict` module to execute our signature.

Other modules define different strategies for executing a task, and trying them out is very simple:

```py
reasoning_haiku_bot = dspy.ChainOfThought(HaikuBot)
result = reasoning_haiku_bot(location="Bodega Bay", mood="mysterious", season="autumn")
print(result.haiku)
```

Which yields:

```
Fog curls over waves  
Leaves whisper, gulls glide at night  
Moon pierces deep mist
```

Instead of `Predict` we used `ChainOfThought`, a module that prompts the LM to reason before delivering a final answer. Our signature, the class-based `HaikuBot`, is the same. The function call, identical.

But behind the scenes, DSPy modified our signature to prompt the model to reason before producing its final poem. The newly added signature output field, `reasoning`, is now populated by the model’s rationale.

We can view this by printing `result.reasoning`:

```
Bodega Bay in autumn evokes a cool, misty atmosphere, and the mysterious mood invites subtle, haunting imagery. I chose fog, waves, and gulls to ground the poem in the specific coastal setting, while autumnal leaves signal the season. The haiku’s 5‑7‑5 syllable structure preserves the traditional form, allowing each image to resonate briefly. The first line establishes the misty shoreline, the second line introduces quiet, nocturnal movement, and the third line offers a luminous, almost spectral image of the moon cutting through the fog—capturing mystery and the fleeting quality of autumn evenings.
```

`ChainOfThought` is simple, but nicely demonstrates the flexibility of DSPy's module layer. Other modules implement other strategies, including model ensembles, coding sandboxes, and *tool-calling*, which the next section explores.

## 6\. Give the program tools with `dspy.ReAct`

### Grounding our program with tools

Our haiku writer is competent, but it could use more information about the locations we name. When asked for a haiku set in Bodega Bay in autumn, our LM makes plausible guesses about the weather, but sometimes gets them wrong. Giving the program tools allows our program to research and ground its references before writing its verse.

### Tools are just Python functions

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

Note that there's a tool in the mix that we didn't define. `finish` is a special tool used by `dspy.ReAct` that the model calls when it's done. `dspy.ReAct` populates that one for us.

### Defining our Wikipedia-wielding haiku bot

In addition to searching Wikipedia, we'll need a tool that lets our agent fetch a given page:

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

1. Searched Wikipedia for "Camp Meeker", which yielded nine relevant results, including "Camp Meeker, California".  
2. Fetched the page content for "Camp Meeker, California".  
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

### ReAct manages an agentic loop

ReAct is a test-time (or inference time) loop strategy. We hand the model a set of tools and a task. The `dspy.ReAct` module instructs the model to reason then act using its tools. When the model calls `finish`, DSPy stops the loop and runs one last synthesis pass to produce the declared output fields.

The model decides how many loops to run, but we can cap the number with the `max_iters` parameter, like so:

```
haiku_bot = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page], max_iters=4)
```

While testing this program, we saw runs where the model made as many as 8 tool calls (investigating Vermont syrup farming) and runs where it skipped research entirely.

### Inspecting the ReAct trajectory

ReAct's returned `Prediction` instance carries a `trajectory` field: a dictionary that records each thought, tool call, and observation (what the tool returned) in order. When an agent does something surprising, the trajectory is the first thing to read.

We can print it like so:

```py
for step, value in result.trajectory.items():
    print(f"{step}: {value}")
```

---

## 7\. Composing your own module

Switching to `dspy.ReAct` turned our single-step program into a multi-step agent. It's a significant change, but `ReAct` itself is simple to understand.

Inside `dspy.ReAct`, other DSPy modules are composed together. Each step where the model considers its inputs and picks the next tool is a `dspy.Predict` module. A bit of code manages the control flow, looping through `Predict` calls until the model calls `finish` or hits `max_iters`. The synthesis step that follows, where the model assembles the answer from everything it's learned, is a `ChainOfThought` module.

Modules are modular. We can compose them together and they just *work*.

To illustrate, let's build a custom module for our haiku writer composed of two modules: one to generate several candidate haikus and one to pick the best verse.

### Composing modules to build an ensemble

LLM-written poems are a roll of the dice. Sometimes their haikus are evocative; other times they're predictable and bland. To increase our program's odds of success, we're going to roll the dice several times, then select the best candidate.

Here's what this looks like:

```py
class HaikuEnsemble(dspy.Module):
    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n  
        # Module 1 generates several haikus
        self.writer = dspy.ReAct(
            "location, season, mood, num_haikus: int -> haikus: list[str]", 
            tools=[wikipedia_search, get_wikipedia_page],
            max_iters=5
        )
        # Module 2 picks the most evocative
        self.judge = dspy.ChainOfThought(
            "location, season, mood, candidates: list[str] -> most_evocative_index: int"
        )

    def forward(self, location: str, season: str, mood: str) -> dspy.Prediction:
        candidates = self.writer(
            location=location, season=season, mood=mood, num_haikus=self.n,
        ).haikus
        verdict = self.judge( 
            location=location, season=season, mood=mood, candidates=candidates,
        )
        return dspy.Prediction(
            haiku=candidates[verdict.most_evocative_index],
            candidates=candidates,
            reasoning=verdict.reasoning,
        )
```

When constructing a module, we need to write two functions:

1. `__init__` sets up our initial state and defines our submodules.  
2. `forward` handles what happens when we call our program, accepting inputs and shepherding through our submodules before returning an assembled output.

Our `HaikuEnsemble` defines two submodules in `__init__`.

1. `writer` is similar to our last `ReAct` program. We've added a new input field, `num_haikus`, specifying how many haikus we want the model to draft. And we've changed our output field to return a `list` of strings.  
2. `judge` is entirely new. It accepts the `location`, `season`, and `mood` inputs in addition to the candidate `haikus`. It selects the most evocative of the bunch.

When we call this program, our `forward` method runs each module in sequence, then returns a single `dspy.Prediction` object containing our results.

If we call:

```py
ensemble = HaikuEnsemble(n=5)
result = ensemble(location="Bodega Bay", season="autumn", mood="inspired")
```

We get the following selected haiku:

```
Mist hugs the harbor
Crimson gulls glide over tide
Leaves whisper to fog
```

And the following reasoning:

```
The prompt asks for the candidate that most vividly evokes Bodega Bay in autumn with an inspired mood. 
 
Bodega Bay is characterized by its foggy harbor, sea gulls, and a crisp coastal environment.  

The first candidate (“Mist hugs the harbor\nCrimson gulls glide over tide\nLeaves whisper to fog”) captures the mist‑laden harbor and the motion of gulls, tying the leaves’ whisper to the tide—a direct reference to the coastal setting and autumnal atmosphere.  

Other candidates highlight fog or forest imagery, sea‑scapes, or harvest motifs but do not simultaneously convey the harbor, gulls, and mist as strongly. Therefore, the first candidate best aligns with the location, season, and mood.
```

### Using a bigger model as our judge

To make this module a true ensemble, let's use a different model to grade the work of our haiku writer. We only need to add one line:

```py
class HaikuEnsemble(dspy.Module):
    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n  
        # Module 1 generates several haikus
        self.writer = dspy.ReAct(
            "location, season, mood, num_haikus: int -> haikus: list[str]", 
            tools=[wikipedia_search, get_wikipedia_page],
            max_iters=5
        )
        # Module 2 picks the most evocative
        self.judge = dspy.ChainOfThought(
            "location, season, mood, candidates: list[str] -> most_evocative_index: int"
        )

    def forward(self, location: str, season: str, mood: str):
        candidates = self.writer(
            location=location, season=season, mood=mood, num_haikus=self.n,
        ).haikus
        # Call a much larger model to evaluate our haikus
        with dspy.context(lm=dspy.LM("openai/gpt-5.4")):
            verdict = self.judge( 
                location=location, season=season, mood=mood, candidates=candidates,
            )
        return dspy.Prediction(
            haiku=candidates[verdict.most_evocative_index],
            candidates=candidates,
            reasoning=verdict.reasoning,
        )
```

The `with dspy.context()` statement allows us to define a new context that sets a new model for the judge call.

### Decompose to isolate, reuse, govern, and optimize

Our haiku task is a small example, but building `HaikuEnsemble` demonstrates how easily we can decompose our programs when necessary. There's no esoteric chaining API; modules are just Python and the DSPy primitives `Signature`, `Module`, and `LM`.

Reasons to decompose appear as our AI programs grow in complexity and we learn their failure modes. For example, we can use custom modules to:

- **Isolate context:** A wandering investigation step can entertain many candidate subjects, but the final haiku writing call should only see the chosen one. Splitting them keeps each module focused on its own job.  
- **Reusable parts across programs:** A well-tuned Wikipedia-research submodule isn't haiku-specific; it could assist any program that needs trusted grounding. Decomposing lets us reuse it across programs.  
- **Route easy work to cheaper models:** Lots of small model calls can gather grounding details quickly and cheaply, while a single call to a stronger model handles the final, nuanced composition.  
- **Design custom control flow:** We could run the final haiku through a syllable-count check with an NLP library, and call the writer again if the meter is off.  
- **Govern independently inspect steps:** Each submodule is its own object, and each call lands in `inspect_history` separately. When the agent does something surprising, we can call one submodule on its own to see exactly what it returned. For haiku writing this is less of a concern, but for high-risk tasks this ability to audit is critical.  
- **More easily evaluate & optimize:** Scoring a single haiku in isolation is hard; "good" cuts across too many dimensions to grade cleanly. Picking the best of three is much easier. Decomposing isolates subtasks we can actually score, which enables evaluation and optimization.

Speaking of evaluation and optimization, it's time for the next section.

---

## 8\. Building metrics for evaluation & optimization

### Why optimizers need metrics

Before DSPy can improve our program automatically, we need to tell it what "better" means. That starts with a metric and a baseline evaluation. In this section we'll see why optimization beats hand-tuning, design a quantifiable metric for our haiku task, and run a baseline score so we have something to beat.

So far we've written DSPy programs by hand. Our signatures and instructions are only as good as what we type, with no examples to learn from. The quality of our haikus depends on the base knowledge a given model has about locations and writing haikus.

*Optimizers* close that gap. An optimizer is an algorithm that improves your program automatically, and DSPy ships several. Some select small input/output pairs (few-shot examples) to include in the prompt as demonstrations. Others rewrite the natural-language instructions in your signatures. A few go further and fine-tune the underlying model. What they share is a loop: run your program many times, keep the version that scores best.

To decide what "better" means, an optimizer needs a *metric*: a Python function that scores a single prediction. There are many ways to define your metric, but most use one or more of these patterns:

- **Labeled data comparisons:** Compare the prediction against a "gold" output, usually labeled by a human expert. For example, a dataset of five-haiku arrays each paired with one ideal pick would let us evaluate and optimize the ensemble judge.  
- **Rule-based checks:** Evaluate with code that checks verifiable properties. For our haiku program, this might involve counting lines and syllables in a haiku.  
- **LLM judges:** Evaluate with another, usually larger, model. We might ask a frontier model how well a smaller model's haiku evokes a place and season, similar to our ensemble's judge. But here the optimizer bakes those judgments into the smaller model's instructions once, instead of calling the judge with every inference.

You can get creative and clever with these patterns. For example, we could measure our program's ability to evoke a specific place by generating sets of haikus from varied location inputs, presenting an LLM judge with the array, and asking it to pick the verse matching a given location.

Today, however, we're going to keep it simple. We're going to use a *quantifiable metric* that checks a haiku's syllable count, line count, tense, first-person usage, and whether it echoes our input terms verbatim.

### Preparing examples from the haiku dataset

We've created a dataset of example inputs by randomly grouping locations, seasons, and mood strings. Click here to download the JSONL file, containing 800 rows.

To prepare them for an evaluation or optimization, we need to convert each record into a `dspy.Example` object, like so:

```py
import json

examples = []
with open("haiku_examples.jsonl") as f:
    for line in f:
        row = json.loads(line)
        examples.append(
            dspy.Example(
                location=row["location"],
                season=row["season"],
                mood=row["mood"],
            ).with_inputs("location", "season", "mood")
        ) 

n = len(examples)
train_end = int(n * 0.75)
val_end = int(n * 0.875)
train, val, test = examples[:train_end], examples[train_end:val_end], examples[val_end:]
```

`.with_inputs` is how you tell DSPy which fields of an `Example` should be passed to the program at call time. In this dataset every field is an input, because our metric will score the haiku against its own rules rather than against a labeled answer. A question-answering dataset is the more typical shape: each row carries `(question, answer)`, you'd call .with\_inputs("question")`, and DSPy would pass only the question to the program while holding` answer\` back for the metric.

### Building our evaluation metric

A metric function for `dspy.Evaluate` accepts the original `example` and the program's `prediction`, and returns a single float. By convention, scores fall in between 0.0 and 1.0 (higher is better).

For our evaluation metrics, let's keep it super simple and only check if our season or mood inputs are being used verbatim in the haiku:

```py
def haiku_score(example, prediction) -> float:
    """
    Penalize verbatim use of the input season string.
    A haiku should evoke the season through imagery, not name it
    directly.
    """
    text = prediction.haiku.lower()
    if example.season.strip().lower() in text:
        return 0.0
    return 1.0
```

Then we call `Evaluate`:

```py
evaluate = dspy.Evaluate(devset=val, metric=haiku_score)
baseline_score = evaluate(haiku_bot)
print(baseline_score)
```

Our `dspy.ReAct` haiku writer, powered by `gpt-5.4-nano`, scores only 30% on this metric. And that makes sense\! Our rule isn't a hard and fast rule of haikus, so our model won't adhere to it driven only by its weights. And our signatures gave no clues this was our objective.

## 9\. Prompt Optimizing with GEPA

### Why we optimize prompts

Different language models respond differently to the same prompt. A request that gets clean structured output from one model can confuse another into rambling. Phrasing that works on today's model can stop working when the provider ships an update next month. [Sometimes a model can't stop talking about goblins](https://openai.com/index/where-the-goblins-came-from/), for some reason. Learning each model's quirks by hand is slow, and the work doesn't transfer.

And even if we stuck with only *one* model, the potential permutations of our word choices and instructions are nearly infinite.

Prompt optimization replaces that hand-tuning loop. We give DSPy a training set and a metric. It then generates variations of your instructions (using an LM), runs your examples with these candidate instructions, and keeps the highest-scoring prompts. We don't have to know which prompt tricks a particular model responds to; the optimizer finds them.

The savings can be large. An optimized program on a smaller, cheaper model can often match a hand-prompted large model on the same task. Reliability improves as well, because the optimizer selects for what the metric rewards. 

[Shopify converted a single-prompt GPT-5 task to DSPy](https://www.youtube.com/watch?v=bxToahwOVpY&t=1404s), moved to a small Qwen model, and optimized with GEPA, yielding a solution \~75x cheaper and \~2x more reliable. [Dropbox used DSPy and prompt optimizers](https://dropbox.tech/machine-learning/optimizing-dropbox-dash-relevance-judge-with-dspy) to move to a smaller model and double program accuracy, labeling “10-100 times more data at the same cost.”

Better yet: when a new model launches next week, we can rerun the optimizer against it and immediately determine if it's worth swapping in.

### GEPA uses reflection to improve instructions

DSPy ships with several prompt optimizers, but today we're going to focus on GEPA.

There are many reasons to like GEPA, but a key feature is it allows our metric to provide text feedback which the LM uses to inform subsequent instructions. Let's update our original metric to demonstrate how this works:

```py
def haiku_score_gepa(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Penalize verbatim use of the input season string.
    A haiku should evoke the season through imagery, not name it
    directly.
    """
    text = prediction.haiku.lower()
    if example.season.strip().lower() in text:
        return dspy.Prediction(
            score=0.0,
            feedback="Don't reference the input season verbatim."
        )
    return dspy.Prediction(score=1.0, feedback=None)
```

Instead of just returning a score, we can tell the instruction-writing LM *why* a prediction failed or succeeded. Our example here is a bit silly, but this feedback ability is powerful. When labeling data to train against, labelers can jot down notes explaining a nuance in a specific record, which can then be passed to GEPA to guide future instructions. When training with an LM judge, the judge can provide detailed feedback for why a prediction didn't meet the criteria.

### Expanding our haiku metric

To give our model more of a challenge, we've built out our metric to check for many conditions we expect from our haikus, specifically:

- Does it have the correct number of lines?  
- Does it have the correct syllable count in each line?  
- Does it avoid repeating inputs verbatim?  
- Does it avoid the first-person voice?  
- Does it have a balanced ratio of parts-of-speech?  
- Does it use few adjectives?  
- Does it use few articles?  
- Does it use the present tense?

We can use the natural-language processing library [spaCy](https://spacy.io/) to assist with most of these measures.

All of these conditions make our metric too long to drop into this walkthrough, but the code is available here. Drop `haiku_metric.py` next to your notebook and add `from haiku_metric import haiku_metric` to follow along.

### Compiling our optimization

With our metric defined, it's time to configure our optimizer:

```py
from haiku_metric import haiku_metric

reflection_lm = dspy.LM("openai/gpt-5.4")

optimizer = dspy.GEPA(
    metric=haiku_metric,
    reflection_lm=reflection_lm,
    auto="light",
    num_threads=2,
)
```

The first thing to note here is `reflection_lm`. GEPA lets us choose a separate LM for reflection and instruction writing. This LM looks at our examples and how they score, then rewrites our prompt in an attempt to improve our scores.

When optimizing smaller models, it's worthwhile to use a larger model as the `reflection_lm`. They're better reasoners and prompters, and their cost isn't a concern since they're called a handful of times during optimization.

In addition to a `reflection_lm`, we set our metric and number of threads. For now we're using two, but depending on your inference provider you may have to tweak this to avoid any rate limits.

The `auto` argument sets our budget. `auto="light"` evaluates around six candidate prompts before stopping. `"medium"` and `"heavy"` options go further, and our GEPA deep dive covers additional levers we can set.

Finally, we compile our optimized program:

```py
optimized_haiku_bot = optimizer.compile(haiku_bot, trainset=train, valset=val)
```

Now it's time to grab a beverage and wait.

Behind the scenes, compile runs an iterative loop.

1. GEPA executes our program on training examples with the student LM and scores each result with our metric.  
2. Examples and their metric results are sent to the reflection LM, which proposes new instructions.  
3. GEPA runs the program again with the new instructions, scores again, and keeps the instruction candidates that score best.

This loop repeats until the budget runs out.

Running this optimizer with `gpt-5.4-nano` and using `gpt-5.4` as our reflection LM, we saw our score climb from 78.1% to 90.1% against our full haiku metric. Compare this to baseline score of `gpt-5.4`, 82.4%.

Once optimized with GEPA, our smaller model is faster, cheaper, and *better* than an unoptimized frontier model.

Let's quickly save our optimized program, then take a look at how our prompt changed.

```py
optimized.save("react_gpt_nano_haiku_optimized.json")
```

### GEPA's improved prompt

As previously discussed, `ReAct` is composed of a `Predict` module that reasons and decides what tools to call and a `ChainOfThought` module that synthesizes the final output. GEPA prompt optimized *both* of these modules.

Inside the JSON file containing our saved program state, we can see both improved instructions.

Our program's synthesis step started off with these instructions, which we defined in our signature docstring:

```
Write a classical haiku given the provided inputs.
```

After compile, these became:

```
Write a classical haiku from three inputs:

Inputs:
- location
- season
- mood

Output requirements:
- Return only the haiku itself.
- Exactly 3 lines.
- No title, no labels, no explanation, no reasoning, no quotation marks.

Primary success criteria, in order:
1. Exact 5-7-5 syllable counts, one line per count.
2. Exactly 3 lines.
3. A concrete seasonal image or cue appropriate to the given season.
4. Do not repeat the input season or mood words verbatim.
5. Keep diction sparse and image-heavy, with strong noun/verb focus.

Haiku style requirements:
- Use a classical haiku approach: brief, image-centered, present tense, emotionally restrained.
- Evoke the location, season, and mood indirectly through concrete imagery rather than naming them outright.
- Build the poem around one small, observable moment tied to the location.
- Prefer concrete nouns and active present-tense verbs.
- Favor lexical density: most words should carry imagery or action.
- Keep adjectives very sparse; avoid piling on descriptors.
- Avoid abstraction, explanation, commentary, and explicit emotional naming.
- Do not use first-person pronouns.
- Keep article use minimal.

Location handling:
- Anchor the poem clearly in the given location with at least one concrete object, surface, sound, or visual detail from that place.
- If the location is unusual or man-made, pair one specific man-made image from the setting with one seasonal sign.
```

It continues (you can view the entire text here), covering season and mood handling, common failure modes, and detailing a final quality check before output.

Fascinatingly, the same program will optimize differently depending on the model.

---

## 10\. Saving a program and reloading it

### Saving the optimized state, or the entire program

`.save(path)` has two modes. Pointing it at a `.json` file stores just the program's learned *state*, the optimized instructions and few-shot demos, not the program's *structure*. Pointing it at a directory with `save_program=True` writes the entire module as a pickled artifact, structure and state together.

The state-only mode is what you usually want. The JSON is small, human-readable, safe to share, and easy to diff in version control. To reload it, you instantiate your program in code as usual and call `.load()` to apply the saved state on top.

Reach for `save_program=True` when whoever is loading the program won't have your Python class definitions handy. Shipping an optimized program to another team is one example; serving it from a different repo than the one you developed it in is another. The cost is that the saved directory contains executable Python, so only load programs from sources you trust.

```py
# state-only — small file, requires re-instantiating the program before loading
optimized.save("haiku_bot.json")

# whole program — directory, rehydrates without you re-defining the class
optimized.save("haiku_bot/", save_program=True)
```

### Reloading a saved program

For the directory form, `dspy.load(path)` rehydrates the full module in one call. For the state-only form, you build a fresh copy of the program in code and call `.load(path)` on it to apply the saved state.

```py
# whole program
loaded = dspy.load("haiku_bot/")

# state-only
fresh = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page])
fresh.load("haiku_bot.json")
```

The save file contains the optimized instructions, demos, and signature metadata. It does not contain the LM client configuration: your API keys, your provider choice, your temperature. That separation is intentional: configure your LM as usual after loading and the same program targets whichever model you point it at today.

## Where to go next

### Compose harder pipelines

If you want multi-step modules with branching control flow, the \[Modules: composing your own\] guide picks up where Section 7 left off.

### Building richer metrics

Our haiku metric was intentionally simple. Programs usually need composite scores that blend syntax checks, semantic similarity, and/or LLM-as-judge rubrics. The \[Metrics: designing and composing\] guide walks through weighting sub-scores, preventing keyword-stuffing, and validating that your metric truly captures what you care about before you let an optimizer chase it.

### Try a different optimizer

If GEPA didn't fit your task, the \[Optimizers: choosing one\] guide walks through when to reach for `BootstrapFewShot`, `MIPROv2`, `BootstrapFinetune`, and the rest.

### Debug a run

Calls and traces are inspectable with `dspy.inspect_history()` and callbacks — see the \[Observability and debugging\] guide.

### Serve in production

Programs can be made async, streamed, and parallelized. The \[Async, streaming, and parallel\] guide covers the surface.  