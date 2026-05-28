# Composing your own module

Switching to `dspy.ReAct` turned our single-step program into a multi-step agent. It’s a significant change, but `ReAct` itself is simple to understand.

Inside [`dspy.ReAct`](../diving-deeper/tools-react-and-mcp.md), other DSPy modules are composed together. Each step where the model considers its inputs and picks the next tool is a `dspy.Predict` module. A bit of code manages the control flow, looping through `Predict` calls until the model calls `finish` or hits `max_iters`. The synthesis step that follows, where the model assembles the answer from everything it’s learned, is a `ChainOfThought` module.

Modules are modular. We can compose them together and they just *work*.

To illustrate, let’s build a custom module for our haiku writer composed of two modules: one to generate several candidate haikus and one to pick the best verse.

## Composing modules to build an ensemble

LLM-written poems are a roll of the dice. Sometimes their haikus are evocative; other times they’re predictable and bland. To increase our program’s odds of success, we’re going to roll the dice several times, then select the best candidate.

Here’s what this looks like:

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

1. `writer` is similar to our last `ReAct` program. We’ve added a new input field, `num_haikus`, specifying how many haikus we want the model to draft. And we’ve changed our output field to return a `list` of strings.  
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

## Using a bigger model as our judge

To make this module a true ensemble, let’s use a different model to grade the work of our haiku writer. We only need to add one line:

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

The [`with dspy.context()`](../diving-deeper/settings-and-context.md) statement allows us to define a new context that sets a new model for the judge call.

## Decompose to isolate, reuse, govern, and optimize

Our haiku task is a small example, but building `HaikuEnsemble` demonstrates how easily we can decompose our programs when necessary. There’s no esoteric chaining API; modules are just Python and the DSPy primitives `Signature`, `Module`, and `LM`.

Reasons to decompose appear as our AI programs grow in complexity and we learn their failure modes. For example, we can use custom modules to:

- **Isolate context:** A wandering investigation step can entertain many candidate subjects, but the final haiku writing call should only see the chosen one. Splitting them keeps each module focused on its own job.  
- **Reusable parts across programs:** A well-tuned Wikipedia-research submodule isn’t haiku-specific; it could assist any program that needs trusted grounding. Decomposing lets us reuse it across programs.  
- **Route easy work to cheaper models:** Lots of small model calls can gather grounding details quickly and cheaply, while a single call to a stronger model handles the final, nuanced composition.  
- **Design custom control flow:** We could run the final haiku through a syllable-count check with an NLP library, and call the writer again if the meter is off.  
- **Govern independently inspect steps:** Each submodule is its own object, and each call lands in `inspect_history` separately. When the agent does something surprising, we can call one submodule on its own to see exactly what it returned. For haiku writing this is less of a concern, but for high-risk tasks this ability to audit is critical.  
- **More easily evaluate and optimize:** Scoring a single haiku in isolation is hard; “good” cuts across too many dimensions to grade cleanly. Picking the best of three is much easier. Decomposing isolates subtasks we can actually score, which enables evaluation and optimization.

See [Modules: composing your own](../diving-deeper/modules.md) for control-flow patterns (branching, retry loops, parallel calls) and composition beyond this example.

Speaking of evaluation and optimization, it’s time for the next section.

---

**Next:** [Metrics →](metrics.md)
