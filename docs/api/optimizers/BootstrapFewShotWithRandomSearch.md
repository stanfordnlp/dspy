---
sidebar_position: 4
---

# teleprompt.BootstrapFewShotWithRandomSearch

### Constructor

The constructor initializes the `BootstrapFewShotWithRandomSearch` class and sets up its attributes. It inherits from the `BootstrapFewShot` class and introduces additional attributes for the random search process.

```python
class BootstrapFewShotWithRandomSearch(BootstrapFewShot):
    def __init__(self, metric, teacher_settings={}, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, num_candidate_programs=16, num_threads=6):
        self.metric = metric
        self.teacher_settings = teacher_settings
        self.max_rounds = max_rounds

        self.num_threads = num_threads

        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.num_candidate_sets = num_candidate_programs
        self.max_num_traces = 1 + int(max_bootstrapped_demos / 2.0 * self.num_candidate_sets)

        self.max_bootstrapped_demos = self.max_num_traces
        self.max_labeled_demos = max_labeled_demos

        print("Going to sample between", self.min_num_samples, "and", self.max_num_samples, "traces per predictor.")
        print("Going to sample", self.max_num_traces, "traces in total.")
        print("Will attempt to train", self.num_candidate_sets, "candidate sets.")
```

**Parameters:**
- `metric` (_callable_, _optional_): Metric function to evaluate examples during bootstrapping. Defaults to `None`.
- `teacher_settings` (_dict_, _optional_): Settings for teacher predictor. Defaults to empty dictionary.
- `max_bootstrapped_demos` (_int_, _optional_): Maximum number of bootstrapped demonstrations per predictor. Defaults to 4.
- `max_labeled_demos` (_int_, _optional_): Maximum number of labeled demonstrations per predictor. Defaults to 16.
- `max_rounds` (_int_, _optional_): Maximum number of bootstrapping rounds. Defaults to 1.
- `num_candidate_programs` (_int_): Number of candidate programs to generate during random search.
- `num_threads` (_int_): Number of threads used for evaluation during random search.

### Method

Refer to [teleprompt.BootstrapFewShot](https://dspy-docs.vercel.app/docs/deep-dive/teleprompter/bootstrap-fewshot) documentation.

## Example

```python
#Assume defined trainset
#Assume defined RAG class
...

#Define teleprompter and include teacher
teacher = dspy.OpenAI(model='gpt-3.5-turbo', api_key = openai.api_key, api_provider = "openai", model_type = "chat")
teleprompter = BootstrapFewShotWithRandomSearch(teacher_settings=dict({'lm': teacher}))

# Compile!
compiled_rag = teleprompter.compile(student=RAG(), trainset=trainset)
```