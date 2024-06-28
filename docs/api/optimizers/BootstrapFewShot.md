---
sidebar_position: 2
---

# teleprompt.BootstrapFewShot

### Constructor

The constructor initializes the `BootstrapFewShot` class and sets up parameters for bootstrapping.

```python
class BootstrapFewShot(Teleprompter):
    def __init__(self, metric=None, metric_threshold=None, teacher_settings={}, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=5):
        self.metric = metric
        self.teacher_settings = teacher_settings

        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
```

**Parameters:**
- `metric` (_callable_, _optional_): Metric function to evaluate examples during bootstrapping. Defaults to `None`.
- `metric_threshold` (_float_, _optional_): Score threshold for metric to determine successful example. Defaults to `None`.
- `teacher_settings` (_dict_, _optional_): Settings for teacher predictor. Defaults to empty dictionary.
- `max_bootstrapped_demos` (_int_, _optional_): Maximum number of bootstrapped demonstrations per predictor. Defaults to 4.
- `max_labeled_demos` (_int_, _optional_): Maximum number of labeled demonstrations per predictor. Defaults to 16.
- `max_rounds` (_int_, _optional_): Maximum number of bootstrapping rounds. Defaults to 1.
- `max_errors` (_int_): Maximum errors permitted during evaluation. Halts run with the latest error message. Defaults to 5. Configure to 1 if no evaluation run error is desired.

### Method

#### `compile(self, student, *, teacher=None, trainset)`

This method compiles the BootstrapFewShot instance by performing bootstrapping to refine the student predictor.

This process includes preparing the student and teacher predictors, which involves creating predictor copies, verifying the student predictor is uncompiled, and compiling the teacher predictor with labeled demonstrations via LabeledFewShot if the teacher predictor hasn't been compiled.

The next stage involves preparing predictor mappings by validating that both the student and teacher predictors have the same program structure and the same signatures but are different objects.

The final stage is performing the bootstrapping iterations.

**Parameters:**
- `student` (_Teleprompter_): Student predictor to be compiled.
- `teacher` (_Teleprompter_, _optional_): Teacher predictor used for bootstrapping. Defaults to `None`.
- `trainset` (_list_): Training dataset used in bootstrapping.

**Returns:**
- The compiled `student` predictor after bootstrapping with refined demonstrations.

### Example

```python
#Assume defined trainset
#Assume defined RAG class
...

#Define teleprompter and include teacher
teacher = dspy.OpenAI(model='gpt-3.5-turbo', api_key = openai.api_key, api_provider = "openai", model_type = "chat")
teleprompter = BootstrapFewShot(teacher_settings=dict({'lm': teacher}))

# Compile!
compiled_rag = teleprompter.compile(student=RAG(), trainset=trainset)
```
