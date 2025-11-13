---
sidebar_position: 999
---

# DSPy Cheatsheet

This page will contain snippets for frequent usage patterns.

## DSPy Programs

### Forcing fresh LM outputs

DSPy caches LM calls. Provide a unique ``rollout_id`` and set a non-zero
``temperature`` (e.g., 1.0) to bypass an existing cache entry while still caching
the new result:

```python
predict = dspy.Predict("question -> answer")
predict(question="1+1", config={"rollout_id": 1, "temperature": 1.0})
```

### dspy.Signature

```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")
```

### dspy.ChainOfThought

```python
generate_answer = dspy.ChainOfThought(BasicQA)

# Call the predictor on a particular input alongside a hint.
question='What is the color of the sky?'
pred = generate_answer(question=question)
```

### dspy.ProgramOfThought

```python
pot = dspy.ProgramOfThought(BasicQA)

question = 'Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?'
result = pot(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ProgramOfThought process): {result.answer}")
```

### dspy.ReAct

```python
react_module = dspy.ReAct(BasicQA)

question = 'Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?'
result = react_module(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ReAct process): {result.answer}")
```

### dspy.Retrieve

```python
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(rm=colbertv2_wiki17_abstracts)

#Define Retrieve Module
retriever = dspy.Retrieve(k=3)

query='When was the first FIFA World Cup held?'

# Call the retriever on a particular query.
topK_passages = retriever(query).passages

for idx, passage in enumerate(topK_passages):
    print(f'{idx+1}]', passage, '\n')
```

### dspy.CodeAct

```python
from dspy import CodeAct

def factorial(n):
    """Calculate factorial of n"""
    if n == 1:
        return 1
    return n * factorial(n-1)

act = CodeAct("n->factorial", tools=[factorial])
result = act(n=5)
result # Returns 120
```

### dspy.Parallel

```python
import dspy

parallel = dspy.Parallel(num_threads=2)
predict = dspy.Predict("question -> answer")
result = parallel(
    [
        (predict, dspy.Example(question="1+1").with_inputs("question")),
        (predict, dspy.Example(question="2+2").with_inputs("question"))
    ]
)
result
```

## DSPy Metrics

### Function as Metric

To create a custom metric you can create a function that returns either a number or a boolean value:

```python
def parse_integer_answer(answer, only_first_line=True):
    try:
        if only_first_line:
            answer = answer.strip().split('\n')[0]

        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split('.')[0]
        answer = ''.join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0

    return answer

# Metric Function
def gsm8k_metric(gold, pred, trace=None) -> int:
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
```

### LLM as Judge

```python
class FactJudge(dspy.Signature):
    """Judge if the answer is factually correct based on the context."""

    context = dspy.InputField(desc="Context for the prediction")
    question = dspy.InputField(desc="Question to be answered")
    answer = dspy.InputField(desc="Answer for the question")
    factually_correct: bool = dspy.OutputField(desc="Is the answer factually correct based on the context?")

judge = dspy.ChainOfThought(FactJudge)

def factuality_metric(example, pred):
    factual = judge(context=example.context, question=example.question, answer=pred.answer)
    return factual.factually_correct
```

## DSPy Evaluation

```python
from dspy.evaluate import Evaluate

evaluate_program = Evaluate(devset=devset, metric=your_defined_metric, num_threads=NUM_THREADS, display_progress=True, display_table=num_rows_to_display)

evaluate_program(your_dspy_program)
```

## DSPy Optimizers

### LabeledFewShot

```python
from dspy.teleprompt import LabeledFewShot

labeled_fewshot_optimizer = LabeledFewShot(k=8)
your_dspy_program_compiled = labeled_fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

### BootstrapFewShot

```python
from dspy.teleprompt import BootstrapFewShot

fewshot_optimizer = BootstrapFewShot(metric=your_defined_metric, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=10)

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

#### Using another LM for compilation, specifying in teacher_settings

```python
from dspy.teleprompt import BootstrapFewShot

fewshot_optimizer = BootstrapFewShot(metric=your_defined_metric, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=10, teacher_settings=dict(lm=gpt4))

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

#### Compiling a compiled program - bootstrapping a bootstrapped program

```python
your_dspy_program_compiledx2 = teleprompter.compile(
    your_dspy_program,
    teacher=your_dspy_program_compiled,
    trainset=trainset,
)
```

#### Saving/loading a compiled program

```python
save_path = './v1.json'
your_dspy_program_compiledx2.save(save_path)
```

```python
loaded_program = YourProgramClass()
loaded_program.load(path=save_path)
```

### BootstrapFewShotWithRandomSearch

Detailed documentation on BootstrapFewShotWithRandomSearch can be found [here](api/optimizers/BootstrapFewShot.md).

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset, valset=devset)

```

Other custom configurations are similar to customizing the `BootstrapFewShot` optimizer.

### Ensemble

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.ensemble import Ensemble

fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)
your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset, valset=devset)

ensemble_optimizer = Ensemble(reduce_fn=dspy.majority)
programs = [x[-1] for x in your_dspy_program_compiled.candidate_programs]
your_dspy_program_compiled_ensemble = ensemble_optimizer.compile(programs[:3])
```

### BootstrapFinetune

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune

#Compile program on current dspy.settings.lm
fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_threads=NUM_THREADS)
your_dspy_program_compiled = tp.compile(your_dspy_program, trainset=trainset[:some_num], valset=trainset[some_num:])

#Configure model to finetune
config = dict(target=model_to_finetune, epochs=2, bf16=True, bsize=6, accumsteps=2, lr=5e-5)

#Compile program on BootstrapFinetune
finetune_optimizer = BootstrapFinetune(metric=your_defined_metric)
finetune_program = finetune_optimizer.compile(your_dspy_program, trainset=some_new_dataset_for_finetuning_model, **config)

finetune_program = your_dspy_program

#Load program and activate model's parameters in program before evaluation
ckpt_path = "saved_checkpoint_path_from_finetuning"
LM = dspy.HFModel(checkpoint=ckpt_path, model=model_to_finetune)

for p in finetune_program.predictors():
    p.lm = LM
    p.activated = False
```

### COPRO

Detailed documentation on COPRO can be found [here](api/optimizers/COPRO.md).

```python
from dspy.teleprompt import COPRO

eval_kwargs = dict(num_threads=16, display_progress=True, display_table=0)

copro_teleprompter = COPRO(prompt_model=model_to_generate_prompts, metric=your_defined_metric, breadth=num_new_prompts_generated, depth=times_to_generate_prompts, init_temperature=prompt_generation_temperature, verbose=False)

compiled_program_optimized_signature = copro_teleprompter.compile(your_dspy_program, trainset=trainset, eval_kwargs=eval_kwargs)
```

### MIPROv2

Note: detailed documentation can be found [here](api/optimizers/MIPROv2.md). `MIPROv2` is the latest extension of `MIPRO` which includes updates such as (1) improvements to instruction proposal and (2) more efficient search with minibatching.

#### Optimizing with MIPROv2

This shows how to perform an easy out-of-the box run with `auto=light`, which configures many hyperparameters for you and performs a light optimization run. You can alternatively set `auto=medium` or `auto=heavy` to perform longer optimization runs. The more detailed `MIPROv2` documentation [here](api/optimizers/MIPROv2.md) also provides more information about how to set hyperparameters by hand.

```python
# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light", # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing program with MIPRO...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
)

# Save optimize program for future use
optimized_program.save(f"mipro_optimized")

# Evaluate optimized program
print(f"Evaluate optimized program...")
evaluate(optimized_program, devset=devset[:])
```

#### Optimizing instructions only with MIPROv2 (0-Shot)

```python
# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light", # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing program with MIPRO...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
)

# Save optimize program for future use
optimized_program.save(f"mipro_optimized")

# Evaluate optimized program
print(f"Evaluate optimized program...")
evaluate(optimized_program, devset=devset[:])
```

### KNNFewShot

```python
from sentence_transformers import SentenceTransformer
from dspy import Embedder
from dspy.teleprompt import KNNFewShot
from dspy import ChainOfThought

knn_optimizer = KNNFewShot(k=3, trainset=trainset, vectorizer=Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode))

qa_compiled = knn_optimizer.compile(student=ChainOfThought("question -> answer"))
```

### BootstrapFewShotWithOptuna

```python
from dspy.teleprompt import BootstrapFewShotWithOptuna

fewshot_optuna_optimizer = BootstrapFewShotWithOptuna(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)

your_dspy_program_compiled = fewshot_optuna_optimizer.compile(student=your_dspy_program, trainset=trainset, valset=devset)
```

Other custom configurations are similar to customizing the `dspy.BootstrapFewShot` optimizer.


### SIMBA

SIMBA, which stands for Stochastic Introspective Mini-Batch Ascent, is a prompt optimizer that accepts arbitrary DSPy programs and proceeds in a sequence of mini-batches seeking to make incremental improvements to the prompt instructions or few-shot examples.

```python
from dspy.teleprompt import SIMBA

simba = SIMBA(metric=your_defined_metric, max_steps=12, max_demos=10)

optimized_program = simba.compile(student=your_dspy_program, trainset=trainset)
```


## DSPy Tools and Utilities

### dspy.Tool

```python
import dspy

def search_web(query: str) -> str:
    """Search the web for information"""
    return f"Search results for: {query}"

tool = dspy.Tool(search_web)
result = tool(query="Python programming")
```

### dspy.streamify

```python
import dspy
import asyncio

predict = dspy.Predict("question->answer")

stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)

async def read_output_stream():
    output_stream = stream_predict(question="Why did a chicken cross the kitchen?")

    async for chunk in output_stream:
        print(chunk)

asyncio.run(read_output_stream())
```


### dspy.asyncify

```python
import dspy

dspy_program = dspy.ChainOfThought("question -> answer")
dspy_program = dspy.asyncify(dspy_program)

asyncio.run(dspy_program(question="What is DSPy"))
```


### Track Usage

```python
import dspy
dspy.configure(track_usage=True)

result = dspy.ChainOfThought(BasicQA)(question="What is 2+2?")
print(f"Token usage: {result.get_lm_usage()}")
```

### dspy.configure_cache

```python
import dspy

# Configure cache settings
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
```

## DSPy `Refine` and `BestofN`

>`dspy.Suggest` and `dspy.Assert` are replaced by `dspy.Refine` and `dspy.BestofN` in DSPy 2.6.

### BestofN

Runs a module up to `N` times with different rollout IDs (bypassing cache) and returns the best prediction, as defined by the `reward_fn`, or the first prediction that passes the `threshold`.

```python
import dspy

qa = dspy.ChainOfThought("question -> answer")
def one_word_answer(args, pred):
    return 1.0 if len(pred.answer) == 1 else 0.0
best_of_3 = dspy.BestOfN(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
best_of_3(question="What is the capital of Belgium?").answer
# Brussels
```

### Refine

Refines a module by running it up to `N` times with different rollout IDs (bypassing cache) and returns the best prediction, as defined by the `reward_fn`, or the first prediction that passes the `threshold`. After each attempt (except the final one), `Refine` automatically generates detailed feedback about the module's performance and uses this feedback as hints for subsequent runs, creating an iterative refinement process.

```python
import dspy

qa = dspy.ChainOfThought("question -> answer")
def one_word_answer(args, pred):
    return 1.0 if len(pred.answer) == 1 else 0.0
best_of_3 = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
best_of_3(question="What is the capital of Belgium?").answer
# Brussels
```

#### Error Handling

By default, `Refine` will try to run the module up to N times until the threshold is met. If the module encounters an error, it will keep going up to N failed attempts. You can change this behavior by setting `fail_count` to a smaller number than `N`.

```python
refine = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0, fail_count=1)
...
refine(question="What is the capital of Belgium?")
# If we encounter just one failed attempt, the module will raise an error.
```

If you want to run the module up to N times without any error handling, you can set `fail_count` to `N`. This is the default behavior.

```python
refine = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0, fail_count=3)
...
refine(question="What is the capital of Belgium?")
```
