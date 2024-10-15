# DSPy Assertions 

## Introduction

Language models (LMs) have transformed how we interact with machine learning, offering vast capabilities in natural language understanding and generation. However, ensuring these models adhere to domain-specific constraints remains a challenge. Despite the growth of techniques like fine-tuning or “prompt engineering”, these approaches are extremely tedious and rely on heavy, manual hand-waving to guide the LMs in adhering to specific constraints. Even DSPy's modularity of programming prompting pipelines lacks mechanisms to effectively and automatically enforce these constraints. 

To address this, we introduce DSPy Assertions, a feature within the DSPy framework designed to automate the enforcement of computational constraints on LMs. DSPy Assertions empower developers to guide LMs towards desired outcomes with minimal manual intervention, enhancing the reliability, predictability, and correctness of LM outputs.

### dspy.Assert and dspy.Suggest API    

We introduce two primary constructs within DSPy Assertions:

- **`dspy.Assert`**:
  - **Parameters**: 
    - `constraint (bool)`: Outcome of Python-defined boolean validation check.
    - `msg (Optional[str])`: User-defined error message providing feedback or correction guidance.
    - `backtrack (Optional[module])`: Specifies target module for retry attempts upon constraint failure. The default backtracking module is the last module before the assertion.
  - **Behavior**: Initiates retry  upon failure, dynamically adjusting the pipeline's execution. If failures persist, it halts execution and raises a `dspy.AssertionError`.

- **`dspy.Suggest`**:
  - **Parameters**: Similar to `dspy.Assert`.
  - **Behavior**: Encourages self-refinement through retries without enforcing hard stops. Logs failures after maximum backtracking attempts and continues execution.

- **dspy.Assert vs. Python Assertions**: Unlike conventional Python `assert` statements that terminate the program upon failure, `dspy.Assert` conducts a sophisticated retry mechanism, allowing the pipeline to adjust. 

Specifically, when a constraint is not met:

- Backtracking Mechanism: An under-the-hood backtracking is initiated, offering the model a chance to self-refine and proceed, which is done through
- Dynamic Signature Modification: internally modifying your DSPy program’s Signature by adding the following fields:
    - Past Output: your model's past output that did not pass the validation_fn
    - Instruction: your user-defined feedback message on what went wrong and what possibly to fix

If the error continues past the `max_backtracking_attempts`, then `dspy.Assert` will halt the pipeline execution, alerting you with an `dspy.AssertionError`. This ensures your program doesn't continue executing with “bad” LM behavior and immediately highlights sample failure outputs for user assessment.

- **dspy.Suggest vs. dspy.Assert**: `dspy.Suggest` on the other hand offers a softer approach. It maintains the same retry backtracking as `dspy.Assert` but instead serves as a gentle nudger. If the model outputs cannot pass the model constraints after the `max_backtracking_attempts`, `dspy.Suggest` will log the persistent failure and continue execution of the program on the rest of the data. This ensures the LM pipeline works in a "best-effort" manner without halting execution. 

- **`dspy.Suggest`** are best utilized as "helpers" during the evaluation phase, offering guidance and potential corrections without halting the pipeline.
- **`dspy.Assert`** are recommended during the development stage as "checkers" to ensure the LM behaves as expected, providing a robust mechanism for identifying and addressing errors early in the development cycle.


## Use Case: Including Assertions in DSPy Programs

We start with using an example of a multi-hop QA SimplifiedBaleen pipeline as defined in the intro walkthrough. 

```python
class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            prev_queries.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        
        pred = self.generate_answer(context=context, question=question)
        pred = dspy.Prediction(context=context, answer=pred.answer)
        return pred

baleen = SimplifiedBaleen()

baleen(question = "Which award did Gary Zukav's first book receive?")
```

To include DSPy Assertions, we simply define our validation functions and declare our assertions following the respective model generation. 

For this use case, suppose we want to impose the following constraints:
    1. Length - each query should be less than 100 characters
    2. Uniqueness - each generated query should differ from previously-generated queries. 
    
We can define these validation checks as boolean functions:

```python
#simplistic boolean check for query length
len(query) <= 100

#Python function for validating distinct queries
def validate_query_distinction_local(previous_queries, query):
    """check if query is distinct from previous queries"""
    if previous_queries == []:
        return True
    if dspy.evaluate.answer_exact_match_str(query, previous_queries, frac=0.8):
        return False
    return True
```

We can declare these validation checks through `dspy.Suggest` statements (as we want to test the program in a best-effort demonstration). We want to keep these after the query generation `query = self.generate_query[hop](context=context, question=question).query`.

```python
dspy.Suggest(
    len(query) <= 100,
    "Query should be short and less than 100 characters",
    target_module=self.generate_query
)

dspy.Suggest(
    validate_query_distinction_local(prev_queries, query),
    "Query should be distinct from: "
    + "; ".join(f"{i+1}) {q}" for i, q in enumerate(prev_queries)),
    target_module=self.generate_query
)
```

It is recommended to define a program with assertions separately than your original program if you are doing comparative evaluation for the effect of assertions. If not, feel free to set Assertions away!

Let's take a look at how the SimplifiedBaleen program will look with Assertions included:

```python
class SimplifiedBaleenAssertions(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query

            dspy.Suggest(
                len(query) <= 100,
                "Query should be short and less than 100 characters",
                target_module=self.generate_query
            )

            dspy.Suggest(
                validate_query_distinction_local(prev_queries, query),
                "Query should be distinct from: "
                + "; ".join(f"{i+1}) {q}" for i, q in enumerate(prev_queries)),
                target_module=self.generate_query
            )

            prev_queries.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        
        if all_queries_distinct(prev_queries):
            self.passed_suggestions += 1

        pred = self.generate_answer(context=context, question=question)
        pred = dspy.Prediction(context=context, answer=pred.answer)
        return pred
```

Now calling programs with DSPy Assertions requires one last step, and that is transforming the program to wrap it with internal assertions backtracking and Retry logic. 

```python
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

baleen_with_assertions = assert_transform_module(SimplifiedBaleenAssertions(), backtrack_handler)

# backtrack_handler is parameterized over a few settings for the backtracking mechanism
# To change the number of max retry attempts, you can do
baleen_with_assertions_retry_once = assert_transform_module(SimplifiedBaleenAssertions(), 
    functools.partial(backtrack_handler, max_backtracks=1))
```

Alternatively, you can also directly call `activate_assertions` on the program with `dspy.Assert/Suggest` statements using the default backtracking mechanism (`max_backtracks=2`):

```python
baleen_with_assertions = SimplifiedBaleenAssertions().activate_assertions()
```

Now let's take a look at the internal LM backtracking by inspecting the history of the LM query generations. Here we see that when a query fails to pass the validation check of being less than 100 characters, its internal `GenerateSearchQuery` signature is dynamically modified during the backtracking+Retry process to include the past query and the corresponding user-defined instruction: `"Query should be short and less than 100 characters"`.


```text
Write a simple search query that will help answer a complex question.

---

Follow the following format.

Context: may contain relevant facts

Question: ${question}

Reasoning: Let's think step by step in order to ${produce the query}. We ...

Query: ${query}

---

Context:
[1] «Kerry Condon | Kerry Condon (born 4 January 1983) is [...]»
[2] «Corona Riccardo | Corona Riccardo (c. 1878October 15, 1917) was [...]»

Question: Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet." ?

Reasoning: Let's think step by step in order to find the answer to this question. First, we need to identify the actress who played Ophelia in a Royal Shakespeare Company production of "Hamlet." Then, we need to find out if this actress also acted in the short film "The Shore."

Query: "actress who played Ophelia in Royal Shakespeare Company production of Hamlet" + "actress in short film The Shore"



Write a simple search query that will help answer a complex question.

---

Follow the following format.

Context: may contain relevant facts

Question: ${question}

Past Query: past output with errors

Instructions: Some instructions you must satisfy

Query: ${query}

---

Context:
[1] «Kerry Condon | Kerry Condon (born 4 January 1983) is an Irish television and film actress, best known for her role as Octavia of the Julii in the HBO/BBC series "Rome," as Stacey Ehrmantraut in AMC's "Better Call Saul" and as the voice of F.R.I.D.A.Y. in various films in the Marvel Cinematic Universe. She is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet."»
[2] «Corona Riccardo | Corona Riccardo (c. 1878October 15, 1917) was an Italian born American actress who had a brief Broadway stage career before leaving to become a wife and mother. Born in Naples she came to acting in 1894 playing a Mexican girl in a play at the Empire Theatre. Wilson Barrett engaged her for a role in his play "The Sign of the Cross" which he took on tour of the United States. Riccardo played the role of Ancaria and later played Berenice in the same play. Robert B. Mantell in 1898 who struck by her beauty also cast her in two Shakespeare plays, "Romeo and Juliet" and "Othello". Author Lewis Strang writing in 1899 said Riccardo was the most promising actress in America at the time. Towards the end of 1898 Mantell chose her for another Shakespeare part, Ophelia im Hamlet. Afterwards she was due to join Augustin Daly's Theatre Company but Daly died in 1899. In 1899 she gained her biggest fame by playing Iras in the first stage production of Ben-Hur.»

Question: Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet." ?

Past Query: "actress who played Ophelia in Royal Shakespeare Company production of Hamlet" + "actress in short film The Shore"

Instructions: Query should be short and less than 100 characters

Query: "actress Ophelia RSC Hamlet" + "actress The Shore"

```


## Assertion-Driven Optimizations

DSPy Assertions work with optimizations that DSPy offers, particularly with `BootstrapFewShotWithRandomSearch`, including the following settings:

- Compilation with Assertions
    This includes assertion-driven example bootstrapping and counterexample bootstrapping during compilation. The teacher model for bootstrapping few-shot demonstrations can make use of DSPy Assertions to offer robust bootstrapped examples for the student model to learn from during inference. In this setting, the student model does not perform assertion aware optimizations (backtracking and retry) during inference.
- Compilation + Inference with Assertions
    -This includes assertion-driven optimizations in both compilation and inference. Now the teacher model offers assertion-driven examples but the student can further optimize with assertions of its own during inference time. 
```python
teleprompter = BootstrapFewShotWithRandomSearch(
    metric=validate_context_and_answer_and_hops,
    max_bootstrapped_demos=max_bootstrapped_demos,
    num_candidate_programs=6,
)

#Compilation with Assertions
compiled_with_assertions_baleen = teleprompter.compile(student = baleen, teacher = baleen_with_assertions, trainset = trainset, valset = devset)

#Compilation + Inference with Assertions
compiled_baleen_with_assertions = teleprompter.compile(student=baleen_with_assertions, teacher = baleen_with_assertions, trainset=trainset, valset=devset)

```