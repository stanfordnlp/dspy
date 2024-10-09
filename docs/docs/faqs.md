---
sidebar_position: 998
---

# FAQs

## Is DSPy right for me? DSPy vs. other frameworks

The **DSPy** philosophy and abstraction differ significantly from other libraries and frameworks, so it's usually straightforward to decide when **DSPy** is (or isn't) the right framework for your usecase. If you're a NLP/AI researcher (or a practitioner exploring new pipelines or new tasks), the answer is generally an invariable **yes**. If you're a practitioner doing other things, please read on.

**DSPy vs. thin wrappers for prompts (OpenAI API, MiniChain, basic templating)** In other words: _Why can't I just write my prompts directly as string templates?_ Well, for extremely simple settings, this _might_ work just fine. (If you're familiar with neural networks, this is like expressing a tiny two-layer NN as a Python for-loop. It kinda works.) However, when you need higher quality (or manageable cost), then you need to iteratively explore multi-stage decomposition, improved prompting, data bootstrapping, careful finetuning, retrieval augmentation, and/or using smaller (or cheaper, or local) models. The true expressive power of building with foundation models lies in the interactions between these pieces. But every time you change one piece, you likely break (or weaken) multiple other components. **DSPy** cleanly abstracts away (_and_ powerfully optimizes) the parts of these interactions that are external to your actual system design. It lets you focus on designing the module-level interactions: the _same program_ expressed in 10 or 20 lines of **DSPy** can easily be compiled into multi-stage instructions for `GPT-4`, detailed prompts for `Llama2-13b`, or finetunes for `T5-base`. Oh, and you wouldn't need to maintain long, brittle, model-specific strings at the core of your project anymore.

**DSPy vs. application development libraries like LangChain, LlamaIndex** LangChain and LlamaIndex target high-level application development; they offer _batteries-included_, pre-built application modules that plug in with your data or configuration. If you'd be happy to use a generic, off-the-shelf prompt for question answering over PDFs or standard text-to-SQL, you will find a rich ecosystem in these libraries. **DSPy** doesn't internally contain hand-crafted prompts that target specific applications. Instead, **DSPy** introduces a small set of much more powerful and general-purpose modules _that can learn to prompt (or finetune) your LM within your pipeline on your data_. when you change your data, make tweaks to your program's control flow, or change your target LM, the **DSPy compiler** can map your program into a new set of prompts (or finetunes) that are optimized specifically for this pipeline. Because of this, you may find that **DSPy** obtains the highest quality for your task, with the least effort, provided you're willing to implement (or extend) your own short program. In short, **DSPy** is for when you need a lightweight but automatically-optimizing programming model — not a library of predefined prompts and integrations. If you're familiar with neural networks: This is like the difference between PyTorch (i.e., representing **DSPy**) and HuggingFace Transformers (i.e., representing the higher-level libraries).

**DSPy vs. generation control libraries like Guidance, LMQL, RELM, Outlines** These are all exciting new libraries for controlling the individual completions of LMs, e.g., if you want to enforce JSON output schema or constrain sampling to a particular regular expression. This is very useful in many settings, but it's generally focused on low-level, structured control of a single LM call. It doesn't help ensure the JSON (or structured output) you get is going to be correct or useful for your task. In contrast, **DSPy** automatically optimizes the prompts in your programs to align them with various task needs, which may also include producing valid structured outputs. That said, we are considering allowing **Signatures** in **DSPy** to express regex-like constraints that are implemented by these libraries.

## Basic Usage

**How should I use DSPy for my task?** We wrote a [eight-step guide](https://dspy-docs.vercel.app/docs/building-blocks/solving_your_task) on this. In short, using DSPy is an iterative process. You first define your task and the metrics you want to maximize, and prepare a few example inputs — typically without labels (or only with labels for the final outputs, if your metric requires them). Then, you build your pipeline by selecting built-in layers (`modules`) to use, giving each layer a `signature` (input/output spec), and then calling your modules freely in your Python code. Lastly, you use a DSPy `optimizer` to compile your code into high-quality instructions, automatic few-shot examples, or updated LM weights for your LM.

**How do I convert my complex prompt into a DSPy pipeline?** See the same answer above.

**What do DSPy optimizers tune?** Or, _what does compiling actually do?_ Each optimizer is different, but they all seek to maximize a metric on your program by updating prompts or LM weights. Current DSPy `optimizers` can inspect your data, simulate traces through your program to generate good/bad examples of each step, propose or refine instructions for each step based on past results, finetune the weights of your LM on self-generated examples, or combine several of these to improve quality or cut cost. We'd love to merge new optimizers that explore a richer space: most manual steps you currently go through for prompt engineering, "synthetic data" generation, or self-improvement can probably generalized into a DSPy optimizer that acts on arbitrary LM programs.

Other FAQs. We welcome PRs to add formal answers to each of these here. You will find the answer in existing issues, tutorials, or the papers for all or most of these.

- **How do I get multiple outputs?**

You can specify multiple output fields. For the short-form signature, you can list multiple outputs as comma separated values, following the "->" indicator (e.g. "inputs -> output1, output2"). For the long-form signature, you can include multiple `dspy.OutputField`s.

- **How can I work with long responses?**

You can specify the generation of long responses as a `dspy.OutputField`. To ensure comprehensive checks of the content within the long-form generations, you can indicate the inclusion of citations per referenced context. Such constraints such as response length or citation inclusion can be stated through Signature descriptions, or concretely enforced through DSPy Assertions. Check out the [LongFormQA notebook](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/longformqa/longformqa_assertions.ipynb) to learn more about **Generating long-form length responses to answer questions**.

- **How can I ensure that DSPy doesn't strip new line characters from my inputs or outputs?**

DSPy uses [Signatures](https://dspy-docs.vercel.app/docs/deep-dive/signature/understanding-signatures) to format prompts passed into LMs. In order to ensure that new line characters aren't stripped from longer inputs, you must specify `format=str` when creating a field.

```python
class UnstrippedSignature(dspy.Signature):
    """Enter some information for the model here."""

    title = dspy.InputField()
    object = dspy.InputField(format=str)
    result = dspy.OutputField(format=str)
```

`object` can now be a multi-line string without issue.

- **How do I define my own metrics? Can metrics return a float?**

You can define metrics as simply Python functions that process model generations and evaluate them based on user-defined requirements. Metrics can compare existent data (e.g. gold labels) to model predictions or they can be used to assess various components of an output using validation feedback from LMs (e.g. LLMs-as-Judges). Metrics can return `bool`, `int`, and `float` types scores. Check out the official [Metrics docs](https://dspy-docs.vercel.app/docs/building-blocks/metrics) to learn more about defining custom metrics and advanced evaluations using AI feedback and/or DSPy programs.

- **How expensive or slow is compiling??**

To reflect compiling metrics, we highlight an experiment for reference, compiling the [`SimplifiedBaleen`](https://dspy-docs.vercel.app/docs/tutorials/simplified-baleen) using the [`dspy.BootstrapFewShotWithRandomSearch`](https://dspy-docs.vercel.app/docs/deep-dive/teleprompter/bootstrap-fewshot) optimizer on the `gpt-3.5-turbo-1106` model over 7 candidate programs and 10 threads. We report that compiling this program takes around 6 minutes with 3200 API calls, 2.7 million input tokens and 156,000 output tokens, reporting a total cost of $3 USD (at the current pricing of the OpenAI model).

Compiling DSPy `optimizers` naturally will incur additional LM calls, but we substantiate this overhead with minimalistic executions with the goal of maximizing performance. This invites avenues to enhance performance of smaller models by compiling DSPy programs with larger models to learn enhanced behavior during compile-time and propagate such behavior to the tested smaller model during inference-time.  


## Deployment or Reproducibility Concerns

- **How do I save a checkpoint of my compiled program?**

Here is an example of saving/loading a compiled module:

```python
cot_compiled = teleprompter.compile(CoT(), trainset=trainset, valset=devset)

#Saving
cot_compiled.save('compiled_cot_gsm8k.json')

#Loading:
cot = CoT()
cot.load('compiled_cot_gsm8k.json')
```

- **How do I export for deployment?**

Exporting DSPy programs is simply saving them as highlighted above!

- **How do I search my own data?**

Open source libraries such as [RAGautouille](https://github.com/bclavie/ragatouille) enable you to search for your own data through advanced retrieval models like ColBERT with tools to embed and index documents. Feel free to integrate such libraries to create searchable datasets while developing your DSPy programs!

- **How do I turn off the cache? How do I export the cache?**

From v2.5, you can turn off the cache by setting `cache` parameter in `dspy.LM` to `False`:

```python
dspy.LM('openai/gpt-4o-mini',  cache=False)
```

Your local cache will be saved to the global env directory `os.environ["DSP_CACHEDIR"]` or for notebooks `os.environ["DSP_NOTEBOOK_CACHEDIR"]`. You can usually set the cachedir to `os.path.join(repo_path, 'cache')` and export this cache from here:
```python
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')
```

:::warning Important
`DSP_CACHEDIR` is responsible for old clients (including dspy.OpenAI, dspy.ColBERTv2, etc.) and `DSPY_CACHEDIR` is responsible for the new dspy.LM client.
:::

In the AWS lambda deployment, you should disable both DSP_* and DSPY_*.


## Advanced Usage

- **How do I parallelize?**
You can parallelize DSPy programs during both compilation and evaluation by specifying multiple thread settings in the respective DSPy `optimizers` or within the `dspy.Evaluate` utility function.

- **How do freeze a module?**

Modules can be frozen by setting their `._compiled` attribute to be True, indicating the module has gone through optimizer compilation and should not have its parameters adjusted. This is handled internally in optimizers such as `dspy.BootstrapFewShot` where the student program is ensured to be frozen before the teacher propagates the gathered few-shot demonstrations in the bootstrapping process. 

- **How do I get JSON output?**

You can specify JSON-type descriptions in the `desc` field of the long-form signature `dspy.OutputField` (e.g. `output = dspy.OutputField(desc='key-value pairs')`).

If you notice outputs are still not conforming to JSON formatting, try Asserting this constraint! Check out [Assertions](https://dspy-docs.vercel.app/docs/building-blocks/assertions) (or the next question!)

- **How do I use DSPy assertions?**

    a) **How to Add Assertions to Your Program**:
    - **Define Constraints**: Use `dspy.Assert` and/or `dspy.Suggest` to define constraints within your DSPy program. These are based on boolean validation checks for the outcomes you want to enforce, which can simply be Python functions to validate the model outputs.
    - **Integrating Assertions**: Keep your Assertion statements following a model generations (hint: following a module layer)

    b) **How to Activate the Assertions**:
    1. **Using `assert_transform_module`**:
        - Wrap your DSPy module with assertions using the `assert_transform_module` function, along with a `backtrack_handler`. This function transforms your program to include internal assertions backtracking and retry logic, which can be customized as well:
        `program_with_assertions = assert_transform_module(ProgramWithAssertions(), backtrack_handler)`
    2. **Activate Assertions**:
        - Directly call `activate_assertions` on your DSPy program with assertions: `program_with_assertions = ProgramWithAssertions().activate_assertions()`

    **Note**: To use Assertions properly, you must **activate** a DSPy program that includes `dspy.Assert` or `dspy.Suggest` statements from either of the methods above. 

## Errors

- **How do I deal with "context too long" errors?**

If you're dealing with "context too long" errors in DSPy, you're likely using DSPy optimizers to include demonstrations within your prompt, and this is exceeding your current context window. Try reducing these parameters (e.g. `max_bootstrapped_demos` and `max_labeled_demos`). Additionally, you can also reduce the number of retrieved passages/docs/embeddings to ensure your prompt is fitting within your model context length.

A more general fix is simply increasing the number of `max_tokens` specified to the LM request (e.g. `lm = dspy.OpenAI(model = ..., max_tokens = ...`).

- **How do I deal with timeouts or backoff errors?**

Firstly, please refer to your LM/RM provider to ensure stable status or sufficient rate limits for your use case!

Additionally, try reducing the number of threads you are testing on as the corresponding servers may get overloaded with requests and trigger a backoff + retry mechanism.

If all variables seem stable, you may be experiencing timeouts or backoff errors due to incorrect payload requests sent to the api providers. Please verify your arguments are compatible with the SDK you are interacting with. 

You can configure backoff times for your LM/RM provider by setting `dspy.settings.backoff_time` while configuring your DSPy workflow. 

```python
dspy.settings.configure(backoff_time = ...)
```

Additionally, if you'd like to set individual backoff times for specific providers, you can do so through the DSPy context manager: 

```python
with dspy.context(backoff_time = ..):
      dspy.OpenAI(...) # example

with dspy.context(backoff_time = ..):
      dspy.AzureOpenAI(...) # example
```

At times, DSPy may have hard-coded arguments that are not relevant for your compatible, in which case, please free to open a PR alerting this or comment out these default settings for your usage. 

## Contributing

**What if I have a better idea for prompting or synthetic data generation?** Perfect. We encourage you to think if it's best expressed as a module or an optimizer, and we'd love to merge it in DSPy so everyone can use it. DSPy is not a complete project; it's an ongoing effort to create structure (modules and optimizers) in place of hacky prompt and pipeline engineering tricks.

**How can I add my favorite LM or vector store?** 

Check out these walkthroughs on setting up a [Custom LM client](https://dspy-docs.vercel.app/docs/deep-dive/language_model_clients/custom-lm-client) and [Custom RM client](https://dspy-docs.vercel.app/docs/deep-dive/retrieval_models_clients/custom-rm-client).
