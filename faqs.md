---
sidebar_position: 1
---

# FAQs

## Basic Usage

**How should I use DSPy for my task?** Using DSPy is an iterative process. You first define your task and the metrics you want to maximize, and prepare a few example inputs — typically without labels (or only with labels for the final outputs, if your metric requires them). Then, you build your pipeline by selecting built-in layers (`modules`) to use, giving each layer a `signature` (input/output spec), and then calling your modules freely in your Python code. Lastly, you use a DSPy `optimizer` to compile your code into high-quality instructions, automatic few-shot examples, or updated LM weights for your LM.


## Is DSPy right for me? DSPy vs. other frameworks

The **DSPy** philosophy and abstraction differ significantly from other libraries and frameworks, so it's usually straightforward to decide when **DSPy** is (or isn't) the right framework for your usecase. If you're a NLP/AI researcher (or a practitioner exploring new pipelines or new tasks), the answer is generally an invariable **yes**. If you're a practitioner doing other things, please read on.

####
<details>
  <summary><h4 style="display: inline">[5.a] DSPy vs. thin wrappers for prompts (OpenAI API, MiniChain, basic templating)</h4></summary>

In other words: _Why can't I just write my prompts directly as string templates?_ Well, for extremely simple settings, this _might_ work just fine. (If you're familiar with neural networks, this is like expressing a tiny two-layer NN as a Python for-loop. It kinda works.)

However, when you need higher quality (or manageable cost), then you need to iteratively explore multi-stage decomposition, improved prompting, data bootstrapping, careful finetuning, retrieval augmentation, and/or using smaller (or cheaper, or local) models. The true expressive power of building with foundation models lies in the interactions between these pieces. But every time you change one piece, you likely break (or weaken) multiple other components.

**DSPy** cleanly abstracts away (_and_ powerfully optimizes) the parts of these interactions that are external to your actual system design. It lets you focus on designing the module-level interactions: the _same program_ expressed in 10 or 20 lines of **DSPy** can easily be compiled into multi-stage instructions for `GPT-4`, detailed prompts for `Llama2-13b`, or finetunes for `T5-base`.

Oh, and you wouldn't need to maintain long, brittle, model-specific strings at the core of your project anymore.

</details>

####
<details>
  <summary><h4 style="display: inline">[5.b] DSPy vs. application development libraries like LangChain, LlamaIndex</h4></summary>


> _Note: If you use LangChain as a thin wrapper around your own prompt strings, refer to answer [5.a] instead._


LangChain and LlamaIndex are popular libraries that target high-level application development with LMs. They offer many _batteries-included_, pre-built application modules that plug in with your data or configuration. In practice, indeed, many usecases genuinely _don't need_ any special components. If you'd be happy to use someone's generic, off-the-shelf prompt for question answering over PDFs or standard text-to-SQL as long as it's easy to set up on your data, then you will probably find a very rich ecosystem in these libraries.

Unlike these libraries, **DSPy** doesn't internally contain hand-crafted prompts that target specific applications you can build. Instead, **DSPy** introduces a very small set of much more powerful and general-purpose modules _that can learn to prompt (or finetune) your LM within your pipeline on your data_.

**DSPy** offers a whole different degree of modularity: when you change your data, make tweaks to your program's control flow, or change your target LM, the **DSPy compiler** can map your program into a new set of prompts (or finetunes) that are optimized specifically for this pipeline. Because of this, you may find that **DSPy** obtains the highest quality for your task, with the least effort, provided you're willing to implement (or extend) your own short program. In short, **DSPy** is for when you need a lightweight but automatically-optimizing programming model — not a library of predefined prompts and integrations.

If you're familiar with neural networks:
> This is like the difference between PyTorch (i.e., representing **DSPy**) and HuggingFace Transformers (i.e., representing the higher-level libraries). If you simply want to use off-the-shelf `BERT-base-uncased` or `GPT2-large` or apply minimal finetuning to them, HF Transformers makes it very straightforward. If, however, you're looking to build your own architecture (or extend an existing one significantly), you have to quickly drop down into something much more modular like PyTorch. Luckily, HF Transformers _is_ implemented in backends like PyTorch. We are similarly excited about high-level wrapper around **DSPy** for common applications. If this is implemented using **DSPy**, your high-level application can also adapt significantly to your data in a way that static prompt chains won't. Please [open an issue](https://github.com/stanfordnlp/dspy/issues/new) if this is something you want to help with.
</details>


####
<details>
  <summary><h4 style="display: inline">[5.c] DSPy vs. generation control libraries like Guidance, LMQL, RELM, Outlines</h4></summary>


Guidance, LMQL, RELM, and Outlines are all exciting new libraries for controlling the individual completions of LMs, e.g., if you want to enforce JSON output schema or constrain sampling to a particular regular expression.

This is very useful in many settings, but it's generally focused on low-level, structured control of a single LM call. It doesn't help ensure the JSON (or structured output) you get is going to be correct or useful for your task.

In contrast, **DSPy** automatically optimizes the prompts in your programs to align them with various task needs, which may also include producing valid structured ouputs. That said, we are considering allowing **Signatures** in **DSPy** to express regex-like constraints that are implemented by these libraries.
</details>



## Optimizers

**What do DSPy optimizers tune?** Each optimizer is different, but they all seek to maximize a metric on your program by updating prompts or LM weights. Current DSPy `optimizers` can inspect your data, simulate traces through your program to generate good/bad examples of each step, propose or refine instructions for each step based on past results, finetune the weights of your LM on self-generated examples, or combine several of these to improve quality or cut cost. We'd love to merge new optimizers that explore a richer space: most manual steps you currently go through for prompt engineering, "synthetic data" generation, or self-improvement can probably generalized into a DSPy optimizer that acts on arbitrary LM programs.





## Contributing

**What if I have a better idea for prompting or synthetic data generation?** Perfect. We encourage you to think if it's best expressed as a module or an optimizer, and we'd love to merge it in DSPy so everyone can use it. DSPy is not a complete project; it's an ongoing effort to create structure (modules and optimizers) in place of hacky prompt and pipeline engineering tricks.
