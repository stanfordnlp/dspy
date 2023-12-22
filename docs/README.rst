.. contents::
   :depth: 3
..

current docs link: https://dspy-ai.readthedocs.io/en/latest/


   <p align="center">


   </p>


   <p align="left">

DSPy: *Programming*—not prompting—Foundation Models
===================================================

Paper —— `DSPy: Compiling Declarative Language Model Calls into
Self-Improving Pipelines <https://arxiv.org/abs/2310.03714>`__

` <https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb>`__

**DSPy** is the framework for solving advanced tasks with language
models (LMs) and retrieval models (RMs). **DSPy** unifies techniques for
**prompting** and **fine-tuning** LMs — and approaches for
**reasoning**, **self-improvement**, and **augmentation with retrieval
and tools**. All of these are expressed through modules that compose and
learn.

To make this possible:

-  **DSPy** provides **composable and declarative modules** for
   instructing LMs in a familiar Pythonic syntax. It upgrades “prompting
   techniques” like chain-of-thought and self-reflection from
   hand-adapted *string manipulation tricks* into truly modular
   *generalized operations that learn to adapt to your task*.

-  **DSPy** introduces an **automatic compiler that teaches LMs** how to
   conduct the declarative steps in your program. Specifically, the
   **DSPy compiler** will internally *trace* your program and then
   **craft high-quality prompts for large LMs (or train automatic
   finetunes for small LMs)** to teach them the steps of your task.

The **DSPy compiler** *bootstraps* prompts and finetunes from minimal
data **without needing manual labels for the intermediate steps** in
your program. Instead of brittle “prompt engineering” with hacky string
manipulation, you can explore a systematic space of modular and
trainable pieces.

For complex tasks, **DSPy** can routinely teach powerful models like
``GPT-3.5`` and local models like ``T5-base`` or ``Llama2-13b`` to be
much more reliable at tasks. **DSPy** will compile the *same program*
into different few-shot prompts and/or finetunes for each LM.

If you want to see **DSPy** in action, `open our intro tutorial
notebook <intro.ipynb>`__.

Table of Contents
-----------------

1. `Installation <#1-installation>`__
2. `Framework
   Syntax <#2-syntax-youre-in-charge-of-the-workflowits-free-form-python-code>`__
3. `Compiling: Two Powerful
   Concepts <#3-two-powerful-concepts-signatures--teleprompters>`__
4. `Tutorials & Documentation <#4-documentation--tutorials>`__
5. `FAQ: Is DSPy right for me? <#5-faq-is-dspy-right-for-me>`__

Analogy to Neural Networks
--------------------------

When we build neural networks, we don’t write manual *for-loops* over
lists of *hand-tuned* floats. Instead, you might use a framework like
`PyTorch <https://pytorch.org/>`__ to compose declarative layers (e.g.,
``Convolution`` or ``Dropout``) and then use optimizers (e.g., SGD or
Adam) to learn the parameters of the network.

Ditto! **DSPy** gives you the right general-purpose modules (e.g.,
``ChainOfThought``, ``Retrieve``, etc.) and takes care of optimizing
their prompts *for your program* and your metric, whatever they aim to
do. Whenever you modify your code, your data, or your validation
constraints, you can *compile* your program again and **DSPy** will
create new effective prompts that fit your changes.

1) Installation
===============

All you need is:

::

   pip install dspy-ai

Or open our intro notebook in Google Colab:
` <https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb>`__

   *Note: If you’re looking for Demonstrate-Search-Predict (DSP), which
   is the previous version of DSPy, you can find it on the*
   `v1 <https://github.com/stanfordnlp/dspy/tree/v1>`__ *branch of this
   repo.*

For the optional Pinecone, Qdrant,
`chromadb <https://github.com/chroma-core/chroma>`__, or
`marqo <https://github.com/marqo-ai/marqo>`__ retrieval integration(s),
include the extra(s) below:

::

   pip install dspy-ai[pinecone]  # or [qdrant] or [chromadb] or [marqo]

2) Syntax: You’re in charge of the workflow—it’s free-form Python code!
=======================================================================

**DSPy** hides tedious prompt engineering, but it cleanly exposes the
important decisions you need to make: **[1]** what’s your system design
going to look like? **[2]** what are the important constraints on the
behavior of your program?

You express your system as free-form Pythonic modules. **DSPy** will
tune the quality of your program *in whatever way* you use foundation
models: you can code with loops, ``if`` statements, or exceptions, and
use **DSPy** modules within any Python control flow you think works for
your task.

Suppose you want to build a simple retrieval-augmented generation (RAG)
system for question answering. You can define your own ``RAG`` program
like this:

.. code:: python

   class RAG(dspy.Module):
       def __init__(self, num_passages=3):
           super().__init__()
           self.retrieve = dspy.Retrieve(k=num_passages)
           self.generate_answer = dspy.ChainOfThought("context, question -> answer")

       def forward(self, question):
           context = self.retrieve(question).passages
           answer = self.generate_answer(context=context, question=question)
           return answer

A program has two key methods, which you can edit to fit your needs.

**Your ``__init__`` method** declares the modules you will use. Here,
``RAG`` will use the built-in ``Retrieve`` for retrieval and
``ChainOfThought`` for generating answers. **DSPy** offers
general-purpose modules that take the shape of *your own* sub-tasks —
and not pre-built functions for specific applications.

Modules that use the LM, like ``ChainOfThought``, require a *signature*.
That is a declarative spec that tells the module what it’s expected to
do. In this example, we use the short-hand signature notation
``context, question -> answer`` to tell ``ChainOfThought`` it will be
given some ``context`` and a ``question`` and must produce an
``answer``. We will discuss more advanced
`signatures <#3a-declaring-the-inputoutput-behavior-of-lms-with-dspysignature>`__
below.

**Your ``forward`` method** expresses any computation you want to do
with your modules. In this case, we use the modules ``self.retrieve``
and ``self.generate_answer`` to search for some ``context`` and then use
the ``context`` and ``question`` to generate the ``answer``!

You can now either use this ``RAG`` program in **zero-shot mode**. Or
**compile** it to obtain higher quality. Zero-shot usage is simple. Just
define an instance of your program and then call it:

.. code:: python

   rag = RAG()  # zero-shot, uncompiled version of RAG
   rag("what is the capital of France?").answer  # -> "Paris"

The next section will discuss how to compile our simple ``RAG`` program.
When we compile it, the **DSPy compiler** will annotate *demonstrations*
of its steps: (1) retrieval, (2) using context, and (3) using
*chain-of-thought* to answer questions. From these demonstrations, the
**DSPy compiler** will make sure it produces an effective few-shot
prompt that works well with your LM, retrieval model, and data. If
you’re working with small models, it’ll finetune your model (instead of
prompting) to do this task.

If you later decide you need another step in your pipeline, just add
another module and compile again. Maybe add a module that takes the chat
history into account during search?

3) Two Powerful Concepts: Signatures & Teleprompters
====================================================

To make it possible to compile any program you write, **DSPy**
introduces two simple concepts: Signatures and Teleprompters.

3.a) Declaring the input/output behavior of LMs with ``dspy.Signature``
-----------------------------------------------------------------------

When we assign tasks to LMs in **DSPy**, we specify the behavior we need
as a **Signature**. A signature is a declarative specification of
input/output behavior of a **DSPy module**.

Instead of investing effort into *how* to get your LM to do a sub-task,
signatures enable you to inform **DSPy** *what* the sub-task is. Later,
the **DSPy compiler** will figure out how to build a complex prompt for
your large LM (or finetune your small LM) specifically for your
signature, on your data, and within your pipeline.

A signature consists of three simple elements:

-  A minimal description of the sub-task the LM is supposed to solve.
-  A description of one or more input fields (e.g., input question) that
   will we will give to the LM.
-  A description of one or more output fields (e.g., the question’s
   answer) that we will expect from the LM.

We support two notations for expressing signatures. The **short-hand
signature notation** is for quick development. You just provide your
module (e.g., ``dspy.ChainOfThought``) with a string with
``input_field_name_1, ... -> output_field_name_1, ...`` with the fields
separated by commas.

In the ``RAG`` class earlier, we saw:

.. code:: python

   self.generate_answer = dspy.ChainOfThought("context, question -> answer")

In many cases, this barebones signature is sufficient. However,
sometimes you need more control. In these cases, we can use the full
notation to express a more fully-fledged signature below.

.. code:: python

   class GenerateSearchQuery(dspy.Signature):
       """Write a simple search query that will help answer a complex question."""

       context = dspy.InputField(desc="may contain relevant facts")
       question = dspy.InputField()
       query = dspy.OutputField()

   ### inside your program's __init__ function
   self.generate_answer = dspy.ChainOfThought(GenerateSearchQuery)

You can optionally provide a ``prefix`` and/or ``desc`` key for each
input or output field to refine or constraint the behavior of modules
using your signature. The description of the sub-task itself is
specified as the docstring (i.e., ``"""Write a simple..."""``).

3.b) Asking **DSPy** to automatically optimize your program with ``dspy.teleprompt.*``
--------------------------------------------------------------------------------------

After defining the ``RAG`` program, we can **compile** it. Compiling a
program will update the parameters stored in each module. For large LMs,
this is primarily in the form of creating and validating good
demonstrations for inclusion in your prompt(s).

Compiling depends on three things: a (potentially tiny) training set, a
metric for validation, and your choice of teleprompter from **DSPy**.
**Teleprompters** are powerful optimizers (included in **DSPy**) that
can learn to bootstrap and select effective prompts for the modules of
any program. (The “tele-” in the name means “at a distance”, i.e.,
automatic prompting at a distance.)

**DSPy** typically requires very minimal labeling. For example, our
``RAG`` pipeline may work well with just a handful of examples that
contain a **question** and its (human-annotated) **answer**. Your
pipeline may involve multiple complex steps: our basic ``RAG`` example
includes a retrieved context, a chain of thought, and the answer.
However, you only need labels for the initial question and the final
answer. **DSPy** will bootstrap any intermediate labels needed to
support your pipeline. If you change your pipeline in any way, the data
bootstrapped will change accordingly!

.. code:: python

   my_rag_trainset = [
     dspy.Example(
       question="Which award did Gary Zukav's first book receive?",
       answer="National Book Award"
     ),
     ...
   ]

Second, define your validation logic, which will express some
constraints on the behavior of your program or individual modules. For
``RAG``, we might express a simple check like this:

.. code:: python

   def validate_context_and_answer(example, pred, trace=None):
       # check the gold label and the predicted answer are the same
       answer_match = example.answer.lower() == pred.answer.lower()

       # check the predicted answer comes from one of the retrieved contexts
       context_match = any((pred.answer.lower() in c) for c in pred.context)

       return answer_match and context_match

Different teleprompters offer various tradeoffs in terms of how much
they optimize cost versus quality, etc. For ``RAG``, we might use the
simple teleprompter called ``BootstrapFewShot``. To do so, we
instantiate the teleprompter itself with a validation function
``my_rag_validation_logic`` and then compile against some training set
``my_rag_trainset``.

.. code:: python

   from dspy.teleprompt import BootstrapFewShot

   teleprompter = BootstrapFewShot(metric=my_rag_validation_logic)
   compiled_rag = teleprompter.compile(RAG(), trainset=my_rag_trainset)

If we now use ``compiled_rag``, it will invoke our LM with rich prompts
with few-shot demonstrations of chain-of-thought retrieval-augmented
question answering on our data.

4) Documentation & Tutorials
============================

While we work on new tutorials and documentation, please check out `our
intro notebook <intro.ipynb>`__.

Or open it directly in free Google Colab:
` <https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb>`__


   <details>


   <summary>


   <h3 style="display: inline">

Module Reference


   </h3>


   </summary>

We have work-in-progress module documentation at `this
PR <https://github.com/stanfordnlp/dspy/pull/93>`__. Please let us know
if anything there is unclear.

Language Model Clients
~~~~~~~~~~~~~~~~~~~~~~

-  ```dspy.OpenAI`` <docs/language_models_client.md#openai>`__
-  ```dspy.Cohere`` <docs/language_models_client.md#cohere>`__
-  ```dspy.TGI`` <docs/language_models_client.md#tgi>`__
-  ```dspy.VLLM`` <docs/language_models_client.md#vllm>`__

Retrieval Model Clients
~~~~~~~~~~~~~~~~~~~~~~~

-  ```dspy.ColBERTv2`` <docs/retrieval_models_client.md#colbertv2>`__
-  ```dspy.AzureCognitiveSearch`` <docs/retrieval_models_client.md#azurecognitivesearch>`__

Signatures
~~~~~~~~~~

-  ``dspy.Signature``
-  ``dspy.InputField``
-  ``dspy.OutputField``

Modules
~~~~~~~

-  ```dspy.Predict`` <docs/modules.md#dspypredict>`__
-  ```dspy.Retrieve`` <docs/modules.md#dspyretrieve>`__
-  ```dspy.ChainOfThought`` <docs/modules.md#dspychainofthought>`__
-  ``dspy.majority`` (functional self-consistency)
-  ``dspy.ProgramOfThought`` `[see open
   PR] <https://github.com/stanfordnlp/dspy/pull/116>`__
-  ```dspy.ReAct`` <docs/modules.md#dspyreact>`__
-  ```dspy.MultiChainComparison`` <docs/modules.md#dspymultichaincomparison>`__
-  ``dspy.SelfCritique`` [coming soon]
-  ``dspy.SelfRevision`` [coming soon]

Teleprompters
~~~~~~~~~~~~~

-  ```dspy.teleprompt.LabeledFewShot`` <docs/teleprompters.md#telepromptlabeledfewshot>`__
-  ```dspy.teleprompt.BootstrapFewShot`` <docs/teleprompters.md#telepromptbootstrapfewshot>`__
-  ```dspy.teleprompt.BootstrapFewShotWithRandomSearch`` <docs/teleprompters.md#telepromptbootstrapfewshotwithrandomsearch>`__
-  ``dspy.teleprompt.LabeledFinetune`` [coming soon]
-  ```dspy.teleprompt.BootstrapFinetune`` <docs/teleprompters.md#telepromptbootstrapfinetune>`__
-  ```dspy.teleprompt.Ensemble`` <docs/teleprompters.md#telepromptensemble>`__
-  ``dspy.teleprompt.kNN`` [coming soon]


   </details>

.. _section-1:


   <details>


   <summary>


   <h3 style="display: inline">

Intro Tutorial [coming soon]


   </h3>


   </summary>

**[Intro-01] Getting Started: High Quality Pipelined Prompts with
Minimal Effort**

**[Intro-02] Using DSPy For Your Own Task: Building Blocks**

**[Intro-03] Adding Complexity: Multi-stage Programs**

**[Intro-04] Adding Complexity for Your Own Task: Design Patterns**


   </details>

.. _section-2:


   <details>


   <summary>


   <h3 style="display: inline">

Advanced Demos [coming soon]


   </h3>


   </summary>

**[Advanced-01] Long-Form QA & Programmatic Evaluation.**

**[Advanced-02] Programmatic Evaluation II & Dataset Creation.**

**[Advanced-03] Compiling & Teleprompters.**

**[Advanced-04] Extending DSPy with Modules or Teleprompters.**

**[Advanced-05]: Agents and General Tool Use in DSPy.**

**[Advanced-06]: Reproducibility, Saving Programs, and Advanced
Caching.**


   </details>

5) FAQ: Is DSPy right for me?
=============================

The **DSPy** philosophy and abstraction differ significantly from other
libraries and frameworks, so it’s usually straightforward to decide when
**DSPy** is (or isn’t) the right framework for your usecase.

If you’re a NLP/AI researcher (or a practitioner exploring new pipelines
or new tasks), the answer is generally an invariable **yes**. If you’re
a practitioner doing other things, please read on.

.. _section-3:


   <details>


   <summary>


   <h4 style="display: inline">

[5.a] DSPy vs. thin wrappers for prompts (OpenAI API, MiniChain, basic
templating)


   </h4>


   </summary>

In other words: *Why can’t I just write my prompts directly as string
templates?* Well, for extremely simple settings, this *might* work just
fine. (If you’re familiar with neural networks, this is like expressing
a tiny two-layer NN as a Python for-loop. It kinda works.)

However, when you need higher quality (or manageable cost), then you
need to iteratively explore multi-stage decomposition, improved
prompting, data bootstrapping, careful finetuning, retrieval
augmentation, and/or using smaller (or cheaper, or local) models. The
true expressive power of building with foundation models lies in the
interactions between these pieces. But every time you change one piece,
you likely break (or weaken) multiple other components.

**DSPy** cleanly abstracts away (*and* powerfully optimizes) the parts
of these interactions that are external to your actual system design. It
lets you focus on designing the module-level interactions: the *same
program* expressed in 10 or 20 lines of **DSPy** can easily be compiled
into multi-stage instructions for ``GPT-4``, detailed prompts for
``Llama2-13b``, or finetunes for ``T5-base``.

Oh, and you wouldn’t need to maintain long, brittle, model-specific
strings at the core of your project anymore.


   </details>

.. _section-4:


   <details>


   <summary>


   <h4 style="display: inline">

[5.b] DSPy vs. application development libraries like LangChain,
LlamaIndex


   </h4>


   </summary>

..

   *Note: If you use LangChain as a thin wrapper around your own prompt
   strings, refer to answer [5.a] instead.*

LangChain and LlamaIndex are popular libraries that target high-level
application development with LMs. They offer many *batteries-included*,
pre-built application modules that plug in with your data or
configuration. In practice, indeed, many usecases genuinely *don’t need*
any special components. If you’d be happy to use someone’s generic,
off-the-shelf prompt for question answering over PDFs or standard
text-to-SQL as long as it’s easy to set up on your data, then you will
probably find a very rich ecosystem in these libraries.

Unlike these libraries, **DSPy** doesn’t internally contain hand-crafted
prompts that target specific applications you can build. Instead,
**DSPy** introduces a very small set of much more powerful and
general-purpose modules *that can learn to prompt (or finetune) your LM
within your pipeline on your data*.

**DSPy** offers a whole different degree of modularity: when you change
your data, make tweaks to your program’s control flow, or change your
target LM, the **DSPy compiler** can map your program into a new set of
prompts (or finetunes) that are optimized specifically for this
pipeline. Because of this, you may find that **DSPy** obtains the
highest quality for your task, with the least effort, provided you’re
willing to implement (or extend) your own short program. In short,
**DSPy** is for when you need a lightweight but automatically-optimizing
programming model — not a library of predefined prompts and
integrations.

If you’re familiar with neural networks: > This is like the difference
between PyTorch (i.e., representing **DSPy**) and HuggingFace
Transformers (i.e., representing the higher-level libraries). If you
simply want to use off-the-shelf ``BERT-base-uncased`` or ``GPT2-large``
or apply minimal finetuning to them, HF Transformers makes it very
straightforward. If, however, you’re looking to build your own
architecture (or extend an existing one significantly), you have to
quickly drop down into something much more modular like PyTorch.
Luckily, HF Transformers *is* implemented in backends like PyTorch. We
are similarly excited about high-level wrapper around **DSPy** for
common applications. If this is implemented using **DSPy**, your
high-level application can also adapt significantly to your data in a
way that static prompt chains won’t. Please `open an
issue <https://github.com/stanfordnlp/dspy/issues/new>`__ if this is
something you want to help with.


   </details>

.. _section-5:


   <details>


   <summary>


   <h4 style="display: inline">

[5.c] DSPy vs. generation control libraries like Guidance, LMQL, RELM,
Outlines


   </h4>


   </summary>

Guidance, LMQL, RELM, and Outlines are all exciting new libraries for
controlling the individual completions of LMs, e.g., if you want to
enforce JSON output schema or constrain sampling to a particular regular
expression.

This is very useful in many settings, but it’s generally focused on
low-level, structured control of a single LM call. It doesn’t help
ensure the JSON (or structured output) you get is going to be correct or
useful for your task.

In contrast, **DSPy** automatically optimizes the prompts in your
programs to align them with various task needs, which may also include
producing valid structured ouputs. That said, we are considering
allowing **Signatures** in **DSPy** to express regex-like constraints
that are implemented by these libraries.


   </details>

Contributors & Acknowledgements
===============================

**DSPy** is led by **Omar Khattab** at Stanford NLP with **Chris Potts**
and **Matei Zaharia**.

Key contributors and team members include **Arnav Singhvi**, **Paridhi
Maheshwari**, **Keshav Santhanam**, **Sri Vardhamanan**, **Eric Zhang**,
**Hanna Moazam**, **Thomas Joshi**, **Saiful Haq**, and **Ashutosh
Sharma**.

**DSPy** includes important contributions from **Rick Battle** and
**Igor Kotenkov**. It reflects discussions with **Lisa Li**, **David
Hall**, **Ashwin Paranjape**, **Heather Miller**, **Chris Manning**,
**Percy Liang**, and many others.

The **DSPy** logo is designed by **Chuyi Zhang**.

📜 Citation & Reading More
==========================

To stay up to date or learn more, follow
`@lateinteraction <https://twitter.com/lateinteraction>`__ on Twitter.

If you use DSPy or DSP in a research paper, please cite our work as
follows:

::

   @article{khattab2023dspy,
     title={DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines},
     author={Khattab, Omar and Singhvi, Arnav and Maheshwari, Paridhi and Zhang, Zhiyuan and Santhanam, Keshav and Vardhamanan, Sri and Haq, Saiful and Sharma, Ashutosh and Joshi, Thomas T. and Moazam, Hanna and Miller, Heather and Zaharia, Matei and Potts, Christopher},
     journal={arXiv preprint arXiv:2310.03714},
     year={2023}
   }
   @article{khattab2022demonstrate,
     title={Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive {NLP}},
     author={Khattab, Omar and Santhanam, Keshav and Li, Xiang Lisa and Hall, David and Liang, Percy and Potts, Christopher and Zaharia, Matei},
     journal={arXiv preprint arXiv:2212.14024},
     year={2022}
   }

You can also read more about the evolution of the framework from
Demonstrate-Search-Predict to DSPy: \* `DSPy: Compiling Declarative
Language Model Calls into Self-Improving
Pipelines <https://arxiv.org/abs/2310.03714>`__ (Academic Paper, Oct
2023) \* `Releasing DSPy, the latest iteration of the
framework <https://twitter.com/lateinteraction/status/1694748401374490946>`__
(Twitter Thread, Aug 2023) \* `Releasing the DSP Compiler
(v0.1) <https://twitter.com/lateinteraction/status/1625231662849073160>`__
(Twitter Thread, Feb 2023) \* `Introducing
DSP <https://twitter.com/lateinteraction/status/1617953413576425472>`__
(Twitter Thread, Jan 2023) \* `Demonstrate-Search-Predict: Composing
retrieval and language models for knowledge-intensive
NLP <https://arxiv.org/abs/2212.14024.pdf>`__ (Academic Paper, Dec 2022)
