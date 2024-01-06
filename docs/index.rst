Welcome to DSPy
##########################

.. image:: images/DSPy8.png
   :width: 460px
   :align: center

DSPy is an innovative framework designed for solving advanced tasks with language models (LMs) and retrieval models (RMs). It unifies techniques for prompting, fine-tuning, and reasoning — along with approaches for self-improvement and augmentation with retrieval and tools — all expressed through modules that compose and learn.


🚀 Why DSPy?
##########################

DSPy provides composable and declarative modules for instructing LMs in a familiar Pythonic syntax, upgrading prompting techniques like chain-of-thought and self-reflection from hand-adapted string manipulation tricks into truly modular generalized operations that learn to adapt to your task.

DSPy introduces an automatic compiler that teaches LMs how to conduct the declarative steps in your program. The DSPy compiler bootstraps prompts and finetunes from minimal data without needing manual labels for the intermediate steps in your program. Instead of brittle "prompt engineering" with hacky string manipulation, you can explore a systematic space of modular and trainable pieces.

For complex tasks, DSPy can routinely teach powerful models like GPT-3.5 and local models like T5-base or Llama2-13b to be much more reliable at tasks. DSPy will compile the same program into different few-shot prompts and/or finetunes for each LM.

To install the library:

.. code-block:: bash

    pip install dspy-ai

For the optional Pinecone, Qdrant, chromadb, or marqo retrieval integration(s), include the extra(s) below:

.. code-block:: bash

    pip install dspy-ai[pinecone]  # or [qdrant] or [chromadb] or [marqo]

👨‍👩‍👧‍👦 Who is DSPy for?
*******************************************

DSPy is designed for NLP/AI researchers and practitioners who require a powerful and flexible framework for composing and optimizing LM-based applications. The intuitive high-level API empowers beginners, while the lower-level APIs allow advanced users to customize and extend modules to fit their needs.

Getting Started
****************

We recommend checking out our `Getting Started Guide <./getting_started/beginner/intro.html>`_ to help you understand how to work with DSPy.

🗺️ Contributing
*****************

# TODO


Community
************
Need help? Have a feature suggestion? Join the DSPy community:

- Discord: https://discord.gg/dspy

Associated projects
-------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started/beginner/intro.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Guides & Tutorials
   :hidden:

   guides/modules.ipynb
   guides/language_models.ipynb
   guides/optimizers.ipynb
   guides/signatures.ipynb

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api_reference/language_models/language_models_client
   api_reference/language_models/using_local_models
   api_reference/modules/modules
   api_reference/primitives/assertions
   api_reference/retrieval/retrieval_models_client
   api_reference/signatures/signatures
   api_reference/teleprompters/teleprompters







Contributors & Acknowledgements
-------------------------------

DSPy is led by Omar Khattab at Stanford NLP with Chris Potts and Matei Zaharia.

Key contributors and team members include Arnav Singhvi, Paridhi Maheshwari, Keshav Santhanam, Sri Vardhamanan, Eric Zhang, Hanna Moazam, Thomas Joshi, Saiful Haq, and Ashutosh Sharma.

DSPy includes important contributions from Rick Battle and Igor Kotenkov. It reflects discussions with Lisa Li, David Hall, Ashwin Paranjape, Heather Miller, Chris Manning, Percy Liang, and many others.

The DSPy logo is designed by Chuyi Zhang.

📜 Citation & Reading More
--------------------------

To stay up to date or learn more, follow [@lateinteraction](https://twitter.com/lateinteraction) on Twitter.

If you use DSPy or DSP in a research paper, please cite our work as follows:

.. code-block:: bibtex

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

You can also read more about the evolution of the framework from Demonstrate-Search-Predict to DSPy:
* [**DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines**](https://arxiv.org/abs/2310.03714) (Academic Paper, Oct 2023)
* [**Releasing DSPy, the latest iteration of the framework**](https://twitter.com/lateinteraction/status/1694748401374490946) (Twitter Thread, Aug 2023)
* [**Releasing the DSP Compiler (v0.1)**](https://twitter.com/lateinteraction/status/1625231662849073160)  (Twitter Thread, Feb 2023)
* [**Introducing DSP**](https://twitter.com/lateinteraction/status/1617953413576425472)  (Twitter Thread, Jan 2023)
* [**Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP**](https://arxiv.org/abs/2212.14024.pdf) (Academic Paper, Dec 2022)

| Beginner |  [**Getting Started**](intro.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb)  |  Introduces the basic building blocks in DSPy. Tackles the task of complex question answering with HotPotQA. |
| Beginner | [**Compiling for Tricky Tasks**](examples/nli/scone/scone.ipynb) | N/A | Teaches LMs to reason about logical statements and negation. Uses GPT-4 to bootstrap few-shot CoT demonstations for GPT-3.5. Establishes a state-of-the-art result on [ScoNe](https://arxiv.org/abs/2305.19426). Contributed by [Chris Potts](https://twitter.com/ChrisGPotts/status/1740033519446057077). |
| Beginner | [**Local Models & Custom Training Data**](skycamp2023.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/skycamp2023.ipynb) | Illustrates two different things together: how to use local models (Llama-2-13B in particular) and how to use your own data examples for training and development.


# TODO: map:
#### Language Model Clients

- [`dspy.OpenAI`](docs/language_models/language_models_client.md#openai)
- [`dspy.Cohere`](docs/language_models/language_models_client.md#cohere)
- [`dspy.TGI`](docs/language_models/language_models_client.md#tgi)
- [`dspy.VLLM`](docs/language_models/language_models_client.md#vllm)

#### Retrieval Model Clients

- [`dspy.ColBERTv2`](docs/retrieval/retrieval_models_client.md#colbertv2)
- [`dspy.AzureCognitiveSearch`](docs/retrieval/retrieval_models_client.md#azurecognitivesearch)


#### Signatures

- `dspy.Signature`
- `dspy.InputField`
- `dspy.OutputField`

#### Modules

- [`dspy.Predict`](docs/modules/modules.md#dspypredict)
- [`dspy.Retrieve`](docs/modules/modules.md#dspyretrieve)
- [`dspy.ChainOfThought`](docs/modules/modules.md#dspychainofthought)
- `dspy.majority` (functional self-consistency)
- `dspy.ProgramOfThought` [[see open PR]](https://github.com/stanfordnlp/dspy/pull/116)
- [`dspy.ReAct`](docs/modules/modules.md#dspyreact)
- [`dspy.MultiChainComparison`](docs/modules/modules.md#dspymultichaincomparison)
- `dspy.SelfCritique` [coming soon]
- `dspy.SelfRevision` [coming soon]


#### Teleprompters

- [`dspy.teleprompt.LabeledFewShot`](docs/Teleprompters/teleprompters.md#telepromptlabeledfewshot)
- [`dspy.teleprompt.BootstrapFewShot`](docs/Teleprompters/teleprompters.md#telepromptbootstrapfewshot)
- [`dspy.teleprompt.BootstrapFewShotWithRandomSearch`](docs/Teleprompters/teleprompters.md#telepromptbootstrapfewshotwithrandomsearch)
- `dspy.teleprompt.LabeledFinetune` [coming soon]
- [`dspy.teleprompt.BootstrapFinetune`](docs/Teleprompters/teleprompters.md#telepromptbootstrapfinetune)
- [`dspy.teleprompt.Ensemble`](docs/Teleprompters/teleprompters.md#telepromptensemble)
- `dspy.teleprompt.kNN` [coming soon]
