Welcome to DSPy
##########################

DSPy is an innovative framework for programmatically harnessing foundation models, providing tools and interfaces in Python and Typescript for enhanced interaction with large language models. Integrating domain-specific data with powerful language models allows users to design tailored applications in the fields of natural language processing, machine learning, and artificial intelligence.

🚀 Why DSPy? 🦙 !
##########################

LlamaIndex is a data framework for `LLM <https://en.wikipedia.org/wiki/Large_language_model>`_-based applications to ingest, structure, and access private or domain-specific data. It's available in Python (these docs) and `Typescript <https://ts.llamaindex.ai/>`_.

🚀 Empowering Applications with Foundation Models
******************

LLMs offer a natural language interface between humans and data. Widely available models come pre-trained on huge amounts of publicly available data like Wikipedia, mailing lists, textbooks, source code and more.

However, while LLMs are trained on a great deal of data, they are not trained on **your** data, which may be private or specific to the problem you're trying to solve. It's behind APIs, in SQL databases, or trapped in PDFs and slide decks.

LlamaIndex solves this problem by connecting to these data sources and adding your data to the data LLMs already have. This is often called Retrieval-Augmented Generation (RAG). RAG enables you to use LLMs to query your data, transform it, and generate new insights. You can ask questions about your data, create chatbots, build semi-autonomous agents, and more. To learn more, check out our Use Cases on the left.

🦙 How can LlamaIndex help?
***************************

LlamaIndex provides the following tools:

- **Data connectors** ingest your existing data from their native source and format. These could be APIs, PDFs, SQL, and (much) more.
- **Data indexes** structure your data in intermediate representations that are easy and performant for LLMs to consume.
- **Engines** provide natural language access to your data. For example:
  - Query engines are powerful retrieval interfaces for knowledge-augmented output.
  - Chat engines are conversational interfaces for multi-message, "back and forth" interactions with your data.
- **Data agents** are foundation model-powered knowledge workers enhanced by tools, including helper functions to API integrations.


👨‍👩‍👧‍👦 Who is DSPy for?
*******************************************

LlamaIndex provides tools for beginners, advanced users, and everyone in between.

Our intuitive high-level API empowers beginners to leverage the capabilities of DSPy to ingest and query their data in 5 lines of code.

For more complex applications, our lower-level APIs allow advanced users to customize and extend any module—data connectors, indices, retrievers, query engines, reranking modules—to fit their needs.

Getting Started
****************

To install the library:

``pip install dspy``

We recommend checking out our `Getting Started Guide <./getting_started/overview.html>`_ to help you navigate the documentation based on your expertise.

🗺️ Ecosystem
************

To download or contribute, find LlamaIndex on:

- Github: https://github.com/[DSPY_REPO_PATH]
- PyPi:

  - DSPy: https://pypi.org/project/dspy/.


- NPM (Typescript/Javascript):
   - Github: https://github.com/[DSPY_TS_REPO_PATH]
   - Docs: https://ts.dspy.ai/
   - DSPy.TS: https://www.npmjs.com/package/dspy

Community
---------
Need help? Have a feature suggestion? Join the LlamaIndex community:

- Twitter: https://twitter.com/dspy_framework
- Discord https://discord.gg/[DSPY_DISCORD_PATH]

Associated projects
-------------------

- 🏡 DSPyHub: https://dspyhub.ai | A large (and growing!) collection of custom data connectors
- 🧪 DSPyLab: https://github.com/[DSPY_LAB_REPO_PATH] | Innovative projects leveraging DSPy capabilities

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/installation.md
   getting_started/reading.md
   getting_started/starter_example.md
   getting_started/concepts.md
   getting_started/customization.rst
   getting_started/discover_llamaindex.md

.. toctree::
   :maxdepth: 2
   :caption: Use Cases
   :hidden:

   use_cases/q_and_a.md
   use_cases/chatbots.md
   use_cases/agents.md
   use_cases/extraction.md
   use_cases/multimodal.md

.. toctree::
   :maxdepth: 2
   :caption: Understanding
   :hidden:

   understanding/understanding.md
   understanding/using_llms/using_llms.md
   understanding/loading/loading.md
   understanding/indexing/indexing.md
   understanding/storing/storing.md
   understanding/querying/querying.md
   understanding/putting_it_all_together/putting_it_all_together.md
   understanding/tracing_and_debugging/tracing_and_debugging.md
   understanding/evaluating/evaluating.md

.. toctree::
   :maxdepth: 2
   :caption: Optimizing
   :hidden:

   optimizing/basic_strategies/basic_strategies.md
   optimizing/advanced_retrieval/advanced_retrieval.md
   optimizing/agentic_strategies/agentic_strategies.md
   optimizing/evaluation/evaluation.md
   optimizing/fine-tuning/fine-tuning.md
   optimizing/production_rag.md
   optimizing/building_rag_from_scratch.md
.. toctree::
   :maxdepth: 2
   :caption: Module Guides
   :hidden:

   module_guides/models/models.md
   module_guides/models/prompts.md
   module_guides/loading/loading.md
   module_guides/indexing/indexing.md
   module_guides/storing/storing.md
   module_guides/querying/querying.md
   module_guides/observability/observability.md
   module_guides/evaluating/root.md
   module_guides/supporting_modules/supporting_modules.md


.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api_reference/index.rst

.. toctree::
   :maxdepth: 2
   :caption: Community
   :hidden:

   community/integrations.md
   community/frequently_asked_questions.md
   community/full_stack_projects.md

.. toctree::
   :maxdepth: 2
   :caption: Contributing
   :hidden:

   contributing/contributing.rst
   contributing/documentation.rst

.. toctree::
   :maxdepth: 2
   :caption: Changes
   :hidden:

   changes/changelog.rst
   changes/deprecated_terms.md
