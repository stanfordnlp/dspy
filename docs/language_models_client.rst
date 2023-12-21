LM Modules Documentation
========================

This documentation provides an overview of the DSPy Language Model
Clients.

Quickstart
----------

.. code:: python

   import dspy

   lm = dspy.OpenAI(model='gpt-3.5-turbo')

   prompt = "Translate the following English text to Spanish: 'Hi, how are you?'"
   completions = lm(prompt, n=5, return_sorted=False)
   for i, completion in enumerate(completions):
       print(f"Completion {i+1}: {completion}")

Supported LM Clients
--------------------

========= ============================
LM Client Jump To
========= ============================
OpenAI    `OpenAI Section <#openai>`__
Cohere    `Cohere Section <#cohere>`__
TGI       `TGI Section <#tgi>`__
VLLM      `VLLM Section <#vllm>`__
========= ============================

OpenAI
------

Usage
~~~~~

.. code:: python

   lm = dspy.OpenAI(model='gpt-3.5-turbo')

Constructor
~~~~~~~~~~~

The constructor initializes the base class ``LM`` and verifies the
provided arguments like the ``api_provider``, ``api_key``, and
``api_base`` to set up OpenAI request retrieval. The ``kwargs``
attribute is initialized with default values for relevant text
generation parameters needed for communicating with the GPT API, such as
``temperature``, ``max_tokens``, ``top_p``, ``frequency_penalty``,
``presence_penalty``, and ``n``.

.. code:: python

   class OpenAI(LM):
       def __init__(
           self,
           model: str = "text-davinci-002",
           api_key: Optional[str] = None,
           api_provider: Literal["openai", "azure"] = "openai",
           model_type: Literal["chat", "text"] = None,
           **kwargs,
       ):

**Parameters:** - ``api_key`` (*Optional[str]*, *optional*): API
provider authentication token. Defaults to None. - ``api_provider``
(*Literal[“openai”, “azure”]*, *optional*): API provider to use.
Defaults to “openai”. - ``model_type`` (*Literal[“chat”, “text”]*):
Specified model type to use. - ``**kwargs``: Additional language model
arguments to pass to the API provider.

Methods
~~~~~~~

``__call__(self, prompt: str, only_completed: bool = True, return_sorted: bool = False, **kwargs) -> List[Dict[str, Any]]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Retrieves completions from OpenAI by calling ``request``.

Internally, the method handles the specifics of preparing the request
prompt and corresponding payload to obtain the response.

After generation, the completions are post-processed based on the
``model_type`` parameter. If the parameter is set to ‘chat’, the
generated content look like ``choice["message"]["content"]``. Otherwise,
the generated text will be ``choice["text"]``.

**Parameters:** - ``prompt`` (*str*): Prompt to send to OpenAI. -
``only_completed`` (*bool*, *optional*): Flag to return only completed
responses and ignore completion due to length. Defaults to True. -
``return_sorted`` (*bool*, *optional*): Flag to sort the completion
choices using the returned averaged log-probabilities. Defaults to
False. - ``**kwargs``: Additional keyword arguments for completion
request.

**Returns:** - ``List[Dict[str, Any]]``: List of completion choices.

Cohere
------

.. _usage-1:

Usage
~~~~~

.. code:: python

   lm = dsp.Cohere(model='command-xlarge-nightly')

.. _constructor-1:

Constructor
~~~~~~~~~~~

The constructor initializes the base class ``LM`` and verifies the
``api_key`` to set up Cohere request retrieval.

.. code:: python

   class Cohere(LM):
       def __init__(
           self,
           model: str = "command-xlarge-nightly",
           api_key: Optional[str] = None,
           stop_sequences: List[str] = [],
       ):

**Parameters:** - ``model`` (*str*): Cohere pretrained models. Defaults
to ``command-xlarge-nightly``. - ``api_key`` (*Optional[str]*,
*optional*): API provider from Cohere. Defaults to None. -
``stop_sequences`` (*List[str]*, *optional*): List of stopping tokens to
end generation.

.. _methods-1:

Methods
~~~~~~~

Refer to ```dspy.OpenAI`` <#openai>`__ documentation.

TGI
---

.. _usage-2:

Usage
~~~~~

.. code:: python

   lm = dspy.HFClientTGI(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")

Prerequisites
~~~~~~~~~~~~~

Refer to the `Text Generation-Inference
Server <https://github.com/stanfordnlp/dspy/blob/local_models_docs/docs/using_local_models.md#text-generation-inference-server>`__
section of the ``Using Local Models`` documentation.

.. _constructor-2:

Constructor
~~~~~~~~~~~

The constructor initializes the ``HFModel`` base class and configures
the client for communicating with the TGI server. It requires a
``model`` instance, communication ``port`` for the server, and the
``url`` for the server to host generate requests. Additional
configuration can be provided via keyword arguments in ``**kwargs``.

.. code:: python

   class HFClientTGI(HFModel):
       def __init__(self, model, port, url="http://future-hgx-1", **kwargs):

**Parameters:** - ``model`` (*HFModel*): Instance of Hugging Face model
connected to the TGI server. - ``port`` (*int*): Port for TGI server. -
``url`` (*str*): Base URL where the TGI server is hosted. -
``**kwargs``: Additional keyword arguments to configure the client.

.. _methods-2:

Methods
~~~~~~~

Refer to ```dspy.OpenAI`` <#openai>`__ documentation.

VLLM
----

.. _usage-3:

Usage
~~~~~

.. code:: python

   lm = dspy.HFClientVLLM(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")

.. _prerequisites-1:

Prerequisites
~~~~~~~~~~~~~

Refer to the `vLLM
Server <https://github.com/stanfordnlp/dspy/blob/local_models_docs/docs/using_local_models.md#vllm-server>`__
section of the ``Using Local Models`` documentation.

.. _constructor-3:

Constructor
~~~~~~~~~~~

Refer to ```dspy.TGI`` <#tgi>`__ documentation. Replace with
``HFClientVLLM``.

.. _methods-3:

Methods
~~~~~~~

Refer to ```dspy.OpenAI`` <#openai>`__ documentation.
