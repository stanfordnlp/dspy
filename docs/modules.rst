dspy.Modules Documentation
==========================

This documentation provides an overview of the DSPy Modules.

DSPy Modules
------------

+-----------------------------------+-----------------------------------+
| Module                            | Jump To                           |
+===================================+===================================+
| Predict                           | `Predict                          |
|                                   | Section <#dspypredict>`__         |
+-----------------------------------+-----------------------------------+
| Retrieve                          | `Retrieve                         |
|                                   | Section <#dspyretrieve>`__        |
+-----------------------------------+-----------------------------------+
| ChainOfThought                    | `ChainOfThought                   |
|                                   | Section <#dspychainofthought>`__  |
+-----------------------------------+-----------------------------------+
| ChainOfThoughtWithHint            | `ChainOfThoughtWithHint           |
|                                   | Section                           |
|                                   |  <#dspychainofthoughtwithhint>`__ |
+-----------------------------------+-----------------------------------+
| MultiChainComparison              | `MultiChainComparison             |
|                                   | Secti                             |
|                                   | on <#dspymultichaincomparison>`__ |
+-----------------------------------+-----------------------------------+
| ReAct                             | `ReAct Section <#dspyreact>`__    |
+-----------------------------------+-----------------------------------+
| Assertion Helpers                 | `Assertion Helpers                |
|                                   | S                                 |
|                                   | ection <#dspyassertionhelpers>`__ |
+-----------------------------------+-----------------------------------+

dspy.Predict
------------

Constructor
~~~~~~~~~~~

The constructor initializes the ``Predict`` class and sets up its
attributes, taking in the ``signature`` and additional config options.
If the ``signature`` is a string, it processes the input and output
fields, generates instructions, and creates a template for the specified
``signature`` type.

.. code:: python

   class Predict(Parameter):
       def __init__(self, signature, **config):
           self.stage = random.randbytes(8).hex()
           self.signature = signature
           self.config = config
           self.reset()

           if isinstance(signature, str):
               inputs, outputs = signature.split("->")
   ## dspy.Assertion Helpers

   ### Assertion Handlers

   The assertion handlers are used to control the behavior of assertions and suggestions in the DSPy framework. They can be used to bypass assertions or suggestions, handle assertion errors, and backtrack suggestions.

   #### `noop_handler(func)`

   This handler is used to bypass assertions and suggestions. When used, both assertions and suggestions will become no-operations (noops).

   #### `bypass_suggest_handler(func)`

   This handler is used to bypass suggestions only. If a suggestion fails, it will be logged but not raised. If an assertion fails, it will be raised.

   #### `bypass_assert_handler(func)`

   This handler is used to bypass assertions only. If an assertion fails, it will be logged but not raised. If a suggestion fails, it will be raised.

   #### `assert_no_except_handler(func)`

   This handler is used to ignore assertion failures and return None.

   #### `suggest_backtrack_handler(func, bypass_suggest=True, max_backtracks=2)`

   This handler is used for backtracking suggestions. It re-runs the latest predictor up to `max_backtracks` times, with updated signature if a suggestion fails.

   #### `handle_assert_forward(assertion_handler, **handler_args)`

   This function is used to handle assertions. It wraps the `forward` method of a module with an assertion handler.

   #### `assert_transform_module(module, assertion_handler=default_assertion_handler, **handler_args)`

   This function is used to transform a module to handle assertions. It replaces the `forward` method of the module with a version that handles assertions.
               inputs, outputs = inputs.split(","), outputs.split(",")
               inputs, outputs = [field.strip() for field in inputs], [field.strip() for field in outputs]

               assert all(len(field.split()) == 1 for field in (inputs + outputs))

               inputs_ = ', '.join([f"`{field}`" for field in inputs])
               outputs_ = ', '.join([f"`{field}`" for field in outputs])

               instructions = f"""Given the fields {inputs_}, produce the fields {outputs_}."""

               inputs = {k: InputField() for k in inputs}
               outputs = {k: OutputField() for k in outputs}

               for k, v in inputs.items():
                   v.finalize(k, infer_prefix(k))
               
               for k, v in outputs.items():
                   v.finalize(k, infer_prefix(k))

               self.signature = dsp.Template(instructions, **inputs, **outputs)
               
               for k, v in outputs.items():
                   v.finalize(k, infer_prefix(k))

               self.signature = dsp.Template(instructions, **inputs, **outputs)

**Parameters:** - ``signature`` (*Any*): Signature of predictive model.
- ``**config`` (*dict*): Additional configuration parameters for model.

Method
~~~~~~

``__call__(self, **kwargs)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method serves as a wrapper for the ``forward`` method. It allows
making predictions using the ``Predict`` class by providing keyword
arguments.

**Paramters:** - ``**kwargs``: Keyword arguments required for
prediction.

**Returns:** - The result of ``forward`` method.

Examples
~~~~~~~~

.. code:: python

   #Define a simple signature for basic question answering
   class BasicQA(dspy.Signature):
       """Answer questions with short factoid answers."""
       question = dspy.InputField()
       answer = dspy.OutputField(desc="often between 1 and 5 words")

   #Pass signature to Predict module
   generate_answer = dspy.Predict(BasicQA)

   # Call the predictor on a particular input.
   question='What is the color of the sky?'
   pred = generate_answer(question=question)

   print(f"Question: {question}")
   print(f"Predicted Answer: {pred.answer}")

dspy.Retrieve
-------------

.. _constructor-1:

Constructor
~~~~~~~~~~~

The constructor initializes the ``Retrieve`` class and sets up its
attributes, taking in ``k`` number of retrieval passages to return for a
query.

.. code:: python

   class Retrieve(Parameter):
       def __init__(self, k=3):
           self.stage = random.randbytes(8).hex()
           self.k = k

**Parameters:** - ``k`` (*Any*): Number of retrieval responses

.. _method-1:

Method
~~~~~~

``__call__(self, *args, **kwargs):``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method serves as a wrapper for the ``forward`` method. It allows
making retrievals on an input query using the ``Retrieve`` class.

**Parameters:** - ``**args``: Arguments required for retrieval. -
``**kwargs``: Keyword arguments required for retrieval.

**Returns:** - The result of the ``forward`` method.

.. _examples-1:

Examples
~~~~~~~~

.. code:: python

   query='When was the first FIFA World Cup held?'

   # Call the retriever on a particular query.
   retrieve = dspy.Retrieve(k=3)
   topK_passages = retrieve(query).passages

   print(f"Top {retrieve.k} passages for question: {query} \n", '-' * 30, '\n')

   for idx, passage in enumerate(topK_passages):
       print(f'{idx+1}]', passage, '\n')

dspy.ChainOfThought
===================

The constructor initializes the ``ChainOfThought`` class and sets up its
attributes. It inherits from the ``Predict`` class and adds specific
functionality for chain of thought processing.

Internally, the class initializes the ``activated`` attribute to
indicate if chain of thought processing has been selected. It extends
the ``signature`` to include additional reasoning steps and an updated
``rationale_type`` when chain of thought processing is activated.

.. code:: python

   class ChainOfThought(Predict):
       def __init__(self, signature, rationale_type=None, activated=True, **config):
           super().__init__(signature, **config)

           self.activated = activated

           signature = self.signature
           *keys, last_key = signature.kwargs.keys()

           DEFAULT_RATIONALE_TYPE = dsp.Type(prefix="Reasoning: Let's think step by step in order to",
                                             desc="${produce the " + last_key + "}. We ...")

           rationale_type = rationale_type or DEFAULT_RATIONALE_TYPE
           
           extended_kwargs = {key: signature.kwargs[key] for key in keys}
           extended_kwargs.update({'rationale': rationale_type, last_key: signature.kwargs[last_key]})
           
           self.extended_signature = dsp.Template(signature.instructions, **extended_kwargs)

**Parameters:** - ``signature`` (*Any*): Signature of predictive model.
- ``rationale_type`` (*dsp.Type*, *optional*): Rationale type for
reasoning steps. Defaults to ``None``. - ``activated`` (*bool*,
*optional*): Flag for activated chain of thought processing. Defaults to
``True``. - ``**config`` (*dict*): Additional configuration parameters
for model.

.. _method-2:

Method
------

``forward(self, **kwargs)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method extends the parent ``Predict`` class’ forward pass while
updating the signature when chain of thought reasoning is activated or
if the language model is a GPT3 model.

**Parameters:** - ``**kwargs``: Keyword arguments required for
prediction.

**Returns:** - The result of the ``forward`` method.

.. _examples-2:

Examples
--------

.. code:: python

   #Define a simple signature for basic question answering
   class BasicQA(dspy.Signature):
       """Answer questions with short factoid answers."""
       question = dspy.InputField()
       answer = dspy.OutputField(desc="often between 1 and 5 words")

   #Pass signature to ChainOfThought module
   generate_answer = dspy.ChainOfThought(BasicQA)

   # Call the predictor on a particular input.
   question='What is the color of the sky?'
   pred = generate_answer(question=question)

   print(f"Question: {question}")
   print(f"Predicted Answer: {pred.answer}")

dspy.ChainOfThoughtWithHint
---------------------------

.. _constructor-2:

Constructor
~~~~~~~~~~~

The constructor initializes the ``ChainOfThoughtWithHint`` class and
sets up its attributes, inheriting from the ``Predict`` class. This
class enhances the ``ChainOfThought`` class by offering an additional
option to provide hints for reasoning. Two distinct signature templates
are created internally depending on the presence of the hint.

.. code:: python

   class ChainOfThoughtWithHint(Predict):
       def __init__(self, signature, rationale_type=None, activated=True, **config):
           super().__init__(signature, **config)

           self.activated = activated

           signature = self.signature
           *keys, last_key = signature.kwargs.keys()

           DEFAULT_HINT_TYPE = dsp.Type(prefix="Hint:", desc="${hint}")

           DEFAULT_RATIONALE_TYPE = dsp.Type(prefix="Reasoning: Let's think step by step in order to",
                                             desc="${produce the " + last_key + "}. We ...")

           rationale_type = rationale_type or DEFAULT_RATIONALE_TYPE
           
           extended_kwargs1 = {key: signature.kwargs[key] for key in keys}
           extended_kwargs1.update({'rationale': rationale_type, last_key: signature.kwargs[last_key]})

           extended_kwargs2 = {key: signature.kwargs[key] for key in keys}
           extended_kwargs2.update({'hint': DEFAULT_HINT_TYPE, 'rationale': rationale_type, last_key: signature.kwargs[last_key]})
           
           self.extended_signature1 = dsp.Template(signature.instructions, **extended_kwargs1)
           self.extended_signature2 = dsp.Template(signature.instructions, **extended_kwargs2)

**Parameters:** - ``signature`` (*Any*): Signature of predictive model.
- ``rationale_type`` (*dsp.Type*, *optional*): Rationale type for
reasoning steps. Defaults to ``None``. - ``activated`` (*bool*,
*optional*): Flag for activated chain of thought processing. Defaults to
``True``. - ``**config`` (*dict*): Additional configuration parameters
for model.

.. _method-3:

Method
~~~~~~

.. _forwardself-kwargs-1:

``forward(self, **kwargs)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method extends the parent ``Predict`` class’s forward pass,
updating the signature dynamically based on the presence of ``hint`` in
the keyword arguments and the ``activated`` attribute.

**Parameters:** - ``**kwargs``: Keyword arguments required for
prediction.

**Returns:** - The result of the ``forward`` method in the parent
``Predict`` class.

.. _examples-3:

Examples
~~~~~~~~

.. code:: python

   #Define a simple signature for basic question answering
   class BasicQA(dspy.Signature):
       """Answer questions with short factoid answers."""
       question = dspy.InputField()
       answer = dspy.OutputField(desc="often between 1 and 5 words")

   #Pass signature to ChainOfThought module
   generate_answer = dspy.ChainOfThoughtWithHint(BasicQA)

   # Call the predictor on a particular input alongside a hint.
   question='What is the color of the sky?'
   hint = "It's what you often see during a sunny day."
   pred = generate_answer(question=question, hint=hint)

   print(f"Question: {question}")
   print(f"Predicted Answer: {pred.answer}")

dspy.MultiChainComparison
-------------------------

.. _constructor-3:

Constructor
~~~~~~~~~~~

The constructor initializes the ``MultiChainComparison`` class and sets
up its attributes. It inherits from the ``Predict`` class and adds
specific functionality for multiple chain comparisons.

The class incorporates multiple student attempt reasonings and concludes
with the selected best reasoning path out of the available attempts.

.. code:: python

   from .predict import Predict
   from ..primitives.program import Module

   import dsp

   class MultiChainComparison(Module):
       def __init__(self, signature, M=3, temperature=0.7, **config):
           super().__init__()

           self.M = M
           signature = Predict(signature).signature
           *keys, last_key = signature.kwargs.keys()

           extended_kwargs = {key: signature.kwargs[key] for key in keys}

           for idx in range(M):
               candidate_type = dsp.Type(prefix=f"Student Attempt #{idx+1}:", desc="${reasoning attempt}")
               extended_kwargs.update({f'reasoning_attempt_{idx+1}': candidate_type})
           
           rationale_type = dsp.Type(prefix="Accurate Reasoning: Thank you everyone. Let's now holistically", desc="${corrected reasoning}")
           extended_kwargs.update({'rationale': rationale_type, last_key: signature.kwargs[last_key]})

           signature = dsp.Template(signature.instructions, **extended_kwargs)
           self.predict = Predict(signature, temperature=temperature, **config)
           self.last_key = last_key

**Parameters:** - ``signature`` (*Any*): Signature of predictive model.
- ``M`` (*int*, *optional*): Number of student reasoning attempts.
Defaults to ``3``. - ``temperature`` (*float*, *optional*): Temperature
parameter for prediction. Defaults to ``0.7``. - ``**config`` (*dict*):
Additional configuration parameters for model.

.. _method-4:

Method
~~~~~~

``forward(self, completions, **kwargs)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method aggregates all the student reasoning attempts and calls the
predict method with extended signatures to get the best reasoning.

**Parameters:** - ``completions``: List of completion objects which
include student reasoning attempts. - ``**kwargs``: Additional keyword
arguments.

**Returns:** - The result of the ``predict`` method for the best
reasoning.

.. _examples-4:

Examples
~~~~~~~~

.. code:: python

   class BasicQA(dspy.Signature):
       """Answer questions with short factoid answers."""
       question = dspy.InputField()
       answer = dspy.OutputField(desc="often between 1 and 5 words")

   # Example completions generated by a model for reference
   completions = [
       dspy.Prediction(rationale="I recall that during clear days, the sky often appears this color.", answer="blue"),
       dspy.Prediction(rationale="Based on common knowledge, I believe the sky is typically seen as this color.", answer="green"),
       dspy.Prediction(rationale="From images and depictions in media, the sky is frequently represented with this hue.", answer="blue"),
   ]

   # Pass signature to MultiChainComparison module
   compare_answers = dspy.MultiChainComparison(BasicQA)

   # Call the MultiChainComparison on the completions
   question = 'What is the color of the sky?'
   final_pred = compare_answers(completions, question=question)

   print(f"Question: {question}")
   print(f"Final Predicted Answer (after comparison): {final_pred.answer}")
   print(f"Final Rationale: {final_pred.rationale}")

dspy.ReAct
----------

.. _constructor-4:

Constructor
~~~~~~~~~~~

The constructor initializes the ``ReAct`` class and sets up its
attributes. It is specifically designed to compose the interleaved steps
of Thought, Action, and Observation.

Internally, the class follows a sequential process: Thoughts (or
reasoning) lead to Actions (such as queries or activities). These
Actions then result in Observations (like results or responses), which
subsequently feedback into the next Thought. This cycle is maintained
for a predefined number of iterations.

.. code:: python

   import dsp
   import dspy
   from ..primitives.program import Module
   from .predict import Predict

   class ReAct(Module):
       def __init__(self, signature, max_iters=5, num_results=3, tools=None):
           ...

**Parameters:** - ``signature`` (*Any*): Signature of the predictive
model. - ``max_iters`` (*int*, *optional*): Maximum number of iterations
for the Thought-Action-Observation cycle. Defaults to ``5``. -
``num_results`` (*int*, *optional*): Number of results to retrieve in
the action step. Defaults to ``3``. - ``tools`` (*List[dspy.Tool]*,
*optional*): List of tools available for actions. If none is provided, a
default ``Retrieve`` tool with ``num_results`` is used.

Methods
~~~~~~~

``_generate_signature(self, iters)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generates a signature for the Thought-Action-Observation cycle based on
the number of iterations.

**Parameters:** - ``iters`` (*int*): Number of iterations.

**Returns:** - A dictionary representation of the signature.

``act(self, output, hop)``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Processes an action and returns the observation or final answer.

**Parameters:** - ``output`` (*dict*): Current output from the Thought.
- ``hop`` (*int*): Current iteration number.

**Returns:** - A string representing the final answer or ``None``.

.. _forwardself-kwargs-2:

``forward(self, **kwargs)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Main method to execute the Thought-Action-Observation cycle for a given
set of input fields.

**Parameters:** - ``**kwargs``: Keyword arguments corresponding to input
fields.

**Returns:** - A ``dspy.Prediction`` object containing the result of the
ReAct process.

.. _examples-5:

Examples
~~~~~~~~

.. code:: python

   # Define a simple signature for basic question answering
   class BasicQA(dspy.Signature):
       """Answer questions with short factoid answers."""
       question = dspy.InputField()
       answer = dspy.OutputField(desc="often between 1 and 5 words")

   # Pass signature to ReAct module
   react_module = dspy.ReAct(BasicQA)

   # Call the ReAct module on a particular input
   question = 'What is the color of the sky?'
   result = react_module(question=question)

   print(f"Question: {question}")
   print(f"Final Predicted Answer (after ReAct process): {result.answer}")
