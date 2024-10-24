# import time
# import dspy
# import ujson
# import random
# import asyncio

# #------------------------------------------------------------------------------
# class Module(dspy.BaseModule, metaclass=dspy.ProgramMeta):
#     def _base_init(self):
#         self._compiled = False

#     def __init__(self):
#         self._compiled = False

#     def __call__(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)
    
# class Predict(Module, dspy.Parameter):
#     def __init__(self, signature, _parse_values=True, **config):
#         self.stage = random.randbytes(8).hex()
#         self.signature = ensure_signature(signature)
#         self.config = config
#         self._parse_values = _parse_values

#     def __call__(self, **kwargs):
#         return self.forward(**kwargs)

#     def forward(self, **kwargs):

#         # Extract the three privileged keyword arguments.
#         signature = ensure_signature(kwargs.pop("signature", self.signature))
#         demos = kwargs.pop("demos", self.demos)
#         config = dict(**self.config, **kwargs.pop("config", {}))

#         # Get the right LM to use.
#         lm = kwargs.pop("lm", self.lm) or dsp.settings.lm
    
#         temperature = 0

#         import dspy

#         if isinstance(lm, dspy.LM):
#             completions = v2_5_generate(lm, config, signature, demos, kwargs, _parse_values=self._parse_values)

# class Adapter:
#     def __call__(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
#         inputs = self.format(signature, demos, inputs)
#         inputs = dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)

#         outputs = lm(**inputs, **lm_kwargs)
#         values = []

#         for output in outputs:
#             value = self.parse(signature, output, _parse_values=_parse_values)
#             assert set(value.keys()) == set(signature.output_fields.keys()), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
#             values.append(value)
        
#         return values
    

# class LM:
#     def __init__(
#             self, 
#             model,
#             model_type='chat', 
#             temperature=0.0,
#             max_tokens=1000,
#             cache=True,
#             launch_kwargs=None,
#             **kwargs
#         ):
#         # Remember to update LM.copy() if you modify the constructor!
#         self.model = model
#         self.model_type = model_type
#         self.cache = cache
#         self.launch_kwargs = launch_kwargs or {}
#         self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

#     def __call__(self, prompt=None, messages=None, **kwargs):
#         # Build the request.
#         cache = kwargs.pop("cache", self.cache)
#         messages = messages or [{"role": "user", "content": prompt}]
#         kwargs = {**self.kwargs, **kwargs}

#         # Make the request and handle LRU & disk caching.
#         if self.model_type == "chat":
#             completion = cached_litellm_completion if cache else litellm_completion
#         else:
#             completion = cached_litellm_text_completion if cache else litellm_text_completion

#         response = completion(ujson.dumps(dict(model=self.model, messages=messages, **kwargs)))
#         outputs = [c.message.content if hasattr(c, "message") else c["text"] for c in response["choices"]]
        
#         return outputs
    
# def v2_5_generate(lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
#     import dspy

#     adapter = dspy.settings.adapter or dspy.ChatAdapter()

#     return adapter(
#         lm, lm_kwargs=lm_kwargs, signature=signature, demos=demos, inputs=inputs, _parse_values=_parse_values
#     )

# async def async_litellm_completion(request, cache={"no-cache": True, "no-store": True}):
#     kwargs = ujson.loads(request)
#     return await litellm.acompletion(cache=cache, **kwargs)

# class QAModule(dspy.Module):
#     def __init__(self):
#         self.predictor = dspy.Predict("question -> restated_question")
#         self.predictor2 = dspy.Predict("question -> answer")
#         super().__init__()

#     def forward(self, question: str):
#         prediction = self.predictor(question=question)
#         prediction2 = self.predictor2(question=prediction.restated_question)
#         return prediction2

# lm = dspy.LM("gpt-4o-mini", cache=False)
# dspy.settings.configure(lm=lm)

# module = QAModule()
# tasks = [module(question=f"What is 1 + {i}?") for i in range(10)]

# asyncio.run(asyncio.gather(*tasks))
