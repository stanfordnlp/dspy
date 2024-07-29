import copy
import random

from langchain_core.pydantic_v1 import Extra
from langchain_core.runnables import Runnable

import dsp
import dspy
from dspy.predict.parameter import Parameter
from dspy.predict.predict import Predict
from dspy.primitives.prediction import Prediction
from dspy.signatures.field import OldInputField, OldOutputField
from dspy.signatures.signature import infer_prefix

# TODO: This class is currently hard to test, because it hardcodes gpt-4 usage:
# gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=4000, model_type='chat')

class Template2Signature(dspy.Signature):
    """You are a processor for prompts. I will give you a prompt template (Python f-string) for an arbitrary task for other LMs.
Your job is to prepare three modular pieces: (i) any essential task instructions or guidelines, (ii) a list of variable names for inputs, (iv) the variable name for output."""

    template = dspy.InputField(format=lambda x: f"```\n\n{x.strip()}\n\n```\n\nLet's now prepare three modular pieces.")
    essential_instructions = dspy.OutputField()
    input_keys = dspy.OutputField(desc='comma-separated list of valid variable names')
    output_key = dspy.OutputField(desc='a valid variable name')


class ShallowCopyOnly:
    def __init__(self, obj): self.obj = obj
    def __getattr__(self, item): return getattr(self.obj, item)
    def __deepcopy__(self, memo): return ShallowCopyOnly(copy.copy(self.obj))


class LangChainPredictMetaClass(type(Predict), type(Runnable)):
    pass

class LangChainPredict(Predict, Runnable, metaclass=LangChainPredictMetaClass):
    class Config: extra = Extra.allow  # Allow extra attributes that are not defined in the model

    def __init__(self, prompt, llm, **config):
        Runnable.__init__(self)
        Parameter.__init__(self)

        self.langchain_llm = ShallowCopyOnly(llm)

        try: langchain_template = '\n'.join([msg.prompt.template for msg in prompt.messages])
        except AttributeError: langchain_template = prompt.template

        self.stage = random.randbytes(8).hex()
        self.signature, self.output_field_key = self._build_signature(langchain_template)
        self.config = config
        self.reset()

    def reset(self):
        self.lm = None
        self.traces = []
        self.train = []
        self.demos = []

    def dump_state(self):
        state_keys = ["lm", "traces", "train", "demos"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

        self.demos = [dspy.Example(**x) for x in self.demos]
    
    def __call__(self, *arg, **kwargs):
        if len(arg) > 0: kwargs = {**arg[0], **kwargs}
        return self.forward(**kwargs)
    
    def _build_signature(self, template):
        gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=4000, model_type='chat')

        with dspy.context(lm=gpt4T): parts = dspy.Predict(Template2Signature)(template=template)

        inputs = {k.strip(): OldInputField() for k in parts.input_keys.split(',')}
        outputs = {k.strip(): OldOutputField() for k in parts.output_key.split(',')}

        for k, v in inputs.items():
            v.finalize(k, infer_prefix(k))  # TODO: Generate from the template at dspy.Predict(Template2Signature)

        for k, v in outputs.items():
            output_field_key = k
            v.finalize(k, infer_prefix(k))

        return dsp.Template(parts.essential_instructions, **inputs, **outputs), output_field_key

    def forward(self, **kwargs):
        # Extract the three privileged keyword arguments.
        signature = kwargs.pop("signature", self.signature)
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))

        prompt = signature(dsp.Example(demos=demos, **kwargs))
        output = self.langchain_llm.invoke(prompt, **config)

        try: content = output.content
        except AttributeError: content = output

        pred = Prediction.from_completions([{self.output_field_key: content}], signature=signature)

        # print('#> len(demos) =', len(demos))
        # print(f"#> {prompt}")
        # print(f"#> PRED = {content}\n\n\n")
        dspy.settings.langchain_history.append((prompt, pred))
            
        if dsp.settings.trace is not None:
            trace = dsp.settings.trace
            trace.append((self, {**kwargs}, pred))

        return output
    
    def invoke(self, d, *args, **kwargs):
        # print(d)
        return self.forward(**d)


# Almost good but need output parsing for the fields!
# TODO: Use template.extract(example, p)

# class LangChainOfThought(LangChainPredict):
#     def __init__(self, signature, **config):
#         super().__init__(signature, **config)

#         signature = self.signature
#         *keys, last_key = signature.kwargs.keys()
#         rationale_type = dsp.Type(prefix="Reasoning: Let's think step by step in order to",
#                                   desc="${produce the " + last_key + "}. We ...")

#         extended_kwargs = {key: signature.kwargs[key] for key in keys}
#         extended_kwargs.update({"rationale": rationale_type, last_key: signature.kwargs[last_key]})
#         self.extended_signature = dsp.Template(signature.instructions, **extended_kwargs)

#     def forward(self, **kwargs):
#         signature = self.extended_signature
#         return super().forward(signature=signature, **kwargs)


class LangChainModule(dspy.Module):
    def __init__(self, lcel):
        super().__init__()
        
        modules = []
        for name, node in lcel.get_graph().nodes.items():
            if isinstance(node.data, LangChainPredict): modules.append(node.data)

        self.modules = modules
        self.chain = lcel
    
    def forward(self, **kwargs):
        output_keys = ['output', self.modules[-1].output_field_key]
        output = self.chain.invoke(dict(**kwargs))
        
        try: output = output.content
        except Exception: pass

        return dspy.Prediction({k: output for k in output_keys})
    
    def invoke(self, d, *args, **kwargs):
        return self.forward(**d).output

