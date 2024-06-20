import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.generic_utils import (
    prompt_to_messages,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.base.query_pipeline.query import InputKeys, OutputKeys, QueryComponent
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline

import dsp
import dspy
from dspy import Predict
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import ensure_signature, make_signature, signature_to_template


def get_formatted_template(predict_module: Predict, kwargs: Dict[str, Any]) -> str:
    """Get formatted template from predict module."""
    # Extract the three privileged keyword arguments.
    signature = ensure_signature(predict_module.signature)
    demos = predict_module.demos

    # All of the other kwargs are presumed to fit a prefix of the signature.
    # That is, they are input variables for the bottom most generation, so
    # we place them inside the input - x - together with the demos.
    x = dsp.Example(demos=demos, **kwargs)

    # Switch to legacy format for dsp.generate
    template = signature_to_template(signature)

    return template(x)


def replace_placeholder(text: str) -> str:
    # Use a regular expression to find and replace ${...} with ${{...}}
    return re.sub(r'\$\{([^\{\}]*)\}', r'${{\1}}', text)


def _input_keys_from_template(template: dsp.Template) -> InputKeys:
    """Get input keys from template."""
    # get only fields that are marked OldInputField and NOT OldOutputField
    # template_vars = list(template.kwargs.keys())
    return [
        k for k, v in template.kwargs.items() if isinstance(v, dspy.signatures.OldInputField)
    ]

def _output_keys_from_template(template: dsp.Template) -> InputKeys:
    """Get output keys from template."""
    # get only fields that are marked OldOutputField and NOT OldInputField
    # template_vars = list(template.kwargs.keys())
    return [
        k for k, v in template.kwargs.items() if isinstance(v, dspy.signatures.OldOutputField)
    ]


class DSPyPromptTemplate(BasePromptTemplate):
    """A prompt template for DSPy.

    Takes in a predict module from DSPy (whether unoptimized or optimized),
    and extracts the relevant prompt template from it given the input.
    
    """

    predict_module: Predict

    def __init__(
        self,
        predict_module: Predict,
        metadata: Optional[Dict[str, Any]] = None,
        template_var_mappings: Optional[Dict[str, Any]] = None,
        function_mappings: Optional[Dict[str, Callable]] = None,
        **kwargs: Any,
    ) -> None:
        template = signature_to_template(predict_module.signature)
        template_vars = _input_keys_from_template(template)
        # print(f"TEMPLATE VARS: {template_vars}")
        # raise Exception

        super().__init__(
            predict_module=predict_module,
            metadata=metadata or {},
            template_vars=template_vars,
            kwargs=kwargs,
            template_var_mappings=template_var_mappings,
            function_mappings=function_mappings,
        )
    
    def partial_format(self, **kwargs: Any) -> "BasePromptTemplate":
        """Returns a new prompt template with the provided kwargs."""
        # NOTE: this is a copy of the implementation in `PromptTemplate`
        output_parser = self.output_parser
        self.output_parser = None

        # get function and fixed kwargs, and add that to a copy
        # of the current prompt object
        prompt = deepcopy(self)
        prompt.kwargs.update(kwargs)

        # NOTE: put the output parser back
        prompt.output_parser = output_parser
        self.output_parser = output_parser
        return prompt

    def format(self, llm: Optional[BaseLLM] = None, **kwargs: Any) -> str:
        """Formats the prompt template."""
        mapped_kwargs = self._map_all_vars(kwargs)
        return get_formatted_template(self.predict_module, mapped_kwargs)

    def format_messages(
        self, llm: Optional[BaseLLM] = None, **kwargs: Any,
    ) -> List[ChatMessage]:
        """Formats the prompt template into chat messages."""
        del llm  # unused
        prompt = self.format(**kwargs)
        return prompt_to_messages(prompt)

    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        """Get template."""
        # get kwarg templates
        kwarg_tmpl_map = {k: "{k}" for k in self.template_vars}

        # get "raw" template with all the values filled in with {var_name} 
        template0 = get_formatted_template(self.predict_module, kwarg_tmpl_map)
        # HACK: there are special 'format' variables of the form ${var_name} that are meant to 
        # prompt the LLM, but we do NOT want to replace with actual prompt variable values. 
        # Replace those with double brackets
        template1 = replace_placeholder(template0)

        return template1


# copied from langchain.py
class Template2Signature(dspy.Signature):
    """You are a processor for prompts. I will give you a prompt template (Python f-string) for an arbitrary task for other LMs.
Your job is to prepare three modular pieces: (i) any essential task instructions or guidelines, (ii) a list of variable names for inputs, (iv) the variable name for output."""

    template = dspy.InputField(format=lambda x: f"```\n\n{x.strip()}\n\n```\n\nLet's now prepare three modular pieces.")
    essential_instructions = dspy.OutputField()
    input_keys = dspy.OutputField(desc='comma-separated list of valid variable names')
    output_key = dspy.OutputField(desc='a valid variable name')


def build_signature(prompt: PromptTemplate) -> dspy.Signature:
    """Attempt to build signature from prompt."""
    # TODO: allow plugging in any llamaindex LLM 
    gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=4000, model_type='chat')

    with dspy.context(lm=gpt4T): 
        parts = dspy.Predict(Template2Signature)(template=prompt.template)

    inputs = {k.strip(): InputField() for k in parts.input_keys.split(',')}
    outputs = {k.strip(): OutputField() for k in parts.output_key.split(',')}

    # dynamically create a pydantic model that subclasses dspy.Signature
    fields = {
        k: (str, v) for k, v in {**inputs, **outputs}.items()
    }
    signature = make_signature(fields, parts.essential_instructions)
    return signature
    

class DSPyComponent(QueryComponent):
    """DSPy Query Component. 
    
    Can take in either a predict module directly.
    TODO: add ability to translate from an existing prompt template / LLM.
    
    """
    predict_module: dspy.Predict
    predict_template: dsp.Template

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        predict_module: dspy.Predict,
    ) -> None:
        """Initialize."""
        return super().__init__(
            predict_module=predict_module,
            predict_template=signature_to_template(predict_module.signature),
        )

    @classmethod
    def from_prompt(
        cls,
        prompt_template: BasePromptTemplate,
        # llm: BaseLLM,
    ) -> "DSPyComponent":
        """Initialize from prompt template.

        LLM is a TODO - currently use DSPy LLM classes.
        
        """
        signature = build_signature(prompt_template)
        predict_module = Predict(signature)
        return cls(predict_module=predict_module)
    
    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: implement
        pass

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        return input

    def _run_component(self, **kwargs: Any) -> Dict:
        """Run component."""
        prediction = self.predict_module(**kwargs)
        return {
            k: getattr(prediction, k) for k in self.output_keys.required_keys
        }

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        # TODO: no async predict module yet
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        input_keys = _input_keys_from_template(self.predict_template)
        return InputKeys.from_keys(input_keys)

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        output_keys = _output_keys_from_template(self.predict_template)
        return OutputKeys.from_keys(output_keys)


class LlamaIndexModule(dspy.Module):
    """A module for LlamaIndex.

    Wraps a QueryPipeline and exposes it as a dspy module for optimization.
    
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, query_pipeline: QueryPipeline) -> None:
        """Initialize."""
        super().__init__()
        self.query_pipeline = query_pipeline
        self.predict_modules = []
        for module in query_pipeline.module_dict.values():
            if isinstance(module, DSPyComponent):
                self.predict_modules.append(module.predict_module)
        

    def forward(self, **kwargs: Any) -> Dict[str, Any]:
        """Forward."""
        output_dict = self.query_pipeline.run(**kwargs, return_values_direct=False)
        return dspy.Prediction(**output_dict)
    
