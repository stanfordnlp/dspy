import random

from pydantic import BaseModel

import dsp
from dspy.predict.parameter import Parameter
from dspy.primitives.program import Module

from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import ensure_signature, signature_to_template

import logging
from functools import lru_cache

@lru_cache(maxsize=None)
def warn_once(msg: str):
    logging.warning(msg)


class Predict(Module, Parameter):
    def __init__(self, signature, _parse_values=True, **config):
        self.stage = random.randbytes(8).hex()
        self.signature = ensure_signature(signature)
        self.config = config
        self._parse_values = _parse_values
        self.reset()

    def reset(self):
        self.lm = None
        self.traces = []
        self.train = []
        self.demos = []

    def dump_state(self, save_verbose=False):
        state_keys = ["lm", "traces", "train"]
        state = {k: getattr(self, k) for k in state_keys}

        state["demos"] = []
        for demo in self.demos:
            demo = demo.copy()

            for field in demo:
                if isinstance(demo[field], BaseModel):
                    demo[field] = demo[field].model_dump_json()

            state["demos"].append(demo)

        # If `save_verbose` save all field metadata as well.
        if save_verbose:
            fields = []
            for field_key in self.signature.fields.keys():
                field_metadata = self.signature.fields[field_key]
                fields.append({
                    "name":  field_key,
                    "field_type": field_metadata.json_schema_extra["__dspy_field_type"],
                    "description": field_metadata.json_schema_extra["desc"],
                    "prefix": field_metadata.json_schema_extra["prefix"]
                })
            state["fields"] = fields
        
        # Cache the signature instructions and the last field's name.
        *_, last_key = self.signature.fields.keys()
        state["signature_instructions"] = self.signature.instructions
        state["signature_prefix"] = self.signature.fields[last_key].json_schema_extra["prefix"]

        # Some special stuff for CoT.
        if hasattr(self, "extended_signature"):
            # Cache the signature instructions and the last field's name.
            state["extended_signature_instructions"] = self.extended_signature.instructions
            state["extended_signature_prefix"] = self.extended_signature.fields[last_key].json_schema_extra['prefix']

        return state

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

        # Reconstruct the signature.
        if "signature_instructions" in state:
            instructions = state["signature_instructions"]
            self.signature = self.signature.with_instructions(instructions)

        if "signature_prefix" in state:
            prefix = state["signature_prefix"]
            *_, last_key = self.signature.fields.keys()
            self.signature = self.signature.with_updated_fields(last_key, prefix=prefix)
        
        # Some special stuff for CoT.
        if "extended_signature_instructions" in state:
            instructions = state["extended_signature_instructions"]
            self.extended_signature = self.extended_signature.with_instructions(instructions)

        if "extended_signature_prefix" in state:
            prefix = state["extended_signature_prefix"]
            *_, last_key = self.extended_signature.fields.keys()
            self.extended_signature = self.extended_signature.with_updated_fields(last_key, prefix=prefix)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        assert not dsp.settings.compiling, "It's no longer ever the case that .compiling is True"

        # Extract the three privileged keyword arguments.
        new_signature = ensure_signature(kwargs.pop("new_signature", None))
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))

        # Get the right LM to use.
        lm = kwargs.pop("lm", self.lm) or dsp.settings.lm
        assert lm is not None, "No LM is loaded."

        # If temperature is 0.0 but its n > 1, set temperature to 0.7.
        temperature = config.get("temperature")
        temperature = lm.kwargs["temperature"] if temperature is None else temperature
        num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1

        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7

        if new_signature is not None:
            signature = new_signature

        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            print(f"WARNING: Not all input fields were provided to module. Present: {present}. Missing: {missing}.")

        import dspy
        if isinstance(lm, dspy.LM):
            completions = v2_5_generate(lm, config, signature, demos, kwargs, _parse_values=self._parse_values)
        else:
            warn_once("\t*** In DSPy 2.5, all LM clients except `dspy.LM` are deprecated. ***\n"
                      f" \t\tYou are using the client {lm.__class__.__name__}, which will be removed in DSPy 2.6.\n"
                      " \t\tChanging the client is straightforward and will let you use new features (Adapters) that"
                      " improve the consistency of LM outputs, especially when using chat LMs. \n\n"
                      " \t\tLearn more about the changes and how to migrate at\n"
                      " \t\thttps://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb")

            if dsp.settings.experimental:
                completions = new_generate(lm, signature, dsp.Example(demos=demos, **kwargs), **config)
            else:
                completions = old_generate(demos, signature, kwargs, config, self.lm, self.stage)

        pred = Prediction.from_completions(completions, signature=signature)

        if kwargs.pop("_trace", True) and dsp.settings.trace is not None:
            trace = dsp.settings.trace
            trace.append((self, {**kwargs}, pred))

        return pred

    def update_config(self, **kwargs):
        self.config = {**self.config, **kwargs}

    def get_config(self):
        return self.config

    def __repr__(self):
        return f"{self.__class__.__name__}({self.signature})"



def old_generate(demos, signature, kwargs, config, lm, stage):
    # Switch to legacy format for dsp.generate
    x = dsp.Example(demos=demos, **kwargs)
    template = signature_to_template(signature)

    if lm is None:
        x, C = dsp.generate(template, **config)(x, stage=stage)
    else:
        # Note: query_only=True means the instructions and examples are not included.
        with dsp.settings.context(lm=lm, query_only=True):
            x, C = dsp.generate(template, **config)(x, stage=stage)

    # assert stage in x, "The generated (input, output) example was not stored"

    completions = []

    for c in C:
        completions.append({})
        for field in template.fields:
            if field.output_variable not in kwargs.keys():
                completions[-1][field.output_variable] = getattr(c, field.output_variable)

    return completions


def new_generate(lm, signature, example, max_depth=6, **kwargs):
    kwargs['stop'] = tuple(kwargs.get('stop', [])) or ('\n---', )

    # Generate and extract the fields.
    template = signature_to_template(signature, adapter=dsp.ExperimentalAdapter)
    prompt = template(example)
    completions = lm(prompt, **kwargs)
    completions = [template.extract(example, p) for p in completions]

    assert all(set(signature.input_fields).issubset(set(c.keys())) for c in completions), "Missing input keys."

    # Find the completions that are most complete.
    field_names = [field.input_variable for field in template.fields]
    for field_idx, key in enumerate(field_names):
        completions_ = [c for c in completions if key in c.keys() and c[key] is not None]
        completions = completions_ or completions
        if len(completions_) == 0: break

    # If none of the completions is completed (i.e., none has the final field set).
    if len(completions_) == 0:
        # Pick the first completion that has gone farthest.
        completion = completions[0]

        for field_idx_ in range(field_idx+1, len(field_names)):
            if field_names[field_idx_] in completion: del completion[field_names[field_idx_]]

        # Recurse with greedy decoding.
        new_kwargs = {**kwargs, "n": 1, "temperature": 0.0,}

        assert max_depth > 0
        return new_generate(lm, signature, completion, max_depth=max_depth-1, **new_kwargs)
    
    # Keep only output fields.
    completions = [{k: v for k, v in c.items() if k in signature.output_fields} for c in completions]

    return completions


def v2_5_generate(lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
    import dspy
    adapter = dspy.settings.adapter or dspy.ChatAdapter()

    return adapter(lm, lm_kwargs=lm_kwargs, signature=signature, demos=demos, inputs=inputs, _parse_values=_parse_values)
    


# TODO: get some defaults during init from the context window?
# # TODO: FIXME: Hmm, I guess expected behavior is that contexts can
# affect execution. Well, we need to determine whether context dominates, __init__ demoninates, or forward dominates.
# Generally, unless overwritten, we'd see n=None, temperature=None.
# That will eventually mean we have to learn them.