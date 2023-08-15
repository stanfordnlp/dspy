import dsp
import random

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction


class Predict(Parameter):
    def __init__(self, signature, **config):
        self.stage = random.randbytes(8).hex()
        self.signature = signature #.signature
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
    
    def __call__(self, **kwargs):
        return self.forward(**kwargs)
    
    def forward(self, **kwargs):
        # Extract the three privileged keyword arguments.
        signature = kwargs.pop("signature", self.signature)
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))

        # If temperature is 0.0 but its n > 1, set temperature to 0.7.
        temperature = config.get("temperature", None)
        temperature = dsp.settings.lm.kwargs['temperature'] if temperature is None else temperature

        num_generations = config.get("n", None)
        num_generations = dsp.settings.lm.kwargs['n'] if num_generations is None else num_generations

        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7
            # print(f"#> Setting temperature to 0.7 since n={num_generations} and prior temperature={temperature}.")

        # All of the other kwargs are presumed to fit a prefix of the signature.

        x = dsp.Example(demos=demos, **kwargs)
        x, C = dsp.generate(signature, **config)(x, stage=self.stage)

        completions = []

        for c in C:
            completions.append({})
            for field in signature.fields:
                if field.output_variable not in kwargs.keys():
                    completions[-1][field.output_variable] = getattr(c, field.output_variable)

        pred = Prediction.from_completions(completions, signature=signature)
            
        if dsp.settings.trace is not None:
            trace = dsp.settings.trace
            trace.append((self, {**kwargs}, pred))

        return pred

    def __repr__(self):
        return f"{self.__class__.__name__}({self.signature})"



# TODO: get some defaults during init from the context window?
# # TODO: FIXME: Hmm, I guess expected behavior is that contexts can
# affect exeuction. Well, we need to determine whether context dominates, __init__ demoninates, or forward dominates.
# Generally, unless overwritten, we'd see n=None, temperature=None.
# That will eventually mean we have to learn them.
