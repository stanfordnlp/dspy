import copy

from dspy.primitives.module import BaseModule


class ProgramMeta(type):
    pass
    # def __call__(cls, *args, **kwargs):
    #     obj = super(ProgramMeta, cls).__call__(*args, **kwargs)

    #     if issubclass(cls, Program) and not getattr(obj, "_program_init_called", False):
    #         obj._base_init()
    #         obj._program_init_called = True
    #     return obj


class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False

    def __init__(self):
        self._compiled = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_predictors(self):
        from dspy.predict.predict import Predict

        named_parameters = self.named_parameters()
        return [
            (name, param)
            for name, param in named_parameters
            if isinstance(param, Predict)
        ]

    def predictors(self):
        return [param for _, param in self.named_predictors()]

    def __repr__(self):
        s = []

        for name, param in self.named_predictors():
            s.append(f"{name} = {param}")

        return "\n".join(s)


class AsyncModule(Module):
    async def __call__(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)


Program = Module
