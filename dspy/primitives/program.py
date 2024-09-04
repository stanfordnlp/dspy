import magicattr
import os

from dspy.primitives.assertions import *
from dspy.primitives.module import BaseModule
from dspy.teleprompt.teleprompt import Teleprompter
from pydantic import BaseModel, RootModel
from typing import List, Tuple, Set, Optional, TypeVar, Type, Union, Literal
import inspect
class ProgramMeta(type):
    def __new__(cls, name, bases, class_dict):
        for attr, value in class_dict.items():
            if attr == "forward" and callable(value):
                original_method= value
                class_dict[attr] = forward_wrapper(original_method,cls,name)
        return type.__new__(cls, name, bases, class_dict)
    # def __call__(cls, *args, **kwargs):
    #     obj = super(ProgramMeta, cls).__call__(*args, **kwargs)

    #     if issubclass(cls, Program) and not getattr(obj, "_program_init_called", False):
    #         obj._base_init()
    #         obj._program_init_called = True
    #     return obj

class PredictorDebugInfo(BaseModel):
    demos : List[dict]
    signature : dict
    extended_signature : Optional[dict] = None 
    type : Literal["PredictorDebugInfo"] = "PredictorDebugInfo"
    unique_id : int

class RetrieveDebugInfo(BaseModel):
    k : int 
    type : Literal["RetrieveDebugInfo"] = "RetrieveDebugInfo"
    unique_id : int

class ModuleDebugInfo(BaseModel):
    unique_id : int
    name : str
    class_name : str
    path : str
    line_num : int
    parameters: List[Tuple[str, Union[PredictorDebugInfo,RetrieveDebugInfo] ]]
    invoked_modules : List[tuple[str, int]]

class ModelDebugInfoGraph(RootModel):
    root : List[ModuleDebugInfo]

class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False

    def __init__(self):
        self._compiled = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_predictors(self):
        from dspy.predict.predict import Predict

        return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]

    def predictors(self):
        return [param for _, param in self.named_predictors()]

    def __repr__(self):
        s = []

        for name, param in self.named_predictors():
            s.append(f"{name} = {param}")

        return "\n".join(s)

    def map_named_predictors(self, func):
        """Applies a function to all named predictors."""
        for name, predictor in self.named_predictors():
            set_attribute_by_name(self, name, func(predictor))
        return self

    def activate_assertions(self, handler=backtrack_handler, **handler_args):
        """
        Activates assertions for the module.
        The default handler is the backtrack_handler.
        """
        assert_transform_module(self, handler, **handler_args)
        return self
    
    def debug_info(module : BaseModule) -> str:
        from dspy.predict.predict import Predict
        from dspy.retrieve.retrieve import Retrieve, RetrieveThenRerank
        from collections import deque
        from collections.abc import Generator
        from dspy.predict.parameter import Parameter
        import itertools
        T = TypeVar('T')
        def named_direct_subobjs(obj, type_ : Type[T]) -> Generator[tuple[str, T], None, None]:
            # this function is very similar to the named_sub_modules 
            # but is only at the base level and will not recursively go find
            # inside another attribute
            queue = deque([])
            seen = set()
            def add_to_queue(name, item):
                if id(item) not in seen:
                    seen.add(id(item))
                    queue.append((name, item))
            for name, item in obj.__dict__.items():
                add_to_queue(f"{name}", item)

            while queue:
                name, item = queue.popleft()

                if isinstance(item, type_):
                    yield name, item

                elif isinstance(item, (list, tuple)):
                    for i, sub_item in enumerate(item):
                        add_to_queue(f"{name}[{i}]", sub_item)

                elif isinstance(item, dict):
                    for key, sub_item in item.items():
                        add_to_queue(f"{name}[{key}]", sub_item)

        ls = []
        def debug_info_inner(module : BaseModule, module_sets: Set[int], name: str):
            unique_id = id(module)
            class_name = type(module).__name__
            path =  os.path.abspath(inspect.getfile(module.__class__))
            line = inspect.findsource(module.__class__)[1]
            module_sets.add(unique_id)
            sub_modules = list(named_direct_subobjs(module, BaseModule))
            non_predict_modules = filter(lambda mod: not isinstance(mod[1], Parameter), sub_modules)
            submodule_info : List[tuple[str, int]]  = []
            for sub_module_name, sub_module in non_predict_modules:
                if id(sub_modules) in module_sets:
                    continue
                submodule_info.append((sub_module_name, id(sub_module)))
                debug_info_inner(sub_module, module_sets, sub_module_name)
            parameters = list(named_direct_subobjs(module, Parameter))
            parameters_infos: List[Tuple[str, Union[PredictorDebugInfo,RetrieveDebugInfo]]] = []
            if isinstance(module, Parameter):
                parameters = itertools.chain([("self as predictor", module)], parameters)
            for param_name, parameter in parameters:
                unique_param_id = id(parameter)*10+1
                if isinstance(parameter, Predict):
                    demos = list(map(lambda demo : demo.toDict(), parameter.demos))
                    signature = parameter.signature.model_json_schema()
                    extended_signature =  parameter.extended_signature.model_json_schema() if hasattr(parameter, "extended_signature") else None
                    info = PredictorDebugInfo(demos = demos, signature=signature, extended_signature=extended_signature, unique_id=unique_param_id)
                elif isinstance(parameter,  Retrieve) or isinstance(parameter, RetrieveThenRerank):
                    k = parameter.k
                    info = RetrieveDebugInfo(k=k, unique_id=unique_param_id)
                if info:
                    parameters_infos.append((param_name, info))
            ls.append(ModuleDebugInfo(unique_id=unique_id, name=name, class_name=class_name, 
                                    path = path, line_num=line, parameters= parameters_infos,
                                    invoked_modules = submodule_info))
        debug_info_inner(module, set(),"current module")
        return ModelDebugInfoGraph(ls).model_dump_json()
    # def __deepcopy__(self, memo):
    #     # memo is a dict of id's to copies already made during the current call
    #     # Check if the object is already copied
    #     if id(self) in memo:
    #         return memo[id(self)]

    #     print(f"Deep copying {self.__class__.__name__}...")

    #     new_copy = copy.copy(self)
    #     memo[id(self)] = new_copy

    #     for k, v in self.__dict__.items():
    #         print(f"Copying attribute {k} of type {type(v)}...")
    #         setattr(new_copy, k, copy.deepcopy(v, memo))
    #         print("Done")

    #     return new_copy
    
def forward_wrapper(method, cls, name):
    def inner_wrapper(self,*args, **kwargs):
        stack = inspect.stack()
        current_id = id(stack[0].frame)
        for frame_info in stack:
            caller_locals = frame_info.frame.f_locals
            if 'self' not in caller_locals:
                continue
            caller = caller_locals['self']
            if isinstance(caller, Module) or isinstance(caller, Teleprompter):
                parent_frame_id = id(frame_info.frame)
                dsp.settings.debug_trace.append({
                    'class_name' : name,
                    'object_id' : id(self),
                    'frame_id' : current_id,
                    'parent_frame_id': parent_frame_id,
                    'args' : args,
                    'kwargs' : kwargs                
                })
        return method(self, *args, **kwargs)
    return inner_wrapper


def set_attribute_by_name(obj, name, value):
    magicattr.set(obj, name, value)

Program = Module