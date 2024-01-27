import copy
import ujson


class BaseModule:
    def __init__(self):
        self._compiled = False

    def named_parameters(self, only_uncompiled=False):
        """
            Unlike PyTorch, handles (non-recursive) lists of parameters too.
        """

        from dspy.predict.parameter import Parameter

        visited = set()
        named_parameters = []

        def add_parameter(param_name, param_value):
            if isinstance(param_value, Parameter) and id(param_value) not in visited:
                # Exclude compiled modules if only_uncompiled is True.
                if (not only_uncompiled or not getattr(param_value, '_compiled', False)):
                        visited.add(id(param_value))
                        named_parameters.append((param_name, param_value))


        for name, value in self.__dict__.items():
            #NOTE(Karel): Move '_compiled' check here so we can toggle entire Modules to be '_compiled'.
            if isinstance(value, Parameter):
                add_parameter(name, value)

            elif isinstance(value, BaseModule):
                for sub_name, param in value.named_parameters(only_uncompiled=only_uncompiled):
                    add_parameter(f"{name}.{sub_name}", param)
            
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    add_parameter(f"{name}[{idx}]", item)

            elif isinstance(value, dict):
                for key, item in value.items():
                    add_parameter(f"{name}['{key}']", item)

        return named_parameters

    def parameters(self):
        return [param for _, param in self.named_parameters()]

    def deepcopy(self):
        return copy.deepcopy(self)

    def reset_copy(self, only_reset_uncompiled=False):
        obj = copy.deepcopy(self)
        
        for param in obj.parameters():
            if only_reset_uncompiled:
                if not hasattr(param, '_compiled') or not param._compiled:
                    param.reset()
            else:
                param.reset()
        
        return obj
    
    def dump_state(self):
        return {name: param.dump_state() for name, param in self.named_parameters()}
    
    def load_state(self, state):
        for name, param in self.named_parameters():
            param.load_state(state[name])
    
    def save(self, path):
        with open(path, "w") as f:
            f.write(ujson.dumps(self.dump_state(), indent=2))
    
    def load(self, path):
        with open(path, "r") as f:
            self.load_state(ujson.loads(f.read()))
