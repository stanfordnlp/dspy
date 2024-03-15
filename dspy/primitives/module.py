import copy
from collections import deque
from collections.abc import Generator

import ujson


class BaseModule:
    def __init__(self):
        pass

    def named_parameters(self):
        """Unlike PyTorch, handles lists of parameters too."""
        from dspy.predict.parameter import Parameter

        # Remove the 'self.' prefix from the names
        return [(name[5:], param) for name, param in self.named_sub_modules(Parameter)]

    def named_sub_modules(self, type_=None, skip_compiled=False) -> Generator[tuple[str, "BaseModule"], None, None]:
        """Find all sub-modules in the module, as well as their names.

        Say self.children[4]['key'].sub_module is a sub-module. Then the name will be
        'children[4][key].sub_module'. But if the sub-module is accessible at different
        paths, only one of the paths will be returned.
        """
        if type_ is None:
            type_ = BaseModule

        queue = deque([("self", self)])
        seen = {id(self)}

        def add_to_queue(name, item):
            if id(item) not in seen:
                seen.add(id(item))
                queue.append((name, item))

        while queue:
            name, item = queue.popleft()
            if isinstance(item, type_):
                yield name, item

            if isinstance(item, BaseModule):
                if skip_compiled and getattr(item, "_compiled", False):
                    continue
                for sub_name, sub_item in item.__dict__.items():
                    add_to_queue(f"{name}.{sub_name}", sub_item)

            elif isinstance(item, (list, tuple)):
                for i, sub_item in enumerate(item):
                    add_to_queue(f"{name}[{i}]", sub_item)

            elif isinstance(item, dict):
                for key, sub_item in item.items():
                    add_to_queue(f"{name}[{key}]", sub_item)

    def parameters(self):
        return [param for _, param in self.named_parameters()]

    def deepcopy(self):
        return copy.deepcopy(self)

    def reset_copy(self):
        obj = copy.deepcopy(self)

        for param in obj.parameters():
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
        with open(path) as f:
            self.load_state(ujson.loads(f.read()))
