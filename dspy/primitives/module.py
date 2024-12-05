import copy
import logging
from collections import deque
from collections.abc import Generator
from pathlib import Path

import cloudpickle
import ujson

from dspy.utils.saving import get_dependency_versions

# NOTE: Note: It's important (temporary decision) to maintain named_parameters that's different in behavior from
# named_sub_modules for the time being.


logger = logging.getLogger(__name__)


class BaseModule:
    def __init__(self):
        pass

    def named_parameters(self):
        """
        Unlike PyTorch, handles (non-recursive) lists of parameters too.
        """

        import dspy
        from dspy.predict.parameter import Parameter

        visited = set()
        named_parameters = []

        def add_parameter(param_name, param_value):
            if isinstance(param_value, Parameter):
                if id(param_value) not in visited:
                    visited.add(id(param_value))
                    param_name = postprocess_parameter_name(param_name, param_value)
                    named_parameters.append((param_name, param_value))

            elif isinstance(param_value, dspy.Module):
                # When a sub-module is pre-compiled, keep it frozen.
                if not getattr(param_value, "_compiled", False):
                    for sub_name, param in param_value.named_parameters():
                        add_parameter(f"{param_name}.{sub_name}", param)

        if isinstance(self, Parameter):
            add_parameter("self", self)

        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                add_parameter(name, value)

            elif isinstance(value, dspy.Module):
                # When a sub-module is pre-compiled, keep it frozen.
                if not getattr(value, "_compiled", False):
                    for sub_name, param in value.named_parameters():
                        add_parameter(f"{name}.{sub_name}", param)

            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    add_parameter(f"{name}[{idx}]", item)

            elif isinstance(value, dict):
                for key, item in value.items():
                    add_parameter(f"{name}['{key}']", item)

        return named_parameters

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
            name = postprocess_parameter_name(name, item)

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
        """Deep copy the module.

        This is a tweak to the default python deepcopy that only deep copies `self.parameters()`, and for other
        attributes, we just do the shallow copy.
        """
        try:
            # If the instance itself is copyable, we can just deep copy it.
            # Otherwise we will have to create a new instance and copy over the attributes one by one.
            return copy.deepcopy(self)
        except Exception:
            pass

        # Create an empty instance.
        new_instance = self.__class__.__new__(self.__class__)
        # Set attribuetes of the copied instance.
        for attr, value in self.__dict__.items():
            if isinstance(value, BaseModule):
                setattr(new_instance, attr, value.deepcopy())
            else:
                try:
                    # Try to deep copy the attribute
                    setattr(new_instance, attr, copy.deepcopy(value))
                except Exception:
                    logging.warning(
                        f"Failed to deep copy attribute '{attr}' of {self.__class__.__name__}, "
                        "falling back to shallow copy or reference copy."
                    )
                    try:
                        # Fallback to shallow copy if deep copy fails
                        setattr(new_instance, attr, copy.copy(value))
                    except Exception:
                        # If even the shallow copy fails, we just copy over the reference.
                        setattr(new_instance, attr, value)

        return new_instance

    def reset_copy(self):
        """Deep copy the module and reset all parameters."""
        new_instance = self.deepcopy()

        for param in new_instance.parameters():
            param.reset()

        return new_instance

    def dump_state(self, save_verbose):
        return {name: param.dump_state(save_verbose) for name, param in self.named_parameters()}

    def load_state(self, state, use_legacy_loading=False):
        for name, param in self.named_parameters():
            if isinstance(param, BaseModule):
                param.load_state(state[name], use_legacy_loading=use_legacy_loading)
            else:
                # `use_legacy_loading` is only applicable for BaseModule instances.
                param.load_state(state[name])

    def save(self, path, save_field_meta=False, state_only=False, metadata=None, use_json=True):
        """Save the module.

        Save the module to a directory or a file. There are two modes:
        - `state_only=True`: Save only the state of the module to a json or pickle file, based on the value of
            `use_json`.
        - `state_only=False`: Save the whole module to a directory via cloudpickle, which contains both the state and
            architecture of the model.

        We also save the dependency versions, so that the loaded model can check if there is a version mismatch on
        critical dependencies or DSPy version.

        Args:
            path (str): Path to the saved state file, which should be a .json or .pkl file when `state_only=True`, and a
                directory when `state_only=False`.
            save_field_meta (bool): Whether to save the field metadata. Only applicable when `state_only=False`.
            state_only (bool): Whether to save only the state of the module.
            metadata (dict): Extra metadata to save.
            use_json (bool): Whether to save the state to a json file. If False, the state is saved to a pickle file.
                Only applicable when `state_only=True`.
        """
        metadata = metadata or {}
        metadata["dependency_versions"] = get_dependency_versions()

        if state_only:
            state = self.dump_state(save_field_meta)
            state["metadata"] = metadata
            if use_json:
                if path.suffix != ".json":
                    raise ValueError(
                        f"`path` must be a json file when `state_only=True` and `use_json=True`, but received: {path}"
                    )
                with open(path, "w") as f:
                    f.write(ujson.dumps(state, indent=2))
            else:
                if path.suffix != ".pkl":
                    raise ValueError(
                        f"`path` must be a pkl file when `state_only=True` and `use_json=False`, but received: {path}"
                    )
                with open(path, "wb") as f:
                    cloudpickle.dump(state, f)
        else:
            if path.suffix:
                raise ValueError(
                    f"`path` must point to a directory without a suffix when `state_only=False`, but received: {path}"
                )
            if path.exists() and not path.is_dir():
                raise NotADirectoryError(f"The path '{path}' exists but is not a directory.")

            if not path.exists():
                # Create the directory (and any parent directories)
                path.mkdir(parents=True)

            try:
                with open(path / "model.pkl", "wb") as f:
                    cloudpickle.dump(self, f)
            except Exception as e:
                raise RuntimeError(
                    f"Saving failed with error: {e}. Please remove the non-picklable attributes from your DSPy model, "
                    "or consider using state-only saving by setting `state_only=True`."
                )

            with open(path / "metadata.json", "w") as f:
                ujson.dump(metadata, f, indent=2)

    def load(self, path, use_legacy_loading=False, use_json=True):
        """Load the saved module.

        Args:
            path (str): Path to the saved state file, which should be a .json file when `use_json=True`, and a .pkl file
                when `use_json=False`.
            use_legacy_loading (bool): Whether to use the legacy loading method. Only use it when you are loading a
                saved state from a version of DSPy prior to v2.5.3.
            use_json (bool): Whether to load the state from a json file. If False, the state is loaded from a pickle
                file.
        """
        path = Path(path)
        if use_json and path.suffix != ".json":
            raise ValueError(f"`path` must be a json file when `use_json=True`, but received: {path}")
        if not use_json and path.suffix != ".pkl":
            raise ValueError(f"`path` must be a pkl file when `use_json=False`, but received: {path}")
        with open(path, "rb") as f:
            if use_json:
                state = ujson.loads(f.read())
            else:
                state = cloudpickle.load(f)

        dependency_versions = get_dependency_versions()
        saved_dependency_versions = state["metadata"]["dependency_versions"]
        for key, saved_version in saved_dependency_versions.items():
            if dependency_versions[key] != saved_version:
                logger.warning(
                    f"There is a mismatch of {key} version between saved model and current environment. "
                    f"You saved with `{key}=={saved_version}`, but now you have "
                    f"`{key}=={dependency_versions[key]}`. This might cause errors or performance downgrade "
                    "on the loaded model, please consider loading the model in the same environment as the "
                    "saving environment."
                )
        self.load_state(state, use_legacy_loading=use_legacy_loading)


def postprocess_parameter_name(name, value):
    # For ChainOfThought backward compatibility, remove ending ._predict if it's there
    if name.endswith("._predict"):
        name = name[:-9]

    if name.endswith(".self"):
        name = name[:-5]

    if name == "_predict":
        return "self"

    return name
