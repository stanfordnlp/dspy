from importlib import import_module

from .logging import get_logger


logger = get_logger(__name__)


class _PatchedModuleObj:
    """Set all the modules components as attributes of the _PatchedModuleObj object."""

    def __init__(self, module, attrs=None):
        attrs = attrs or []
        if module is not None:
            for key in module.__dict__:
                if key in attrs or not key.startswith("__"):
                    setattr(self, key, getattr(module, key))
        self._original_module = module._original_module if isinstance(module, _PatchedModuleObj) else module


class patch_submodule:
    """
    Patch a submodule attribute of an object, by keeping all other submodules intact at all levels.

    Example::

        >>> import importlib
        >>> from datasets.load import dataset_module_factory
        >>> from datasets.streaming import patch_submodule, xjoin
        >>>
        >>> dataset_module = dataset_module_factory("snli")
        >>> snli_module = importlib.import_module(dataset_module.module_path)
        >>> patcher = patch_submodule(snli_module, "os.path.join", xjoin)
        >>> patcher.start()
        >>> assert snli_module.os.path.join is xjoin
    """

    _active_patches = []

    def __init__(self, obj, target: str, new, attrs=None):
        self.obj = obj
        self.target = target
        self.new = new
        self.key = target.split(".")[0]
        self.original = {}
        self.attrs = attrs or []

    def __enter__(self):
        *submodules, target_attr = self.target.split(".")

        # Patch modules:
        # it's used to patch attributes of submodules like "os.path.join";
        # in this case we need to patch "os" and "os.path"

        for i in range(len(submodules)):
            try:
                submodule = import_module(".".join(submodules[: i + 1]))
            except ModuleNotFoundError:
                continue
            # We iterate over all the globals in self.obj in case we find "os" or "os.path"
            for attr in self.obj.__dir__():
                obj_attr = getattr(self.obj, attr)
                # We don't check for the name of the global, but rather if its value *is* "os" or "os.path".
                # This allows to patch renamed modules like "from os import path as ospath".
                if obj_attr is submodule or (
                    isinstance(obj_attr, _PatchedModuleObj) and obj_attr._original_module is submodule
                ):
                    self.original[attr] = obj_attr
                    # patch at top level
                    setattr(self.obj, attr, _PatchedModuleObj(obj_attr, attrs=self.attrs))
                    patched = getattr(self.obj, attr)
                    # construct lower levels patches
                    for key in submodules[i + 1 :]:
                        setattr(patched, key, _PatchedModuleObj(getattr(patched, key, None), attrs=self.attrs))
                        patched = getattr(patched, key)
                    # finally set the target attribute
                    setattr(patched, target_attr, self.new)

        # Patch attribute itself:
        # it's used for builtins like "open",
        # and also to patch "os.path.join" we may also need to patch "join"
        # itself if it was imported as "from os.path import join".

        if submodules:  # if it's an attribute of a submodule like "os.path.join"
            try:
                attr_value = getattr(import_module(".".join(submodules)), target_attr)
            except (AttributeError, ModuleNotFoundError):
                return
            # We iterate over all the globals in self.obj in case we find "os.path.join"
            for attr in self.obj.__dir__():
                # We don't check for the name of the global, but rather if its value *is* "os.path.join".
                # This allows to patch renamed attributes like "from os.path import join as pjoin".
                if getattr(self.obj, attr) is attr_value:
                    self.original[attr] = getattr(self.obj, attr)
                    setattr(self.obj, attr, self.new)
        elif target_attr in globals()["__builtins__"]:  # if it'a s builtin like "open"
            self.original[target_attr] = globals()["__builtins__"][target_attr]
            setattr(self.obj, target_attr, self.new)
        else:
            raise RuntimeError(f"Tried to patch attribute {target_attr} instead of a submodule.")

    def __exit__(self, *exc_info):
        for attr in list(self.original):
            setattr(self.obj, attr, self.original.pop(attr))

    def start(self):
        """Activate a patch."""
        self.__enter__()
        self._active_patches.append(self)

    def stop(self):
        """Stop an active patch."""
        try:
            self._active_patches.remove(self)
        except ValueError:
            # If the patch hasn't been started this will fail
            return None

        return self.__exit__()
