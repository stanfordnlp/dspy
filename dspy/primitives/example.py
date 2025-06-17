class Example:
    def __init__(self, base=None, **kwargs):
        # Internal storage and other attributes
        self._store = {}
        self._demos = []
        self._input_keys = None

        # Initialize from a base Example if provided
        if base and isinstance(base, type(self)):
            self._store = base._store.copy()

        # Initialize from a dict if provided
        elif base and isinstance(base, dict):
            self._store = base.copy()

        # Update with provided kwargs
        self._store.update(kwargs)

    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError
        if key in self._store:
            return self._store[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key.startswith("_") or key in dir(self.__class__):
            super().__setattr__(key, value)
        else:
            self._store[key] = value

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __contains__(self, key):
        return key in self._store

    def __len__(self):
        return len([k for k in self._store if not k.startswith("dspy_")])

    def __repr__(self):
        # return f"Example({self._store})" + f" (input_keys={self._input_keys}, demos={self._demos})"
        d = {k: v for k, v in self._store.items() if not k.startswith("dspy_")}
        return f"Example({d})" + f" (input_keys={self._input_keys})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, Example) and self._store == other._store

    def __hash__(self):
        return hash(tuple(self._store.items()))

    def keys(self, include_dspy=False):
        return [k for k in self._store.keys() if not k.startswith("dspy_") or include_dspy]

    def values(self, include_dspy=False):
        return [v for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

    def items(self, include_dspy=False):
        return [(k, v) for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

    def get(self, key, default=None):
        return self._store.get(key, default)

    def with_inputs(self, *keys):
        copied = self.copy()
        copied._input_keys = set(keys)
        return copied

    def inputs(self):
        if self._input_keys is None:
            raise ValueError("Inputs have not been set for this example. Use `example.with_inputs()` to set them.")

        # return items that are in input_keys
        d = {key: self._store[key] for key in self._store if key in self._input_keys}
        # return type(self)(d)
        new_instance = type(self)(base=d)
        new_instance._input_keys = self._input_keys  # Preserve input_keys in new instance
        return new_instance

    def labels(self):
        # return items that are NOT in input_keys
        input_keys = self.inputs().keys()
        d = {key: self._store[key] for key in self._store if key not in input_keys}
        return type(self)(d)

    def __iter__(self):
        return iter(dict(self._store))

    def copy(self, **kwargs):
        return type(self)(base=self, **kwargs)

    def without(self, *keys):
        copied = self.copy()
        for key in keys:
            del copied[key]
        return copied

    def toDict(self):  # noqa: N802
        return self._store.copy()
