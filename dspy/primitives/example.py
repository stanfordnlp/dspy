from pydantic import BaseModel


class Example:
    """Hold a single data row with named fields, like one row of a dataset.

    If you've used HuggingFace datasets or pandas DataFrames, think of an
    `Example` as one row: each field is a column value you access by name.
    A list of `Example` objects is how you pass a trainset or devset to DSPy
    optimizers and evaluators.

    Create one with keyword arguments, a dictionary, or another `Example`.
    Call `with_inputs` to declare which fields are inputs and which are labels.
    DSPy modules and optimizers rely on that split: they call
    `example.inputs()` to feed your program and `example.labels()` to check
    its output.

    Examples:
        Keyword arguments:

        >>> import dspy
        >>> example = dspy.Example(
        ...     question="What is the capital of France?",
        ...     answer="Paris",
        ... )
        >>> example.question
        'What is the capital of France?'
        >>> example.answer
        'Paris'

        From a dictionary:

        >>> data = {"question": "What is 2+2?", "answer": "4"}
        >>> example = dspy.Example(data)
        >>> example["question"]
        'What is 2+2?'

        From another `Example`:

        >>> original = dspy.Example(question="Hello", answer="World")
        >>> copy = dspy.Example(original)
        >>> copy.question
        'Hello'

        Input/output separation:

        >>> example = dspy.Example(
        ...     question="What is the weather?",
        ...     answer="It's sunny",
        ... ).with_inputs("question")
        >>> example.inputs().question
        'What is the weather?'
        >>> example.labels().answer
        "It's sunny"

        Dictionary-like operations:

        >>> example = dspy.Example(name="Alice", age=30)
        >>> "name" in example
        True
        >>> example.get("city", "Unknown")
        'Unknown'

    See Also:
        [`dspy.Prediction`][dspy.Prediction]: An `Example` subclass returned
            by DSPy modules, with completions tracking.
    """

    def __init__(self, base=None, **kwargs):
        """Create an `Example` from keyword arguments, a dict, or another `Example`.

        Args:
            base: A dictionary or `Example` to copy fields from.
                When `None`, starts empty.
            **kwargs: Field names and values. These are merged on top of
                anything in `base`.
        """
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
        """Return field names, like `dict.keys()`.

        Args:
            include_dspy: If `True`, include internal fields prefixed
                with `dspy_`. Normally you can ignore these.

        Examples:
            >>> import dspy
            >>> dspy.Example(question="Why?", answer="Because.").keys()
            ['question', 'answer']
        """
        return [k for k in self._store.keys() if not k.startswith("dspy_") or include_dspy]

    def values(self, include_dspy=False):
        """Return field values, like `dict.values()`.

        Args:
            include_dspy: If `True`, include internal fields prefixed
                with `dspy_`.

        Examples:
            >>> import dspy
            >>> dspy.Example(question="Why?", answer="Because.").values()
            ['Why?', 'Because.']
        """
        return [v for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

    def items(self, include_dspy=False):
        """Return `(field_name, value)` pairs, like `dict.items()`.

        Args:
            include_dspy: If `True`, include internal fields prefixed
                with `dspy_`.

        Examples:
            >>> import dspy
            >>> dspy.Example(question="Why?", answer="Because.").items()
            [('question', 'Why?'), ('answer', 'Because.')]
        """
        return [(k, v) for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

    def get(self, key, default=None):
        """Return the value for `key`, or `default` if the field doesn't exist.

        Args:
            key: Field name to look up.
            default: Value to return when `key` is missing.

        Examples:
            >>> import dspy
            >>> ex = dspy.Example(name="Alice")
            >>> ex.get("name")
            'Alice'
            >>> ex.get("city", "Unknown")
            'Unknown'
        """
        return self._store.get(key, default)

    def with_inputs(self, *keys):
        """Mark which fields are inputs and return a new `Example`.

        Fields not listed here are treated as labels (expected outputs).
        DSPy optimizers and evaluators use this split: they pass
        `example.inputs()` to your program and compare the output against
        `example.labels()`.

        Args:
            *keys: Names of the input fields.

        Returns:
            A copy of this `Example` with the input keys set.

        Examples:
            >>> import dspy
            >>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
            >>> ex.inputs().keys()
            ['question']
            >>> ex.labels().keys()
            ['answer']
        """
        copied = self.copy()
        copied._input_keys = set(keys)
        return copied

    def inputs(self):
        """Return a new `Example` containing only the input fields.

        Requires `with_inputs` to have been called first.

        Raises:
            ValueError: If `with_inputs` was not called on this example.

        Examples:
            >>> import dspy
            >>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
            >>> ex.inputs()
            Example({'question': 'Why?'}) (input_keys={'question'})
        """
        if self._input_keys is None:
            raise ValueError("Inputs have not been set for this example. Use `example.with_inputs()` to set them.")

        # return items that are in input_keys
        d = {key: self._store[key] for key in self._store if key in self._input_keys}
        # return type(self)(d)
        new_instance = type(self)(base=d)
        new_instance._input_keys = self._input_keys  # Preserve input_keys in new instance
        return new_instance

    def labels(self):
        """Return a new `Example` containing only the label (non-input) fields.

        Requires `with_inputs` to have been called first, since labels are
        everything that is *not* an input.

        Examples:
            >>> import dspy
            >>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
            >>> ex.labels()
            Example({'answer': 'Because.'}) (input_keys=None)
        """
        # return items that are NOT in input_keys
        input_keys = self.inputs().keys()
        d = {key: self._store[key] for key in self._store if key not in input_keys}
        return type(self)(d)

    def __iter__(self):
        return iter(dict(self._store))

    def copy(self, **kwargs):
        """Return a shallow copy, optionally overriding fields.

        Args:
            **kwargs: Fields to add or override in the copy.

        Examples:
            >>> import dspy
            >>> ex = dspy.Example(question="Why?", answer="Because.")
            >>> ex.copy(answer="No reason.")
            Example({'question': 'Why?', 'answer': 'No reason.'}) (input_keys=None)
        """
        return type(self)(base=self, **kwargs)

    def without(self, *keys):
        """Return a copy with the specified fields removed.

        Args:
            *keys: Field names to drop.

        Examples:
            >>> import dspy
            >>> ex = dspy.Example(question="Why?", answer="Because.", source="web")
            >>> ex.without("source")
            Example({'question': 'Why?', 'answer': 'Because.'}) (input_keys=None)
        """
        copied = self.copy()
        for key in keys:
            del copied[key]
        return copied

    def toDict(self):  # noqa: N802
        """Convert to a plain dictionary, recursively serializing nested objects.

        Nested `Example` objects, Pydantic models, lists, and dicts are
        converted so the result is JSON-friendly.

        Examples:
            >>> import dspy
            >>> dspy.Example(question="Why?", answer="Because.").toDict()
            {'question': 'Why?', 'answer': 'Because.'}
        """
        def convert_to_serializable(value):
            if hasattr(value, "toDict"):
                return value.toDict()
            elif isinstance(value, BaseModel):
                # Handle Pydantic models (e.g., dspy.History)
                return value.model_dump()
            elif isinstance(value, list):
                return [convert_to_serializable(item) for item in value]
            elif isinstance(value, dict):
                return {k: convert_to_serializable(v) for k, v in value.items()}
            else:
                return value

        serializable_store = {}
        for k, v in self._store.items():
            serializable_store[k] = convert_to_serializable(v)

        return serializable_store
