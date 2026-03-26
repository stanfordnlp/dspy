from pydantic import BaseModel


class Example:
    """A flexible data container for DSPy examples and training data with named fields.

    An `Example` is roughly one row from a HuggingFace dataset or pandas
    `DataFrame`. It behaves a lot like a dictionary or dot-access record: you
    can read fields with `example["question"]` or `example.question`.

    In DSPy, lists of `Example` objects are your trainset, devset, and testset.
    Most examples are built from keyword arguments or an existing record, then
    tagged with `with_inputs(...)` to say which fields should be fed into a
    module. The remaining fields are labels or metadata.

    When you write evaluation code, custom optimizers, or training loops, use
    `example.inputs()` for the fields you want to pass to a module, and use
    `example.labels()` for the fields you want to compare against the module's
    output.

    Examples:
        Build one from keyword arguments:

        >>> import dspy
        >>> example = dspy.Example(
        ...     question="What is the capital of France?",
        ...     answer="Paris",
        ... ).with_inputs("question")
        >>> example.question
        'What is the capital of France?'
        >>> example.answer
        'Paris'
        >>> example.inputs().toDict()
        {'question': 'What is the capital of France?'}

        Build one from an existing record:

        >>> record = {"question": "What is 2+2?", "answer": "4"}
        >>> example = dspy.Example(**record).with_inputs("question")
        >>> example["question"]
        'What is 2+2?'
        >>> example.labels().answer
        '4'

        Mark which fields are inputs:

        >>> example = dspy.Example(
        ...     question="What is the weather?",
        ...     answer="It's sunny",
        ... ).with_inputs("question")
        >>> example.inputs().question
        'What is the weather?'
        >>> example.labels().answer
        "It's sunny"

        Use examples in a trainset:

        >>> trainset = [
        ...     dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        ...     dspy.Example(question="What is 3+3?", answer="6").with_inputs("question"),
        ... ]
        >>> trainset[0].inputs().toDict()
        {'question': 'What is 2+2?'}

        Use an example in a metric:

        >>> def exact_match_metric(example, pred, trace=None):
        ...     return example.answer.lower() == pred.answer.lower()
        >>> gold = dspy.Example(question="What is 1+1?", answer="2").with_inputs("question")
        >>> pred = dspy.Prediction(answer="2")
        >>> exact_match_metric(gold, pred)
        True

        Use it like a dictionary:

        >>> example = dspy.Example(name="Alice", age=30).with_inputs("name")
        >>> "name" in example
        True
        >>> example.get("city", "Unknown")
        'Unknown'

    See Also:
        [`dspy.Evaluate`][dspy.Evaluate]: Evaluate a program on a list of
            `Example`s.
        [`Metrics`](../../learn/evaluation/metrics.md): Write metric functions
            that compare an `Example` with a prediction.
    """
    # TODO: Add a `dspy.datasets.DataLoader` API link once the datasets API
    # pages exist.

    def __init__(self, base=None, **kwargs):
        """Create an `Example` from fields or from an existing record.

        In the common case, pass fields as keyword arguments, like
        `dspy.Example(question="...", answer="...")`. Use `base` when you
        already have a dictionary or another `Example` and want to copy its
        fields before adding or overriding a few values.

        Args:
            base: A dictionary or `Example` to copy fields from before applying
                `**kwargs`. When `None`, starts with no fields.
            **kwargs: Field names and values to store on the example. If a field
                appears in both `base` and `**kwargs`, the value from
                `**kwargs` wins.
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
