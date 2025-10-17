class Example:
    """A flexible data container for DSPy examples and training data.

    The `Example` class is the standard data format used in DSPy evaluation and optimization.

    Key features:
        - Dictionary-like access patterns (item access, iteration, etc.)
        - Flexible initialization from dictionaries, other `Example` instances, or keyword arguments
        - Input/output field separation for training data
        - Serialization support for saving/loading `Example` instances
        - Immutable operations that return new `Example` instances

    Examples:

        Basic usage with keyword arguments:

        ```python
        import dspy

        # Create an example with input and output fields
        example = dspy.Example(
            question="What is the capital of France?",
            answer="Paris",
        )
        print(example.question)  # "What is the capital of France?"
        print(example.answer)   # "Paris"
        ```

        Initialize from a dictionary:

        ```python
        data = {"question": "What is 2+2?", "answer": "4"}
        example = dspy.Example(data)
        print(example["question"])  # "What is 2+2?"
        ```

        Copy from another Example:

        ```python
        original = dspy.Example(question="Hello", answer="World")
        copy = dspy.Example(original)
        print(copy.question)  # "Hello"
        ```

        Working with input/output separation:

        ```python
        # Mark which fields are inputs for training
        example = dspy.Example(
            question="What is the weather?",
            answer="It's sunny",
        ).with_inputs("question")

        # Get only input fields
        inputs = example.inputs()
        print(inputs.question)  # "What is the weather?"

        # Get only output fields (labels)
        labels = example.labels()
        print(labels.answer)  # "It's sunny"
        ```

        Dictionary-like operations:

        ```python
        example = dspy.Example(name="Alice", age=30)

        # Check if key exists
        if "name" in example:
            print("Name field exists")

        # Get with default value
        city = example.get("city", "Unknown")
        print(city)  # "Unknown"
        ```
    """

    def __init__(self, base=None, **kwargs):
        """Initialize an Example instance.

        Args:
            base: Optional base data source. Can be:
                - Another Example instance (copies its data)
                - A dictionary (copies its key-value pairs)
                - None (creates empty Example)
            **kwargs: Additional key-value pairs to store in the Example.
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
        def convert_to_serializable(value):
            if hasattr(value, "toDict"):
                return value.toDict()
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
