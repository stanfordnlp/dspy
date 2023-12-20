from dspy.primitives.example import Example


class Prediction(Example):
    """
    The Prediction class is a subclass of the Example class.
    It represents a prediction made by the model.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a new instance of the Prediction class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        
        del self._demos
        del self._input_keys

        self._completions = None
    
    @classmethod
    def from_completions(cls, list_or_dict, signature=None):
        """
        Create a new instance of the Prediction class from completions.

        Args:
            list_or_dict (list or dict): The completions to use for creating the Prediction instance.
            signature (str, optional): The signature of the completions. Defaults to None.

        Returns:
            Prediction: A new instance of the Prediction class.
        """
        obj = cls()
        obj._completions = Completions(list_or_dict, signature=signature)
        obj._store = {k: v[0] for k, v in obj._completions.items()}

        return obj
    
    def __repr__(self):
        """
        Get a string representation of the Prediction instance.

        Returns:
            str: A string representation of the Prediction instance.
        """
        store_repr = ',\n    '.join(f"{k}={repr(v)}" for k, v in self._store.items())

        if self._completions is None or len(self._completions) == 1:
            return f"Prediction(\n    {store_repr}\n)"
        
        num_completions = len(self._completions)
        return f"Prediction(\n    {store_repr},\n    completions=Completions(...)\n) ({num_completions-1} completions omitted)"
        
    def __str__(self):
        """
        Get a string representation of the Prediction instance.

        Returns:
            str: A string representation of the Prediction instance.
        """
        return self.__repr__()

    @property
    def completions(self):
        """
        Get the completions of the Prediction instance.

        Returns:
            Completions: The completions of the Prediction instance.
        """
        return self._completions


class Completions:
    def __init__(self, list_or_dict, signature=None):
        self.signature = signature

        if isinstance(list_or_dict, list):
            kwargs = {}
            for arg in list_or_dict:
                for k, v in arg.items():
                    kwargs.setdefault(k, []).append(v)
        else:
            kwargs = list_or_dict

        assert all(isinstance(v, list) for v in kwargs.values()), "All values must be lists"

        if kwargs:
            length = len(next(iter(kwargs.values())))
            assert all(len(v) == length for v in kwargs.values()), "All lists must have the same length"

        self._completions = kwargs

    def items(self):
        return self._completions.items()

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError("Index out of range")
            
            return Prediction(**{k: v[key] for k, v in self._completions.items()})
        
        return self._completions[key]

    def __getattr__(self, name):
        if name in self._completions:
            return self._completions[name]
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __len__(self):
        # Return the length of the list for one of the keys
        # It assumes all lists have the same length
        return len(next(iter(self._completions.values())))

    def __contains__(self, key):
        return key in self._completions

    def __repr__(self):
        items_repr = ',\n    '.join(f"{k}={repr(v)}" for k, v in self._completions.items())
        return f"Completions(\n    {items_repr}\n)"

    def __str__(self):
        # return str(self._completions)
        return self.__repr__()
