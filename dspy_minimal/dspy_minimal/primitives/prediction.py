from typing import Any, List


class Prediction:
    """A simplified prediction class."""
    
    def __init__(self, **kwargs):
        self._data = kwargs
        self._lm_usage = None
        
    @classmethod
    def from_completions(cls, completions, signature=None):
        """Create a prediction from completions."""
        # Handle the new adapter format where completions is a list of dicts
        if isinstance(completions, list) and len(completions) > 0:
            # Take the first completion
            first_completion = completions[0]
            if isinstance(first_completion, dict):
                return cls(**first_completion)
            else:
                return cls(content=str(first_completion))
        elif isinstance(completions, dict):
            # Handle case where completions is a single dict
            return cls(**completions)
        else:
            # Fallback for other formats
            return cls(content=str(completions))
        
    def __getitem__(self, key):
        return self._data[key]
        
    def __setitem__(self, key, value):
        self._data[key] = value
        
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def __getattr__(self, name):
        """Allow attribute-style access to data fields."""
        if name in self._data:
            return self._data[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Allow attribute-style setting of data fields."""
        if name in ['_data', '_lm_usage']:
            super().__setattr__(name, value)
        else:
            self._data[name] = value
        
    def set_lm_usage(self, usage):
        """Set LM usage information."""
        self._lm_usage = usage
        
    def get_lm_usage(self):
        """Get LM usage information."""
        return self._lm_usage
        
    def __repr__(self):
        return f"Prediction({self._data})"
        
    def __str__(self):
        # Return just the first value if there's only one, otherwise return the full representation
        if len(self._data) == 1:
            return str(next(iter(self._data.values())))
        else:
            return self.__repr__() 